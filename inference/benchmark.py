"""
benchmark.py — Injectable decode-step micro-benchmark for DeepSeek-V3.2-Exp.

Measures the 5 ops that dominate a single decode step and returns a single
`score_ms` (geometric mean of per-op median times, lower is better).
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from kernel import act_quant, fp8_index, USE_TORCH_FP8_FALLBACK
from model import weight_dequant, block_size

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
STRICT_TOL = 1e-3   # index ops
LOOSE_TOL  = 1.0    # gemm / quant ops

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_value(cfg_dict: Dict[str, Any], key: str, default):
    return cfg_dict.get(key, default)


def _shapes_from_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    dim              = _cfg_value(cfg_dict, "dim",              7168)
    q_lora_rank      = _cfg_value(cfg_dict, "q_lora_rank",      1536)
    kv_lora_rank     = _cfg_value(cfg_dict, "kv_lora_rank",     512)
    n_heads          = _cfg_value(cfg_dict, "n_heads",          128)
    qk_nope_head_dim = _cfg_value(cfg_dict, "qk_nope_head_dim", 128)
    qk_rope_head_dim = _cfg_value(cfg_dict, "qk_rope_head_dim", 64)
    v_head_dim       = _cfg_value(cfg_dict, "v_head_dim",       128)
    index_n_heads    = _cfg_value(cfg_dict, "index_n_heads",    64)
    index_head_dim   = _cfg_value(cfg_dict, "index_head_dim",   128)

    return dict(
        dim=dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        n_heads=n_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        # derived
        wq_b_n=n_heads * (qk_nope_head_dim + qk_rope_head_dim),  # 128 * 192 = 24576
        wkv_b_n=n_heads * (qk_nope_head_dim + v_head_dim),       # 128 * 256 = 32768
    )


# ---------------------------------------------------------------------------
# Tensor factory
# ---------------------------------------------------------------------------

def make_tensors(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-create all input and weight tensors needed by the benchmark.

    Returns a dict with keys:
        act_x                    — (1, dim) bfloat16
        wq_b_fp8, wq_b_s         — fp8 weight + scale for wq_b gemm
        wkv_b_fp8, wkv_b_s      — fp8 weight + scale for wkv_b gemm
        q_fp8, q_s               — (1, 1, index_n_heads, index_head_dim) fp8 + scale
        k2k_fp8, k2k_s           — ctx=2048 key cache fp8 + scale
        k16k_fp8, k16k_s         — ctx=16384 key cache fp8 + scale
        shapes                   — the shapes dict
    """
    s = _shapes_from_cfg(cfg_dict)
    device = "cuda"
    bfp = torch.bfloat16

    def make_fp8_weight(n, k):
        w = torch.randn(n, k, dtype=bfp, device=device)
        # scale shape: (n//block_size, k//block_size)
        scale = torch.rand(n // block_size, k // block_size, dtype=torch.float32, device=device).add_(0.1)
        fp8 = w.to(torch.float8_e4m3fn)
        return fp8, scale

    def make_fp8_index_cache(b, n_ctx, d):
        raw = torch.randn(b, n_ctx, d, dtype=bfp, device=device)
        fp8 = raw.to(torch.float8_e4m3fn)
        scale = torch.rand(b, n_ctx, dtype=torch.float32, device=device).add_(0.1)
        return fp8, scale

    # act_quant input
    act_x = torch.randn(1, s["dim"], dtype=bfp, device=device)

    # fp8 gemm weights
    wq_b_fp8,  wq_b_s  = make_fp8_weight(s["wq_b_n"],  s["q_lora_rank"])
    wkv_b_fp8, wkv_b_s = make_fp8_weight(s["wkv_b_n"], s["kv_lora_rank"])

    # shared activation tensors for gemm correctness checks — same inputs used by
    # both reference and all experiment closures so diffs reflect precision only
    wq_b_a_fp8,  wq_b_a_s  = act_quant(torch.randn(1, s["q_lora_rank"],  dtype=bfp, device=device))
    wkv_b_a_fp8, wkv_b_a_s = act_quant(torch.randn(1, s["kv_lora_rank"], dtype=bfp, device=device))

    # fp8 index
    q_raw  = torch.randn(1, 1, s["index_n_heads"], s["index_head_dim"], dtype=bfp, device=device)
    q_fp8  = q_raw.to(torch.float8_e4m3fn)
    q_s    = torch.rand(1, 1, s["index_n_heads"], dtype=torch.float32, device=device).add_(0.1)

    k2k_fp8,  k2k_s  = make_fp8_index_cache(1, 2048,  s["index_head_dim"])
    k16k_fp8, k16k_s = make_fp8_index_cache(1, 16384, s["index_head_dim"])

    return dict(
        act_x=act_x,
        wq_b_fp8=wq_b_fp8,   wq_b_s=wq_b_s,
        wkv_b_fp8=wkv_b_fp8, wkv_b_s=wkv_b_s,
        wq_b_a_fp8=wq_b_a_fp8,   wq_b_a_s=wq_b_a_s,
        wkv_b_a_fp8=wkv_b_a_fp8, wkv_b_a_s=wkv_b_a_s,
        q_fp8=q_fp8,   q_s=q_s,
        k2k_fp8=k2k_fp8,   k2k_s=k2k_s,
        k16k_fp8=k16k_fp8, k16k_s=k16k_s,
        shapes=s,
    )


# ---------------------------------------------------------------------------
# Reference implementations (used for correctness checks)
# ---------------------------------------------------------------------------

def _ref_act_quant(tensors):
    return act_quant(tensors["act_x"])


def _ref_fp8_gemm_wq_b(tensors):
    from kernel import fp8_gemm
    return fp8_gemm(tensors["wq_b_a_fp8"], tensors["wq_b_a_s"],
                    tensors["wq_b_fp8"],   tensors["wq_b_s"])


def _ref_fp8_gemm_wkv_b(tensors):
    from kernel import fp8_gemm
    return fp8_gemm(tensors["wkv_b_a_fp8"], tensors["wkv_b_a_s"],
                    tensors["wkv_b_fp8"],   tensors["wkv_b_s"])


def _ref_fp8_index(q_fp8, q_s, k_fp8, k_s):
    return fp8_index(q_fp8, q_s, k_fp8, k_s)


# ---------------------------------------------------------------------------
# Default hot-path implementations (match what the model actually runs)
# ---------------------------------------------------------------------------

def _default_act_quant_fn(tensors):
    return act_quant(tensors["act_x"])


def _make_default_fp8_gemm_wq_b(tensors):
    """Returns a closure that runs the wq_b gemm on the shared pre-quantised activation."""
    a_fp8 = tensors["wq_b_a_fp8"]
    a_s   = tensors["wq_b_a_s"]
    b_fp8 = tensors["wq_b_fp8"]
    b_s   = tensors["wq_b_s"]

    def fn(_tensors):
        from kernel import fp8_gemm
        return fp8_gemm(a_fp8, a_s, b_fp8, b_s)

    return fn


def _make_default_fp8_gemm_wkv_b(tensors):
    a_fp8 = tensors["wkv_b_a_fp8"]
    a_s   = tensors["wkv_b_a_s"]
    b_fp8 = tensors["wkv_b_fp8"]
    b_s   = tensors["wkv_b_s"]

    def fn(_tensors):
        from kernel import fp8_gemm
        return fp8_gemm(a_fp8, a_s, b_fp8, b_s)

    return fn


def _make_default_fp8_index(q_fp8, q_s, k_fp8, k_s):
    def fn(_tensors):
        return fp8_index(q_fp8, q_s, k_fp8, k_s)
    return fn


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_op(fn, tensors, warmup: int, iters: int) -> float:
    """Returns median wall time in milliseconds."""
    for _ in range(warmup):
        fn(tensors)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(tensors)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check(name: str, out, ref, tol: float) -> bool:
    try:
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(ref, tuple):
            ref = ref[0]
        if out is None:
            return False
        diff = (out.float() - ref.float()).abs().max().item()
        ok = diff <= tol
        if not ok:
            print(f"  [FAIL] {name}: max_diff={diff:.4f} tol={tol}")
        return ok
    except Exception as e:
        print(f"  [ERROR] {name} correctness check: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg_dict: Dict[str, Any],
    overrides: Dict[str, Any] = {},
    warmup: int = 2,
    iters: int = 3,
    tensors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the 5-op decode-step benchmark.

    Parameters
    ----------
    cfg_dict  : model config dict (keys from config_671B_v3.2.json)
    overrides : dict mapping op name → closure with signature fn(tensors) → tensor
                Keys: act_quant_fn, fp8_gemm_wq_b_fn, fp8_gemm_wkv_b_fn, fp8_index_fn
                Each closure should have pre-cached state baked in.
    warmup    : number of warmup iterations per op
    iters     : number of timed iterations per op
    tensors   : pre-created tensors from make_tensors(); created here if None

    Returns
    -------
    dict with keys: score_ms, ops (per-op dict), all_ok
    """
    torch.set_default_dtype(torch.bfloat16)

    if tensors is None:
        tensors = make_tensors(cfg_dict)

    s = tensors["shapes"]

    # ---- build default fns (will be replaced by overrides) ----------------
    default_wq_b_fn  = _make_default_fp8_gemm_wq_b(tensors)
    default_wkv_b_fn = _make_default_fp8_gemm_wkv_b(tensors)
    default_idx2k_fn = _make_default_fp8_index(tensors["q_fp8"], tensors["q_s"],
                                                tensors["k2k_fp8"],  tensors["k2k_s"])
    default_idx16k_fn= _make_default_fp8_index(tensors["q_fp8"], tensors["q_s"],
                                                tensors["k16k_fp8"], tensors["k16k_s"])

    act_quant_fn      = overrides.get("act_quant_fn",      lambda t: act_quant(t["act_x"]))
    fp8_gemm_wq_b_fn  = overrides.get("fp8_gemm_wq_b_fn",  default_wq_b_fn)
    fp8_gemm_wkv_b_fn = overrides.get("fp8_gemm_wkv_b_fn", default_wkv_b_fn)
    fp8_index_2k_fn   = overrides.get("fp8_index_fn",      default_idx2k_fn)
    # 16k index uses same override key but a separate default
    fp8_index_16k_fn  = overrides.get("fp8_index_16k_fn",  default_idx16k_fn)

    # ---- reference outputs for correctness --------------------------------
    ref_act  = act_quant(tensors["act_x"])
    ref_wq   = default_wq_b_fn(tensors)
    ref_wkv  = default_wkv_b_fn(tensors)
    ref_idx2k  = default_idx2k_fn(tensors)
    ref_idx16k = default_idx16k_fn(tensors)

    # ---- correctness -------------------------------------------------------
    checks = {
        "act_quant":    _check("act_quant",    act_quant_fn(tensors),      ref_act,   LOOSE_TOL),
        "fp8_gemm_wq_b": _check("fp8_gemm_wq_b", fp8_gemm_wq_b_fn(tensors), ref_wq,  LOOSE_TOL),
        "fp8_gemm_wkv_b": _check("fp8_gemm_wkv_b", fp8_gemm_wkv_b_fn(tensors), ref_wkv, LOOSE_TOL),
        "fp8_index_2k":  _check("fp8_index_2k",  fp8_index_2k_fn(tensors),  ref_idx2k,  STRICT_TOL),
        "fp8_index_16k": _check("fp8_index_16k", fp8_index_16k_fn(tensors), ref_idx16k, STRICT_TOL),
    }
    all_ok = all(checks.values())

    # ---- timing ------------------------------------------------------------
    op_fns = [
        ("act_quant",      act_quant_fn),
        ("fp8_gemm_wq_b",  fp8_gemm_wq_b_fn),
        ("fp8_gemm_wkv_b", fp8_gemm_wkv_b_fn),
        ("fp8_index_2k",   fp8_index_2k_fn),
        ("fp8_index_16k",  fp8_index_16k_fn),
    ]

    ops = {}
    for op_name, fn in op_fns:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Clear cache between ops
        ms = _time_op(fn, tensors, warmup, iters)
        ops[op_name] = dict(ms=ms, ok=checks[op_name])

    # geometric mean
    log_sum = sum(math.log(v["ms"]) for v in ops.values())
    score_ms = math.exp(log_sum / len(ops))

    # ---- pretty table ------------------------------------------------------
    col_w = 20
    print(f"\n{'Op':<{col_w}} {'median_ms':>10}  {'ok':>5}")
    print("-" * (col_w + 20))
    for op_name, info in ops.items():
        status = "PASS" if info["ok"] else "FAIL"
        print(f"{op_name:<{col_w}} {info['ms']:>10.4f}  {status:>5}")
    print("-" * (col_w + 20))
    valid_str = "" if all_ok else "  [INVALID — tolerance failures]"
    print(f"SCORE: {score_ms:.4f} ms{valid_str}\n")

    return dict(score_ms=score_ms, ops=ops, all_ok=all_ok)
