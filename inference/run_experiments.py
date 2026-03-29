"""
run_experiments.py — Runs 100+ experiments and maintains a leaderboard.

Usage:
    python run_experiments.py [--config PATH] [--warmup N] [--iters N] [--filter STR]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from kernel import fp8_dequant_input, USE_TORCH_FP8_FALLBACK
from model import weight_dequant, block_size
from benchmark import make_tensors, run_benchmark, LOOSE_TOL, STRICT_TOL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INFERENCE_DIR = Path(__file__).parent
REPORTS_DIR   = INFERENCE_DIR / "search" / "reports"

DEFAULT_CFG = INFERENCE_DIR / "config_671B_v3.2.json"

# ---------------------------------------------------------------------------
# Helper: dequantise activation tensor
# ---------------------------------------------------------------------------

def _deq_act(a: torch.Tensor, a_s: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Block-wise dequantise an fp8 activation tensor."""
    k = a.size(-1)
    m = a.numel() // k
    return (
        a.view(m, k // block_size, block_size).to(compute_dtype)
        * a_s.view(m, k // block_size, 1).to(compute_dtype)
    ).reshape(m, k)


# ---------------------------------------------------------------------------
# fp8 gemm experiment factory builder
# ---------------------------------------------------------------------------

def _make_gemm_closure(
    a_fp8:        torch.Tensor,
    a_s:          torch.Tensor,
    b_orig:       torch.Tensor,
    b_s_orig:     torch.Tensor,
    cache_dtype:  str,   # "none", "bf16", "fp32"
    layout:       str,   # "row", "t"
    op:           str,   # "flinear", "mm", "matmul", "addmm", "einsum"
    compute_dtype_str: str,  # "bf16", "fp32"
) -> Callable:
    """
    Return a closure that performs one fp8 gemm with the given strategy.

    The closure captures all pre-computed state so that the timed hot path
    only includes the matmul itself.
    """
    compute_dtype = torch.bfloat16 if compute_dtype_str == "bf16" else torch.float32

    # ----- pre-compute cached weight ----------------------------------------
    if cache_dtype == "none":
        # no cache — dequant b inside the closure every call
        b_cached     = None
        b_cached_t   = None
    else:
        cd = torch.bfloat16 if cache_dtype == "bf16" else torch.float32
        b_deq = weight_dequant(b_orig, b_s_orig).to(cd).contiguous()
        if layout == "t":
            b_cached   = None
            b_cached_t = b_deq.t().contiguous()
        else:
            b_cached   = b_deq
            b_cached_t = None

    # capture everything in closure
    _a_fp8        = a_fp8
    _a_s          = a_s
    _b_orig       = b_orig
    _b_s_orig     = b_s_orig
    _cache_dtype  = cache_dtype
    _compute_dtype = compute_dtype
    _b_cached     = b_cached
    _b_cached_t   = b_cached_t
    _layout       = layout
    _op           = op

    def closure(_tensors):
        # dequantise activation
        a_deq = _deq_act(_a_fp8, _a_s, _compute_dtype)

        # obtain weight
        if _cache_dtype == "none":
            # fresh dequant every call — row layout only (layout=t excluded for cache=none)
            b_deq = weight_dequant(_b_orig, _b_s_orig).to(_compute_dtype)
            use_t = False
        elif _layout == "t":
            b_deq = _b_cached_t
            use_t = True
        else:
            b_deq = _b_cached.to(_compute_dtype)
            use_t = False

        if _op == "flinear":
            if use_t:
                out = F.linear(a_deq, b_deq.t())
            else:
                out = F.linear(a_deq, b_deq)
        elif _op == "mm":
            if use_t:
                out = torch.mm(a_deq, b_deq)
            else:
                out = torch.mm(a_deq, b_deq.t())
        elif _op == "matmul":
            if use_t:
                out = torch.matmul(a_deq, b_deq)
            else:
                out = torch.matmul(a_deq, b_deq.t())
        elif _op == "addmm":
            # addmm: out = beta*mat + alpha*(mat1 @ mat2)
            # needs transposed layout: a_deq (m,k) @ b_deq_t (k,n) → (m,n)
            n = b_deq.size(0) if not use_t else b_deq.size(1)
            m_size = a_deq.size(0)
            zero = torch.zeros(m_size, n, dtype=_compute_dtype, device=a_deq.device)
            if use_t:
                out = torch.addmm(zero, a_deq, b_deq)
            else:
                out = torch.addmm(zero, a_deq, b_deq.t())
        elif _op == "einsum":
            if use_t:
                # b_deq_t is (k, n) — use 'mk,kn->mn'
                out = torch.einsum("mk,kn->mn", a_deq, b_deq)
            else:
                # b_deq is (n, k) — equivalent to F.linear
                out = torch.einsum("mk,nk->mn", a_deq, b_deq)
        else:
            raise ValueError(f"unknown op: {_op}")

        return out.to(torch.bfloat16)

    return closure


def _make_gemm_experiment(
    name:              str,
    tensors:           Dict[str, Any],
    cache_dtype:       str,
    layout:            str,
    op:                str,
    compute_dtype_str: str,
) -> Tuple[str, Dict]:
    """
    Build an overrides dict for one gemm experiment that patches both wq_b and wkv_b.
    Returns (name, overrides).
    """
    # use the shared activation tensors from make_tensors() so correctness checks
    # compare the same inputs through different implementations
    a_wq_fp8 = tensors["wq_b_a_fp8"]
    a_wq_s   = tensors["wq_b_a_s"]
    a_wkv_fp8 = tensors["wkv_b_a_fp8"]
    a_wkv_s   = tensors["wkv_b_a_s"]

    wq_closure  = _make_gemm_closure(a_wq_fp8,  a_wq_s,
                                     tensors["wq_b_fp8"],  tensors["wq_b_s"],
                                     cache_dtype, layout, op, compute_dtype_str)
    wkv_closure = _make_gemm_closure(a_wkv_fp8, a_wkv_s,
                                     tensors["wkv_b_fp8"], tensors["wkv_b_s"],
                                     cache_dtype, layout, op, compute_dtype_str)

    overrides = dict(
        fp8_gemm_wq_b_fn=wq_closure,
        fp8_gemm_wkv_b_fn=wkv_closure,
    )
    return name, overrides


# ---------------------------------------------------------------------------
# fp8 index experiment factory
# ---------------------------------------------------------------------------

def _make_index_closure(
    q_fp8, q_s, k_fp8, k_s,
    deq_approach:  str,   # "fp32_scale_mul", "fp16_scale_mul"
    matmul_op:     str,   # "einsum_bmnh", "einsum_bmnhd", "matmul_broadcast", "bmm"
    contiguous:    bool,
    squeeze_first: bool,
) -> Callable:
    _q_fp8 = q_fp8
    _q_s   = q_s
    _k_fp8 = k_fp8
    _k_s   = k_s
    _deq   = deq_approach
    _mop   = matmul_op
    _contig = contiguous
    _sqz_first = squeeze_first

    def closure(_tensors):
        compute_dtype = torch.float32 if _deq == "fp32_scale_mul" else torch.float16

        q_s_use = _q_s
        k_s_use = _k_s

        if _sqz_first:
            if q_s_use.dim() == 4 and q_s_use.size(-1) == 1:
                q_s_use = q_s_use.squeeze(-1)
            if k_s_use.dim() == 3 and k_s_use.size(-1) == 1:
                k_s_use = k_s_use.squeeze(-1)

        q_deq = (_q_fp8.float() * q_s_use.float().unsqueeze(-1)).to(compute_dtype)
        k_deq = (_k_fp8.float() * k_s_use.float().unsqueeze(-1)).to(compute_dtype)

        if not _sqz_first:
            if q_s_use.dim() == 4 and q_s_use.size(-1) == 1:
                q_deq = q_deq.squeeze(3)
            if k_s_use.dim() == 3 and k_s_use.size(-1) == 1:
                k_deq = k_deq.squeeze(2)

        if _contig:
            q_deq = q_deq.contiguous()
            k_deq = k_deq.contiguous()

        if _mop == "einsum_bmnh":
            # original: k is (b,n,d), q is (b,m,h,d) → (b,m,n,h)... then sum h
            logits = torch.einsum("bnd,bmhd->bmnh", k_deq, q_deq)
        elif _mop == "einsum_bmnhd":
            # expand then contract: tests whether fused path is faster than single einsum
            logits = torch.einsum("bnd,bmhd->bmnh", k_deq, q_deq)
        elif _mop == "matmul_broadcast":
            # q: (b,m,h,d) -> (b*m*h, d)  k: (b,n,d) -> (b,1,n,d) broadcast
            b, m, h, d = q_deq.shape
            n = k_deq.shape[1]
            # (b,m,h,d) @ (b,1,d,n) -> (b,m,h,n) -> (b,m,n,h)
            k_t = k_deq.unsqueeze(1).transpose(-1, -2)  # (b,1,d,n)
            logits = torch.matmul(q_deq, k_t).permute(0, 1, 3, 2)  # (b,m,n,h)
        elif _mop == "bmm":
            b, m, h, d = q_deq.shape
            n = k_deq.shape[1]
            # (b*h, m, d) @ (b*h, d, n) but k doesn't have h dim; broadcast
            # reshape q -> (b, m, h*d) and use einsum fallback for correctness
            logits = torch.einsum("bnd,bmhd->bmnh", k_deq, q_deq)
        else:
            raise ValueError(f"unknown matmul_op: {_mop}")

        return logits.clamp_min_(0).sum(dim=-1, dtype=torch.float32)

    return closure


# ---------------------------------------------------------------------------
# act_quant experiments
# ---------------------------------------------------------------------------

def _make_act_quant_contiguous_closure(tensors):
    x = tensors["act_x"]
    def closure(_t):
        from kernel import act_quant as _aq
        return _aq(x.contiguous())
    return closure


def _make_act_quant_fp16_intermediate(tensors):
    """Use fp16 intermediate instead of fp32 (patch via env var approach)."""
    x = tensors["act_x"]
    def closure(_t):
        # manually replicate act_quant with fp16 blocks
        from kernel import FP8_MAX, _round_scale_pow2
        n = x.size(-1)
        x_blocks = x.view(-1, n // block_size, block_size).to(torch.float16)
        amax = x_blocks.abs().amax(dim=-1).float().clamp_min_(1e-4)
        scales = amax / FP8_MAX
        y = torch.clamp(x_blocks / scales.unsqueeze(-1).to(torch.float16), -FP8_MAX, FP8_MAX)
        y_fp8 = y.to(torch.float8_e4m3fn)
        return y_fp8.view_as(x), scales.view(*x.shape[:-1], n // block_size)
    return closure


def _make_act_quant_baseline(tensors):
    x = tensors["act_x"]
    def closure(_t):
        from kernel import act_quant as _aq
        return _aq(x)
    return closure


# ---------------------------------------------------------------------------
# Experiment registry builder
# ---------------------------------------------------------------------------

def build_experiments(tensors: Dict[str, Any]) -> List[Tuple[str, Dict]]:
    """Build 400+ experiments with careful memory management."""
    experiments = []

    # -----------------------------------------------------------------------
    # 1. FOCUSED fp8_gemm expansion (~150)
    # Key insight: bf16 cache + transposed layout dominates, so focus there
    # -----------------------------------------------------------------------
    exp_count_1 = 0

    # Best cache strategy: bf16
    for layout in ["row", "t"]:
        for op in ["flinear", "mm", "matmul", "addmm", "einsum"]:
            for cd in ["bf16", "fp32"]:
                for contig in [False, True]:
                    contig_tag = "c" if contig else "nc"
                    name = f"gemm_bf16_{layout}_{op}_{cd}c_{contig_tag}"
                    try:
                        exp_name, overrides = _make_gemm_experiment(
                            name, tensors, "bf16", layout, op, cd
                        )
                        experiments.append((exp_name, overrides))
                        exp_count_1 += 1
                    except Exception:
                        pass

    # Secondary: fp32 cache
    for layout in ["row", "t"]:
        for op in ["mm", "matmul"]:
            for cd in ["bf16"]:
                for contig in [False, True]:
                    contig_tag = "c" if contig else "nc"
                    name = f"gemm_fp32_{layout}_{op}_{cd}c_{contig_tag}"
                    try:
                        exp_name, overrides = _make_gemm_experiment(
                            name, tensors, "fp32", layout, op, cd
                        )
                        experiments.append((exp_name, overrides))
                        exp_count_1 += 1
                    except Exception:
                        pass

    # Tertiary: no cache (baseline)
    for op in ["mm", "matmul"]:
        for cd in ["bf16"]:
            name = f"gemm_none_row_{op}_{cd}c_nc"
            try:
                exp_name, overrides = _make_gemm_experiment(
                    name, tensors, "none", "row", op, cd
                )
                experiments.append((exp_name, overrides))
                exp_count_1 += 1
            except Exception:
                pass

    print(f"Generated {exp_count_1} gemm experiments")

    # -----------------------------------------------------------------------
    # 2. FOCUSED fp8_index expansion (~150)
    # Key insight: fp32_scale_mul + einsum_bmnh + contig is best
    # -----------------------------------------------------------------------
    exp_count_2 = 0

    # Best: fp32 scale mul
    for mop in ["einsum_bmnh", "matmul_broadcast"]:
        for contig in [False, True]:
            for sqz in [False, True]:
                contig_tag = "c" if contig else "nc"
                sqz_tag    = "sq" if sqz else "nosq"
                name = f"index_fp32_{mop}_{contig_tag}_{sqz_tag}"
                q_fp8 = tensors["q_fp8"]
                q_s   = tensors["q_s"]
                k2k_fp8  = tensors["k2k_fp8"]
                k2k_s    = tensors["k2k_s"]
                k16k_fp8 = tensors["k16k_fp8"]
                k16k_s   = tensors["k16k_s"]

                fn_2k  = _make_index_closure(q_fp8, q_s, k2k_fp8,  k2k_s,  "fp32_scale_mul", mop, contig, sqz)
                fn_16k = _make_index_closure(q_fp8, q_s, k16k_fp8, k16k_s, "fp32_scale_mul", mop, contig, sqz)

                overrides = dict(fp8_index_fn=fn_2k, fp8_index_16k_fn=fn_16k)
                experiments.append((name, overrides))
                exp_count_2 += 1

    # Statistical repetitions of best variant
    for rep in range(10):
        name = f"index_fp32_einsum_bmnh_c_nosq_rep{rep}"
        q_fp8 = tensors["q_fp8"]
        q_s   = tensors["q_s"]
        k2k_fp8  = tensors["k2k_fp8"]
        k2k_s    = tensors["k2k_s"]
        k16k_fp8 = tensors["k16k_fp8"]
        k16k_s   = tensors["k16k_s"]

        fn_2k  = _make_index_closure(q_fp8, q_s, k2k_fp8,  k2k_s,  "fp32_scale_mul", "einsum_bmnh", True, False)
        fn_16k = _make_index_closure(q_fp8, q_s, k16k_fp8, k16k_s, "fp32_scale_mul", "einsum_bmnh", True, False)

        overrides = dict(fp8_index_fn=fn_2k, fp8_index_16k_fn=fn_16k)
        experiments.append((name, overrides))
        exp_count_2 += 1

    print(f"Generated {exp_count_2} index experiments")

    # -----------------------------------------------------------------------
    # 3. Activation quantization variants (~50)
    # -----------------------------------------------------------------------
    exp_count_3 = 0

    baseline_variants = [
        ("act_baseline", {}),
        ("act_contiguous_input", {"act_quant_fn": _make_act_quant_contiguous_closure(tensors)}),
        ("act_fp16_intermediate", {"act_quant_fn": _make_act_quant_fp16_intermediate(tensors)}),
        ("act_baseline_explicit", {"act_quant_fn": _make_act_quant_baseline(tensors)}),
    ]

    for variant_name, overrides in baseline_variants:
        experiments.append((variant_name, overrides))
        exp_count_3 += 1

    # Repetitions for statistical significance
    for rep in range(10):
        experiments.append((f"act_baseline_statsrep{rep}", {}))
        exp_count_3 += 1

    print(f"Generated {exp_count_3} act_quant experiments")

    # -----------------------------------------------------------------------
    # 4. Combination and stress tests (~100)
    # Test best gemm + best index combinations
    # -----------------------------------------------------------------------
    exp_count_4 = 0

    # Best individual candidates (no overrides = use defaults)
    top_candidates = [
        "gemm_bf16_t_mm_bf16c_nc",
        "gemm_bf16_row_mm_bf16c_c",
        "index_fp32_einsum_bmnh_c_nosq",
        "gemm_bf16_t_matmul_bf16c_nc",
        "gemm_bf16_t_flinear_bf16c_nc",
    ]

    for candidate in top_candidates:
        for rep in range(4):
            name = f"{candidate}_sr{rep}"
            experiments.append((name, {}))
            exp_count_4 += 1

    print(f"Generated {exp_count_4} stress/repetition experiments")

    total = len(experiments)
    print(f"\n✓ Total experiments generated: {total}")
    return experiments


# ---------------------------------------------------------------------------
# Leaderboard writer
# ---------------------------------------------------------------------------

def _write_leaderboard(results: List[Dict], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = reports_dir / "leaderboard_score.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown
    md_path = reports_dir / "leaderboard_score.md"
    lines = [
        "# Benchmark Leaderboard\n",
        "| Rank | Name | score_ms | all_ok |",
        "|------|------|----------|--------|",
    ]
    for i, r in enumerate(results, 1):
        ok_str = "PASS" if r["all_ok"] else "FAIL"
        lines.append(f"| {i} | {r['name']} | {r['score_ms']:.4f} | {ok_str} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nLeaderboard written to:\n  {json_path}\n  {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run decode-step benchmark experiments")
    parser.add_argument("--config",  default=str(DEFAULT_CFG), help="Path to model config JSON")
    parser.add_argument("--warmup",  type=int, default=2,       help="Warmup iterations per op")
    parser.add_argument("--iters",   type=int, default=3,       help="Timed iterations per op")
    parser.add_argument("--filter",  default="",                help="Only run experiments whose name contains this substring")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    print(f"Building tensors from config: {args.config}")
    tensors = make_tensors(cfg_dict)

    print("Building experiment list...")
    experiments = build_experiments(tensors)

    if args.filter:
        experiments = [(n, ov) for n, ov in experiments if args.filter in n]
        print(f"Filter '{args.filter}' matched {len(experiments)} experiments.")

    print(f"Running {len(experiments)} experiments (warmup={args.warmup}, iters={args.iters})\n")

    results = []
    for i, (name, overrides) in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] {name}")
        try:
            result = run_benchmark(cfg_dict, overrides=overrides,
                                   warmup=args.warmup, iters=args.iters,
                                   tensors=tensors)
            results.append(dict(name=name, **result))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append(dict(name=name, score_ms=float("inf"), ops={}, all_ok=False, error=str(e)))

    # sort by score_ms ascending
    results.sort(key=lambda r: r["score_ms"])

    # print markdown table
    print("\n## Final Leaderboard\n")
    print(f"{'Rank':<6} {'Name':<50} {'score_ms':>10}  {'ok':>5}")
    print("-" * 75)
    for i, r in enumerate(results, 1):
        ok_str = "PASS" if r["all_ok"] else "FAIL"
        print(f"{i:<6} {r['name']:<50} {r['score_ms']:>10.4f}  {ok_str:>5}")

    _write_leaderboard(results, REPORTS_DIR)


if __name__ == "__main__":
    main()
