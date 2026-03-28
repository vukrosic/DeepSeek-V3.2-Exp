#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel import act_quant, fp8_dequant_input, fp8_gemm, fp8_gemm_cached_weight  # noqa: E402
from model import ModelArgs, block_size, weight_dequant  # noqa: E402


TARGETS = {
    "mla_wq_b": {
        "baseline_kind": "fp8_gemm",
        "in_features": lambda args: args.q_lora_rank,
        "out_features": lambda args: args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim),
    },
    "mla_wkv_b": {
        "baseline_kind": "cached_bf16",
        "in_features": lambda args: args.kv_lora_rank,
        "out_features": lambda args: args.n_heads * (args.qk_nope_head_dim + args.v_head_dim),
    },
    "indexer_wq_b": {
        "baseline_kind": "cached_fp32",
        "in_features": lambda args: args.q_lora_rank,
        "out_features": lambda args: args.index_n_heads * args.index_head_dim,
    },
    "indexer_wk": {
        "baseline_kind": "cached_fp32",
        "in_features": lambda args: args.dim,
        "out_features": lambda args: args.index_head_dim,
    },
    "mla_wq_a": {
        "baseline_kind": "cached_fp32",
        "in_features": lambda args: args.dim,
        "out_features": lambda args: args.q_lora_rank,
    },
    "mla_wkv_a": {
        "baseline_kind": "cached_fp32",
        "in_features": lambda args: args.dim,
        "out_features": lambda args: args.kv_lora_rank + args.qk_rope_head_dim,
    },
}


def load_args(config_path: Path) -> ModelArgs:
    with open(config_path) as f:
        return ModelArgs(**json.load(f))


def benchmark_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    times_ms.sort()
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "p95_ms": times_ms[min(len(times_ms) - 1, math.ceil(len(times_ms) * 0.95) - 1)],
    }


def tensor_check(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    exact = torch.equal(a, b)
    out: Dict[str, Any] = {"exact": bool(exact)}
    if exact:
        out["max_abs_diff"] = 0.0
        out["mean_abs_diff"] = 0.0
        return out
    diff = (a.float() - b.float()).abs()
    out["max_abs_diff"] = float(diff.max().item())
    out["mean_abs_diff"] = float(diff.mean().item())
    return out


def gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def make_candidates() -> List[Dict[str, str]]:
    weight_variants = [
        "dequant_each_call_row",
        "cache_bf16_row",
        "cache_fp32_row",
        "cache_bf16_t",
        "cache_fp32_t",
    ]
    input_variants = [
        "a_float",
        "a_float_contig",
        "a_bf16",
        "a_bf16_contig",
    ]
    op_variants = [
        "flinear",
        "mm",
        "matmul",
        "addmm",
        "einsum",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for weight_variant in weight_variants:
        for input_variant in input_variants:
            for op_variant in op_variants:
                idx += 1
                out.append(
                    {
                        "id": f"cand_{idx:03d}",
                        "weight_variant": weight_variant,
                        "input_variant": input_variant,
                        "op_variant": op_variant,
                    }
                )
    assert len(out) == 100
    return out


def materialize_input(a_fp8: torch.Tensor, a_scale: torch.Tensor, variant: str) -> torch.Tensor:
    a = fp8_dequant_input(a_fp8, a_scale).view(a_fp8.numel() // a_fp8.size(-1), a_fp8.size(-1))
    if variant == "a_float":
        return a.float()
    if variant == "a_float_contig":
        return a.float().contiguous()
    if variant == "a_bf16":
        return a.to(torch.bfloat16)
    if variant == "a_bf16_contig":
        return a.to(torch.bfloat16).contiguous()
    raise ValueError(f"unknown input variant: {variant}")


def target_dtype(input_variant: str, weight_variant: str) -> torch.dtype:
    if input_variant.startswith("a_float") or "fp32" in weight_variant:
        return torch.float32
    return torch.bfloat16


def cast_pair(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return a.to(dtype), b.to(dtype)


def run_op(a: torch.Tensor, weight_variant: str, op_variant: str, cache: Dict[str, Any]) -> torch.Tensor:
    if weight_variant == "dequant_each_call_row":
        if op_variant in {"flinear", "einsum"}:
            w = cache["row_dequant_each_call"]()
        else:
            w = cache["row_dequant_each_call"]().t()
    elif weight_variant == "cache_bf16_row":
        w = cache["bf16_row"] if op_variant in {"flinear", "einsum"} else cache["bf16_row"].t()
    elif weight_variant == "cache_fp32_row":
        w = cache["fp32_row"] if op_variant in {"flinear", "einsum"} else cache["fp32_row"].t()
    elif weight_variant == "cache_bf16_t":
        w = cache["bf16_t"].t() if op_variant in {"flinear", "einsum"} else cache["bf16_t"]
    elif weight_variant == "cache_fp32_t":
        w = cache["fp32_t"].t() if op_variant in {"flinear", "einsum"} else cache["fp32_t"]
    else:
        raise ValueError(f"unknown weight variant: {weight_variant}")

    dtype = target_dtype(cache["input_variant"], weight_variant)
    a_cast, w_cast = cast_pair(a, w, dtype)
    if op_variant == "flinear":
        out = F.linear(a_cast, w_cast)
    elif op_variant == "mm":
        out = a_cast @ w_cast
    elif op_variant == "matmul":
        out = torch.matmul(a_cast, w_cast)
    elif op_variant == "addmm":
        bias = cache["zeros"].to(dtype)
        out = torch.addmm(bias, a_cast, w_cast)
    elif op_variant == "einsum":
        out = torch.einsum("mk,nk->mn", a_cast, w_cast)
    else:
        raise ValueError(f"unknown op variant: {op_variant}")
    return out.to(torch.get_default_dtype())


def evaluate_candidate(
    candidate: Dict[str, str],
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    ref: torch.Tensor,
    cache: Dict[str, Any],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    cache["input_variant"] = candidate["input_variant"]

    def fn() -> torch.Tensor:
        a = materialize_input(a_fp8, a_scale, candidate["input_variant"])
        return run_op(a, candidate["weight_variant"], candidate["op_variant"], cache)

    out = fn()
    bench = benchmark_cuda(fn, warmup, iters)
    check = tensor_check(out, ref)
    return {
        **candidate,
        "exact": check["exact"],
        "max_abs_diff": check["max_abs_diff"],
        "mean_abs_diff": check["mean_abs_diff"],
        "mean_ms": bench["mean_ms"],
        "median_ms": bench["median_ms"],
        "min_ms": bench["min_ms"],
        "max_ms": bench["max_ms"],
        "p95_ms": bench["p95_ms"],
    }


def family_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for family in [
        "dequant_each_call_row",
        "cache_bf16_row",
        "cache_fp32_row",
        "cache_bf16_t",
        "cache_fp32_t",
    ]:
        family_rows = [r for r in results if r["weight_variant"] == family]
        exact_rows = [r for r in family_rows if r["exact"]]
        exact_rows.sort(key=lambda r: r["mean_ms"])
        out[family] = {
            "exact_count": len(exact_rows),
            "total_count": len(family_rows),
            "best_exact": exact_rows[0] if exact_rows else None,
        }
    return out


def make_baseline(
    baseline_kind: str,
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bf16_row: torch.Tensor,
    fp32_row: torch.Tensor,
) -> tuple[torch.Tensor, Callable[[], torch.Tensor]]:
    if baseline_kind == "fp8_gemm":
        fn = lambda: fp8_gemm(a_fp8, a_scale, weight_fp8, weight_scale)
        return fn(), fn
    if baseline_kind == "cached_bf16":
        fn = lambda: fp8_gemm_cached_weight(a_fp8, a_scale, bf16_row)
        return fn(), fn
    if baseline_kind == "cached_fp32":
        fn = lambda: fp8_gemm_cached_weight(a_fp8, a_scale, fp32_row)
        return fn(), fn
    raise ValueError(f"unknown baseline kind: {baseline_kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config_671B_v3.2.json")
    parser.add_argument("--prefill-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_args = load_args(args.config)
    spec = TARGETS[args.target]
    m = args.prefill_len
    k = spec["in_features"](model_args)
    n = spec["out_features"](model_args)

    act_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).contiguous()
    a_fp8, a_scale = act_quant(act_bf16, block_size, model_args.scale_fmt)

    weight_fp32 = torch.randn(n, k, device="cuda", dtype=torch.float32).clamp_(-448, 448)
    weight_fp8 = weight_fp32.to(torch.float8_e4m3fn).contiguous()
    weight_scale = torch.rand(
        (n + block_size - 1) // block_size,
        (k + block_size - 1) // block_size,
        device="cuda",
        dtype=torch.float32,
    ).mul_(0.05).add_(1e-4).contiguous()

    bf16_row = weight_dequant(weight_fp8, weight_scale).contiguous()
    fp32_row = bf16_row.float().contiguous()
    bf16_t = bf16_row.t().contiguous()
    fp32_t = fp32_row.t().contiguous()

    baseline, baseline_fn = make_baseline(
        spec["baseline_kind"], a_fp8, a_scale, weight_fp8, weight_scale, bf16_row, fp32_row
    )
    baseline_bench = benchmark_cuda(baseline_fn, args.warmup, args.iters)

    cache: Dict[str, Any] = {
        "bf16_row": bf16_row,
        "fp32_row": fp32_row,
        "bf16_t": bf16_t,
        "fp32_t": fp32_t,
        "row_dequant_each_call": lambda: weight_dequant(weight_fp8, weight_scale).contiguous(),
        "zeros": torch.zeros((m, n), device="cuda", dtype=torch.float32),
    }

    candidates = make_candidates()
    results = [
        evaluate_candidate(candidate, a_fp8, a_scale, baseline, cache, args.warmup, args.iters)
        for candidate in candidates
    ]
    exact_results = [r for r in results if r["exact"]]
    exact_results.sort(key=lambda r: r["mean_ms"])
    overall_results = sorted(results, key=lambda r: r["mean_ms"])

    summary = {
        "target": args.target,
        "device": torch.cuda.get_device_name(0),
        "config": str(args.config),
        "shape": {"m": m, "k": k, "n": n},
        "baseline_kind": spec["baseline_kind"],
        "baseline_current": baseline_bench,
        "baseline_cache_sizes_gib": {
            "bf16_row": gib(bf16_row.numel() * bf16_row.element_size()),
            "fp32_row": gib(fp32_row.numel() * fp32_row.element_size()),
        },
        "candidate_count": len(results),
        "exact_count": len(exact_results),
        "inexact_count": len(results) - len(exact_results),
        "best_exact": exact_results[0] if exact_results else None,
        "top_10_exact": exact_results[:10],
        "top_10_overall": overall_results[:10],
        "family_summary": family_summary(results),
        "results": results,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target": summary["target"],
                "candidate_count": summary["candidate_count"],
                "exact_count": summary["exact_count"],
                "best_exact": summary["best_exact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
