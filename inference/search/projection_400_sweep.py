#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # staging and candidate listing can run without GPU deps
    torch = None
    F = None


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch is not None:
    from kernel import act_quant, fp8_dequant_input, fp8_gemm, fp8_gemm_cached_weight  # noqa: E402
    from model import ModelArgs, block_size, weight_dequant  # noqa: E402
else:  # pragma: no cover - exercised only in torch-less staging environments
    act_quant = None
    fp8_dequant_input = None
    fp8_gemm = None
    fp8_gemm_cached_weight = None
    ModelArgs = Any
    block_size = 128
    weight_dequant = None


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


INPUT_VARIANTS = [
    "a_float",
    "a_float_contig",
    "a_float_clone",
    "a_bf16",
    "a_bf16_contig",
]

CACHE_LAYOUT_VARIANTS = [
    "row_view",
    "row_clone",
    "t_view",
    "t_contig",
]

CACHE_DTYPE_VARIANTS = [
    "bf16",
    "fp32",
]

OP_VARIANTS = [
    "flinear",
    "mm",
    "matmul",
    "addmm",
    "einsum",
    "flinear_t",
    "mm_t",
    "matmul_t",
    "addmm_t",
    "einsum_t",
]


@dataclass(frozen=True)
class CandidateSpec:
    index: int
    input_variant: str
    cache_layout: str
    cache_dtype: str
    op_variant: str

    @property
    def label(self) -> str:
        return f"cand_{self.index:03d}"


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


def make_candidates() -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    idx = 0
    for input_variant in INPUT_VARIANTS:
        for cache_layout in CACHE_LAYOUT_VARIANTS:
            for cache_dtype in CACHE_DTYPE_VARIANTS:
                for op_variant in OP_VARIANTS:
                    idx += 1
                    out.append(
                        CandidateSpec(
                            index=idx,
                            input_variant=input_variant,
                            cache_layout=cache_layout,
                            cache_dtype=cache_dtype,
                            op_variant=op_variant,
                        )
                    )
    assert len(out) == 400
    return out


def select_candidates(
    candidates: Sequence[CandidateSpec],
    candidate_ids: Sequence[str],
    candidate_offset: int,
    candidate_limit: int,
) -> List[CandidateSpec]:
    if candidate_ids:
        wanted = set(candidate_ids)
        selected = [candidate for candidate in candidates if candidate.label in wanted]
        missing = sorted(wanted - {candidate.label for candidate in selected})
        if missing:
            raise SystemExit(f"unknown candidate ids: {', '.join(missing)}")
        return selected
    offset = max(candidate_offset, 0)
    if candidate_limit <= 0:
        return list(candidates[offset:])
    return list(candidates[offset : offset + candidate_limit])


def materialize_input(a_fp8: torch.Tensor, a_scale: torch.Tensor, variant: str) -> torch.Tensor:
    a = fp8_dequant_input(a_fp8, a_scale).view(a_fp8.numel() // a_fp8.size(-1), a_fp8.size(-1))
    if variant == "a_float":
        return a.float()
    if variant == "a_float_contig":
        return a.float().contiguous()
    if variant == "a_float_clone":
        return a.float().clone().contiguous()
    if variant == "a_bf16":
        return a.to(torch.bfloat16)
    if variant == "a_bf16_contig":
        return a.to(torch.bfloat16).contiguous()
    raise ValueError(f"unknown input variant: {variant}")


def materialize_weight(base: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "row_view":
        return base.view_as(base)
    if layout == "row_clone":
        return base.clone()
    if layout == "t_view":
        return base.t()
    if layout == "t_contig":
        return base.t().contiguous()
    raise ValueError(f"unknown cache layout: {layout}")


def target_dtype(input_variant: str, cache_dtype: str) -> torch.dtype:
    if input_variant.startswith("a_float") or cache_dtype == "fp32":
        return torch.float32
    return torch.bfloat16


def run_op(
    a: torch.Tensor,
    candidate: CandidateSpec,
    cache: Dict[str, Any],
) -> torch.Tensor:
    if candidate.cache_dtype == "bf16":
        base = cache["bf16"]
    elif candidate.cache_dtype == "fp32":
        base = cache["fp32"]
    else:
        raise ValueError(f"unknown cache dtype: {candidate.cache_dtype}")

    weight = materialize_weight(base, candidate.cache_layout)
    dtype = target_dtype(candidate.input_variant, candidate.cache_dtype)
    a_cast = a.to(dtype)
    w_cast = weight.to(dtype)
    row_weight = w_cast if candidate.cache_layout.startswith("row") else w_cast.t()
    t_weight = w_cast if candidate.cache_layout.startswith("t") else w_cast.t()

    if candidate.op_variant == "flinear":
        out = F.linear(a_cast, row_weight)
    elif candidate.op_variant == "mm":
        out = a_cast @ row_weight.t()
    elif candidate.op_variant == "matmul":
        out = torch.matmul(a_cast, row_weight.t())
    elif candidate.op_variant == "addmm":
        bias = cache["zeros"].to(dtype)
        out = torch.addmm(bias, a_cast, row_weight.t())
    elif candidate.op_variant == "einsum":
        out = torch.einsum("mk,nk->mn", a_cast, row_weight)
    elif candidate.op_variant == "flinear_t":
        out = F.linear(a_cast, t_weight.t())
    elif candidate.op_variant == "mm_t":
        out = a_cast @ t_weight
    elif candidate.op_variant == "matmul_t":
        out = torch.matmul(a_cast, t_weight)
    elif candidate.op_variant == "addmm_t":
        bias = cache["zeros"].to(dtype)
        out = torch.addmm(bias, a_cast, t_weight)
    elif candidate.op_variant == "einsum_t":
        out = torch.einsum("mk,kn->mn", a_cast, t_weight)
    else:
        raise ValueError(f"unknown op variant: {candidate.op_variant}")
    return out.to(torch.get_default_dtype())


def evaluate_candidate(
    candidate: CandidateSpec,
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    ref: torch.Tensor,
    cache: Dict[str, Any],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    def fn() -> torch.Tensor:
        a = materialize_input(a_fp8, a_scale, candidate.input_variant)
        return run_op(a, candidate, cache)

    out = fn()
    bench = benchmark_cuda(fn, warmup, iters)
    check = tensor_check(out, ref)
    payload = asdict(candidate)
    payload["id"] = candidate.label
    payload["label"] = candidate.label
    return {
        **payload,
        "exact": check["exact"],
        "max_abs_diff": check["max_abs_diff"],
        "mean_abs_diff": check["mean_abs_diff"],
        "mean_ms": bench["mean_ms"],
        "median_ms": bench["median_ms"],
        "min_ms": bench["min_ms"],
        "max_ms": bench["max_ms"],
        "p95_ms": bench["p95_ms"],
    }


def variant_summary(results: List[Dict[str, Any]], key: str, variants: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        family_rows = [r for r in results if r[key] == variant]
        exact_rows = [r for r in family_rows if r["exact"]]
        exact_rows.sort(key=lambda r: r["mean_ms"])
        out[variant] = {
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
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=400)
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    candidates = make_candidates()
    if args.list_candidates:
        print(json.dumps([asdict(candidate) | {"id": candidate.label} for candidate in candidates], indent=2))
        return

    try:
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
            "bf16": bf16_row,
            "fp32": fp32_row,
            "bf16_t": bf16_t,
            "fp32_t": fp32_t,
            "zeros": torch.zeros((m, n), device="cuda", dtype=torch.float32),
        }

        selected_candidates = select_candidates(
            candidates,
            args.candidate_id,
            args.candidate_offset,
            args.candidate_limit,
        )
        if not selected_candidates:
            raise SystemExit("selected candidate set is empty")

        results = []
        for candidate in selected_candidates:
            try:
                results.append(
                    evaluate_candidate(candidate, a_fp8, a_scale, baseline, cache, args.warmup, args.iters)
                )
            except Exception as exc:
                results.append(
                    {
                        **(asdict(candidate) | {"id": candidate.label, "label": candidate.label}),
                        "exact": False,
                        "max_abs_diff": None,
                        "mean_abs_diff": None,
                        "mean_ms": None,
                        "median_ms": None,
                        "min_ms": None,
                        "max_ms": None,
                        "p95_ms": None,
                        "runtime_error": str(exc),
                    }
                )
        exact_results = [r for r in results if r["exact"] and not r.get("runtime_error")]
        exact_results.sort(key=lambda r: r["mean_ms"])
        overall_results = sorted(
            results,
            key=lambda r: (r["mean_ms"] is None, float("inf") if r["mean_ms"] is None else r["mean_ms"]),
        )

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
            "input_summary": variant_summary(results, "input_variant", INPUT_VARIANTS),
            "layout_summary": variant_summary(results, "cache_layout", CACHE_LAYOUT_VARIANTS),
            "cache_dtype_summary": variant_summary(results, "cache_dtype", CACHE_DTYPE_VARIANTS),
            "op_summary": variant_summary(results, "op_variant", OP_VARIANTS),
            "results": results,
        }
    except Exception as exc:
        summary = {
            "target": args.target,
            "config": str(args.config),
            "candidate_count": 0,
            "exact_count": 0,
            "inexact_count": 0,
            "best_exact": None,
            "top_10_exact": [],
            "top_10_overall": [],
            "results": [],
            "runtime_error": str(exc),
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
