#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel import act_quant  # noqa: E402
from model import ModelArgs, block_size  # noqa: E402


FP8_MAX = 448.0
FP8_MAX_INV = 1.0 / FP8_MAX
DEFAULT_CASES = ["index_q", "kv_cache"]


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
    out = {"exact": bool(exact)}
    if exact:
        out["max_abs_diff"] = 0.0
        out["mean_abs_diff"] = 0.0
        return out
    diff = (a.float() - b.float()).abs()
    out["max_abs_diff"] = float(diff.max().item())
    out["mean_abs_diff"] = float(diff.mean().item())
    return out


def ceil_pow2_scale_cpu(x: torch.Tensor) -> torch.Tensor:
    x_np = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32, copy=False))
    bits = x_np.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    man_bits = bits & ((1 << 23) - 1)
    ceil_log2 = exp - 127 + (man_bits != 0)
    out_bits = ((ceil_log2 + 127).astype(np.uint32)) << 23
    out_np = out_bits.view(np.float32)
    return torch.from_numpy(out_np).to(device=x.device, dtype=torch.float32)


def act_quant_reference(
    x: torch.Tensor, block: int = block_size, scale_fmt: str | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block == 0
    n = x.size(-1)
    x_blocks = x.view(-1, n // block, block).float()
    amax = x_blocks.abs().amax(dim=-1).clamp_min_(1e-4)
    scales = amax / FP8_MAX
    if scale_fmt is not None:
        scales = ceil_pow2_scale_cpu(scales)
    y = torch.clamp(x_blocks / scales.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y.view_as(x), scales.view(*x.shape[:-1], n // block)


def make_case_shapes(model_args: ModelArgs, batch_size: int, seq_len: int) -> Dict[str, Tuple[int, ...]]:
    return {
        "index_q": (batch_size, seq_len, model_args.index_n_heads, model_args.index_head_dim),
        "kv_cache": (batch_size, seq_len, model_args.kv_lora_rank),
        "mla_input_x": (batch_size, seq_len, model_args.dim),
        "mla_qr": (batch_size, seq_len, model_args.q_lora_rank),
        "mla_kv": (batch_size, seq_len, model_args.kv_lora_rank),
        "decode_input_x": (batch_size, 1, model_args.dim),
        "decode_qr": (batch_size, 1, model_args.q_lora_rank),
        "decode_kv": (batch_size, 1, model_args.kv_lora_rank),
    }


def reduce_impl(x_blocks: torch.Tensor, reduction: str) -> torch.Tensor:
    abs_x = x_blocks.abs()
    if reduction == "amax":
        return abs_x.amax(dim=-1)
    if reduction == "amax_keepdim":
        return abs_x.amax(dim=-1, keepdim=True).squeeze(-1)
    if reduction == "max_values":
        return abs_x.max(dim=-1).values
    if reduction == "torch_amax":
        return torch.amax(abs_x, dim=-1)
    if reduction == "torch_max":
        return torch.max(abs_x, dim=-1).values
    raise ValueError(f"unknown reduction variant: {reduction}")


def scale_impl(amax: torch.Tensor, scale: str) -> torch.Tensor:
    fp8_max_tensor = torch.tensor(FP8_MAX, device=amax.device, dtype=amax.dtype)
    inv_tensor = torch.tensor(FP8_MAX_INV, device=amax.device, dtype=amax.dtype)
    if scale == "div_const":
        return amax / FP8_MAX
    if scale == "torch_div_const":
        return torch.div(amax, FP8_MAX)
    if scale == "div_tensor":
        return torch.div(amax, fp8_max_tensor)
    if scale == "mul_inv_const":
        return amax * FP8_MAX_INV
    if scale == "mul_inv_tensor":
        return torch.mul(amax, inv_tensor)
    raise ValueError(f"unknown scale variant: {scale}")


def clamp_impl(raw: torch.Tensor, clamp: str) -> torch.Tensor:
    lo = -FP8_MAX
    hi = FP8_MAX
    if clamp == "clamp":
        return torch.clamp(raw, lo, hi)
    if clamp == "minimum_maximum":
        lo_t = torch.tensor(lo, device=raw.device, dtype=raw.dtype)
        hi_t = torch.tensor(hi, device=raw.device, dtype=raw.dtype)
        return torch.minimum(torch.maximum(raw, lo_t), hi_t)
    if clamp == "clip":
        return torch.clip(raw, lo, hi)
    if clamp == "where":
        hi_t = torch.tensor(hi, device=raw.device, dtype=raw.dtype)
        lo_t = torch.tensor(lo, device=raw.device, dtype=raw.dtype)
        return torch.where(raw > hi, hi_t, torch.where(raw < lo, lo_t, raw))
    raise ValueError(f"unknown clamp variant: {clamp}")


def materialize_variants(
    y: torch.Tensor, scales: torch.Tensor, x: torch.Tensor
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    return {
        "view": (y.view_as(x), scales.view(*x.shape[:-1], x.size(-1) // block_size)),
        "reshape": (y.reshape_as(x), scales.reshape(*x.shape[:-1], x.size(-1) // block_size)),
    }


def candidate_space() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    idx = 0
    for reduction in ["amax", "amax_keepdim", "max_values", "torch_amax", "torch_max"]:
        for scale in ["div_const", "torch_div_const", "div_tensor", "mul_inv_const", "mul_inv_tensor"]:
            for clamp in ["clamp", "minimum_maximum", "clip", "where"]:
                for materialize in ["view", "reshape"]:
                    idx += 1
                    out.append(
                        {
                            "id": f"cand_{idx:03d}",
                            "reduction": reduction,
                            "scale": scale,
                            "clamp": clamp,
                            "materialize": materialize,
                        }
                    )
    assert len(out) == 200
    return out


def select_candidates(
    candidates: Sequence[Dict[str, str]],
    candidate_ids: Sequence[str],
    candidate_offset: int,
    candidate_limit: int,
) -> List[Dict[str, str]]:
    if candidate_ids:
        wanted = set(candidate_ids)
        selected = [candidate for candidate in candidates if candidate["id"] in wanted]
        missing = sorted(wanted - {candidate["id"] for candidate in selected})
        if missing:
            raise SystemExit(f"unknown candidate ids: {', '.join(missing)}")
        return selected
    offset = max(candidate_offset, 0)
    if candidate_limit <= 0:
        return list(candidates[offset:])
    return list(candidates[offset : offset + candidate_limit])


def act_quant_candidate(
    x: torch.Tensor,
    scale_fmt: str | None,
    reduction: str,
    scale: str,
    clamp: str,
    materialize: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.size(-1)
    x_blocks = x.view(-1, n // block_size, block_size).float()
    amax = reduce_impl(x_blocks, reduction).clamp_min_(1e-4)
    scales = scale_impl(amax, scale)
    if scale_fmt is not None:
        mantissa, exponent = torch.frexp(scales)
        rounded_exponent = exponent - mantissa.eq(0.5).to(exponent.dtype)
        scales = torch.ldexp(torch.ones_like(scales), rounded_exponent)
    raw = x_blocks / scales.unsqueeze(-1)
    y = clamp_impl(raw, clamp).to(torch.float8_e4m3fn)
    y_out, s_out = materialize_variants(y, scales, x)[materialize]
    return y_out, s_out


def run_candidate(
    candidate: Dict[str, str],
    x: torch.Tensor,
    scale_fmt: str | None,
    ref_y: torch.Tensor,
    ref_s: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    def fn() -> Tuple[torch.Tensor, torch.Tensor]:
        return act_quant_candidate(
            x,
            scale_fmt,
            candidate["reduction"],
            candidate["scale"],
            candidate["clamp"],
            candidate["materialize"],
        )

    y, s = fn()
    bench = benchmark_cuda(fn, warmup, iters)
    y_check = tensor_check(y, ref_y)
    s_check = tensor_check(s, ref_s)
    exact = y_check["exact"] and s_check["exact"]
    diff = max(float(y_check["max_abs_diff"]), float(s_check["max_abs_diff"]))
    return {
        **candidate,
        "exact": bool(exact),
        "max_abs_diff": diff,
        "mean_ms": bench["mean_ms"],
        "median_ms": bench["median_ms"],
        "min_ms": bench["min_ms"],
        "max_ms": bench["max_ms"],
        "p95_ms": bench["p95_ms"],
        "y_check": y_check,
        "s_check": s_check,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config_671B_v3.2.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-len", "--seq-len", dest="seq_len", type=int, default=256)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--json-out", type=Path, default=SEARCH_ROOT / "results_act_sweep.json")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_args = load_args(args.config)
    all_cases = make_case_shapes(model_args, args.batch_size, args.seq_len)

    if args.list_cases:
        print(json.dumps(all_cases, indent=2))
        return

    candidates = candidate_space()
    if args.list_candidates:
        print(json.dumps(candidates, indent=2))
        return

    case_names = args.case or list(DEFAULT_CASES)
    missing_cases = [name for name in case_names if name not in all_cases]
    if missing_cases:
        raise SystemExit(f"unknown cases: {', '.join(missing_cases)}")

    selected_candidates = select_candidates(
        candidates,
        args.candidate_id,
        args.candidate_offset,
        args.candidate_limit,
    )
    if not selected_candidates:
        raise SystemExit("selected candidate set is empty")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        torch.cuda.set_device(0)
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        results: List[Dict[str, Any]] = []
        baseline_per_case: Dict[str, Dict[str, Any]] = {}

        for case_name in case_names:
            shape = all_cases[case_name]
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()
            ref_y, ref_s = act_quant_reference(x, block_size, model_args.scale_fmt)
            baseline_fn = lambda: act_quant(x, block_size, model_args.scale_fmt)
            baseline_per_case[case_name] = {
                "shape": list(shape),
                "benchmark": benchmark_cuda(baseline_fn, args.warmup, args.iters),
                "y_check": tensor_check(ref_y, ref_y),
                "s_check": tensor_check(ref_s, ref_s),
            }
            for candidate in selected_candidates:
                record = run_candidate(
                    candidate,
                    x,
                    model_args.scale_fmt,
                    ref_y,
                    ref_s,
                    args.warmup,
                    args.iters,
                )
                record["case"] = case_name
                record["shape_dims"] = list(shape)
                results.append(record)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in results:
            grouped.setdefault(record["id"], []).append(record)

        summary_rows: List[Dict[str, Any]] = []
        for candidate in selected_candidates:
            rows = grouped[candidate["id"]]
            exact = all(row["exact"] for row in rows)
            summary_rows.append(
                {
                    **candidate,
                    "exact": exact,
                    "mean_ms": statistics.fmean(row["mean_ms"] for row in rows),
                    "median_ms": statistics.median(row["mean_ms"] for row in rows),
                    "max_abs_diff": max(float(row["max_abs_diff"]) for row in rows),
                    "case_results": rows,
                }
            )

        exact_rows = [row for row in summary_rows if row["exact"]]
        exact_rows.sort(key=lambda row: row["mean_ms"])
        summary_rows.sort(key=lambda row: row["mean_ms"])

        payload = {
            "device": torch.cuda.get_device_name(0),
            "config": str(args.config),
            "scale_fmt": model_args.scale_fmt,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "cases": case_names,
            "candidate_offset": args.candidate_offset,
            "candidate_limit": args.candidate_limit,
            "selected_candidate_count": len(selected_candidates),
            "total_candidate_space": len(candidates),
            "case_count": len(case_names),
            "baseline_per_case": baseline_per_case,
            "exact_count": len(exact_rows),
            "inexact_count": len(summary_rows) - len(exact_rows),
            "best_exact": exact_rows[0] if exact_rows else None,
            "top_10_exact": exact_rows[:10],
            "top_10_overall": summary_rows[:10],
            "results": summary_rows,
        }
    except Exception as exc:
        payload = {
            "config": str(args.config),
            "scale_fmt": model_args.scale_fmt,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "cases": case_names,
            "candidate_offset": args.candidate_offset,
            "candidate_limit": args.candidate_limit,
            "selected_candidate_count": len(selected_candidates),
            "total_candidate_space": len(candidates),
            "case_count": len(case_names),
            "exact_count": 0,
            "inexact_count": 0,
            "best_exact": None,
            "top_10_exact": [],
            "top_10_overall": [],
            "results": [],
            "runtime_error": str(exc),
            "traceback": traceback.format_exc(),
        }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "selected_candidate_count": payload["selected_candidate_count"],
        "case_count": payload["case_count"],
        "exact_count": payload["exact_count"],
        "best_exact": payload["best_exact"],
    }
    if "runtime_error" in payload:
        summary["runtime_error"] = payload["runtime_error"]
    print(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
