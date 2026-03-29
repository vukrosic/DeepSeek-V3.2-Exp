#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

try:
    import torch
except ModuleNotFoundError:  # staging can run without GPU deps
    torch = None


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parent
DEFAULT_CONFIG = REPO_ROOT / "config_671B_v3.2.json"
DEFAULT_STAGE_ROOT = SEARCH_ROOT / "staging" / "gauss-index-batch-20260328"
DEFAULT_OWNER = "gauss"
DEFAULT_TASK_ID = "03_fp8_index_exact"
DEFAULT_PREFILL_LEN = 256
DEFAULT_DECODE_CONTEXT = 2048
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
REMOTE_PYTHON = "/venv/main/bin/python3"

Q_VARIANTS = [
    "fp32_contig",
    "fp32_clone",
    "bf16_contig",
    "bf16_clone",
    "fp16_contig",
]
K_VARIANTS = [
    "fp32_contig",
    "fp32_clone",
    "bf16_contig",
    "fp16_contig",
]
DOT_VARIANTS = [
    "einsum_kq",
    "einsum_qk",
    "matmul_broadcast",
    "bmm_broadcast",
    "broadcast_reduce",
]
ACCUM_VARIANTS = [
    "sum_fp32",
    "sum_default",
]
SHAPE_VARIANTS = [
    "decode",
    "prefill_small",
    "prefill_mid",
]


@dataclass(frozen=True)
class CandidateSpec:
    index: int
    q_variant: str
    k_variant: str
    dot_variant: str
    accum_variant: str

    @property
    def label(self) -> str:
        return f"cand-{self.index:03d}"


def load_args(config_path: Path) -> Dict[str, int]:
    with open(config_path) as f:
        return json.load(f)


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


def make_shapes(args: Dict[str, int], prefill_len: int, decode_context: int) -> Dict[str, Dict[str, int]]:
    return {
        "decode": {
            "b": 1,
            "m": 1,
            "h": args["index_n_heads"],
            "d": args["index_head_dim"],
            "n": decode_context,
        },
        "prefill_small": {
            "b": 1,
            "m": min(prefill_len, 64),
            "h": args["index_n_heads"],
            "d": args["index_head_dim"],
            "n": min(prefill_len, 512),
        },
        "prefill_mid": {
            "b": 1,
            "m": min(prefill_len, 128),
            "h": args["index_n_heads"],
            "d": args["index_head_dim"],
            "n": min(decode_context, 1024),
        },
    }


def make_candidates() -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    for idx, (q_variant, k_variant, dot_variant, accum_variant) in enumerate(
        product(Q_VARIANTS, K_VARIANTS, DOT_VARIANTS, ACCUM_VARIANTS),
        start=1,
    ):
        out.append(
            CandidateSpec(
                index=idx,
                q_variant=q_variant,
                k_variant=k_variant,
                dot_variant=dot_variant,
                accum_variant=accum_variant,
            )
        )
    assert len(out) == 200
    return out


def load_candidate(candidate_id: str) -> CandidateSpec:
    for candidate in make_candidates():
        if candidate.label == candidate_id:
            return candidate
    raise SystemExit(f"unknown candidate id: {candidate_id}")


def candidate_payload(candidate: CandidateSpec) -> Dict[str, Any]:
    payload = asdict(candidate)
    payload["label"] = candidate.label
    payload["id"] = candidate.label
    return payload


def select_candidates(
    candidates: Iterable[CandidateSpec],
    candidate_ids: List[str],
    candidate_offset: int,
    candidate_limit: int,
) -> List[CandidateSpec]:
    candidate_list = list(candidates)
    if candidate_ids:
        wanted = set(candidate_ids)
        selected = [candidate for candidate in candidate_list if candidate.label in wanted]
        missing = sorted(wanted - {candidate.label for candidate in selected})
        if missing:
            raise SystemExit(f"unknown candidate ids: {', '.join(missing)}")
        return selected
    offset = max(candidate_offset, 0)
    if candidate_limit <= 0:
        return candidate_list[offset:]
    return candidate_list[offset : offset + candidate_limit]


def _dequantize_fp8(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    scale_f = scale.float()
    while scale_f.dim() < x.dim():
        scale_f = scale_f.unsqueeze(-1)
    out = x.float() * scale_f
    return out.to(dtype)


def _materialize_variant(x: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "fp32_contig":
        return x.float().contiguous()
    if variant == "fp32_clone":
        return x.float().clone().contiguous()
    if variant == "bf16_contig":
        return x.to(torch.bfloat16).contiguous()
    if variant == "bf16_clone":
        return x.to(torch.bfloat16).clone().contiguous()
    if variant == "fp16_contig":
        return x.to(torch.float16).contiguous()
    raise ValueError(f"unknown materialization variant: {variant}")


def _dot_kernel(q: torch.Tensor, k: torch.Tensor, variant: str) -> torch.Tensor:
    common_dtype = torch.promote_types(q.dtype, k.dtype)
    if q.dtype != common_dtype:
        q = q.to(common_dtype)
    if k.dtype != common_dtype:
        k = k.to(common_dtype)
    if variant == "einsum_kq":
        return torch.einsum("bnd,bmhd->bmnh", k, q)
    if variant == "einsum_qk":
        return torch.einsum("bmhd,bnd->bmnh", q, k)
    if variant == "matmul_broadcast":
        return torch.matmul(k.unsqueeze(1), q.permute(0, 1, 3, 2))
    if variant == "bmm_broadcast":
        b, m, h, d = q.shape
        n = k.size(1)
        lhs = k.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, n, d)
        rhs = q.reshape(b * m, h, d).transpose(1, 2)
        return torch.bmm(lhs, rhs).view(b, m, n, h)
    if variant == "broadcast_reduce":
        return (k.unsqueeze(2) * q.unsqueeze(1)).sum(dim=-1)
    raise ValueError(f"unknown dot variant: {variant}")


def exact_index_reference(q_fp8: torch.Tensor, q_s: torch.Tensor, k_fp8: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    if q_s.dim() == 4 and q_s.size(-1) == 1:
        q_s = q_s.squeeze(-1)
    if k_s.dim() == 3 and k_s.size(-1) == 1:
        k_s = k_s.squeeze(-1)
    logits = torch.einsum("bnd,bmhd->bmnh", k_fp8.float(), q_fp8.float())
    logits = logits.clamp_min_(0) * q_s.float().unsqueeze(2)
    return logits.sum(dim=-1, dtype=torch.float32) * k_s.float().unsqueeze(1)


def run_candidate(
    candidate: CandidateSpec,
    shape_name: str,
    args: Dict[str, int],
    prefill_len: int,
    decode_context: int,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    if torch is None:
        raise SystemExit("torch is required for run mode")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from kernel import act_quant  # noqa: E402
    from model import block_size  # noqa: E402

    shapes = make_shapes(args, prefill_len, decode_context)
    shape = shapes[shape_name]
    b, m, h, d, n = shape["b"], shape["m"], shape["h"], shape["d"], shape["n"]

    q_bf16 = torch.randn(b, m, h, d, device="cuda", dtype=torch.bfloat16).contiguous()
    k_bf16 = torch.randn(b, n, d, device="cuda", dtype=torch.bfloat16).contiguous()
    q_fp8, q_s = act_quant(q_bf16, block_size, args["scale_fmt"])
    k_fp8, k_s = act_quant(k_bf16, block_size, args["scale_fmt"])
    q_s = q_s.contiguous()
    k_s = k_s.squeeze(-1).contiguous()

    q_deq = _materialize_variant(_dequantize_fp8(q_fp8, q_s, torch.float32), candidate.q_variant)
    k_deq = _materialize_variant(_dequantize_fp8(k_fp8, k_s, torch.float32), candidate.k_variant)

    def candidate_fn() -> torch.Tensor:
        logits = _dot_kernel(q_deq, k_deq, candidate.dot_variant)
        logits = logits.clamp_min_(0)
        logits = logits * q_s.float().unsqueeze(2)
        if candidate.accum_variant == "sum_fp32":
            return logits.sum(dim=-1, dtype=torch.float32) * k_s.float().unsqueeze(1)
        return logits.sum(dim=-1) * k_s.float().unsqueeze(1)

    ref = exact_index_reference(q_fp8, q_s, k_fp8, k_s)
    out = candidate_fn()
    bench = benchmark_cuda(candidate_fn, warmup, iters)
    return {
        "shape": shape_name,
        "candidate": candidate_payload(candidate),
        "benchmark": bench,
        "check": tensor_check(out, ref),
        "shape_dims": shape,
    }


def build_queue_manifest(
    stage_root: Path,
    shape_name: str,
    candidates: List[CandidateSpec],
    shard_index: int,
    repo_root: Path,
    owner: str,
) -> Dict[str, Any]:
    batch_name = stage_root.name
    first = candidates[0].label
    last = candidates[-1].label
    shard_label = f"sh{shard_index:03d}-{first}-to-{last}"
    run_dir = f"search/runs/{batch_name}-{shape_name}-{shard_label}"
    rel_result = f"{run_dir}/results.json"
    candidate_flags = " ".join(f"--candidate-id {candidate.label}" for candidate in candidates)
    manifest = {
        "id": f"{batch_name}-{shape_name}-{shard_label}",
        "owner": owner,
        "priority": 80,
        "task_id": DEFAULT_TASK_ID,
        "run_dir": run_dir,
        "cwd": "/workspace/DeepSeek-V3.2-Exp/inference",
        "command": (
            "PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference "
            f"{REMOTE_PYTHON} search/index_200_sweep.py run "
            f"--shape {shape_name} "
            f"{candidate_flags} "
            f"--json-out {rel_result}"
        ),
        "result_paths": [rel_result],
        "tags": ["gpu", "exact", "search", "index", "staged"],
        "notes": (
            f"Staged exact index shard {shard_label} for {shape_name}; "
            f"generated by {Path(__file__).name}."
        ),
    }
    return manifest


def _is_statically_valid(shape_name: str, candidate: CandidateSpec) -> bool:
    # broadcast_reduce returns b,n,m,h order and only stays shape-safe when m == 1.
    return not (shape_name != "decode" and candidate.dot_variant == "broadcast_reduce")


def write_stage_artifacts(
    stage_root: Path,
    repo_root: Path,
    prefill_len: int,
    decode_context: int,
    owner: str,
    shard_size: int,
) -> Dict[str, Any]:
    shapes = make_shapes(load_args(DEFAULT_CONFIG), prefill_len, decode_context)
    candidates = make_candidates()
    stage_root.mkdir(parents=True, exist_ok=True)
    manifests_root = stage_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    created: List[str] = []
    skipped: List[str] = []
    for shape_name in SHAPE_VARIANTS:
        shape_dir = manifests_root / shape_name
        shape_dir.mkdir(parents=True, exist_ok=True)
        valid_candidates = []
        for candidate in candidates:
            if not _is_statically_valid(shape_name, candidate):
                skipped.append(f"{shape_name}/{candidate.label}")
                continue
            valid_candidates.append(candidate)
        for shard_index, offset in enumerate(range(0, len(valid_candidates), shard_size), start=1):
            shard = valid_candidates[offset : offset + shard_size]
            manifest = build_queue_manifest(stage_root, shape_name, shard, shard_index, repo_root, owner)
            path = shape_dir / f"{shard_index:03d}.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            created.append(str(path.relative_to(stage_root.parent)))

    readme = stage_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Gauss Index Batch",
                "",
                "This staging area holds queue-compatible manifests for the exact fp8 index sweep.",
                "",
                "Scope:",
                "- 3 shape families: `decode`, `prefill_small`, `prefill_mid`",
                "- 200 candidate variants per shape before static filtering",
                f"- shard size: `{shard_size}` candidate(s) per manifest",
                f"- {len(created)} staged experiments total after static filtering",
                "",
                "This directory is intentionally separate from `search/queue/`.",
                "Do not submit these manifests automatically.",
                "",
                "To inspect a manifest, open `manifests/<shape>/NNN.json`.",
            ]
        )
        + "\n"
    )

    index_payload = {
        "batch": stage_root.name,
        "config": str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
        "prefill_len": prefill_len,
        "decode_context": decode_context,
        "shapes": shapes,
        "candidate_count_per_shape": len(candidates),
        "valid_manifest_count_per_shape": {
            shape_name: sum(1 for candidate in candidates if _is_statically_valid(shape_name, candidate))
            for shape_name in SHAPE_VARIANTS
        },
        "shard_size": shard_size,
        "total_manifests": len(created),
        "manifest_paths": created,
        "skipped_static_invalid": skipped,
        "owner": owner,
    }
    (stage_root / "manifest_index.json").write_text(json.dumps(index_payload, indent=2) + "\n")
    return index_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--shape", choices=SHAPE_VARIANTS, required=True)
    run.add_argument("--candidate-id", action="append", default=[])
    run.add_argument("--candidate-offset", type=int, default=0)
    run.add_argument("--candidate-limit", type=int, default=200)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--prefill-len", type=int, default=DEFAULT_PREFILL_LEN)
    run.add_argument("--decode-context", type=int, default=DEFAULT_DECODE_CONTEXT)
    run.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    run.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    run.add_argument("--json-out", type=Path, required=True)

    stage = sub.add_parser("stage")
    stage.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    stage.add_argument("--prefill-len", type=int, default=DEFAULT_PREFILL_LEN)
    stage.add_argument("--decode-context", type=int, default=DEFAULT_DECODE_CONTEXT)
    stage.add_argument("--owner", default=DEFAULT_OWNER)
    stage.add_argument("--shard-size", type=int, default=1)

    args = parser.parse_args()

    if args.cmd == "run":
        if torch is None:
            raise SystemExit("torch is required for run mode")
        torch.cuda.set_device(0)
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(20260328)
        torch.cuda.manual_seed_all(20260328)
        model_args = load_args(args.config)
        selected_candidates = select_candidates(
            make_candidates(),
            args.candidate_id,
            args.candidate_offset,
            args.candidate_limit,
        )
        if not selected_candidates:
            raise SystemExit("selected candidate set is empty")
        try:
            results = []
            for candidate in selected_candidates:
                try:
                    record = run_candidate(
                        candidate,
                        args.shape,
                        model_args,
                        args.prefill_len,
                        args.decode_context,
                        args.warmup,
                        args.iters,
                    )
                except Exception as exc:
                    record = {
                        "shape": args.shape,
                        "candidate": candidate_payload(candidate),
                        "benchmark": None,
                        "check": {"exact": False},
                        "shape_dims": make_shapes(
                            model_args,
                            args.prefill_len,
                            args.decode_context,
                        )[args.shape],
                        "runtime_error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                results.append(record)
            exact_results = [row for row in results if row["check"]["exact"] and not row.get("runtime_error")]
            exact_results.sort(key=lambda row: row["benchmark"]["mean_ms"])
            result = {
                "shape": args.shape,
                "selected_candidate_count": len(selected_candidates),
                "candidate_ids": [candidate.label for candidate in selected_candidates],
                "exact_count": len(exact_results),
                "best_exact": exact_results[0] if exact_results else None,
                "results": results,
            }
        except Exception as exc:
            result = {
                "shape": args.shape,
                "selected_candidate_count": len(selected_candidates),
                "candidate_ids": [candidate.label for candidate in selected_candidates],
                "exact_count": 0,
                "best_exact": None,
                "results": [],
                "runtime_error": str(exc),
                "traceback": traceback.format_exc(),
            }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        payload = {
            "shape": args.shape,
            "selected_candidate_count": result["selected_candidate_count"],
            "exact_count": result["exact_count"],
            "best_exact_candidate": (
                result["best_exact"]["candidate"].get("label")
                or result["best_exact"]["candidate"].get("id")
                if result.get("best_exact")
                else None
            ),
        }
        if "runtime_error" in result:
            payload["runtime_error"] = result["runtime_error"]
        print(json.dumps(payload, indent=2))
    elif args.cmd == "stage":
        summary = write_stage_artifacts(
            args.stage_root,
            REPO_ROOT,
            args.prefill_len,
            args.decode_context,
            args.owner,
            max(args.shard_size, 1),
        )
        print(json.dumps(
            {
                "stage_root": str(args.stage_root),
                "total_manifests": summary["total_manifests"],
                "candidate_count_per_shape": summary["candidate_count_per_shape"],
                "valid_manifest_count_per_shape": summary["valid_manifest_count_per_shape"],
                "shard_size": summary["shard_size"],
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
