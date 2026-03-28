#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = REPO_ROOT / "search" / "queue"
REPORTS_DIR = REPO_ROOT / "search" / "reports"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def resolve_result_path(cwd: Path, result_path: str) -> Path:
    candidate = Path(result_path)
    if candidate.is_absolute():
        return candidate
    return cwd / candidate


def projection_record(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    best = result.get("best_exact")
    baseline = result.get("baseline_current")
    if not isinstance(best, dict) or not isinstance(baseline, dict):
        return None
    best_ms = best.get("mean_ms")
    baseline_ms = baseline.get("mean_ms")
    if not best_ms or not baseline_ms:
        return None
    return {
        "id": manifest["id"],
        "task_id": manifest["task_id"],
        "target": result.get("target"),
        "shape": result.get("shape"),
        "speedup": baseline_ms / best_ms,
        "baseline_ms": baseline_ms,
        "best_ms": best_ms,
        "candidate_id": candidate_id_from_payload(best),
        "weight_variant": best.get("weight_variant"),
        "cache_layout": best.get("cache_layout"),
        "cache_dtype": best.get("cache_dtype"),
        "input_variant": best.get("input_variant"),
        "op_variant": best.get("op_variant"),
    }


def candidate_id_from_payload(candidate: dict[str, Any] | None) -> str | None:
    if not isinstance(candidate, dict):
        return None
    label = candidate.get("label") or candidate.get("id")
    if isinstance(label, str) and label:
        return label
    index = candidate.get("index")
    if isinstance(index, int):
        return f"cand-{index:03d}"
    return None


def index_records(manifest: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result.get("results"), list):
        if not result["results"] and result.get("runtime_error"):
            return [
                {
                    "id": manifest["id"],
                    "task_id": manifest["task_id"],
                    "shape": result.get("shape"),
                    "candidate_id": None,
                    "mean_ms": None,
                    "exact": False,
                    "runtime_error": result.get("runtime_error"),
                    "batch_failure": True,
                }
            ]
        rows = []
        for entry in result["results"]:
            benchmark = entry.get("benchmark") or {}
            check = entry.get("check") or {}
            rows.append(
                {
                    "id": manifest["id"],
                    "task_id": manifest["task_id"],
                    "shape": entry.get("shape") or result.get("shape"),
                    "candidate_id": candidate_id_from_payload(entry.get("candidate")),
                    "mean_ms": benchmark.get("mean_ms"),
                    "exact": check.get("exact"),
                    "runtime_error": entry.get("runtime_error") or result.get("runtime_error"),
                    "batch_failure": False,
                }
            )
        return rows
    benchmark = result.get("benchmark") or {}
    check = result.get("check") or {}
    return [
        {
            "id": manifest["id"],
            "task_id": manifest["task_id"],
            "shape": result.get("shape"),
            "candidate_id": candidate_id_from_payload(result.get("candidate")),
            "mean_ms": benchmark.get("mean_ms"),
            "exact": check.get("exact"),
            "runtime_error": result.get("runtime_error"),
            "batch_failure": False,
        }
    ]


def act_record(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    best = result.get("best_exact")
    return {
        "id": manifest["id"],
        "task_id": manifest["task_id"],
        "cases": result.get("cases"),
        "selected_candidate_count": result.get("selected_candidate_count"),
        "exact_count": result.get("exact_count"),
        "best_exact_mean_ms": best.get("mean_ms") if isinstance(best, dict) else None,
        "runtime_error": result.get("runtime_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, default=QUEUE_DIR)
    parser.add_argument("--json-out", type=Path, default=REPORTS_DIR / "queue-live-summary.json")
    parser.add_argument("--md-out", type=Path, default=REPORTS_DIR / "queue-live-summary.md")
    args = parser.parse_args()

    queue_dir = args.queue_dir.resolve()
    buckets = {name: list((queue_dir / name).glob("*.json")) for name in ["pending", "running", "completed", "failed"]}

    counts = {name: len(paths) for name, paths in buckets.items()}
    failures_by_task = Counter()
    failures_by_owner = Counter()
    projection_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    act_rows: list[dict[str, Any]] = []

    for path in buckets["failed"]:
        manifest = load_json(path)
        if not manifest:
            continue
        failures_by_task[manifest.get("task_id", "unknown")] += 1
        failures_by_owner[manifest.get("owner", "unknown")] += 1

    for path in buckets["completed"]:
        manifest = load_json(path)
        if not manifest:
            continue
        cwd = Path(manifest["cwd"])
        result_paths = manifest.get("result_paths") or []
        if not result_paths:
            continue
        result_path = resolve_result_path(cwd, result_paths[0])
        result = load_json(result_path)
        if not result:
            continue
        task_id = manifest.get("task_id")
        if task_id == "02_fp8_gemm_exact":
            row = projection_record(manifest, result)
            if row:
                projection_rows.append(row)
        elif task_id == "03_fp8_index_exact":
            index_rows.extend(index_records(manifest, result))
        elif task_id == "01_act_quant_exact":
            act_rows.append(act_record(manifest, result))

    projection_by_id: dict[str, dict[str, Any]] = {}
    for row in projection_rows:
        existing = projection_by_id.get(row["id"])
        if existing is None or row["speedup"] > existing["speedup"]:
            projection_by_id[row["id"]] = row
    projection_rows = sorted(projection_by_id.values(), key=lambda row: row["speedup"], reverse=True)
    exact_index = [row for row in index_rows if row.get("exact") is True and not row.get("runtime_error")]
    index_batch_failures = sum(1 for row in index_rows if row.get("batch_failure"))
    index_candidate_runtime_errors = sum(
        1 for row in index_rows if row.get("runtime_error") and not row.get("batch_failure")
    )
    act_runtime_errors = sum(1 for row in act_rows if row.get("runtime_error"))

    payload = {
        "counts": counts,
        "failures_by_task": dict(failures_by_task),
        "failures_by_owner": dict(failures_by_owner),
        "projection_top_speedups": projection_rows[:20],
        "projection_completed_count": len(projection_rows),
        "index_completed_count": len(index_rows),
        "index_exact_count": len(exact_index),
        "index_batch_failure_count": index_batch_failures,
        "index_candidate_runtime_error_count": index_candidate_runtime_errors,
        "act_completed_count": len(act_rows),
        "act_runtime_error_count": act_runtime_errors,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Queue Live Summary",
        "",
        "## Counts",
        f"- pending: {counts['pending']}",
        f"- running: {counts['running']}",
        f"- completed: {counts['completed']}",
        f"- failed: {counts['failed']}",
        "",
        "## Failed Jobs",
    ]
    if failures_by_task:
        for task_id, count in failures_by_task.most_common():
            lines.append(f"- {task_id}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Projection Leaders",
        ]
    )
    if projection_rows:
        for row in projection_rows[:15]:
            shape = row.get("shape") or {}
            prefill = shape.get("m") if isinstance(shape, dict) else None
            lines.append(
                "- "
                f"{row['target']} prefill={prefill}: {row['speedup']:.3f}x "
                f"via "
                f"{row.get('weight_variant') or row.get('cache_layout')} / "
                f"{row.get('cache_dtype') or row.get('input_variant')} / "
                f"{row['op_variant']}"
            )
    else:
        lines.append("- none yet")
    lines.extend(
        [
            "",
            "## Index Batch",
            f"- completed records: {len(index_rows)}",
            f"- exact records: {len(exact_index)}",
            f"- batch-failure records: {index_batch_failures}",
            f"- candidate runtime-error records: {index_candidate_runtime_errors}",
            "",
            "## Act Batch",
            f"- completed records: {len(act_rows)}",
            f"- runtime-error records: {act_runtime_errors}",
        ]
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
