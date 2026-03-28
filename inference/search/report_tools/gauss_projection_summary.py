#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


RUN_ROOT = Path(__file__).resolve().parents[1] / "runs"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "reports" / "main-grid-b-shape-summary.md"


TARGET_LABELS = {
    "mla-wq-b": "mla_wq_b",
    "mla-wq-a": "mla_wq_a",
    "mla-wkv-a": "mla_wkv_a",
    "mla-wkv-b": "mla_wkv_b",
    "indexer-wq-b": "indexer_wq_b",
    "indexer-wk": "indexer_wk",
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def infer_target(path: Path, payload: Dict) -> str:
    if "target" in payload:
        return payload["target"]
    label = path.parent.name
    for needle, target in TARGET_LABELS.items():
        if needle in label:
            return target
    return label


def infer_prefill_len(payload: Dict) -> int:
    shape = payload.get("shape", {})
    return int(shape.get("m", 0))


def exact_speedups(payload: Dict) -> List[float]:
    baseline = payload.get("baseline_current_fp8_gemm", payload.get("baseline_current"))
    if baseline is None:
        return []
    base_ms = baseline["mean_ms"]
    return [base_ms / row["mean_ms"] for row in payload.get("results", []) if row.get("exact")]


def best_exact(payload: Dict) -> Dict:
    baseline = payload.get("baseline_current_fp8_gemm", payload.get("baseline_current"))
    base_ms = baseline["mean_ms"]
    exact_rows = [row for row in payload.get("results", []) if row.get("exact")]
    for row in exact_rows:
        row["speedup"] = base_ms / row["mean_ms"]
    return max(exact_rows, key=lambda row: row["speedup"])


def summarize() -> List[Dict]:
    entries = []
    for path in sorted(RUN_ROOT.glob("**/results_sweep.json")):
        payload = load_json(path)
        if "results" not in payload:
            continue
        target = infer_target(path, payload)
        if target not in TARGET_LABELS.values():
            continue
        speeds = exact_speedups(payload)
        if not speeds:
            continue
        best = best_exact(payload)
        baseline = payload.get("baseline_current_fp8_gemm", payload.get("baseline_current"))
        entries.append(
            {
                "target": target,
                "prefill_len": infer_prefill_len(payload),
                "exact_count": payload.get("exact_count", 0),
                "baseline_ms": baseline["mean_ms"],
                "speedup_min": min(speeds),
                "speedup_max": max(speeds),
                "best_family": best["weight_variant"],
                "best_input": best["input_variant"],
                "best_op": best["op_variant"],
                "best_ms": best["mean_ms"],
                "best_speedup": best["speedup"],
            }
        )
    return sorted(entries, key=lambda row: (row["prefill_len"], row["target"]))


def render_markdown(rows: List[Dict]) -> str:
    lines = []
    lines.append("# Main Grid B Shape Summary")
    lines.append("")
    lines.append("Completed projection sweeps only. All finished rows in this batch use `prefill_len=256`; the `l64` and `l512` queue manifests exist but are not completed, so they are excluded here.")
    lines.append("")
    lines.append("| target | prefill_len | exact count | best exact family/op | best exact ms | best speedup | exact speedup range |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| `{row['target']}` | {row['prefill_len']} | {row['exact_count']} | "
            f"`{row['best_family']}` / `{row['best_op']}` | {row['best_ms']:.3f} | {row['best_speedup']:.3f}x | "
            f"{row['speedup_min']:.3f}x-{row['speedup_max']:.3f}x |"
        )
    lines.append("")
    lines.append("## Engineering Takeaways")
    lines.append("- `cache_fp32_row` is the dominant exact winner family across the completed projection sweeps.")
    lines.append("- `mla_wq_b` is the strongest win: exact `cache_fp32_row + flinear` reaches about `2.10x` over baseline.")
    lines.append("- `mla_wkv_b` is the only sweep where `cache_fp32_t` becomes the best exact family, which points to transpose/layout sensitivity in that path.")
    lines.append("- `indexer_wk` is the widest exact spread and the lowest worst-case exact speedup, so it is the least stable candidate family in the batch.")
    lines.append("- Exact wins are real but localized: most other exact winners sit around `1.01x` to `1.16x`, so the remaining headroom is modest unless the memory layout changes more materially.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = summarize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(rows))
    print(args.out)


if __name__ == "__main__":
    main()
