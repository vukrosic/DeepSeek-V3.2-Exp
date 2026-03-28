#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, UTC
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/workspace/DeepSeek-V3.2-Exp/inference"
RUNS_DIR = LOCAL_ROOT / "search" / "runs"
STAGING_DIR = LOCAL_ROOT / "search" / "staging"
REMOTE_QUEUE = LOCAL_ROOT / "search" / "queue" / "remote_queue.py"
TASK_ID = "02_fp8_gemm_exact"
SWEEP_SCRIPT = "search/projection_100_sweep.py"

DEFAULT_TARGETS = [
    "mla_wq_b",
    "mla_wkv_b",
    "indexer_wq_b",
    "indexer_wk",
    "mla_wq_a",
    "mla_wkv_a",
]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def build_manifest(
    batch_tag: str,
    owner: str,
    priority: int,
    target: str,
    prefill_len: int,
    sweep_script: str,
    candidate_offset: int,
    candidate_limit: int,
) -> dict:
    sweep_stem = Path(sweep_script).stem
    candidate_tag = f"o{candidate_offset}-n{candidate_limit}" if candidate_limit > 0 else f"o{candidate_offset}-all"
    run_slug = f"{utc_stamp()}-{TASK_ID}-{target}-l{prefill_len}-{sweep_stem}-{candidate_tag}"
    run_dir_rel = Path("search") / "runs" / run_slug
    json_rel = run_dir_rel / "results_sweep.json"
    candidate_flags = ""
    if candidate_offset > 0:
        candidate_flags += f" --candidate-offset {candidate_offset}"
    if candidate_limit > 0:
        candidate_flags += f" --candidate-limit {candidate_limit}"
    return {
        "id": f"{batch_tag}-{target}-l{prefill_len}-{sweep_stem}-{candidate_tag}",
        "owner": owner,
        "priority": priority,
        "task_id": TASK_ID,
        "run_dir": str(run_dir_rel),
        "cwd": REMOTE_ROOT,
        "command": (
            f"PYTHONPATH={REMOTE_ROOT} python3 {sweep_script} --target {target} "
            f"--prefill-len {prefill_len}{candidate_flags} --json-out {json_rel}"
        ),
        "result_paths": [str(json_rel)],
        "tags": [
            "gpu",
            "exact",
            sweep_stem,
            target,
            f"prefill_{prefill_len}",
            candidate_tag,
        ],
        "notes": (
            f"Exact projection sweep via {sweep_stem} for {target} "
            f"at prefill_len={prefill_len} with candidate window {candidate_tag} "
            f"on the RTX 3090 fallback path."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--owner", default="main")
    parser.add_argument("--priority-base", type=int, default=120)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument("--sweep-script", default=SWEEP_SCRIPT)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    batch_root = STAGING_DIR / f"{args.batch_tag}-{utc_stamp()}"
    manifests_dir = batch_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    priority = args.priority_base
    for target in args.targets:
        for prefill_len in args.lengths:
            manifest = build_manifest(
                args.batch_tag,
                args.owner,
                priority,
                target,
                prefill_len,
                args.sweep_script,
                args.candidate_offset,
                args.candidate_limit,
            )
            manifests.append(manifest)
            priority += 1

    for manifest in manifests:
        run_dir = LOCAL_ROOT / manifest["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / f"{manifest['id']}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.submit:
        subprocess.run(
            ["python3", str(REMOTE_QUEUE), "submit-dir", str(manifests_dir)],
            check=True,
        )

    summary = {
        "batch_tag": args.batch_tag,
        "owner": args.owner,
        "targets": args.targets,
        "lengths": args.lengths,
        "sweep_script": args.sweep_script,
        "candidate_offset": args.candidate_offset,
        "candidate_limit": args.candidate_limit,
        "manifest_count": len(manifests),
        "submitted": bool(args.submit),
        "manifests_dir": str(manifests_dir),
    }
    (batch_root / "README.md").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
