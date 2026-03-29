#!/usr/bin/env python3
"""
Agent-side helper for staging and submitting exact kernel search batches.

This script is intentionally thin: it only wraps existing queue tooling and
keeps the focus on exact kernel work (GEMM + index).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import List


QUEUE_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = QUEUE_DIR.parents[1]  # inference/
REMOTE_QUEUE = QUEUE_DIR / "remote_queue.py"
GEMM_BATCH = QUEUE_DIR / "generate_projection_batch.py"
INDEX_SWEEP = LOCAL_ROOT / "search" / "index_200_sweep.py"

DEFAULT_GEMM_TARGETS = [
    "mla_wq_b",
    "mla_wkv_b",
    "indexer_wq_b",
    "indexer_wk",
]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def cmd_status(args: argparse.Namespace) -> None:
    run(["python3", str(REMOTE_QUEUE), "status"])


def cmd_tail(args: argparse.Namespace) -> None:
    run(["python3", str(REMOTE_QUEUE), "tail", "--lines", str(args.lines)])


def cmd_submit(args: argparse.Namespace) -> None:
    run(["python3", str(REMOTE_QUEUE), "submit", str(Path(args.manifest).resolve())])


def cmd_submit_dir(args: argparse.Namespace) -> None:
    run(["python3", str(REMOTE_QUEUE), "submit-dir", str(Path(args.manifest_dir).resolve())])


def cmd_gemm_stage(args: argparse.Namespace) -> None:
    batch_tag = args.batch_tag or f"gemm-batch-{utc_stamp()}"
    cmd = [
        "python3",
        str(GEMM_BATCH),
        "--batch-tag",
        batch_tag,
        "--owner",
        args.owner,
        "--priority-base",
        str(args.priority_base),
    ]
    if args.targets:
        cmd.extend(["--targets", *args.targets])
    if args.lengths:
        cmd.extend(["--lengths", *[str(v) for v in args.lengths]])
    if args.submit:
        cmd.append("--submit")
    run(cmd)


def cmd_index_stage(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root) if args.stage_root else (
        LOCAL_ROOT / "search" / "staging" / f"index-batch-{utc_stamp()}"
    )
    cmd = [
        "python3",
        str(INDEX_SWEEP),
        "stage",
        "--stage-root",
        str(stage_root),
        "--prefill-len",
        str(args.prefill_len),
        "--decode-context",
        str(args.decode_context),
        "--owner",
        args.owner,
        "--shard-size",
        str(args.shard_size),
    ]
    run(cmd)
    if args.submit:
        manifests_dir = stage_root / "manifests"
        run(["python3", str(REMOTE_QUEUE), "submit-dir", str(manifests_dir)])


def cmd_bench_smoke(args: argparse.Namespace) -> None:
    batch_tag = args.batch_tag or f"bench-batch-{utc_stamp()}"
    run_slug = f"{utc_stamp()}-{batch_tag}-{args.name}"
    run_dir_rel = Path("search") / "runs" / run_slug
    json_rel = run_dir_rel / "benchmark.json"
    manifest = {
        "id": f"{batch_tag}-{args.name}",
        "owner": args.owner,
        "priority": args.priority_base,
        "task_id": "07_end_to_end_exact",
        "run_dir": str(run_dir_rel),
        "cwd": "/workspace/DeepSeek-V3.2-Exp/inference",
        "command": (
            "PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference /venv/main/bin/python3 -c "
            "\"import json, pathlib; from benchmark import run_benchmark; "
            "cfg = json.load(open('config_671B_v3.2.json')); "
            f"res = run_benchmark(cfg, warmup={args.warmup}, iters={args.iters}); "
            f"path = pathlib.Path('{json_rel}'); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.write_text(json.dumps(res, indent=2) + '\\n'); "
            "print(json.dumps({'json_out': str(path), 'score_ms': res['score_ms'], 'all_ok': res['all_ok']}, indent=2))\""
        ),
        "result_paths": [str(json_rel)],
        "tags": ["gpu", "exact", "bench", args.name],
        "notes": "Single smoke benchmark for the default decode-step kernel path.",
    }
    run_dir = LOCAL_ROOT / manifest["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_root = LOCAL_ROOT / "search" / "staging" / f"{batch_tag}-{utc_stamp()}"
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_root / f"{manifest['id']}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.submit:
        run(["python3", str(REMOTE_QUEUE), "submit", str(manifest_path)])
    else:
        print(manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="Show remote queue status.")
    status.set_defaults(func=cmd_status)

    tail = sub.add_parser("tail", help="Tail remote queue runner log.")
    tail.add_argument("--lines", type=int, default=60)
    tail.set_defaults(func=cmd_tail)

    submit = sub.add_parser("submit", help="Submit a single manifest to the remote queue.")
    submit.add_argument("manifest")
    submit.set_defaults(func=cmd_submit)

    submit_dir = sub.add_parser("submit-dir", help="Submit a directory of manifests.")
    submit_dir.add_argument("manifest_dir")
    submit_dir.set_defaults(func=cmd_submit_dir)

    gemm = sub.add_parser("gemm-stage", help="Stage (and optionally submit) GEMM projection sweep.")
    gemm.add_argument("--batch-tag")
    gemm.add_argument("--owner", default="agent")
    gemm.add_argument("--priority-base", type=int, default=120)
    gemm.add_argument("--targets", nargs="+", default=DEFAULT_GEMM_TARGETS)
    gemm.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    gemm.add_argument("--submit", action="store_true")
    gemm.set_defaults(func=cmd_gemm_stage)

    index = sub.add_parser("index-stage", help="Stage (and optionally submit) exact index sweep.")
    index.add_argument("--stage-root")
    index.add_argument("--owner", default="agent")
    index.add_argument("--prefill-len", type=int, default=256)
    index.add_argument("--decode-context", type=int, default=2048)
    index.add_argument("--shard-size", type=int, default=5)
    index.add_argument("--submit", action="store_true")
    index.set_defaults(func=cmd_index_stage)

    bench = sub.add_parser("bench-smoke", help="Run a single benchmark smoke job for the default decode path.")
    bench.add_argument("--batch-tag")
    bench.add_argument("--owner", default="agent")
    bench.add_argument("--name", default="default-decode")
    bench.add_argument("--priority-base", type=int, default=110)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--iters", type=int, default=3)
    bench.add_argument("--submit", action="store_true")
    bench.set_defaults(func=cmd_bench_smoke)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
