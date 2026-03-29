#!/usr/bin/env python3
import argparse
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable, List


LOCAL_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/workspace/DeepSeek-V3.2-Exp/inference"
REMOTE_PYTHON = "/venv/main/bin/python3"
RUNS_DIR = LOCAL_ROOT / "search" / "runs"
STAGING_DIR = LOCAL_ROOT / "search" / "staging"
REMOTE_QUEUE = LOCAL_ROOT / "search" / "queue" / "remote_queue.py"
TASK_ID = "01_act_quant_exact"
SWEEP_SCRIPT = "search/act_200_sweep.py"

DEFAULT_CASES = [
    "index_q",
    "kv_cache",
    "mla_input_x",
    "mla_qr",
    "mla_kv",
    "decode_input_x",
    "decode_qr",
    "decode_kv",
]
DEFAULT_LENGTHS = [128, 256, 512, 1024]
DEFAULT_CANDIDATE_WINDOWS = ["0:50", "50:50", "100:50", "150:50"]


@dataclass(frozen=True)
class CandidateWindow:
    offset: int
    limit: int

    @property
    def tag(self) -> str:
        return f"o{self.offset}-all" if self.limit <= 0 else f"o{self.offset}-n{self.limit}"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def parse_window(spec: str) -> CandidateWindow:
    if ":" not in spec:
        raise SystemExit(f"invalid candidate window '{spec}', expected OFFSET:LIMIT")
    offset_text, limit_text = spec.split(":", 1)
    try:
        offset = int(offset_text)
        limit = int(limit_text)
    except ValueError as exc:
        raise SystemExit(f"invalid candidate window '{spec}', expected integers") from exc
    if offset < 0:
        raise SystemExit(f"candidate window offset must be non-negative: {spec}")
    if limit < 0:
        raise SystemExit(f"candidate window limit must be non-negative: {spec}")
    return CandidateWindow(offset=offset, limit=limit)


def candidate_window_specs(raw_windows: Iterable[str]) -> List[CandidateWindow]:
    specs = [parse_window(spec) for spec in raw_windows]
    if not specs:
        specs = [parse_window(spec) for spec in DEFAULT_CANDIDATE_WINDOWS]
    return specs


def build_manifest(
    batch_tag: str,
    owner: str,
    priority: int,
    case: str,
    prefill_len: int,
    window: CandidateWindow,
    config: str,
    batch_size: int,
    warmup: int,
    iters: int,
    seed: int,
    sweep_script: str,
) -> dict:
    sweep_stem = Path(sweep_script).stem
    candidate_tag = window.tag
    run_slug = f"{utc_stamp()}-{TASK_ID}-{case}-l{prefill_len}-{sweep_stem}-{candidate_tag}"
    run_dir_rel = Path("search") / "runs" / run_slug
    json_rel = run_dir_rel / "results_act_sweep.json"
    candidate_flags = f" --candidate-offset {window.offset}"
    if window.limit > 0:
        candidate_flags += f" --candidate-limit {window.limit}"
    else:
        candidate_flags += " --candidate-limit 0"
    return {
        "id": f"{batch_tag}-{case}-l{prefill_len}-{sweep_stem}-{candidate_tag}",
        "owner": owner,
        "priority": priority,
        "task_id": TASK_ID,
        "run_dir": str(run_dir_rel),
        "cwd": REMOTE_ROOT,
        "command": (
            f"PYTHONPATH={REMOTE_ROOT} {REMOTE_PYTHON} {sweep_script} "
            f"--config {config} "
            f"--batch-size {batch_size} "
            f"--prefill-len {prefill_len} "
            f"--case {case}"
            f"{candidate_flags} "
            f"--warmup {warmup} "
            f"--iters {iters} "
            f"--seed {seed} "
            f"--json-out {json_rel}"
        ),
        "result_paths": [str(json_rel)],
        "tags": [
            "gpu",
            "exact",
            "act",
            sweep_stem,
            case,
            f"prefill_{prefill_len}",
            candidate_tag,
        ],
        "notes": (
            f"Queue-safe act quant sweep via {sweep_stem} for {case} "
            f"at prefill_len={prefill_len} with candidate window {candidate_tag} "
            f"on the RTX 3090 fallback path."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--owner", default="main")
    parser.add_argument("--priority-base", type=int, default=120)
    parser.add_argument("--config", default="config_671B_v3.2.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--candidate-window", action="append", default=[])
    parser.add_argument("--sweep-script", default=SWEEP_SCRIPT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    batch_root = STAGING_DIR / f"{args.batch_tag}-{utc_stamp()}"
    manifests_dir = batch_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    windows = candidate_window_specs(args.candidate_window)
    manifests = []
    priority = args.priority_base
    for case in args.cases:
        for prefill_len in args.lengths:
            for window in windows:
                manifest = build_manifest(
                    args.batch_tag,
                    args.owner,
                    priority,
                    case,
                    prefill_len,
                    window,
                    args.config,
                    args.batch_size,
                    args.warmup,
                    args.iters,
                    args.seed,
                    args.sweep_script,
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
        "cases": args.cases,
        "lengths": args.lengths,
        "candidate_windows": [window.tag for window in windows],
        "sweep_script": args.sweep_script,
        "manifest_count": len(manifests),
        "submitted": bool(args.submit),
        "manifests_dir": str(manifests_dir),
    }
    (batch_root / "README.md").write_text(json.dumps(summary, indent=2) + "\n")
    (batch_root / "manifest_index.json").write_text(json.dumps({
        "batch_tag": args.batch_tag,
        "owner": args.owner,
        "cases": args.cases,
        "lengths": args.lengths,
        "candidate_windows": [window.__dict__ for window in windows],
        "manifest_count": len(manifests),
        "manifests": manifests,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
