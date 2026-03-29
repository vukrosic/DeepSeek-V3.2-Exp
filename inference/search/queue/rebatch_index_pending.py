#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from queue_runner import LOCK_PATH, PENDING_DIR, dump_json, load_json, pid_is_alive


INDEX_TASK_ID = "03_fp8_index_exact"
DEFAULT_SHARD_SIZE = 20
REMOTE_PYTHON = "/venv/main/bin/python3"
ID_PATTERN = re.compile(r"^gauss-index-batch-20260328-(?P<shape>[^-]+)-cand-(?P<cand>\d+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.index_200_sweep import load_candidate  # noqa: E402


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def lock_is_held() -> bool:
    if not LOCK_PATH.exists():
        return False
    try:
        payload = load_json(LOCK_PATH)
    except Exception:
        return True
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_lock() -> bool:
    if not LOCK_PATH.exists():
        return False
    try:
        payload = load_json(LOCK_PATH)
    except Exception:
        LOCK_PATH.unlink(missing_ok=True)
        return True
    pid = payload.get("pid")
    if not isinstance(pid, int) or not pid_is_alive(pid):
        LOCK_PATH.unlink(missing_ok=True)
        return True
    return False


def acquire_lock_with_retry() -> None:
    while True:
        cleanup_stale_lock()
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "acquired_at": utc_now()}, f)
                f.write("\n")
            return
        except FileExistsError:
            time.sleep(0.2)


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def parse_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception:
        return None
    if payload.get("task_id") != INDEX_TASK_ID:
        return None
    match = ID_PATTERN.match(payload.get("id", ""))
    if not match:
        return None
    command = payload.get("command", "")
    argv = shlex.split(command)
    if "search/index_200_sweep.py" not in command or "--candidate-id" not in argv:
        return None
    shape = match.group("shape")
    candidates = []
    for idx, token in enumerate(argv):
        if token == "--candidate-id" and idx + 1 < len(argv):
            candidates.append(argv[idx + 1])
    if len(candidates) != 1:
        return None
    return {
        "path": path,
        "payload": payload,
        "shape": shape,
        "candidate_id": candidates[0],
        "owner": payload.get("owner", "unknown"),
        "priority": int(payload.get("priority", 80)),
    }


def is_statically_valid(shape: str, candidate_id: str) -> bool:
    return not (shape != "decode" and load_candidate(candidate_id).dot_variant == "broadcast_reduce")


def build_shard_manifest(
    shape: str,
    candidate_ids: list[str],
    shard_index: int,
    shard_size: int,
    owner: str,
    priority: int,
) -> dict[str, Any]:
    first = candidate_ids[0]
    last = candidate_ids[-1]
    shard_label = f"sh{shard_index:03d}-{first}-to-{last}"
    run_dir = f"search/runs/gauss-index-batch-20260328-{shape}-{shard_label}"
    result_path = f"{run_dir}/results.json"
    candidate_flags = " ".join(f"--candidate-id {candidate_id}" for candidate_id in candidate_ids)
    return {
        "id": f"gauss-index-batch-20260328-{shape}-{shard_label}",
        "owner": owner,
        "priority": priority,
        "task_id": INDEX_TASK_ID,
        "run_dir": run_dir,
        "cwd": "/workspace/DeepSeek-V3.2-Exp/inference",
        "command": (
            "PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference "
            f"{REMOTE_PYTHON} search/index_200_sweep.py run "
            f"--shape {shape} "
            f"{candidate_flags} "
            f"--json-out {result_path}"
        ),
        "result_paths": [result_path],
        "tags": ["gpu", "exact", "search", "index", "rebatched", f"shard:{shard_size}"],
        "notes": (
            f"Rebatched exact index shard for {shape}: {first}..{last} "
            f"({len(candidate_ids)} candidate(s) per queue job)."
        ),
        "submitted_at": utc_now(),
        "status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--archive-root", type=Path, default=PENDING_DIR.parent / "archived_pending")
    args = parser.parse_args()
    shard_size = max(args.shard_size, 1)

    acquire_lock_with_retry()
    try:
        staged = []
        for path in sorted(PENDING_DIR.glob("*.json")):
            parsed = parse_manifest(path)
            if parsed is not None:
                staged.append(parsed)

        if not staged:
            print(json.dumps({"rewritten": 0, "archived": 0, "shards": 0}, indent=2))
            return

        by_group: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in staged:
            group_key = (row["shape"], row["owner"], row["priority"])
            by_group.setdefault(group_key, []).append(row)

        archive_dir = args.archive_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-index-rebatch")
        archive_dir.mkdir(parents=True, exist_ok=True)

        for row in staged:
            target = archive_dir / row["path"].name
            row["path"].rename(target)

        rewritten = []
        pruned_invalid = []
        for (shape, owner, priority), rows in sorted(by_group.items()):
            rows.sort(key=lambda row: int(row["candidate_id"].split("-")[-1]))
            candidate_ids = []
            for row in rows:
                if is_statically_valid(shape, row["candidate_id"]):
                    candidate_ids.append(row["candidate_id"])
                else:
                    pruned_invalid.append(row["payload"]["id"])
            for shard_index, offset in enumerate(range(0, len(candidate_ids), shard_size), start=1):
                shard_ids = candidate_ids[offset : offset + shard_size]
                manifest = build_shard_manifest(shape, shard_ids, shard_index, shard_size, owner, priority)
                filename = f"{100000 - manifest['priority']:06d}__{manifest['submitted_at'].replace(':', '').replace('+00:00', 'Z')}__{manifest['id']}.json"
                path = PENDING_DIR / filename
                dump_json(path, manifest)
                rewritten.append(path.name)

        print(
            json.dumps(
                {
                    "archived": len(staged),
                    "shards": len(rewritten),
                    "pruned_invalid": len(pruned_invalid),
                    "archive_dir": str(archive_dir),
                    "rewritten_ids": rewritten[:10],
                },
                indent=2,
            )
        )
    finally:
        release_lock()


if __name__ == "__main__":
    main()
