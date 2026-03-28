#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Iterable

from queue_runner import COMPLETED_DIR, FAILED_DIR, LOCK_PATH, PENDING_DIR, dump_json, load_json, pid_is_alive


LOCAL_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = LOCAL_ROOT / "search" / "runs"
DEFAULT_BATCH_ROOT = LOCAL_ROOT / "search" / "staging" / "gauss-index-batch-20260328"
DEFAULT_TASK_ID = "03_fp8_index_exact"
RETRY_SUFFIX_RE = re.compile(r"^(?P<base>.+)-retry(?P<num>\d+)$")


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


def acquire_lock_with_retry() -> None:
    while True:
        if LOCK_PATH.exists():
            try:
                payload = load_json(LOCK_PATH)
            except Exception:
                LOCK_PATH.unlink(missing_ok=True)
                continue
            pid = payload.get("pid")
            if not isinstance(pid, int) or not pid_is_alive(pid):
                LOCK_PATH.unlink(missing_ok=True)
                continue
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


def batch_prefix_from_root(batch_root: Path) -> str:
    index_path = batch_root / "manifest_index.json"
    if index_path.is_file():
        try:
            payload = load_json(index_path)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            batch = payload.get("batch")
            if isinstance(batch, str) and batch:
                return batch
    return batch_root.name


def safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def manifest_sources() -> Iterable[Path]:
    for bucket_dir in [COMPLETED_DIR, FAILED_DIR]:
        if bucket_dir.exists():
            yield from sorted(bucket_dir.glob("*.json"))
    if RUNS_DIR.exists():
        yield from sorted(RUNS_DIR.glob("**/queue_manifest.json"))


def resolve_local_result_path(manifest: dict[str, Any], raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return LOCAL_ROOT / candidate


def load_result_payload(manifest: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    result_paths = manifest.get("result_paths") or []
    if not isinstance(result_paths, list) or not result_paths:
        return None, None
    raw_result_path = result_paths[0]
    if not isinstance(raw_result_path, str):
        return None, None
    result_path = resolve_local_result_path(manifest, raw_result_path)
    if not result_path.exists():
        return result_path, None
    return result_path, safe_load_json(result_path)


def manifest_status(manifest: dict[str, Any]) -> str:
    status = manifest.get("status")
    return status if isinstance(status, str) else ""


def result_status(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    status = result.get("status")
    return status if isinstance(status, str) else ""


def empty_result_reasons(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, dict):
        return ["result_payload_missing"]

    reasons: list[str] = []
    status = result_status(result)
    if status in {"initialized", "empty"}:
        reasons.append(f"result_status_{status}")

    if isinstance(result.get("results"), list) and not result["results"]:
        reasons.append("results_list_empty")

    if isinstance(result.get("variants"), list) and not result["variants"]:
        if status in {"initialized", "empty"}:
            reasons.append("variants_list_empty")

    candidate_count = result.get("candidate_count")
    if isinstance(candidate_count, int) and candidate_count == 0:
        reasons.append("candidate_count_zero")

    exact_count = result.get("exact_count")
    inexact_count = result.get("inexact_count")
    if isinstance(exact_count, int) and isinstance(inexact_count, int) and exact_count == 0 and inexact_count == 0:
        if "candidate_count" in result or "results" in result or "top_10_exact" in result or "top_10_overall" in result:
            reasons.append("no_exact_or_inexact_rows")

    top_exact = result.get("top_10_exact")
    top_overall = result.get("top_10_overall")
    if isinstance(top_exact, list) and isinstance(top_overall, list) and not top_exact and not top_overall:
        if status in {"initialized", "empty"} or candidate_count == 0:
            reasons.append("top_10_lists_empty")

    if result.get("best_exact") is None and result.get("winner") is None:
        if status in {"initialized", "empty"} and not reasons:
            reasons.append("no_best_candidate")

    return reasons


def batch_failure_reasons(manifest: dict[str, Any], result: dict[str, Any] | None, result_path: Path | None) -> list[str]:
    reasons: list[str] = []

    status = manifest_status(manifest)
    if status == "failed":
        reasons.append("manifest_status_failed")

    returncode = manifest.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        reasons.append(f"returncode_{returncode}")

    if manifest.get("startup_error"):
        reasons.append("startup_error")

    missing_result_paths = manifest.get("missing_result_paths")
    if isinstance(missing_result_paths, list) and missing_result_paths:
        reasons.append("missing_result_paths")

    if result_path is not None and not result_path.exists():
        reasons.append("result_file_missing")
    if result_path is not None and result_path.exists() and result is None:
        reasons.append("result_payload_unreadable")

    if isinstance(result, dict):
        result_state = result_status(result)
        if result_state == "failed":
            reasons.append("result_status_failed")
        if result.get("runtime_error"):
            reasons.append("result_runtime_error")

    return reasons


def identify_retry_jobs(
    batch_root: Path,
    task_id: str,
    batch_prefix: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for source_path in manifest_sources():
        manifest = safe_load_json(source_path)
        if manifest is None:
            continue

        job_id = manifest.get("id")
        if not isinstance(job_id, str) or job_id in seen_ids:
            continue
        seen_ids.add(job_id)

        if manifest.get("task_id") != task_id:
            continue
        if not job_id.startswith(batch_prefix):
            continue

        result_path, result_payload = load_result_payload(manifest)
        failure_reasons = batch_failure_reasons(manifest, result_payload, result_path)
        empty_reasons = empty_result_reasons(result_payload)

        retry_reasons: list[str] = []
        retry_kind = ""
        if failure_reasons:
            retry_reasons = failure_reasons
            retry_kind = "batch_failure"
        elif empty_reasons:
            retry_reasons = empty_reasons
            retry_kind = "empty_result"

        if not retry_reasons:
            continue

        jobs.append(
            {
                "source_path": source_path,
                "manifest": manifest,
                "result_path": result_path,
                "result_payload": result_payload,
                "retry_kind": retry_kind,
                "retry_reasons": retry_reasons,
            }
        )

    return sorted(jobs, key=lambda row: row["manifest"]["id"])


def existing_ids() -> set[str]:
    ids: set[str] = set()
    if PENDING_DIR.exists():
        for path in sorted(PENDING_DIR.glob("*.json")):
            manifest = safe_load_json(path)
            if manifest is None:
                continue
            job_id = manifest.get("id")
            if isinstance(job_id, str):
                ids.add(job_id)
    for path in manifest_sources():
        manifest = safe_load_json(path)
        if manifest is None:
            continue
        job_id = manifest.get("id")
        if isinstance(job_id, str):
            ids.add(job_id)
    return ids


def ids_from_dir(root: Path) -> set[str]:
    ids: set[str] = set()
    if not root.exists():
        return ids
    for path in sorted(root.glob("*.json")):
        manifest = safe_load_json(path)
        if manifest is None:
            continue
        job_id = manifest.get("id")
        if isinstance(job_id, str):
            ids.add(job_id)
    return ids


def next_retry_id(base_id: str, used_ids: set[str]) -> tuple[str, int]:
    highest = 0
    for candidate_id in used_ids:
        match = RETRY_SUFFIX_RE.match(candidate_id)
        if match and match.group("base") == base_id:
            highest = max(highest, int(match.group("num")))
    attempt = highest + 1
    while True:
        retry_id = f"{base_id}-retry{attempt:02d}"
        if retry_id not in used_ids:
            return retry_id, attempt
        attempt += 1


def rewrite_command(
    command: str,
    source_run_dir: str,
    source_result_paths: list[str],
    target_run_dir: str,
    target_result_paths: list[str],
) -> str:
    rewritten = command.replace(source_run_dir, target_run_dir)
    for source_path, target_path in zip(source_result_paths, target_result_paths):
        rewritten = rewritten.replace(source_path, target_path)
    return rewritten


def build_retry_manifest(
    manifest: dict[str, Any],
    attempt: int,
    retry_reasons: list[str],
    retry_kind: str,
    output_kind: str,
    output_root: Path,
    priority_bump: int,
) -> tuple[dict[str, Any], Path]:
    source_id = manifest["id"]
    retry_id = f"{source_id}-retry{attempt:02d}"
    source_run_dir = str(manifest["run_dir"])
    target_run_dir = f"search/runs/{retry_id}"

    source_result_paths = [path for path in manifest.get("result_paths", []) if isinstance(path, str)]
    if not source_result_paths:
        source_result_paths = [f"{source_run_dir}/results.json"]
    target_result_paths = [f"{target_run_dir}/{Path(path).name}" for path in source_result_paths]

    command = manifest.get("command", "")
    if not isinstance(command, str):
        command = ""
    command = rewrite_command(command, source_run_dir, source_result_paths, target_run_dir, target_result_paths)

    priority = manifest.get("priority", 80)
    if not isinstance(priority, int):
        priority = 80
    priority += priority_bump

    submitted_at = utc_now()
    notes = manifest.get("notes", "")
    if not isinstance(notes, str):
        notes = ""
    retry_note = f"Retry of {source_id} after {retry_kind}: {', '.join(retry_reasons)}."
    if notes:
        notes = f"{retry_note} Original notes: {notes}"
    else:
        notes = retry_note

    tags = manifest.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tag_list = [tag for tag in tags if isinstance(tag, str)]
    for tag in ["retry", f"retry_kind:{retry_kind}"]:
        if tag not in tag_list:
            tag_list.append(tag)

    payload = dict(manifest)
    payload.update(
        {
            "id": retry_id,
            "priority": priority,
            "run_dir": target_run_dir,
            "command": command,
            "result_paths": target_result_paths,
            "tags": tag_list,
            "notes": notes,
            "submitted_at": submitted_at,
            "status": "pending",
            "retry_of": source_id,
            "retry_attempt": attempt,
            "retry_kind": retry_kind,
            "retry_reasons": retry_reasons,
            "source_run_dir": source_run_dir,
        }
    )

    if output_kind == "pending":
        filename = f"{100000 - priority:06d}__{submitted_at.replace(':', '').replace('+00:00', 'Z')}__{retry_id}.json"
        output_path = output_root / filename
    else:
        output_path = output_root / f"{retry_id}.json"

    return payload, output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--emit", choices=["staging", "pending"], default="staging")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--priority-bump", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_root = args.batch_root
    batch_prefix = batch_prefix_from_root(batch_root)
    output_root = args.output_root
    if output_root is None:
        output_root = PENDING_DIR if args.emit == "pending" else batch_root / "retry_manifests"
    output_root = output_root.resolve()

    jobs = identify_retry_jobs(batch_root, args.task_id, batch_prefix)
    used_ids = existing_ids()
    if args.emit == "staging":
        used_ids.update(ids_from_dir(output_root))
    used_ids.update(job["manifest"]["id"] for job in jobs)

    retries: list[dict[str, Any]] = []
    should_lock = args.emit == "pending" and not args.dry_run
    if should_lock:
        acquire_lock_with_retry()
    try:
        for job in jobs:
            source_id = job["manifest"]["id"]
            retry_id, attempt = next_retry_id(source_id, used_ids)
            used_ids.add(retry_id)
            payload, output_path = build_retry_manifest(
                job["manifest"],
                attempt,
                job["retry_reasons"],
                job["retry_kind"],
                args.emit,
                output_root,
                args.priority_bump,
            )
            payload["id"] = retry_id
            if not args.dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                dump_json(output_path, payload)
            retries.append(
                {
                    "source_id": source_id,
                    "retry_id": retry_id,
                    "retry_kind": job["retry_kind"],
                    "retry_reasons": job["retry_reasons"],
                    "output_path": str(output_path),
                    "result_path": str(job["result_path"]) if job["result_path"] is not None else None,
                }
            )

        summary = {
            "batch_root": str(batch_root),
            "batch_prefix": batch_prefix,
            "task_id": args.task_id,
            "emit": args.emit,
            "dry_run": args.dry_run,
            "scanned_jobs": len(list(manifest_sources())),
            "retry_candidates": len(jobs),
            "retry_manifests": len(retries),
            "output_root": str(output_root),
            "retries": retries[:20],
        }

        print(json.dumps(summary, indent=2))
    finally:
        if should_lock:
            release_lock()


if __name__ == "__main__":
    main()
