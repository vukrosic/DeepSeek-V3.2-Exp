#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List


QUEUE_DIR = Path(__file__).resolve().parent
PENDING_DIR = QUEUE_DIR / "pending"
RUNNING_DIR = QUEUE_DIR / "running"
COMPLETED_DIR = QUEUE_DIR / "completed"
FAILED_DIR = QUEUE_DIR / "failed"
LOGS_DIR = QUEUE_DIR / "logs"
LOCK_PATH = QUEUE_DIR / ".gpu_queue_lock"

REQUIRED_KEYS = {
    "id",
    "owner",
    "priority",
    "task_id",
    "run_dir",
    "cwd",
    "command",
    "result_paths",
    "tags",
    "notes",
}

BUCKET_DIRS = {
    "pending": PENDING_DIR,
    "running": RUNNING_DIR,
    "completed": COMPLETED_DIR,
    "failed": FAILED_DIR,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_dirs() -> None:
    for path in [PENDING_DIR, RUNNING_DIR, COMPLETED_DIR, FAILED_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock_payload() -> Dict[str, Any] | None:
    if not LOCK_PATH.is_file():
        return None
    try:
        return load_json(LOCK_PATH)
    except (json.JSONDecodeError, OSError):
        return None


def lock_status() -> Dict[str, Any] | None:
    if LOCK_PATH.is_dir():
        return {
            "path": str(LOCK_PATH),
            "kind": "legacy_dir",
            "stale": not current_running(),
        }
    payload = read_lock_payload()
    if payload is None:
        return None
    pid = payload.get("pid")
    return {
        "path": str(LOCK_PATH),
        "kind": "pid_file",
        "pid": pid,
        "acquired_at": payload.get("acquired_at"),
        "stale": not isinstance(pid, int) or not pid_is_alive(pid),
    }


def validate_manifest(payload: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise SystemExit(f"manifest missing keys: {', '.join(missing)}")
    if not isinstance(payload["result_paths"], list) or not payload["result_paths"]:
        raise SystemExit("manifest result_paths must be a non-empty list")
    if not isinstance(payload["priority"], int):
        raise SystemExit("manifest priority must be an int")


def normalize_relative_path(raw_path: str, field: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise SystemExit(f"manifest {field} must be relative to cwd")
    if any(part == ".." for part in path.parts):
        raise SystemExit(f"manifest {field} must not escape cwd")
    return path


def validate_submission_hygiene(payload: Dict[str, Any]) -> None:
    cwd = Path(payload["cwd"])
    if not cwd.is_absolute():
        raise SystemExit("manifest cwd must be an absolute path")
    if not cwd.exists():
        raise SystemExit(f"manifest cwd does not exist: {cwd}")

    run_dir_rel = normalize_relative_path(payload["run_dir"], "run_dir")
    run_dir = cwd / run_dir_rel
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"manifest run_dir is not empty: {run_dir}")

    for raw_result_path in payload["result_paths"]:
        result_rel = normalize_relative_path(raw_result_path, "result_paths")
        result_path = cwd / result_rel
        try:
            result_path.relative_to(run_dir)
        except ValueError:
            raise SystemExit(
                f"manifest result_path must be inside run_dir: {raw_result_path}"
            )
        if result_path.exists():
            raise SystemExit(f"manifest result_path already exists: {result_path}")


def manifest_filename(payload: Dict[str, Any]) -> str:
    created = payload.get("submitted_at", utc_now()).replace(":", "").replace("+00:00", "Z")
    return f"{100000 - payload['priority']:06d}__{created}__{payload['id']}.json"


def existing_job_buckets(job_id: str) -> Dict[str, List[Path]]:
    matches: Dict[str, List[Path]] = {}
    for bucket_name, bucket_dir in BUCKET_DIRS.items():
        bucket_matches: List[Path] = []
        for path in bucket_dir.glob("*.json"):
            try:
                payload = load_json(path)
            except (json.JSONDecodeError, OSError):
                continue
            if payload.get("id") == job_id:
                bucket_matches.append(path)
        if bucket_matches:
            matches[bucket_name] = sorted(bucket_matches)
    return matches


def cmd_submit(manifest_path: str, allow_failed_retry: bool = False) -> None:
    ensure_dirs()
    source = Path(manifest_path)
    payload = load_json(source)
    validate_manifest(payload)
    validate_submission_hygiene(payload)
    existing = existing_job_buckets(payload["id"])
    blocked = [bucket for bucket in ["pending", "running", "completed"] if bucket in existing]
    if blocked:
        raise SystemExit(f"job id already exists in {', '.join(blocked)}")
    if "failed" in existing and not allow_failed_retry:
        raise SystemExit("job id already exists in failed; explicit retry required")
    payload.setdefault("submitted_at", utc_now())
    payload["status"] = "pending"
    target = PENDING_DIR / manifest_filename(payload)
    dump_json(target, payload)
    print(target)


def acquire_lock() -> None:
    if LOCK_PATH.is_dir():
        # Compatibility cleanup for the original directory lock.
        if current_running():
            raise SystemExit("queue runner lock is already held")
        shutil.rmtree(LOCK_PATH, ignore_errors=True)
    elif LOCK_PATH.exists():
        payload = read_lock_payload()
        pid = payload.get("pid") if payload else None
        if isinstance(pid, int) and pid_is_alive(pid):
            raise SystemExit("queue runner lock is already held")
        LOCK_PATH.unlink(missing_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit("queue runner lock is already held")
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "acquired_at": utc_now()}, f)
        f.write("\n")


def release_lock() -> None:
    if LOCK_PATH.is_dir():
        shutil.rmtree(LOCK_PATH, ignore_errors=True)
    else:
        LOCK_PATH.unlink(missing_ok=True)


def pending_manifests() -> List[Path]:
    return sorted(PENDING_DIR.glob("*.json"))


def current_running() -> List[Path]:
    return sorted(RUNNING_DIR.glob("*.json"))


def reconcile_state() -> List[str]:
    repairs: List[str] = []
    completed_names = {path.name for path in COMPLETED_DIR.glob("*.json")}
    failed_names = {path.name for path in FAILED_DIR.glob("*.json")}
    for path in list(RUNNING_DIR.glob("*.json")):
        if path.name in completed_names or path.name in failed_names:
            path.unlink()
            repairs.append(f"removed stale running duplicate: {path.name}")
    return repairs


def verify_result_paths(payload: Dict[str, Any]) -> List[str]:
    missing = []
    for rel_path in payload["result_paths"]:
        if not (Path(payload["cwd"]) / rel_path).exists():
            missing.append(rel_path)
    return missing


def prune_pending() -> Dict[str, int]:
    removed_completed = 0
    removed_running = 0
    removed_pending_duplicates = 0

    completed_ids = set()
    running_ids = set()
    for bucket_name, target_set in [("completed", completed_ids), ("running", running_ids)]:
        for path in BUCKET_DIRS[bucket_name].glob("*.json"):
            try:
                payload = load_json(path)
            except (json.JSONDecodeError, OSError):
                continue
            job_id = payload.get("id")
            if job_id:
                target_set.add(job_id)

    grouped_pending: Dict[str, List[Path]] = {}
    for path in sorted(PENDING_DIR.glob("*.json")):
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        job_id = payload.get("id")
        if not job_id:
            continue
        grouped_pending.setdefault(job_id, []).append(path)

    for job_id, paths in grouped_pending.items():
        if job_id in completed_ids:
            for path in paths:
                path.unlink(missing_ok=True)
                removed_completed += 1
            continue
        if job_id in running_ids:
            for path in paths:
                path.unlink(missing_ok=True)
                removed_running += 1
            continue
        for path in paths[1:]:
            path.unlink(missing_ok=True)
            removed_pending_duplicates += 1

    return {
        "removed_pending_for_completed_ids": removed_completed,
        "removed_pending_for_running_ids": removed_running,
        "removed_pending_duplicate_ids": removed_pending_duplicates,
    }


def cmd_status() -> None:
    ensure_dirs()
    repairs = reconcile_state()
    running = current_running()
    payload = {
        "pending": len(list(PENDING_DIR.glob("*.json"))),
        "running": len(running),
        "completed": len(list(COMPLETED_DIR.glob("*.json"))),
        "failed": len(list(FAILED_DIR.glob("*.json"))),
        "running_job": load_json(running[0]) if running else None,
        "repairs": repairs,
        "lock": lock_status(),
    }
    print(json.dumps(payload, indent=2))


def run_job(manifest_path: Path) -> int:
    payload = load_json(manifest_path)
    job_id = payload["id"]
    log_path = LOGS_DIR / f"{job_id}.log"
    payload["status"] = "running"
    payload["started_at"] = utc_now()
    dump_json(manifest_path, payload)

    with open(log_path, "a") as log_file:
        log_file.write(f"[queue] start {job_id} {payload['started_at']}\n")
        startup_error = None
        try:
            proc = subprocess.run(
                payload["command"],
                shell=True,
                cwd=payload["cwd"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            startup_error = str(exc)
            log_file.write(f"[queue] startup error: {startup_error}\n")
            proc = subprocess.CompletedProcess(
                args=payload["command"],
                returncode=127,
            )
        finished_at = utc_now()
        missing_paths = []
        if proc.returncode == 0:
            missing_paths = verify_result_paths(payload)
            if missing_paths:
                proc = subprocess.CompletedProcess(
                    args=payload["command"],
                    returncode=2,
                )
                log_file.write(
                    f"[queue] missing declared result paths: {', '.join(missing_paths)}\n"
                )
        payload["finished_at"] = finished_at
        payload["returncode"] = proc.returncode
        payload["log_path"] = str(log_path.relative_to(QUEUE_DIR.parent))
        payload["status"] = "completed" if proc.returncode == 0 else "failed"
        if startup_error:
            payload["startup_error"] = startup_error
        if missing_paths:
            payload["missing_result_paths"] = missing_paths
        dump_json(manifest_path, payload)
        log_file.write(f"[queue] finish {job_id} {finished_at} returncode={proc.returncode}\n")
        return proc.returncode


def cmd_run_next(quiet_idle: bool = False) -> bool:
    ensure_dirs()
    acquire_lock()
    try:
        repairs = reconcile_state()
        for repair in repairs:
            print(repair)
        if current_running():
            raise SystemExit("a job is already in running/")
        pending = pending_manifests()
        if not pending:
            if not quiet_idle:
                print("no pending jobs")
            return False
        source = pending[0]
        leased = RUNNING_DIR / source.name
        source.rename(leased)
        returncode = run_job(leased)
        destination_dir = COMPLETED_DIR if returncode == 0 else FAILED_DIR
        leased.rename(destination_dir / leased.name)
        print((destination_dir / leased.name).relative_to(QUEUE_DIR.parent))
        return True
    finally:
        release_lock()


def cmd_prune_pending() -> None:
    ensure_dirs()
    acquire_lock()
    try:
        repairs = prune_pending()
        print(json.dumps(repairs, indent=2))
    finally:
        release_lock()


def cmd_loop(poll_seconds: float) -> None:
    ensure_dirs()
    while True:
        try:
            cmd_run_next(quiet_idle=True)
        except SystemExit as exc:
            message = str(exc)
            if message != "queue runner lock is already held":
                print(message, file=sys.stderr)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("manifest_path")
    submit.add_argument("--allow-failed-retry", action="store_true")

    sub.add_parser("status")
    sub.add_parser("run-next")
    sub.add_parser("reconcile")
    sub.add_parser("prune-pending")

    loop = sub.add_parser("loop")
    loop.add_argument("--poll-seconds", type=float, default=2.0)

    args = parser.parse_args()
    if args.cmd == "submit":
        cmd_submit(args.manifest_path, allow_failed_retry=args.allow_failed_retry)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "run-next":
        cmd_run_next()
    elif args.cmd == "reconcile":
        ensure_dirs()
        for repair in reconcile_state():
            print(repair)
    elif args.cmd == "prune-pending":
        cmd_prune_pending()
    elif args.cmd == "loop":
        cmd_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
