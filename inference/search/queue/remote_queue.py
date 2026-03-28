#!/usr/bin/env python3
import argparse
import datetime as dt
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path


GPU = Path("/root/auto-research/gpu")
REMOTE_ROOT = "/workspace/DeepSeek-V3.2-Exp/inference"


def gpu_run(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(GPU), "run", command],
        text=True,
        capture_output=True,
        check=True,
    )


def cmd_status() -> None:
    proc = gpu_run(
        f"cd {REMOTE_ROOT} && PYTHONPATH={REMOTE_ROOT} python3 search/queue/queue_runner.py status"
    )
    print(proc.stdout.strip())


def cmd_submit(manifest_path: str) -> None:
    local = Path(manifest_path).resolve()
    remote_tmp = f"/workspace/{local.name}"
    subprocess.run([str(GPU), "upload", str(local), remote_tmp], check=True, text=True)
    proc = gpu_run(
        f"cd {REMOTE_ROOT} && PYTHONPATH={REMOTE_ROOT} python3 search/queue/queue_runner.py submit {shlex.quote(remote_tmp)}"
    )
    print(proc.stdout.strip())


def cmd_submit_dir(manifest_dir: str) -> None:
    local = Path(manifest_dir).resolve()
    if not local.is_dir():
        raise SystemExit(f"not a directory: {local}")
    suffix = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    remote_tmp = f"/workspace/{local.name}-{suffix}"
    subprocess.run([str(GPU), "upload", str(local), remote_tmp], check=True, text=True)
    proc = gpu_run(
        " && ".join(
            [
                f"cd {REMOTE_ROOT}",
                f"find {shlex.quote(remote_tmp)} -type f -name '*.json' | sort | while read -r f; do "
                f"PYTHONPATH={REMOTE_ROOT} python3 search/queue/queue_runner.py submit \"$f\"; "
                "done",
            ]
        )
    )
    print(proc.stdout.strip())


def cmd_tail(lines: int) -> None:
    proc = gpu_run(f"cd {REMOTE_ROOT} && tail -n {lines} search/queue/runner.out")
    print(proc.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    submit = sub.add_parser("submit")
    submit.add_argument("manifest_path")
    submit_dir = sub.add_parser("submit-dir")
    submit_dir.add_argument("manifest_dir")
    tail = sub.add_parser("tail")
    tail.add_argument("--lines", type=int, default=40)
    args = parser.parse_args()

    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "submit":
        cmd_submit(args.manifest_path)
    elif args.cmd == "submit-dir":
        cmd_submit_dir(args.manifest_dir)
    elif args.cmd == "tail":
        cmd_tail(args.lines)


if __name__ == "__main__":
    main()
