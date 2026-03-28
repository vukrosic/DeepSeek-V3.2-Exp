# GPU Experiment Queue

This queue is the only approved path for running benchmark or search jobs on the remote GPU.

## Why This Exists

Microbenchmarks collide with each other. If two agents run timing jobs at the same time, the numbers are contaminated and the search becomes noise. The queue makes one process the exclusive GPU owner and forces everything else to submit jobs instead of free-running experiments.

## Model

- many agents may prepare experiments in parallel
- exactly one queue runner may execute GPU jobs at a time
- every GPU experiment is a manifest in `pending/`
- the runner leases one manifest, runs it, writes logs, and moves it to `completed/` or `failed/`

## Layout

- `pending/`: queued jobs waiting to run
- `running/`: the currently leased job
- `completed/`: finished successful jobs
- `failed/`: finished failed jobs
- `logs/`: stdout and stderr capture per job
- `templates/`: manifest templates
- `queue_runner.py`: submission, status, and execution CLI
- `rebatch_index_pending.py`: backlog compaction tool for one-candidate index queues
- `RULES.md`: agent rules and submission contract

## Core Rule

Agents do not manually start remote GPU benchmarks anymore. Agents submit manifests. The queue runner owns the GPU.

## Basic Commands

Show local queue folders:
```bash
cd inference
python3 search/queue/queue_runner.py status
```

Show remote queue status:
```bash
cd inference
python3 search/queue/remote_queue.py status
```

Submit a local manifest into the remote queue:
```bash
cd inference
python3 search/queue/remote_queue.py submit path/to/manifest.json
```

Submit a directory tree of manifests into the remote queue:
```bash
cd inference
python3 search/queue/remote_queue.py submit-dir path/to/manifests/
```

Submit a manifest into the local queue folders:
```bash
cd inference
python3 search/queue/queue_runner.py submit path/to/manifest.json
```

Run one job:
```bash
cd inference
python3 search/queue/queue_runner.py run-next
```

Prune queued duplicates that already have a `completed/` or `running/` copy:
```bash
cd inference
python3 search/queue/queue_runner.py prune-pending
```

Run as the single queue worker:
```bash
cd inference
python3 search/queue/queue_runner.py loop --poll-seconds 2
```

## Remote Usage

On the GPU machine, start one long-lived runner from the repo root:

```bash
cd /workspace/DeepSeek-V3.2-Exp/inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py loop --poll-seconds 2
```

If you want it detached:

```bash
cd /workspace/DeepSeek-V3.2-Exp/inference
nohup bash -lc 'PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py loop --poll-seconds 2' > search/queue/runner.out 2>&1 &
```

Tail the remote runner log from local:

```bash
cd inference
python3 search/queue/remote_queue.py tail --lines 60
```

Rebatch a pending one-candidate index backlog into shard jobs:
```bash
cd /workspace/DeepSeek-V3.2-Exp/inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/rebatch_index_pending.py --shard-size 20
```

Build a live queue snapshot and compact report on the remote repo:
```bash
cd /workspace/DeepSeek-V3.2-Exp/inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/report_tools/queue_snapshot.py
```

## Submission Contract

Every manifest must:

- have a unique `id`
- name its `owner`
- name the `task_id`
- point to a single new `run_dir`
- provide a `cwd`
- provide the exact `command`
- list the expected `result_paths`
- write only inside its `run_dir` plus queue logs

The queue runner now rejects duplicate ids already present in `pending/`, `running/`, or `completed/`.
If you intentionally want to retry a failed job, use a new `id` or pass `--allow-failed-retry` to the local `queue_runner.py submit` command.
It also rejects stale or malformed submissions when:

- `cwd` is missing or not absolute
- `run_dir` is absolute, escapes `cwd`, or is already populated
- any `result_path` is absolute, escapes `cwd`, lies outside `run_dir`, or already exists

Read `RULES.md` before submitting jobs.
