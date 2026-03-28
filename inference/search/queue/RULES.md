# Queue Rules

These are the operating rules for agents using the single-GPU experiment queue.

## Non-Negotiable Rules

- one queue runner owns the GPU
- agents do not manually launch remote benchmark jobs outside the queue
- one manifest equals one experiment job
- one job writes only inside its own run folder and the queue log folder
- no job may mutate shared reports directly while it is running
- shared reports are updated only after results are reviewed

## What Agents May Do In Parallel

- read code
- build scripts
- create run folders
- write notes
- generate manifests
- analyze completed results

## What Agents May Not Do In Parallel

- run remote timing jobs directly
- run two GPU benchmark jobs at once
- write into another agent's run folder
- overwrite queue state files by hand

## Submission Rules

Before submitting, the agent must:

1. have a run folder
2. have the experiment script already synced or ready to sync
3. know the exact output files the command will write
4. know the exactness gate for the experiment
5. confirm the command is idempotent enough to survive a retry
6. confirm `run_dir` is new or empty and that no declared `result_paths` already exist
7. confirm `run_dir` and `result_paths` are relative to `cwd`

## Manifest Rules

Every manifest must include:

- `id`
- `owner`
- `priority`
- `task_id`
- `run_dir`
- `cwd`
- `command`
- `result_paths`
- `tags`
- `notes`

Path rules:

- `cwd` must be an absolute path on the target machine
- `run_dir` must be relative to `cwd`
- every `result_path` must be relative to `cwd`
- every `result_path` must live under `run_dir`
- manifests that point at already-populated run folders or existing results should be treated as stale and rejected, not submitted

Priority rules:

- higher `priority` runs first
- if priorities tie, older jobs run first
- use high priority only for blockers

## Result Rules

- raw benchmark output goes into the run folder
- queue execution logs go into `search/queue/logs/`
- successful manifests move to `completed/`
- failed manifests move to `failed/`
- completed jobs do not update the leaderboard automatically

## Recommended Agent Flow

1. create or choose a run folder
2. implement the experiment locally
3. validate the command locally if possible
4. submit a manifest
5. wait for completion
6. inspect raw results
7. only then update shared reports

## Collision Policy

If a job needs a special environment, large memory, or long exclusive time, say so in `notes` and `tags`.

If two jobs would interfere semantically, they still go through the same queue. The queue serializes them.

## Failure Policy

- if a job fails, inspect the log before retrying
- do not blindly resubmit the same broken command repeatedly
- retries should use a new manifest id unless you are explicitly reusing the failed id with `--allow-failed-retry`
- if a previously failed id is still pending elsewhere, prune or remove the duplicate before retrying

## Duplicate Policy

- duplicate ids across `pending/`, `running/`, `completed/`, and `failed/` are operational debt
- the queue runner rejects submissions whose id already exists in `pending/`, `running/`, or `completed/`
- `prune-pending` is the safe cleanup path for stale duplicate ids already in the queue
