# Automation Guide

This file is the GitHub-facing handoff for contributors who want to continue the 3090 optimization work with scripts, queues, or coding agents.

## Goal

Keep improving the inference path on an RTX 3090 without turning the repo into an unreviewable pile of one-off experiments.

Use the repo in two layers:

- shared code and accepted wins live in `inference/kernel.py` and `inference/model.py`
- search, queueing, prompts, and reports live under `inference/search/`

## Read This First

1. [`README.md`](README.md)
2. [`PROCESS.md`](PROCESS.md)
3. [`KERNEL_SPEED_SEARCH_SKILL.md`](KERNEL_SPEED_SEARCH_SKILL.md)
4. [`AGENT_PLAYBOOK.md`](AGENT_PLAYBOOK.md)
5. [`queue/README.md`](queue/README.md)
6. [`reports/current-3090-baseline.md`](reports/current-3090-baseline.md)
7. [`reports/leaderboard.md`](reports/leaderboard.md)

## Acceptance Lanes

This repo now uses two correctness lanes:

- `exact`
  Use this for routing-sensitive paths such as index selection, masks, top-k, or anything where tiny drift can flip behavior.
- `near-exact`
  Use this only for smooth algebra kernels such as projection or activation paths, with an explicit tolerance and a downstream confirmation step.

Do not silently move a task from `exact` to `near-exact`.

## Useful Files

- search scripts
  - `act_200_sweep.py`
  - `act_400_sweep.py`
  - `projection_100_sweep.py`
  - `projection_400_sweep.py`
  - `index_200_sweep.py`
  - `index_400_sweep.py`
- queue tooling
  - `queue/queue_runner.py`
  - `queue/remote_queue.py`
  - `queue/generate_projection_batch.py`
  - `queue/generate_act_batch.py`
  - `queue/rebatch_index_pending.py`
  - `queue/retry_failed_index_batches.py`
- reports and process docs
  - `PROCESS.md`
  - `AGENT_PLAYBOOK.md`
  - `reports/current-3090-baseline.md`
  - `reports/leaderboard.md`
- agent prompts
  - `prompts/01_search_operator.md`
  - `prompts/02_verification_and_benchmarking.md`
  - `prompts/03_experiment_design.md`

## Standard Workflow

1. Pick one task family.
2. Confirm the current accepted baseline.
3. Build a narrow candidate family.
4. Validate locally with `py_compile` and shape checks.
5. Stage queue manifests, not ad hoc GPU commands.
6. Submit through the queue.
7. Inspect raw results before touching shared reports.
8. Promote only accepted winners into shared code.

## Queue Commands

Status:

```bash
cd inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py status
```

Loop:

```bash
cd inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py loop --poll-seconds 0.5
```

Projection batch generation:

```bash
cd inference
python3 search/queue/generate_projection_batch.py \
  --batch-tag my-proj-batch \
  --owner my-name \
  --sweep-script search/projection_400_sweep.py \
  --candidate-offset 0 \
  --candidate-limit 200 \
  --lengths 1 2 4 8 16 32 64 128
```

Act batch generation:

```bash
cd inference
python3 search/queue/generate_act_batch.py \
  --batch-tag my-act-batch \
  --owner my-name \
  --sweep-script search/act_400_sweep.py \
  --cases index_q kv_cache mla_input_x \
  --lengths 256 1024 \
  --candidate-window 0:100 \
  --candidate-window 100:100
```

Index batch staging:

```bash
cd inference
python3 search/index_400_sweep.py stage \
  --stage-root search/staging/my-index-batch \
  --owner my-name \
  --shard-size 25
```

Submit a manifest directory:

```bash
cd inference
python3 search/queue/remote_queue.py submit-dir search/staging/my-batch/manifests
```

Refresh the live summary:

```bash
cd inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/report_tools/queue_snapshot.py
```

## Recommended Agent Split

On a single GPU, parallelize only the preparation work:

- one queue owner
- one or more candidate-generation agents
- one verification/reporting agent
- one docs/handoff agent

Do not let multiple agents run timing loops directly on the same remote GPU.

## What To Publish

If you want this repo to stay GitHub-clean:

- commit docs, prompts, scripts, and curated markdown reports
- do not commit generated queue state, run folders, or local staging output
- regenerate staged manifests instead of versioning them

## Good Next Steps

- continue `mla_wq_b` and `mla_wkv_b` shape crossover work
- keep strict search on index/routing paths
- use the `near-exact` lane only for smooth algebra kernels
- collapse repeated manual analysis into reusable report tools
