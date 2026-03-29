# Beginner Guide For Auto Research

This repo is optimized for kernel and inference speed research. If you are new here, do not start by editing the model blindly. Follow this order.

## What This Repo Is For

- exact or declared near-exact kernel speed work
- queue-safe benchmark execution on the shared GPU
- reproducible experiment batches and result tracking

## Read First

1. [`/root/DeepSeek-V3.2-Exp/README.md`](/root/DeepSeek-V3.2-Exp/README.md)
2. [`README.md`](/root/DeepSeek-V3.2-Exp/inference/search/README.md)
3. [`PROCESS.md`](/root/DeepSeek-V3.2-Exp/inference/search/PROCESS.md)
4. [`AGENT_PLAYBOOK.md`](/root/DeepSeek-V3.2-Exp/inference/search/AGENT_PLAYBOOK.md)
5. [`queue/README.md`](/root/DeepSeek-V3.2-Exp/inference/search/queue/README.md)
6. [`queue/RULES.md`](/root/DeepSeek-V3.2-Exp/inference/search/queue/RULES.md)

## Simple Working Loop

1. Pick one hotspot from `search/tasks/`.
2. Check the current baseline in `search/reports/`.
3. Create a run folder with `search/search_runner.py init-run`.
4. Make the smallest exact or declared near-exact change.
5. Validate locally with `py_compile` or the repo harness.
6. If the job needs GPU timing, submit through `search/queue/`.
7. Inspect the log and result JSON.
8. Keep only changes that beat the right baseline and pass the task gate.

## Good Starting Tasks

- `01_act_quant_exact`
- `02_fp8_gemm_exact`
- `03_fp8_index_exact`

These are the core kernel paths and the best place to learn the workflow.

## What Not To Do

- do not run ad hoc remote timing loops
- do not bypass the queue on the shared GPU
- do not change routing, mask semantics, or output shapes
- do not promote a candidate without an exactness result

## Queue Shortcut

If you need a queue-safe agent entry point, use:

```bash
cd inference
python3 search/queue/queue_agent.py status
python3 search/queue/queue_agent.py gemm-stage --submit
python3 search/queue/queue_agent.py bench-smoke --submit
```

## Result Files

- raw runs live under `search/runs/`
- staged manifests live under `search/staging/`
- queue logs live under `search/queue/logs/`
- accepted reports live under `search/reports/`

