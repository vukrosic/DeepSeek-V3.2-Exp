# Kernel Search Workspace

This directory is the control plane for speed work that must preserve functionality.

## Start Here

- `AGENT_PLAYBOOK.md`: handoff document for any agent continuing the search
- `AUTOMATION.md`: GitHub-facing continuation guide for contributors and agents
- `KERNEL_SPEED_SEARCH_SKILL.md`: repo-contained kernel optimization workflow
- `PROCESS.md`: end-to-end workflow for kernel research, testing, queue usage, and rollout
- `PLAN.md`: decomposition, task boundaries, and dependency order
- `reports/current-3090-baseline.md`: active baseline, constraints, and current bottlenecks
- `reports/leaderboard.md`: accepted, rejected, and pending candidates
- `queue/README.md`: single-GPU queue model, runner commands, and submission flow
- `queue/RULES.md`: rules for agents so experiments do not collide
- `prompts/01_search_operator.md`: execution prompt for a search worker
- `prompts/02_verification_and_benchmarking.md`: verification prompt
- `prompts/03_experiment_design.md`: planning prompt

## Rules
- No output-shape changes.
- No routing changes.
- No mask-semantics changes.
- No sampling changes.
- No precision changes unless the task manifest explicitly says it is an optional experiment and not the default path.
- A candidate is accepted only if it passes the declared acceptance lane for the task.
- Routing-sensitive paths stay in the `exact` lane.
- Smooth algebra paths may use a declared `near-exact` lane, but only with explicit tolerance and downstream confirmation.

## Layout
- `tasks/`: independent search manifests
- `templates/`: manifest template for new search tracks
- `runs/`: per-run work folders created by the runner
- `reports/`: summaries and conclusions
- `baselines/`: captured benchmark JSON and notes
- `search_runner.py`: small CLI to list tasks and create run folders
- `PLAN.md`: dependency order and operating rules for the full search program
- `PROCESS.md`: exactness/test/queue/rollout workflow for continuing agents

## Search Strategy
Each task is searched independently and has:
- a strict scope
- acceptance constraints
- benchmark targets
- allowed change surface
- dependencies on earlier tasks only where necessary

## Current Decomposition
1. `01_act_quant_exact`
2. `02_fp8_gemm_exact`
3. `03_fp8_index_exact`
4. `04_weight_dequant_cache_exact`
5. `05_attention_exact`
6. `06_moe_dispatch_exact`
7. `07_end_to_end_exact`

## Usage
List tasks:
```bash
cd inference
python3 search/search_runner.py list
```

Validate task graph:
```bash
cd inference
python3 search/search_runner.py validate
```

Print dependency graph:
```bash
cd inference
python3 search/search_runner.py graph
```

Show one task:
```bash
cd inference
python3 search/search_runner.py show 02_fp8_gemm_exact
```

Create a run folder:
```bash
cd inference
python3 search/search_runner.py init-run 02_fp8_gemm_exact --label ampere-cutlass-prototype
```

## Notes
- `kernel_phase0_harness.py` is the main exactness + baseline harness.
- `kernel_algorithm_search.py` is the current micro-search script for GEMM and index variants.
- `projection_100_sweep.py` and `projection_400_sweep.py` are the current projection-search frontends.
- `act_200_sweep.py` and `act_400_sweep.py` are the current act-quant search frontends; `queue/generate_act_batch.py` stages queue-safe act batches for either script.
- `index_200_sweep.py` and `index_400_sweep.py` are the current exact fp8-index search frontends.
- `queue/generate_projection_batch.py` stages large queue-safe projection batches for either sweep script.
- `queue/rebatch_index_pending.py` collapses old single-candidate index jobs into shard jobs without touching completed work.
- `queue/retry_failed_index_batches.py` emits retry manifests for batch-level index failures after the underlying bug is fixed.
- This workspace does not force a single backend. TileLang, CUDA, CUTLASS, Triton, or cuBLASLt are all valid if the task's exactness gate passes.
