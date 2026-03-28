# AI Search Playbook

This file is the handoff document for any agent continuing the speed search in this repo.

## Mission

Improve inference speed without intentionally changing functionality. Faster outputs are not enough. A candidate only counts if it preserves the default behavior of the code path it replaces and beats the current accepted baseline.

## Start Here

1. Read `README.md` in the inference root.
2. Read `search/README.md`.
3. Read `search/AUTOMATION.md`.
4. Read `search/KERNEL_SPEED_SEARCH_SKILL.md`.
5. Read `search/PROCESS.md`.
6. Read `search/PLAN.md`.
7. Read `search/reports/current-3090-baseline.md`.
8. Read `search/reports/leaderboard.md`.
9. Pick exactly one search task from `search/tasks/`.

## Ground Truth Constraints

- Device used for the active search: RTX 3090 (`sm_86`)
- Active comparison baseline on this GPU: exact CUDA fallback path
- The original TileLang FP8 path in this repo does not compile on `sm_86`
- `torch.compile` / Inductor is also not a working default path here for this FP8 format
- Global dequantized-weight caching is not viable on a 24 GiB card

These are not opinions. Treat them as current operating facts unless you disprove them with a recorded run.

## Operating Rules

- Work on one task at a time.
- Do not hide semantic changes inside a speed patch.
- Do not claim a win from a lower-precision variant unless that variant is explicitly a non-default experiment.
- Keep exploratory code local to the run folder until it wins.
- Promote only accepted winners into shared code.
- Update the leaderboard after every accepted or rejected significant candidate.
- Keep strict and near-exact winners labeled separately.

## Search Loop

1. Pick one hotspot.
2. Capture the current accepted baseline.
3. Propose two to five small exact candidates.
4. Reject losers quickly with cheap correctness tests using the active lane.
5. Benchmark survivors on the target shapes.
6. Keep only the winner.
7. Re-run the relevant model-path or end-to-end check.
8. Write the result down in the run folder and leaderboard.

Small cheap experiments are for eliminating losers fast before scaling winners. Do not start with a giant rewrite if a microbench can kill the idea in five minutes.

## Required Artifacts Per Candidate

- one run folder under `search/runs/`
- `task_snapshot.json`
- `notes.md`
- `results.json`
- benchmark command or script name
- exactness result
- keep or reject decision

## File Discipline

- Shared code lives in files like `kernel.py`, `model.py`, or dedicated search scripts.
- Stable operating docs live in `search/`.
- One-off experiments should move into a run folder or a dedicated search script once they stop being trivial.
- If you create a new benchmark script that will be reused, keep it in `search/` or the inference root and document it.

## Acceptance Lanes

- `exact`: bitwise or task-defined exact equality; still required for routing-sensitive paths.
- `near-exact`: bounded numeric drift is allowed only for smooth algebra kernels and only with a declared tolerance plus downstream confirmation.

If you are working on index selection, routing, masking, or anything that can flip a branch, stay in the `exact` lane.

## What Counts As A Real Win

- The changed code path passes the active acceptance lane for that task.
- The benchmark beats the current accepted baseline for the same shape and setup.
- The memory tradeoff is understood and recorded.
- The new default path is maintainable enough that another agent can continue from it.

## What Does Not Count

- A speedup measured against the wrong baseline
- A win that comes from silently changing semantics
- A win that only works on toy shapes if the real hotspot uses different ones
- A win that requires an infeasible model-wide cache on 24 GiB VRAM
- A benchmark with no exactness result

## Command Skeleton

```bash
cd inference
python3 search/search_runner.py validate
python3 search/search_runner.py list
python3 search/search_runner.py show 02_fp8_gemm_exact
python3 search/search_runner.py init-run 02_fp8_gemm_exact --label short-name
```

## Prompt Files

- `search/prompts/01_search_operator.md`
- `search/prompts/02_verification_and_benchmarking.md`
- `search/prompts/03_experiment_design.md`

Use those when you want another agent to execute a bounded part of the search without improvising the rules.
