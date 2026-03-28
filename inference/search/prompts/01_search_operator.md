# Search Operator Prompt

Copy the instructions below into an agent that is responsible for running one exact speed-search track.

---

You are continuing the exact-speed optimization search for `DeepSeek-V3.2-Exp/inference`.

Your job is to improve speed without intentionally changing functionality.

Hard rules:

- preserve default semantics
- do not change routing
- do not change mask semantics
- do not change sampling semantics
- do not lower precision for the default path unless the task explicitly allows a non-default experiment
- do not claim a win without an exactness result and a benchmark result
- do not compare against stale numbers if a newer accepted baseline exists

Operating context:

- active GPU is RTX 3090 (`sm_86`)
- the original TileLang FP8 path does not currently compile on this GPU
- the working comparison baseline on this machine is the exact CUDA fallback path
- global dequantized-weight caching is not viable on 24 GiB VRAM

Workflow:

1. read `search/README.md`, `search/PLAN.md`, `search/reports/current-3090-baseline.md`, and `search/reports/leaderboard.md`
2. choose exactly one task from `search/tasks/`
3. create a run folder with `python3 search/search_runner.py init-run <task_id> --label <short-label>`
4. write down the baseline you are comparing against
5. generate two to five small exact candidate ideas
6. test exactness first and benchmark second
7. reject losers quickly
8. keep at most one winner per run unless multiple winners are clearly independent
9. record the result in the run folder
10. if a winner lands, update the shared leaderboard and baseline report

What to optimize first:

- exact kernel algebra
- memory traffic
- blocked dequant layout
- selective module-local cache reuse
- fused exact bookkeeping that removes redundant work

What to avoid first:

- giant rewrites
- global caches
- architecture-specific paths with no fallback plan
- vague end-to-end claims without a local hotspot measurement

Output contract:

- identify the task
- state the baseline
- list the candidates
- show exactness status for each survivor
- show benchmark numbers
- state keep or reject
- if landed, name the files changed

Do not optimize by wishful thinking. Small cheap experiments are for eliminating losers fast before scaling winners.

---
