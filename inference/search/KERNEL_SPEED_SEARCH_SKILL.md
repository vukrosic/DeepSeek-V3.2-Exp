# Kernel Speed Search Skill

This is a repo-contained version of the kernel-speed-search workflow used during the RTX 3090 rescue and optimization pass.

Use it when the goal is to improve inference speed without quietly changing model behavior.

## Goal

Preserve functionality, improve speed, prove the win, and keep the experiment system clean enough that another contributor or agent can continue from your work.

## When To Use This

Use this workflow when you are:

- optimizing hot-path tensor code
- comparing kernel or layout variants
- building CUDA, Triton, TileLang, or PyTorch fallback alternatives
- running algorithm search on quantized, attention, MoE, or projection paths
- operating on one shared GPU where benchmark collisions would invalidate results

Do not use this workflow as a license to change semantics, precision defaults, routing, masking, or cache meaning without saying so explicitly.

## Operating Principles

- Measure before and after every real change.
- Keep search code separate from accepted shared code until a win is proven.
- Prefer shape-aware and hardware-aware choices over one global guess.
- Keep strict and near-exact results labeled separately.
- Serialize GPU timing through one queue owner when only one benchmark GPU exists.

## Minimal Research Loop

1. Identify one hotspot.
2. Record the active baseline for the same shape and path.
3. Generate a small candidate family.
4. Define the acceptance lane before timing.
5. Run local validation first.
6. Queue remote GPU timing instead of running ad hoc benchmarks.
7. Reject losers quickly.
8. Promote only verified winners into shared code.
9. Update the leaderboard and notes.

## Acceptance Lanes

- `exact`
  Required for routing-sensitive or discontinuous paths such as top-k, masks, index selection, and expert routing.
- `near-exact`
  Allowed only for smooth algebra paths such as projection or activation kernels, with explicit tolerance plus downstream confirmation.

Never relabel a candidate after the fact just to keep a faster result alive.

## What Counts As A Real Win

- same shape and setup as the accepted baseline
- passes the declared acceptance gate
- repeatable timing win
- memory or complexity tradeoff recorded
- understandable enough that another contributor can continue the work

## What To Read In This Repo

Start in this order:

1. `README.md`
2. `search/README.md`
3. `search/AUTOMATION.md`
4. `search/PROCESS.md`
5. `search/AGENT_PLAYBOOK.md`
6. `search/reports/current-3090-baseline.md`
7. `search/reports/leaderboard.md`

## Useful Commands

```bash
cd inference
python3 search/search_runner.py validate
python3 search/search_runner.py list
python3 search/search_runner.py show 02_fp8_gemm_exact
python3 search/search_runner.py init-run 02_fp8_gemm_exact --label my-candidate-family
```

Queue status:

```bash
cd inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py status
```

Queue loop:

```bash
cd inference
PYTHONPATH=/workspace/DeepSeek-V3.2-Exp/inference python3 search/queue/queue_runner.py loop --poll-seconds 0.5
```

## Repo Hygiene

- commit code, docs, prompts, and curated reports
- do not commit queue runtime state, local staging output, or generated run artifacts
- keep machine-generated JSON and logs out of Git
- regenerate staged manifests instead of versioning them

## Current Ground Truth For The 3090

- GPU: RTX 3090 (`sm_86`)
- baseline on this machine: exact CUDA fallback path
- original TileLang FP8 path in this repo does not compile on `sm_86`
- one GPU means one timing owner at a time

Treat those as current facts unless you disprove them with a recorded run.
