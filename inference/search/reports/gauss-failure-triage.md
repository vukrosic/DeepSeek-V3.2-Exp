# Gauss Index Failure Triage

## Scope

This triage looked for existing failed queue jobs and logs related to the staged Gauss fp8 index batch.

Files inspected:
- `search/queue/failed/`
- `search/queue/logs/`
- `search/staging/gauss-index-batch-20260328/`
- `search/index_200_sweep.py`

Helper used:
- `python3 search/report_tools/popper_failed_gauss.py`

## Current State

- There are **0 actual failed queue manifests** in `search/queue/failed/`.
- There are **0 queue log files** in `search/queue/logs/`.
- The Gauss staging batch exists and contains **600 staged manifests**:
  - `200` for `decode`
  - `200` for `prefill_small`
  - `200` for `prefill_mid`

Conclusion: there is currently **no executed queue-failure set to retry**. What exists is a staged batch with one concrete static failure family and one metadata issue.

## Failure Modes

### 1. Static hard-failure family: `broadcast_reduce` on prefill shapes

Count:
- `80` staged manifests
  - `40` in `prefill_small`
  - `40` in `prefill_mid`

Cause:
- `index_200_sweep.py` defines `broadcast_reduce` as:
  - [index_200_sweep.py](/root/DeepSeek-V3.2-Exp/inference/search/index_200_sweep.py#L209)
- That expression returns logits in `b,n,m,h` order.
- The rest of the pipeline assumes `b,m,n,h` order:
  - scaling with `q_s.float().unsqueeze(2)` at [index_200_sweep.py](/root/DeepSeek-V3.2-Exp/inference/search/index_200_sweep.py#L255)
  - reference layout at [index_200_sweep.py](/root/DeepSeek-V3.2-Exp/inference/search/index_200_sweep.py#L214)

Why it matters:
- In `decode`, `m=1`, so the bad layout can still broadcast and may not explode immediately.
- In `prefill_small` and `prefill_mid`, `m>1`, so this family is structurally wrong and should be treated as a hard reject before any submission.

Example manifests:
- `search/staging/gauss-index-batch-20260328/manifests/prefill_mid/cand-009.json`
- `search/staging/gauss-index-batch-20260328/manifests/prefill_mid/cand-010.json`
- `search/staging/gauss-index-batch-20260328/manifests/prefill_mid/cand-019.json`

### 2. Metadata cleanup issue: owner is not `gauss`

Count:
- `600` staged manifests

Cause:
- `DEFAULT_OWNER = "codex"` at [index_200_sweep.py](/root/DeepSeek-V3.2-Exp/inference/search/index_200_sweep.py#L25)
- manifests are emitted from that default at [index_200_sweep.py](/root/DeepSeek-V3.2-Exp/inference/search/index_200_sweep.py#L285)

Why it matters:
- This is not a runtime failure.
- It does break attribution and makes later retry/cleanup work ambiguous if someone expects these to be Gauss-owned jobs.

## What Is Not Present

- No queue runner return codes for these jobs
- No queue execution logs for these jobs
- No completed or failed queue manifests for these jobs

So there is nothing in queue state to "retry" yet.

## Retry / Cleanup Plan

1. Do not retry from `search/queue/failed/`.
   There is nothing there.

2. Quarantine the known-bad prefill `broadcast_reduce` family before any future submission.
   Practical effect: drop `80` manifests from the staged batch.

3. Treat the remaining staged pool as the real candidate set.
   Count after removing the hard-failure family: `520` manifests.

4. Fix or override ownership metadata before any future submission if Gauss attribution matters.
   This is a cleanup step, not a queue fix.

5. If queue failures appear later, compare them against the static-bad list first.
   The first suspects should be the prefill `broadcast_reduce` manifests, not the whole batch.

## Bottom Line

- **Actual executed failures:** `0`
- **Static hard-failure manifests to exclude:** `80`
- **Metadata-misattributed manifests:** `600`
- **Usable staged manifests after cleanup:** `520`
