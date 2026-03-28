# Kernel Research Process

This document is the operating manual for speed research in this repo.

The goal is simple:

- preserve functionality
- preserve default semantics
- improve speed through exact or declared near-exact kernel work, memory-layout changes, dispatch changes, or execution-policy changes

If a candidate wins only because it changes precision, routing, masks, cache semantics, or other behavior that can change outputs, it is not a default-path win.

## Scope

This process covers:

- queue ownership on the single research GPU
- agent orchestration without benchmark collisions
- exactness and benchmark gates
- failure handling and retry discipline
- rollout rules for promoting research into shared code

This document does not replace the queue command docs. Read these alongside it:

- `search/README.md`
- `search/AGENT_PLAYBOOK.md`
- `search/queue/README.md`
- `search/queue/RULES.md`

## Current Ground Truth

Treat these as the active working assumptions until a recorded run disproves them:

- active research GPU: RTX 3090 (`sm_86`)
- default comparison path on this GPU: exact CUDA fallback
- original TileLang FP8 path does not currently compile on this GPU
- `torch.compile` is not the default path for this FP8 setup
- one GPU means one timing owner at a time

## Mission And Acceptance Standard

The research program has two separate outputs:

1. exact hot-path wins
2. experiment-system wins

Keep them separate. A faster queue, better sharding, or better reporting is useful, but it is not a kernel win.

A result counts as accepted only when:

- the comparison uses the same shape and setup as the current accepted baseline
- the exactness gate for that task passes
- the benchmark win is real and repeatable
- the memory or complexity cost is recorded
- another agent can understand why it was accepted without reverse-engineering the run

## Single-GPU Ownership

The remote GPU is single-owner. That rule is not optional.

- exactly one queue runner owns the GPU
- all remote timing jobs go through `search/queue/`
- agents do not free-run remote benchmarks or ad hoc timing loops
- agents may do CPU-side preparation, local validation, staging, and analysis in parallel

The reason is measurement quality. Two remote timing jobs at once contaminate the numbers and invalidate the search.

## Agent Roles

Use parallel agents, but give them non-colliding roles.

The primary queue owner may:

- keep the runner alive
- submit approved manifests
- inspect queue health
- review completed runs
- land accepted code changes

Search agents may:

- read code and identify bottlenecks
- build candidate families
- write scripts and manifests
- prepare run folders
- analyze completed results
- summarize winners and losers

Search agents may not:

- manually run remote timing commands outside the queue
- write into another agent's run folder
- edit queue state by hand
- update shared reports before results are reviewed

Verification agents may:

- reproduce exactness checks locally
- review benchmark hygiene
- audit failed runs
- challenge weak claims before rollout

## Do Not Fight Other People's Changes

You are not alone in this repo.

- do not revert changes you did not make
- read the current files before assuming the old process still applies
- if a shared script changed, adapt your manifest or experiment to that current version
- if another agent already owns a run folder or staged batch, do not reuse it

The default assumption is that other diffs are intentional unless they directly break the current task.

## End-To-End Loop

Use this loop for every serious optimization idea:

1. Pick one hotspot or one experiment-system bottleneck.
2. Confirm the current accepted baseline for the same shape, path, and hardware.
3. Write the smallest experiment that can disprove the idea quickly.
4. Define the exactness gate before benchmarking.
5. Run local validation first.
6. Submit remote GPU work through the queue.
7. Reject losers immediately.
8. Promote only exact, measured winners.
9. Update shared reports only after the decision is real.

Do not skip directly from a code idea to a leaderboard entry.

## Test Ladder

Every meaningful change should move through this ladder:

1. `py_compile`
   Use this for scripts, queue tools, and import safety.
2. local shape and manifest validation
   Confirm the candidate runs, paths are valid, and batching logic is correct.
3. micro exactness
   Compare old vs new using the same tensors, same shapes, and same task gate.
4. micro benchmark
   Use the same harness, same shape, same warmup, and same iteration policy as the accepted baseline.
5. remote compile or import sanity
   Verify the changed files import on the actual GPU host.
6. queue-safe remote execution
   If GPU timing is involved, run through the queue.
7. shared-path or end-to-end confirmation
   Required before claiming a model-path rollout is complete.

If a candidate fails early, stop early. Do not push a broken idea down the ladder.

## Exactness Gates

Exactness is defined per task, not by vague intuition.

At minimum, every task must specify:

- the reference path
- the exact tensors or outputs being compared
- the allowed tolerance
- whether equality must be exact or whether a bounded tolerance is accepted for that task

Default rule in this repo is now tiered, not uniformly bitwise:

1. strict-semantic kernels
   Keep the hard gate for discontinuous control paths such as top-k selection, routing, masking, index selection, or anything where tiny numeric drift can flip a branch or expert choice.
2. numerically-equivalent kernels
   For smooth algebra paths such as plain GEMM/layout/cache variants, allow a bounded tolerance if the semantics are unchanged and the tolerance is declared before benchmarking.
3. behavioral confirmation
   If a path moves from bitwise-equal to bounded-tolerance acceptance, require an extra downstream confirmation at the next meaningful level up.

In practice:

- `fp8_index`, routing, masks, and similar control paths stay strict
- projection and act-style algebra kernels may use a declared tolerance lane
- never silently relax a gate just to keep a faster candidate alive
- never mix exact and tolerant wins into one unnamed leaderboard bucket

When using a tolerance lane, record:

- the chosen `atol` / `rtol` or other comparison rule
- why that tolerance is appropriate for the dtype and operation
- what downstream confirmation was used
- whether the result is an exact win or a near-exact win

Reject candidates immediately if they:

- change output shapes
- change routing or mask behavior
- depend on lower precision as the reason they are faster
- require semantic changes to batching, cache meaning, or control flow without explicit approval

## Benchmark Hygiene

Benchmark quality matters as much as the kernel idea.

Use these rules:

- compare the same shape, same device, same code path, and same measurement harness
- warm up before timing
- use enough iterations to suppress noise
- avoid mixed benchmark scopes such as timing setup in one version and not the other
- record memory tradeoffs for cached-weight or workspace-heavy candidates
- separate search overhead from kernel time when interpreting results
- keep queue load serialized so timings are not polluted by another job

Do not claim a win from:

- one noisy measurement
- different shapes
- different warmup or iteration counts
- a benchmark that includes unrelated setup only on one side
- a candidate that is faster only because the baseline path was misconfigured

## When To Write Kernels

Write or change kernel code only when the hot path is actually the limiter.

Do not start with a new kernel if the real bottleneck is:

- repeated Python startup per candidate
- one-candidate jobs flooding the queue
- stale queue debt
- bad manifest design
- measuring the wrong shapes

Fix packaging first if packaging dominates wall time.

## Experiment Design Rules

Design experiments so they kill bad ideas cheaply.

- vary one dimension at a time when possible
- search families, then tune survivors
- use real hotspot shapes, not toy shapes
- record memory cost next to time
- check crossover behavior, not just one best case
- keep the change surface narrow so the result is attributable

For batched or sharded jobs:

- batch independent candidates together only when the exactness and result format stay clear
- keep per-candidate results, not just one batch-level summary
- isolate candidate failures so one bad candidate does not waste the rest of the shard

## Near-Exact Policy

Near-exact acceptance is allowed only when all of the following are true:

- the kernel is on a smooth numeric path rather than a discontinuous control path
- the candidate keeps the same intended algorithm semantics
- the speedup is real enough to justify the extra validation burden
- the tolerance is written down before the comparison is run

Near-exact is not a free pass for:

- bf16 or fp16 drift in routing-sensitive code
- reordering that changes top-k decisions
- candidates that are only “close” on toy shapes
- candidates that pass one loose tensor check but destabilize the next-level output

Treat near-exact as a separate acceptance lane, not a silent weakening of the exact lane.

## Queue Submission Process

Every remote GPU job should follow this flow:

1. choose or create a fresh run folder
2. make sure the experiment script is already present on the target repo state
3. validate the command locally if possible
4. define expected result files before submission
5. create one manifest with a unique `id`
6. submit through the queue tooling
7. wait for completion
8. inspect raw results and logs before any report update

Every manifest must:

- have a unique `id`
- name its `owner`
- name the `task_id`
- point to a single new `run_dir`
- provide an absolute `cwd`
- provide the exact `command`
- declare the expected `result_paths`
- write only inside its own `run_dir` plus queue logs

If any of those are unclear, the job is not ready to submit.

## Queue Safety Rules

These are the practical non-negotiables:

- one runner owns the GPU
- one manifest equals one experiment job
- one job writes only into its own run folder and queue logs
- no job mutates shared reports while running
- completed jobs do not auto-update the leaderboard

If the backlog becomes dominated by tiny one-candidate jobs, fix the packaging before submitting more of them.

## Rebatch And Sharding Policy

Rebatch when all of the following are true:

- the queue is dominated by one task family
- per-job wall time is mostly process startup, setup, or queue overhead
- the candidates are independent
- per-candidate result recording can be preserved

Rebatching is allowed to reduce queue overhead.
Rebatching is not allowed to hide failures or erase per-candidate visibility.

A good shard:

- keeps candidates independent
- records each candidate outcome
- preserves the original exactness gate
- can survive one bad candidate without losing the whole batch

## Failure Handling

Failures are expected. Sloppy retries are not.

If a job fails:

1. inspect the queue log
2. inspect the raw result files
3. decide whether the failure is candidate-specific, script-level, or queue-level
4. fix the root cause before retrying
5. resubmit with a new manifest id unless the queue flow explicitly allows reuse

Failure classes to separate:

- invalid candidate math or shape logic
- broken script or import path
- stale or malformed manifest
- environment issue on the remote host
- queue packaging issue such as too many tiny jobs

Use failure results to improve the search system. Do not repeatedly resubmit the same broken command.

## Artifact Hygiene

Keep source artifacts and machine artifacts separate:

- keep scripts, manifests worth reusing, and process docs versionable
- ignore generated logs, local benchmark dumps, queue runtime buckets, and machine-written summary JSON
- do not hand-edit files inside `search/runs/`; rerun or restage instead

## Rollout Policy

There are three valid end states for a candidate:

1. rejected
2. pending more verification
3. accepted for rollout

Land a shared-path change only when:

- the task exactness gate passes
- the win beats the current accepted baseline for the same shape and path
- the memory cost is acceptable and recorded
- the code path is understandable enough for the next agent to maintain
- the rollout does not silently change default semantics

If the win is real but the memory cost is too high for the default path, keep it as:

- a documented pending result
- an opt-in path
- or a search lead for later shape-aware rollout

## Required Artifacts

Every serious experiment should leave behind:

- the script or command used
- the manifest
- the raw exactness result
- the raw benchmark result
- the log if the job failed
- the keep or reject decision
- a note about memory or complexity tradeoffs if relevant

Without those artifacts, the result is not durable.

## Documentation And Reporting

Update shared docs only after the result is reviewed.

When the accepted technical picture changes, update:

- `inference/README.md`
- `search/reports/current-3090-baseline.md`
- `search/reports/leaderboard.md`

When the operating process changes, update:

- `search/README.md`
- `search/AGENT_PLAYBOOK.md`
- `search/queue/README.md`
- `search/queue/RULES.md`

## Practical Checklist For The Next Agent

Before starting:

- read the current queue status
- read the current baseline and leaderboard
- check whether the hotspot is still the hotspot
- verify no one else already staged the same work

Before submitting a GPU job:

- define the exactness gate
- use a fresh run folder
- confirm result paths
- confirm the job is queue-safe
- prefer sharded jobs if startup dominates

Before claiming a win:

- show the baseline
- show the new result
- show the exactness status
- show the memory tradeoff if any
- state accepted, rejected, or pending

## Current Priorities

1. keep exact projection and index wins moving into shape-aware shared code when the memory cost is small
2. keep the queue focused on high-value exact work
3. prevent stale manifests and malformed submissions from polluting backlog
4. keep batched search jobs failure-tolerant and queue-efficient
5. preserve a clean handoff path for the next agent
