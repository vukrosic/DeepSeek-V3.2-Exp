# Exact Search Plan

This workspace splits speed work into independent search tracks so each hotspot can be searched, benchmarked, and accepted separately.

## Objective
- Preserve model functionality.
- Preserve default semantics.
- Improve speed only through better kernels, schedules, memory reuse, or exact algorithmic reformulations.

## Search Units
1. `01_act_quant_exact`
   Baseline quantization path and exactness harness.
2. `02_fp8_gemm_exact`
   GEMM kernel search only.
3. `03_fp8_index_exact`
   Index-score kernel search only.
4. `04_weight_dequant_cache_exact`
   Exact caching policy experiments for dequantized weights.
5. `05_attention_exact`
   Attention score/apply path, but only with exact mask and softmax semantics.
6. `06_moe_dispatch_exact`
   MoE dispatch and expert accumulation only.
7. `07_end_to_end_exact`
   Full-path validation after individual winners are selected.

## Dependency Order
```text
01_act_quant_exact
|-- 02_fp8_gemm_exact
|   `-- 04_weight_dequant_cache_exact
|-- 03_fp8_index_exact
|   `-- 05_attention_exact
`-- 07_end_to_end_exact

06_moe_dispatch_exact
`-- 07_end_to_end_exact
```

## Folder Structure
- `tasks/`: one manifest per independent search track
- `templates/`: manifest template for adding new tracks
- `runs/`: timestamped run folders, one candidate batch per folder
- `baselines/`: stable captured benchmark JSON and notes
- `reports/`: conclusions, accepted winners, and cross-task summaries

## Run Workflow
1. Capture or confirm the baseline for the task.
2. Create a run folder with `search_runner.py init-run`.
3. Implement or prototype one exact candidate.
4. Run exactness first, benchmark second.
5. Record variant-level results in that run folder.
6. Mark the candidate `keep`, `reject`, or `revisit`.
7. Promote only the winner into the shared path.
8. Re-run `07_end_to_end_exact` after any accepted change.

## Isolation Rules
- One run folder should focus on one task only.
- Cross-task edits are not allowed except in `07_end_to_end_exact`.
- A task may depend on another task's winner, but should not silently redefine its semantics.
- If a search idea needs wider behavior changes, it belongs in a new manifest, not inside an existing exact task.

## Acceptance Rules
- Default path must pass the task's exactness gate.
- Benchmark wins must be measured against the current accepted baseline for that task.
- Temporary compatibility or exploratory paths are allowed, but they do not replace the default path unless they pass exactness and benchmark gates.
