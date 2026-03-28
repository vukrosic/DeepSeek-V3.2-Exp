# Current 3090 Baseline

## Environment
- GPU: RTX 3090 (`sm_86`)
- Repo path on remote: `/workspace/DeepSeek-V3.2-Exp/inference`
- Current default path: exact CUDA fallback in `kernel.py`

## Exact Baseline Numbers
- `act_quant.index_q`: `0.206 ms`
- `act_quant.kv_cache`: `0.202 ms`
- `fp8_gemm.wq_b_prefill`: `2.216 ms`
- `fp8_gemm.wkv_b_prefill`: `1.066 ms`
- `fp8_index.decode`: `0.165 ms`
- `fp8_index.prefill_small`: `0.163 ms`

## Validity
- All current harness checks pass exactly.
- `max_abs_diff = 0.0` for the current exact default path.

## Accepted Exact Improvement
- The blocked weight dequant path was simplified to an algebraically identical broadcasted block multiply.
- This preserves outputs exactly and improves the real fallback path on the 3090.
- Measured gain:
  - `fp8_gemm.wq_b_prefill`: `2.604 ms -> 2.216 ms`
  - `fp8_gemm.wkv_b_prefill`: `1.258 ms -> 1.066 ms`
- Fallback weight dequant now also handles non-divisible tail blocks exactly.
- This fixes the fallback path for shapes like `wkv_a = 576 x 7168`, which previously failed in the exact torch fallback.
- Additional exact model-path improvements now landed on top of the kernel-level baseline:
  - `MLA.wq_a + MLA.wkv_a`: shared input quant plus fp32 cached weights: about `1.431 ms -> 0.734 ms`
  - `wkv_b` cached bf16 dequant in fallback prefill: about `1.030 ms -> 0.662 ms`
  - `Indexer`: shared precomputed `x_fp8` / `qr_fp8` reuse: about `1.616 ms -> 1.237 ms`
  - `Indexer` projections: fp32 cached dequantized weights over bf16 cache: about `0.242 ms -> 0.191 ms`
  - `Indexer` cached `wq_b` + `wk` fallback path over fresh dequant: about `1.688 ms -> 1.603 ms`
  - shared `act_quant(x)` for `MLA.wq_a` + `MLA.wkv_a`: about `1.427 ms -> 1.309 ms`

## Exact Search Findings
- Fastest exact GEMM formulation on this GPU remains cached `F.linear`: about `1.047 ms`.
- Faster cached formulations using `mm`, `matmul`, `addmm`, or `einsum` are not exact here and drift by up to `0.015625`.
- A dedicated `MLA.wq_b` sweep over `100` candidate algorithms found `40` exact and `60` inexact variants.
- Best exact `MLA.wq_b` candidate from that sweep is row-major cached `fp32` weight plus `F.linear`: `2.217 ms -> 1.056 ms`.
- All top-10 exact `MLA.wq_b` results came from the `cache_fp32_row` family.
- The fake-fast cluster around `0.424-0.464 ms` used bf16 cached-weight variants and all drifted by `0.015625`, so they remain rejected.
- A single-GPU queue is now in place under `search/queue/` so agents submit jobs instead of colliding on benchmark runs.
- A queued batch of `500` additional exact projection candidates completed with `0` failures across:
  - `MLA.wkv_b`
  - `Indexer.wq_b`
  - `Indexer.wk`
  - `MLA.wq_a`
  - `MLA.wkv_a`
- Best exact winners from that queued batch:
  - `MLA.wkv_b`: `0.659 ms -> 0.513 ms`, `cache_fp32_t + F.linear`
  - `Indexer.wq_b`: `0.422 ms -> 0.390 ms`, `cache_fp32_row + mm`
  - `Indexer.wk`: `0.127 ms -> 0.110 ms`, `cache_fp32_row + matmul`
  - `MLA.wq_a`: `0.383 ms -> 0.355 ms`, `cache_fp32_row + F.linear`
  - `MLA.wkv_a`: `0.210 ms -> 0.208 ms`, `cache_fp32_row + F.linear`
- A separate `fp8_index` sweep over `200` candidates found `60` exact and `140` inexact variants.
- Best exact `fp8_index` candidate from that sweep is `q_fp32_contig + k_fp32_contig + matmul_broadcast`: `0.168 ms -> 0.144 ms`.
- The `broadcast_sum` index family was rejected because it drifted by `3.0517578125e-05` and does not pass the exact gate.
- A separate `fp8_index` prefill-small sweep over `200` candidates found `40` exact and `160` inexact variants.
- Best exact prefill-small index candidate is `q_fp32_contig + k_fp32_contig + einsum_qk`: `0.193 ms -> 0.188 ms`.
- The faster `einsum_kq` prefill-small variant was rejected because it drifted by `6.103515625e-05`.
- A separate `act_quant` sweep over `200` candidates found all candidates exact, but none beat the current shipped fallback, so that lane is currently closed as a reject.
- Current exact component breakdown for the representative `wq_b` shape:
  - `a_dequant`: about `0.046 ms`
  - `b_dequant`: about `1.179 ms`
  - exact cached `F.linear`: about `0.960 ms`
- Conclusion: exact search should still focus on weight dequant and cache policy, not on more dense matmul reformulations.

## Important Constraint
- The original TileLang FP8 path in this repo does not compile on `sm_86`.
- Therefore, all exact algorithm search on this GPU should compare against the exact CUDA fallback baseline until an Ampere-native exact kernel exists.
- `torch.compile` / Inductor is also blocked for this path on `sm_86` because Triton rejects the repo's FP8 dtype on this architecture.

## Cache Constraint
- A global dequantized-weight cache is not viable on this 24 GiB GPU.
- See `weight_dequant_cache_feasibility.md` for the quantified budget analysis.
- The useful exact cache strategy on this GPU is therefore selective, module-local reuse, not a model-wide dequantized-weight cache.
- The new module-local hot caches are targeted at `MLA` input projections and `Indexer` projections, not MoE experts or model-wide dense weights.

## Search Priority
1. `02_fp8_gemm_exact`
2. `03_fp8_index_exact`
3. `04_weight_dequant_cache_exact`
4. `01_act_quant_exact`
5. `05_attention_exact`
6. `06_moe_dispatch_exact`
