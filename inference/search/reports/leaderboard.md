# Exact Speed Leaderboard

All entries below preserve functionality. Faster-but-inexact candidates are listed separately and are not accepted.

## Accepted

| rank | area | benchmark | before ms | after ms | speedup | exact |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `MLA.wq_a + MLA.wkv_a` | shared input quant + fp32 cached weights | 1.431 | 0.734 | 1.95x | yes |
| 2 | `MLA.wkv_b` prefill path | cached dequantized weight reuse | 1.030 | 0.662 | 1.56x | yes |
| 3 | `Indexer` path | shared precomputed `x_fp8` / `qr_fp8` reuse | 1.616 | 1.237 | 1.31x | yes |
| 4 | `fp8_gemm.wq_b_prefill` | exact weight dequant simplification | 2.604 | 2.216 | 1.17x | yes |
| 5 | `fp8_gemm.wkv_b_prefill` | exact weight dequant simplification | 1.258 | 1.066 | 1.18x | yes |
| 6 | `Indexer` projections | fp32 cached dequantized weights over bf16 cache | 0.242 | 0.191 | 1.27x | yes |
| 7 | `Indexer` projections | cached dequantized weights over fresh dequant path | 1.688 | 1.603 | 1.05x | yes |
| 8 | `MLA.wq_a + MLA.wkv_a` | shared input quant over duplicate quantization | 1.427 | 1.309 | 1.09x | yes |

## Rejected

| area | candidate | result | reason |
| --- | --- | --- | --- |
| `act_quant` 200-sweep | all 200 exact variants | rejected | exact but none beat the shipped fallback baseline |
| exact GEMM | cached `mm` / `matmul` / `addmm` / `einsum` | rejected | faster but not exact, drift up to `0.015625` |
| `MLA.wq_b` 100-sweep | transposed cached-weight families | rejected | all `cache_bf16_t` and `cache_fp32_t` variants drifted by `0.015625` |
| `MLA.wq_b` 100-sweep | bf16 cached-weight fast cluster | rejected | fastest variants were `0.424-0.464 ms` but drifted by `0.015625` |
| `fp8_index` 200-sweep | `broadcast_sum` family | rejected | drifted by `3.0517578125e-05` and does not pass exact gate |
| `fp8_index` prefill 200-sweep | `einsum_kq` fast variant | rejected | drifted by `6.103515625e-05` and does not pass exact gate |
| compile path | `torch.compile` / Inductor | rejected | Triton rejects this FP8 dtype on `sm_86` |
| cache policy | global dequantized-weight cache | rejected | not viable on 24 GiB VRAM |
| MLA prefill | reuse first `kv` quantization for `wkv_b` | rejected | requantization changes scales and outputs |

## Pending

| area | candidate | before ms | after ms | speedup | note |
| --- | --- | ---: | ---: | ---: | --- |
| `MLA.wq_b` | bf16 cached dequantized weight | 2.217 | 1.321 | 1.68x | exact, from 100-candidate sweep, but about `4.29 GiB` across `61` layers |
| `MLA.wq_b` | fp32 cached dequantized weight | 2.217 | 1.056 | 2.10x | exact, best of 100 tested candidates, but about `8.58 GiB` across `61` layers |
| `MLA.wkv_b` prefill | fp32 cached dequantized weight | 1.110 | 0.584 | 1.90x | exact in prefill, but current decode path rejects mixed `bf16` / `fp32` `einsum` |
| `MLA.wkv_b` | fp32 transposed cached weight + `F.linear` | 0.659 | 0.513 | 1.29x | exact, best of queued 100-candidate sweep, but not landed yet |
| `Indexer.wq_b` | fp32 row cache + `mm` | 0.422 | 0.390 | 1.08x | exact, best of queued 100-candidate sweep |
| `Indexer.wk` | fp32 row cache + `matmul` | 0.127 | 0.110 | 1.16x | exact, best of queued 100-candidate sweep |
| `MLA.wq_a` | fp32 row cache + `F.linear` | 0.383 | 0.355 | 1.08x | exact, best of queued 100-candidate sweep |
| `MLA.wkv_a` | fp32 row cache + `F.linear` | 0.210 | 0.208 | 1.01x | exact, best of queued 100-candidate sweep |
| `fp8_index` | `q_fp32_contig + k_fp32_contig + matmul_broadcast` | 0.168 | 0.144 | 1.16x | exact, best of 200-candidate index sweep |
| `fp8_index` prefill-small | `q_fp32_contig + k_fp32_contig + einsum_qk` | 0.193 | 0.188 | 1.03x | exact, best of 200-candidate prefill-small sweep |

## Notes

- Device: RTX 3090 (`sm_86`)
- Repo path: `/workspace/DeepSeek-V3.2-Exp/inference`
- Kernel-level numbers come from the exact fallback path and harness/search scripts.
- The `MLA.wq_b` sweep tested `100` candidate algorithms: `40` exact and `60` inexact.
- A queued batch of `5 x 100 = 500` additional projection candidates completed with `0` queue failures.
- The separate `fp8_index` sweep tested `200` candidates: `60` exact and `140` inexact.
- The separate `act_quant` sweep tested `200` candidates: all exact, but none beat the shipped fallback.
- The separate `fp8_index` prefill-small sweep tested `200` candidates: `40` exact and `160` inexact.
- Some module-path A/B benches use `rotate_activation = identity` on both sides because `fast_hadamard_transform` is not available in the current remote Python 3.12 environment. The comparison is still valid for the changed code region because both baseline and candidate use the same patch.
