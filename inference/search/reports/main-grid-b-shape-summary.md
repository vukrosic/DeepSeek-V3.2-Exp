# Main Grid B Shape Summary

Completed projection sweeps only. All finished rows in this batch use `prefill_len=256`; the `l64` and `l512` queue manifests exist but are not completed, so they are excluded here.

| target | prefill_len | exact count | best exact family/op | best exact ms | best speedup | exact speedup range |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `indexer_wk` | 256 | 40 | `cache_fp32_row` / `matmul` | 0.110 | 1.157x | 0.032x-1.157x |
| `indexer_wq_b` | 256 | 40 | `cache_fp32_row` / `mm` | 0.390 | 1.081x | 0.511x-1.081x |
| `mla_wkv_a` | 256 | 40 | `cache_fp32_row` / `flinear` | 0.208 | 1.010x | 0.518x-1.010x |
| `mla_wkv_b` | 256 | 70 | `cache_fp32_t` / `flinear` | 0.513 | 1.286x | 0.576x-1.286x |
| `mla_wq_a` | 256 | 40 | `cache_fp32_row` / `flinear` | 0.355 | 1.081x | 0.528x-1.081x |
| `mla_wq_b` | 256 | 40 | `cache_fp32_row` / `flinear` | 1.056 | 2.100x | 0.952x-2.100x |

## Engineering Takeaways
- `cache_fp32_row` is the dominant exact winner family across the completed projection sweeps.
- `mla_wq_b` is the strongest win: exact `cache_fp32_row + flinear` reaches about `2.10x` over baseline.
- `mla_wkv_b` is the only sweep where `cache_fp32_t` becomes the best exact family, which points to transpose/layout sensitivity in that path.
- `indexer_wk` is the widest exact spread and the lowest worst-case exact speedup, so it is the least stable candidate family in the batch.
- Exact wins are real but localized: most other exact winners sit around `1.01x` to `1.16x`, so the remaining headroom is modest unless the memory layout changes more materially.
