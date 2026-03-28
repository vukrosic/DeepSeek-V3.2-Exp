# Weight Dequant Cache Feasibility

This report estimates the memory cost of caching dequantized fp8 weights for the current DeepSeek V3.2 config.

## Headline
- Full fp8 weight set in bf16 cache form: 1.22 TiB
- Full fp8 weight set in fp32 cache form: 2.43 TiB
- Non-routed fp8 weights only in bf16 cache form: 28.23 GiB
- Non-routed fp8 weights only in fp32 cache form: 56.46 GiB
- One dense transformer block in bf16 cache form: 1.09 GiB (fits on a 24 GiB GPU)
- One MoE routed-expert bank in bf16 cache form: 21.00 GiB (fits on a 24 GiB GPU)

## Exact Config
- `n_layers` = 61
- `n_dense_layers` = 3
- `n_heads` = 128
- `n_routed_experts` = 256
- `n_shared_experts` = 1
- `n_activated_experts` = 8
- `q_lora_rank` = 1536
- `kv_lora_rank` = 512
- `qk_nope_head_dim` = 128
- `qk_rope_head_dim` = 64
- `v_head_dim` = 128
- `inter_dim` = 18432
- `moe_inter_dim` = 2048
- `dim` = 7168

## Per-Group Costs
| group | instances | shape | fp8 weight | scale | bf16 cache | fp32 cache |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mla.wq_a` | 61 | `7168 x 1536` | 640.50 MiB | 0.16 MiB | 1.25 GiB | 2.50 GiB |
| `mla.wq_b` | 61 | `1536 x 24576` | 2.14 GiB | 0.54 MiB | 4.29 GiB | 8.58 GiB |
| `mla.wkv_a` | 61 | `7168 x 576` | 240.19 MiB | 0.05 MiB | 480.38 MiB | 960.75 MiB |
| `mla.wkv_b` | 61 | `512 x 32768` | 976.00 MiB | 0.24 MiB | 1.91 GiB | 3.81 GiB |
| `mla.wo` | 61 | `16384 x 7168` | 6.67 GiB | 1.67 MiB | 13.34 GiB | 26.69 GiB |
| `dense_mlp.w1` | 3 | `7168 x 18432` | 378.00 MiB | 0.09 MiB | 756.00 MiB | 1.48 GiB |
| `dense_mlp.w2` | 3 | `18432 x 7168` | 378.00 MiB | 0.09 MiB | 756.00 MiB | 1.48 GiB |
| `dense_mlp.w3` | 3 | `7168 x 18432` | 378.00 MiB | 0.09 MiB | 756.00 MiB | 1.48 GiB |
| `moe.shared_experts.w1` | 58 | `7168 x 2048` | 812.00 MiB | 0.20 MiB | 1.59 GiB | 3.17 GiB |
| `moe.shared_experts.w2` | 58 | `2048 x 7168` | 812.00 MiB | 0.20 MiB | 1.59 GiB | 3.17 GiB |
| `moe.shared_experts.w3` | 58 | `7168 x 2048` | 812.00 MiB | 0.20 MiB | 1.59 GiB | 3.17 GiB |
| `moe.experts.w1` | 14848 | `7168 x 2048` | 203.00 GiB | 50.75 MiB | 406.00 GiB | 812.00 GiB |
| `moe.experts.w2` | 14848 | `2048 x 7168` | 203.00 GiB | 50.75 MiB | 406.00 GiB | 812.00 GiB |
| `moe.experts.w3` | 14848 | `7168 x 2048` | 203.00 GiB | 50.75 MiB | 406.00 GiB | 812.00 GiB |

## Conclusions
- Caching every routed expert weight is not feasible on a 24 GiB GPU.
- A single dense transformer block can fit, but only as a narrowly scoped layer-local cache.
- The dense non-expert fp8 weights are too large to keep fully resident as dequantized bf16/fp32 copies across the full model.
- Any useful cache policy must be selective, layer-local, or windowed rather than global.

## Artifacts
- JSON summary: `weight_dequant_cache_feasibility.json`
- This markdown report: `weight_dequant_cache_feasibility.md`
