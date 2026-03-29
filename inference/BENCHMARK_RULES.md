# Benchmark Rules

## What `score_ms` Is

`score_ms` is the **geometric mean of the median wall-clock times** (in milliseconds)
across the 5 decode-step ops listed below.

```
score_ms = exp( mean( log(median_ms_op_i) for i in 1..5 ) )
```

Lower is better.  A score is only valid if **all 5 correctness checks pass**.

---

## The 5 Ops and Their Shapes

All shapes are for the 671B DeepSeek-V3.2 config
(`config_671B_v3.2.json`).  Constant parameters:

| Symbol | Value |
|--------|-------|
| `dim` | 7168 |
| `q_lora_rank` | 1536 |
| `kv_lora_rank` | 512 |
| `n_heads` | 128 |
| `qk_nope_head_dim` | 128 |
| `qk_rope_head_dim` | 64 |
| `v_head_dim` | 128 |
| `index_n_heads` | 64 |
| `index_head_dim` | 128 |
| `block_size` | 128 |

| # | Op key | Description | Key tensor shapes |
|---|--------|-------------|-------------------|
| 1 | `act_quant` | Block-wise FP8 quantisation of a single token | input `(1, 7168)` bf16 |
| 2 | `fp8_gemm_wq_b` | FP8 GEMM for W_Q_B projection | m=1, k=1536, n=24576 (128×192) |
| 3 | `fp8_gemm_wkv_b` | FP8 GEMM for W_KV_B projection | m=1, k=512, n=32768 (128×256) |
| 4 | `fp8_index_2k` | FP8 index score, short context | b=1, m=1, h=64, d=128, ctx=2048 |
| 5 | `fp8_index_16k` | FP8 index score, long context | b=1, m=1, h=64, d=128, ctx=16384 |

---

## Tolerance Policy

| Op family | Tolerance constant | Value |
|-----------|-------------------|-------|
| `act_quant`, `fp8_gemm_*` | `LOOSE_TOL` | 1.0 |
| `fp8_index_*` | `STRICT_TOL` | 1e-3 |

The check computes `(output.float() - reference.float()).abs().max()` and
asserts it is `<= tolerance`.

If any check fails, the run is marked `all_ok=False` and the score is
**invalid** — it will still be printed/recorded but labelled FAIL.

---

## How to Add a New Experiment

1. Write a **factory function** with signature:

   ```python
   def my_factory(tensors: dict) -> dict:
       # tensors comes from benchmark.make_tensors(cfg_dict)
       # pre-compute any expensive state here (e.g. dequantise weights)
       cached_weight = ...

       def my_fn(tensors):
           # hot path — only this is timed
           return cached_weight @ ...

       return {"fp8_gemm_wq_b_fn": my_fn, "fp8_gemm_wkv_b_fn": my_fn}
   ```

2. Add a `(name, factory)` pair to the experiment list in
   `run_experiments.py`:

   ```python
   experiments.append(("my_experiment", my_factory(tensors)))
   ```

### Override keys

| Key | Signature | Replaces |
|-----|-----------|----------|
| `act_quant_fn` | `fn(tensors) -> (fp8, scale)` | `act_quant` |
| `fp8_gemm_wq_b_fn` | `fn(tensors) -> bf16 tensor` | wq_b gemm |
| `fp8_gemm_wkv_b_fn` | `fn(tensors) -> bf16 tensor` | wkv_b gemm |
| `fp8_index_fn` | `fn(tensors) -> fp32 tensor` | 2k index |
| `fp8_index_16k_fn` | `fn(tensors) -> fp32 tensor` | 16k index |

Each closure should have **pre-computed/cached state already baked in**
so that only the hot path is measured.

---

## How to Run

```bash
# Run all 100+ experiments with default settings (warmup=2, iters=3)
python run_experiments.py

# Custom config
python run_experiments.py --config /path/to/config.json

# Adjust timing
python run_experiments.py --warmup 5 --iters 10

# Filter to a subset
python run_experiments.py --filter gemm_fp32
python run_experiments.py --filter index_fp16

# Show help
python run_experiments.py --help
```

Results are written to:
- `search/reports/leaderboard_score.json`
- `search/reports/leaderboard_score.md`

---

## Rules and Constraints

1. **Must pass tolerance check** — a result with `all_ok=False` is invalid
   and should not be promoted to production.

2. **Reference hardware** — scores are measured on an **RTX 3090 (sm_86)**.
   Scores are not comparable across GPU generations.

3. **No TileLang kernels** — TileLang does not compile on sm_86
   (`sm_86 < sm_90`).  All experiments must use PyTorch ops only.
   The existing `kernel.py` automatically falls back to pure-PyTorch
   implementations on pre-Hopper devices
   (`USE_TORCH_FP8_FALLBACK=True` on sm_86).

4. **Closures must be self-contained** — the timed hot path may not
   perform expensive initialisation (weight dequantisation, memory
   allocation, etc.).  All such work must happen inside the factory,
   before the closure is returned.

5. **Reproducibility** — tensor seeds are fixed by `make_tensors()`;
   do not replace those tensors with different random data in a factory.

6. **Output dtype** — every override must return `bfloat16` (or a tuple
   whose first element is `bfloat16`) to match the reference.
