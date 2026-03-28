# DeepSeek V3.2

## Current Optimization Status

This inference repo now includes a measured RTX 3090 optimization workspace for the DeepSeek V3.1 / V3.2-Exp inference path. Accepted changes preserve functionality and default semantics. Faster-but-drifting candidates are tracked separately and not landed as defaults.

### Current Search Snapshot

- Tested device: RTX 3090 (`sm_86`)
- Current measured baseline path on this GPU: exact CUDA fallback in `kernel.py`
- Constraint: the original TileLang FP8 path in this repo does not compile on `sm_86`, so exact search on this machine compares against the exact fallback path until an Ampere-native exact kernel exists

### Best Accepted Exact Wins

| area | change | before ms | after ms | speedup |
| --- | --- | ---: | ---: | ---: |
| `MLA.wq_a + MLA.wkv_a` | shared input quant + fp32 cached weights | 1.431 | 0.734 | 1.95x |
| `MLA.wkv_b` prefill path | cached dequantized weight reuse | 1.030 | 0.662 | 1.56x |
| `Indexer` path | shared precomputed `x_fp8` / `qr_fp8` reuse | 1.616 | 1.237 | 1.31x |
| `fp8_gemm.wq_b_prefill` | exact weight dequant simplification | 2.604 | 2.216 | 1.17x |
| `fp8_gemm.wkv_b_prefill` | exact weight dequant simplification | 1.258 | 1.066 | 1.18x |

### Search Docs

- Speed leaderboard and decision log: `search/reports/leaderboard.md`
- Current 3090 baseline and constraints: `search/reports/current-3090-baseline.md`
- Search workspace overview: `search/README.md`
- GitHub-facing automation handoff: `search/AUTOMATION.md`
- Repo-contained kernel-search skill: `search/KERNEL_SPEED_SEARCH_SKILL.md`
- Research workflow and testing process: `search/PROCESS.md`
- Search operating rules: `search/PLAN.md`
- AI handoff and execution docs: `search/AGENT_PLAYBOOK.md`
- Single-GPU experiment queue and submission rules: `search/queue/README.md`

### What Was Landed

- Exact blocked weight dequant simplification in `kernel.py` and `model.py`
- Exact tail-block handling for non-`128`-divisible fallback shapes
- Shared `act_quant(x)` reuse across the `MLA` input projections
- Selective module-local dequantized-weight caches for the `MLA` input path and `Indexer` path
- Shared `x_fp8` and `qr_fp8` reuse through the `Indexer` path
- Search tooling, reports, and a persistent leaderboard so future optimization work is measured instead of guessed

### Search Policy

- preserve functionality
- preserve default semantics
- keep routing-sensitive paths strict
- allow a separate near-exact lane for smooth algebra kernels when tolerance is explicit and downstream behavior is rechecked
- reject fake wins that rely on hidden behavior changes, unsupported dtypes, or undeclared drift
- favor small cheap experiments to eliminate losers quickly, then scale only the winners
- re-run the active acceptance gate before claiming a speed win

## Usage

First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```
