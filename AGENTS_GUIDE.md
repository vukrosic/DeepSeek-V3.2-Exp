# Multi-Agent Kernel Optimization Guide

## Overview

This guide explains how to scale the DeepSeek-V3.2-Exp kernel optimization from 100s to 1000s of experiments using multiple Claude Code agents in parallel.

**Current Best Result:** `0.2301 ms` (geometric mean of 5 decode-step ops)
- Strategy: `gemm_bf16_t_matmul_bf16c_nc`
- Speedup vs baseline: **~2.0x**

## Architecture

### Core Components

1. **Benchmark Framework** (`/root/DeepSeek-V3.2-Exp/inference/`)
   - `benchmark.py` — Single experiment runner (measures 5 ops, returns score_ms)
   - `run_experiments.py` — Experiment set builder + orchestration
   - `kernel.py`, `model.py` — The actual CUDA kernels being optimized

2. **Queue System** (TBD - implement per instructions below)
   - Central experiment queue to prevent collisions
   - Results aggregation point
   - Leaderboard maintenance

3. **GPU Resource**
   - vast.ai RTX 3090 (24GB)
   - SSH: `ssh6.vast.ai:31117` (key: `~/.ssh/vast_ai_ed25519`)
   - Workspace: `/workspace/DeepSeek-V3.2-Exp/`
   - Venv: `/venv/main/bin/activate`
   - Each experiment: ~2-3 seconds (warmup=2, iters=3)

## How to Generate Experiments

### Pattern 1: Parameter Space Cartesian Product

```python
# Combinatorial exploration
cache_dtypes = ["bf16", "fp32"]
layouts = ["row", "t"]
ops = ["mm", "matmul"]
compute_dtypes = ["bf16", "fp32"]

for cache in cache_dtypes:
    for layout in layouts:
        for op in ops:
            for cd in compute_dtypes:
                # Build 1 experiment per combination
                name = f"gemm_{cache}_{layout}_{op}_{cd}c"
                # = 2 * 2 * 2 * 2 = 16 experiments
```

**Why this works:** Each dimension has small domain, multiplicative growth = dense coverage.

### Pattern 2: Focused Deep Dives

Once you find a good region, drill down:

```python
# Best cache strategy: bf16
# Test all ops + dtypes + contig variants

layouts = ["row", "t"]
ops = ["flinear", "mm", "matmul", "addmm", "einsum"]  # 5 ops
contig_variants = [False, True]  # 2 variants
compute_dtypes = ["bf16", "fp32"]  # 2 dtypes

# Per layout: 5 * 2 * 2 = 20 experiments
# Per cache: 2 * 20 = 40 experiments
# = ~100 focused experiments per cache strategy
```

### Pattern 3: Repetitions for Statistical Significance

```python
# Run best candidate 10+ times to measure variance
best_candidate = "gemm_bf16_t_matmul_bf16c_nc"

for rep in range(10):
    name = f"{best_candidate}_rep{rep}"
    # = 10 repetitions of same config
    # Reveals measurement noise, reveals consistency
```

### Pattern 4: Variant Mutation

Tiny changes to top candidates:

```python
# If top candidate: gemm_bf16_t_mm_bf16c_nc
# Try mutations:

base = "gemm_bf16_t_mm_bf16c_nc"
mutations = [
    "gemm_bf16_t_mm_bf16c_c",      # Toggle contig
    "gemm_bf16_t_matmul_bf16c_nc", # Try matmul instead of mm
    "gemm_bf16_row_mm_bf16c_nc",   # Try row layout
]

for mut in mutations:
    # Each mutation = 1 experiment
    # = 3 mutations per top candidate
```

## Generating 1000+ Experiments

### Strategy 1: Expand Each Dimension

**Current:** ~150 gemm experiments
**Target:** 300+ gemm experiments

Approach:
```python
# Add new dimensions:
- block_sizes: [64, 128, 256]           # 3x multiplier
- contiguous_variants: [False, True]    # 2x multiplier
- cast_orderings: [early, late, mixed]  # 3x multiplier

# If you combine 150 * (3/2 avg expansion) = ~200+
```

### Strategy 2: Full Cartesian Product on Best Region

```python
# Focus ONLY on best strategies, remove dead paths

best_configs = {
    "cache": ["bf16"],           # ONLY bf16 (skip fp32, none)
    "layout": ["row", "t"],      # Both
    "ops": ["mm", "matmul"],     # Top 2 (skip flinear, addmm, einsum)
    "cd": ["bf16"],              # ONLY bf16 (skip fp32)
    "contig": [False, True],     # Both
}

# Product: 1 * 2 * 2 * 1 * 2 = 8 base configs
# Multiply by reps (5-10x each) = 40-80
# Multiply by index variants = 40-80 + 40-80 = 120
# Multiply by act_quant reps = 120 + 50 = 170

# Then add: mutations, shape variants, scheduling = 300+
```

### Strategy 3: Index Kernel Expansion

**Current:** ~16 index experiments
**Target:** 200+ index experiments

Key parameters:
```python
# Expand deq_approaches
deq_approaches = ["fp32_scale_mul", "fp16_scale_mul", "mixed_scale"]
# 3x

# Expand matmul operations
matmul_ops = [
    "einsum_bmnh",           # Current best
    "einsum_bmnhd",          # Variant
    "matmul_broadcast",      # Current good
    "bmm",                   # Variant
    "custom_reshape_mm",     # New: reshape then mm
    "batched_gemm",          # New: cuBLAS batched
]
# 6x

# Expand squeezing patterns
squeeze_patterns = [
    "nosqueeze",
    "squeeze_first",
    "squeeze_after",
    "squeeze_both",
]
# 4x

# Result: 3 * 6 * 4 = 72 index core configs
# + reps (3-5x each) = 200+ experiments
```

### Strategy 4: Activation Quantization Exploration

**Current:** ~15 act_quant experiments
**Target:** 150+ experiments

```python
# Base variants
variants = [
    "baseline",
    "contiguous_input",
    "fp16_intermediate",
    "bf16_output",
    "cached_scales",
    "cached_amax",
    "cached_both",
]
# 7 variants

# Each variant: 10-20 repetitions
# Each variant: try different block sizes [64, 128, 256]
# = 7 * 15 * 3 = 315 experiments (can trim to ~100-150)
```

## Queue Architecture (Implement This)

### File-Based Queue (Simple, No DB)

**Location:** `/workspace/queue/`

**Files:**
```
queue/
├── pending/
│   ├── exp_001.json
│   ├── exp_002.json
│   └── ...
├── running/
│   ├── exp_001.json  (locked by agent PID)
│   └── ...
├── completed/
│   ├── exp_001_result.json
│   └── ...
└── manifest.json (global status)
```

**Experiment JSON format:**
```json
{
  "id": "exp_001",
  "name": "gemm_bf16_t_mm_bf16c_nc",
  "type": "gemm",
  "params": {
    "cache_dtype": "bf16",
    "layout": "t",
    "op": "mm",
    "compute_dtype": "bf16",
    "contig": false
  },
  "priority": 1,
  "created_at": "2026-03-29T12:00:00Z",
  "timeout_seconds": 30
}
```

**Result JSON:**
```json
{
  "id": "exp_001",
  "name": "gemm_bf16_t_mm_bf16c_nc",
  "score_ms": 0.2253,
  "all_ok": true,
  "ops": {
    "act_quant": {"ms": 0.21, "ok": true},
    "fp8_gemm_wq_b": {"ms": 0.23, "ok": true},
    ...
  },
  "completed_at": "2026-03-29T12:00:05Z",
  "agent_id": "agent_1",
  "duration_seconds": 5
}
```

### Agent Workflow

1. **Lock an experiment:**
   ```bash
   mv queue/pending/exp_001.json queue/running/exp_001_${AGENT_ID}.json
   ```

2. **Run it:**
   ```bash
   python run_experiments.py --filter "gemm_bf16_t_mm_bf16c_nc" --iters 3
   ```

3. **Write result:**
   ```bash
   mv queue/running/exp_001_${AGENT_ID}.json queue/completed/exp_001_result.json
   ```

4. **Update manifest:**
   ```python
   manifest["completed"] += 1
   manifest["remaining"] = len(os.listdir("queue/pending"))
   ```

### Aggregation

**Script:** `aggregate_results.py`

```python
import json
from pathlib import Path

results = []
for result_file in Path("queue/completed").glob("*_result.json"):
    with open(result_file) as f:
        results.append(json.load(f))

# Sort by score_ms
results.sort(key=lambda r: r["score_ms"])

# Write leaderboard
with open("leaderboard_final.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Top 10:")
for i, r in enumerate(results[:10], 1):
    print(f"{i:3d}. {r['name']:<50s} {r['score_ms']:>10.4f}")
```

## Agent Task Assignment Strategy

### Parallel Experiment Domains

**Agent 1: Gemm Cache Strategy Deep Dive**
- Focus: All combinations of bf16 cache with different layouts/ops
- ~200 experiments
- Est. time: 400-600 seconds (~7-10 min)

**Agent 2: Index Kernel Variants**
- Focus: All dequant + matmul operation combinations
- ~150 experiments
- Est. time: 300-450 seconds (~5-7 min)

**Agent 3: Act Quant + Repetitions**
- Focus: Activation quantization variants + statistical reps
- ~150 experiments
- Est. time: 300-450 seconds (~5-7 min)

**Agent 4: Top Candidate Mutations**
- Focus: Take leaderboard top 10, mutate and test
- ~200 experiments
- Est. time: 400-600 seconds (~7-10 min)

**Agent 5: New Ideas / Speculative**
- Focus: Try novel combinations not covered by others
- ~300 experiments
- Est. time: 600-900 seconds (~10-15 min)

**Total:** ~1000 experiments
**Parallel runtime:** ~10-15 minutes (with 5 agents)
**Sequential equivalent:** ~30-50 minutes

## Implementation Checklist

- [ ] Create `/workspace/queue/pending/`, `running/`, `completed/` directories
- [ ] Write `exp_generator.py` to create 1000 experiment JSONs
- [ ] Write `queue_agent.py` template for agents to execute
- [ ] Write `aggregate_results.py` for final leaderboard
- [ ] Create `manifest.json` to track global progress
- [ ] Test with 1 agent first, then scale to 5

## Code Template for Agent

```python
#!/usr/bin/env python3
"""Agent: Execute queued kernel optimization experiments."""

import json
import os
import subprocess
from pathlib import Path

QUEUE_DIR = Path("/workspace/queue")
AGENT_ID = os.environ.get("AGENT_ID", "agent_1")

def claim_experiment():
    """Atomically claim next experiment."""
    pending = list(QUEUE_DIR.glob("pending/*.json"))
    if not pending:
        return None

    exp_file = pending[0]
    lock_file = QUEUE_DIR / "running" / f"{exp_file.stem}_{AGENT_ID}.json"

    # Atomic rename = lock
    exp_file.rename(lock_file)

    with open(lock_file) as f:
        return json.load(f)

def run_experiment(exp):
    """Execute single experiment."""
    filter_str = exp["name"]

    result = subprocess.run(
        f"cd /workspace/DeepSeek-V3.2-Exp/inference && "
        f"source /venv/main/bin/activate && "
        f"python run_experiments.py --filter '{filter_str}' --iters 3",
        shell=True,
        capture_output=True,
        timeout=exp["timeout_seconds"]
    )

    # Parse output leaderboard_score.json
    leaderboard_path = Path("/workspace/DeepSeek-V3.2-Exp/inference/search/reports/leaderboard_score.json")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            results = json.load(f)
            if results:
                return results[0]  # Best result

    return None

def main():
    while True:
        exp = claim_experiment()
        if exp is None:
            print(f"[{AGENT_ID}] No more experiments. Exiting.")
            break

        print(f"[{AGENT_ID}] Running: {exp['name']}")

        try:
            result = run_experiment(exp)
            if result:
                result["agent_id"] = AGENT_ID
                result_file = QUEUE_DIR / "completed" / f"{exp['id']}_result.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"[{AGENT_ID}] ✓ {exp['name']}: {result['score_ms']:.4f}")
        except Exception as e:
            print(f"[{AGENT_ID}] ✗ Error: {e}")

        # Clean up lock
        lock_file = QUEUE_DIR / "running" / f"{exp['id']}_{AGENT_ID}.json"
        if lock_file.exists():
            lock_file.unlink()

if __name__ == "__main__":
    main()
```

## Key Insights

1. **Focus > Breadth:** 300 high-quality experiments beat 1000 random ones
2. **Repetitions Matter:** Same config 5x reveals measurement variance
3. **Memory Management:** Add `torch.cuda.empty_cache()` between experiments
4. **Timeout Handling:** Set 30-second timeout per experiment
5. **Atomic Operations:** Use file moves, not writes, for queue locking
6. **Leaderboard Freshness:** Update incrementally, not at the end

## Progress Tracking

**Completed so far:**
- Run 1: 110 experiments → Best: 0.2364 ms
- Run 2: 268 experiments → Best: 0.2253 ms
- Run 3: 102 experiments → Best: 0.2301 ms
- **Target:** 1000 experiments → Best: < 0.22 ms (estimated)

## Next Steps

1. **Immediate:** Use this guide to generate 1000 experiment queue
2. **Short-term:** Spawn 5 agents with different domains
3. **Medium-term:** Analyze results for patterns, identify new promising regions
4. **Long-term:** Integrate best kernel into model.py, measure full inference speedup

## Contact / Continuation

If you're a new agent continuing this work:
1. Read the current leaderboard: `/workspace/DeepSeek-V3.2-Exp/inference/search/reports/leaderboard_score.json`
2. Check queue status: `ls /workspace/queue/pending/ | wc -l`
3. Pick a domain from "Agent Task Assignment Strategy" above
4. Generate experiments in that domain
5. Submit to queue
6. Execute using provided template
7. Update global leaderboard

---

**Last updated:** 2026-03-29
**Best known result:** 0.2301 ms (score_ms, geometric mean of 5 ops)
**Speedup:** ~2.0x from baseline (0.4529 ms)
