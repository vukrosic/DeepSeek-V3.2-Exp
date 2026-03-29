# DeepSeek-V3.2-Exp

## RTX 3090 Fork Notes

This repo now includes a practical RTX 3090 (`sm_86`) rescue and optimization pass for the DeepSeek V3.1 / V3.2-Exp inference path. The starting point on a 3090 was not a healthy baseline: the default TileLang FP8 inference path in this repo did not compile on Ampere, so the provided implementation was not directly usable as-is on this class of GPU.

What was changed in this fork:

- built a working CUDA fallback path for the 3090 instead of relying on the non-compiling TileLang FP8 path
- optimized the fallback path without intentionally changing model semantics
- added a search/benchmark/control-plane under [`inference/search`](inference/search) so kernel work is measured, queued, and reproducible
- documented the process, queue rules, prompts, and continuation workflow so other agents or contributors can keep the search going

Main issues we hit and how they were solved:

| Issue | Why it mattered on 3090 | What we changed |
| :--- | :--- | :--- |
| TileLang FP8 kernels did not compile on `sm_86` | the default inference implementation was not usable on the target GPU | added a working exact CUDA fallback path in [`inference/kernel.py`](inference/kernel.py) and wired it through [`inference/model.py`](inference/model.py) |
| fallback path spent too much time in weight dequant and repeated setup | the working path was correct but slower than it needed to be | simplified exact dequant math, added selective cached-weight reuse, and reused shared quantized activations where safe |
| search jobs could collide or waste GPU time | one GPU means one bad runner design ruins all measurements | added a single-owner experiment queue, shard/rebatch tools, retry tools, and queue-safe manifest generation |
| one strict bitwise-equality rule was too blunt for every kernel | it rejected useful smooth-algebra candidates but should stay strict for routing-sensitive code | split policy into strict `exact` and declared `near-exact` lanes, while keeping index/routing/mask paths strict |

Where the 3090 work landed:

- shared code improvements in [`inference/kernel.py`](inference/kernel.py) and [`inference/model.py`](inference/model.py)
- kernel and algorithm search scripts in [`inference/search`](inference/search)
- queue and automation tooling in [`inference/search/queue`](inference/search/queue)
- baseline and leaderboard reports in [`inference/search/reports`](inference/search/reports)

## Beginner Auto-Research Path

If you are new to this repo, use this order:

1. Read [`inference/search/BEGINNER_GUIDE.md`](inference/search/BEGINNER_GUIDE.md).
2. Read [`inference/search/README.md`](inference/search/README.md).
3. Read [`inference/search/PROCESS.md`](inference/search/PROCESS.md).
4. Pick one task from [`inference/search/tasks`](inference/search/tasks).
5. Use [`inference/search/search_runner.py`](inference/search/search_runner.py) to validate and create a run folder.
6. Use [`inference/search/queue/queue_agent.py`](inference/search/queue/queue_agent.py) for queue-safe GPU work.
7. Keep changes local until they win exactness and benchmark checks.

Minimal entry commands:

```bash
cd inference
python3 search/search_runner.py validate
python3 search/search_runner.py list
python3 search/search_runner.py show 02_fp8_gemm_exact
python3 search/queue/queue_agent.py status
```

Best measured results so far on the 3090:

- `mla_wq_b`: up to about `5.90x` speedup at `m=1`, with strong wins still visible through larger prefill lengths
- `mla_wkv_b`: about `1.84x` at small `m`
- medium-length `mla_wq_b` shapes remain meaningfully faster, for example about `1.43x` at `m=384`
- strict index-path work is instrumented and has been searched, but no accepted strict winner has been promoted yet

If you want to continue this work or automate more of it, start here:

- search workspace overview: [`inference/search/README.md`](inference/search/README.md)
- GitHub-facing continuation guide: [`inference/search/AUTOMATION.md`](inference/search/AUTOMATION.md)
- repo-contained kernel-search skill: [`inference/search/KERNEL_SPEED_SEARCH_SKILL.md`](inference/search/KERNEL_SPEED_SEARCH_SKILL.md)
- process and acceptance policy: [`inference/search/PROCESS.md`](inference/search/PROCESS.md)
- agent handoff: [`inference/search/AGENT_PLAYBOOK.md`](inference/search/AGENT_PLAYBOOK.md)
- reusable agent prompts: [`inference/search/prompts`](inference/search/prompts)
- queue model and submission rules: [`inference/search/queue/README.md`](inference/search/queue/README.md)

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction


We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

This experimental release represents our ongoing research into more efficient transformer architectures, particularly focusing on improving computational efficiency when processing extended text sequences.

<div align="center">
 <img src="cost.jpg" >
</div>

- DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time, delivering substantial improvements in long-context training and inference efficiency while maintaining virtually identical model output quality.


- To rigorously evaluate the impact of introducing sparse attention, we deliberately aligned the training configurations of DeepSeek-V3.2-Exp with V3.1-Terminus. Across public benchmarks in various domains, DeepSeek-V3.2-Exp demonstrates performance on par with V3.1-Terminus.


| Benchmark | DeepSeek-V3.1-Terminus | DeepSeek-V3.2-Exp |
| :--- | :---: | :---: |
| **Reasoning Mode w/o Tool Use** | | |
| MMLU-Pro | 85.0 | 85.0 |
| GPQA-Diamond | 80.7 | 79.9 |
| Humanity's Last Exam | 21.7 | 19.8 |
| LiveCodeBench | 74.9 | 74.1 |
| AIME 2025 | 88.4 | 89.3 |
| HMMT 2025 | 86.1 | 83.6 |
| Codeforces | 2046 | 2121 |
| Aider-Polyglot | 76.1 | 74.5 |
| **Agentic Tool Use** | | |
| BrowseComp | 38.5 | 40.1 |
| BrowseComp-zh | 45.0 | 47.9 |
| SimpleQA | 96.8 | 97.1 |
| SWE Verified | 68.4 | 67.8 |
| SWE-bench Multilingual | 57.8 | 57.9 |
| Terminal-bench | 36.7 | 37.7 |

## Update

- 2025.11.17: **We have identified that previous versions of the inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded model performance.** Specifically, the input tensor to RoPE in the indexer module requires a non-interleaved layout, whereas RoPE in the MLA module expects an interleaved layout. This issue has now been resolved. Please refer to the updated version of the inference demo code and take note of this implementation detail.

## Open-Source Kernels

For TileLang kernels with **better readability and research-purpose design**, please refer to [TileLang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32).

For **high-performance CUDA kernels**, indexer logit kernels (including paged versions) are available in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/200). Sparse attention kernels are released in [FlashMLA](https://github.com/deepseek-ai/FlashMLA/pull/98).



## How to Run Locally

### HuggingFace
We provide an updated inference demo code in the [inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) folder to help the community quickly get started with our model and understand its architectural details.

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

### SGLang

#### Installation with Docker

```
# H200
docker pull lmsysorg/sglang:dsv32

# MI350
docker pull lmsysorg/sglang:dsv32-rocm

# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

#### Launch Command
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

### vLLM

vLLM provides day-0 support of DeepSeek-V3.2-Exp. See the [recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html) for up-to-date details.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{deepseekai2024deepseekv32,
      title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
      author={DeepSeek-AI},
      year={2025},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
