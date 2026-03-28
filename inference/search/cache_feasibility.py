import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT.parent / "config_671B_v3.2.json"
DEFAULT_REPORT = ROOT / "reports" / "weight_dequant_cache_feasibility.md"
DEFAULT_JSON = ROOT / "reports" / "weight_dequant_cache_feasibility.json"
BLOCK_SIZE = 128
GPU_BUDGET_BYTES = 24 * 1024**3


@dataclass(frozen=True)
class WeightGroup:
    name: str
    instances: int
    rows: int
    cols: int

    @property
    def elements(self) -> int:
        return self.instances * self.rows * self.cols

    @property
    def fp8_weight_bytes(self) -> int:
        return self.elements

    @property
    def scale_bytes(self) -> int:
        scale_rows = self.rows // BLOCK_SIZE
        scale_cols = self.cols // BLOCK_SIZE
        return self.instances * scale_rows * scale_cols * 4

    def dequant_bytes(self, dtype_bytes: int) -> int:
        return self.elements * dtype_bytes


def load_config(path: Path) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def tib(num_bytes: int) -> float:
    return num_bytes / (1024**4)


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**4:
        return f"{tib(num_bytes):.2f} TiB"
    if num_bytes >= 1024**3:
        return f"{gib(num_bytes):.2f} GiB"
    return f"{mib(num_bytes):.2f} MiB"


def build_groups(cfg: Dict[str, int]) -> Tuple[List[WeightGroup], List[WeightGroup]]:
    dim = cfg["dim"]
    inter_dim = cfg["inter_dim"]
    moe_inter_dim = cfg["moe_inter_dim"]
    n_layers = cfg["n_layers"]
    n_dense_layers = cfg["n_dense_layers"]
    n_moe_layers = n_layers - n_dense_layers
    n_heads = cfg["n_heads"]
    q_lora_rank = cfg["q_lora_rank"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_nope_head_dim = cfg["qk_nope_head_dim"]
    qk_rope_head_dim = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    n_routed_experts = cfg["n_routed_experts"]

    mla = [
        WeightGroup("mla.wq_a", n_layers, dim, q_lora_rank),
        WeightGroup("mla.wq_b", n_layers, q_lora_rank, n_heads * qk_head_dim),
        WeightGroup("mla.wkv_a", n_layers, dim, kv_lora_rank + qk_rope_head_dim),
        WeightGroup("mla.wkv_b", n_layers, kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)),
        WeightGroup("mla.wo", n_layers, n_heads * v_head_dim, dim),
    ]

    dense_mlp = [
        WeightGroup("dense_mlp.w1", n_dense_layers, dim, inter_dim),
        WeightGroup("dense_mlp.w2", n_dense_layers, inter_dim, dim),
        WeightGroup("dense_mlp.w3", n_dense_layers, dim, inter_dim),
    ]

    moe_shared = [
        WeightGroup("moe.shared_experts.w1", n_moe_layers, dim, moe_inter_dim),
        WeightGroup("moe.shared_experts.w2", n_moe_layers, moe_inter_dim, dim),
        WeightGroup("moe.shared_experts.w3", n_moe_layers, dim, moe_inter_dim),
    ]

    moe_routed = [
        WeightGroup("moe.experts.w1", n_moe_layers * n_routed_experts, dim, moe_inter_dim),
        WeightGroup("moe.experts.w2", n_moe_layers * n_routed_experts, moe_inter_dim, dim),
        WeightGroup("moe.experts.w3", n_moe_layers * n_routed_experts, dim, moe_inter_dim),
    ]

    return mla + dense_mlp + moe_shared + moe_routed, mla + dense_mlp + moe_shared


def summarize(groups: List[WeightGroup]) -> Dict[str, int]:
    out = {
        "instances": 0,
        "elements": 0,
        "fp8_weight_bytes": 0,
        "scale_bytes": 0,
        "bf16_cache_bytes": 0,
        "fp32_cache_bytes": 0,
    }
    for g in groups:
        out["instances"] += g.instances
        out["elements"] += g.elements
        out["fp8_weight_bytes"] += g.fp8_weight_bytes
        out["scale_bytes"] += g.scale_bytes
        out["bf16_cache_bytes"] += g.dequant_bytes(2)
        out["fp32_cache_bytes"] += g.dequant_bytes(4)
    return out


def budget_flag(num_bytes: int) -> str:
    return "fits" if num_bytes <= GPU_BUDGET_BYTES else "does not fit"


def render_report(cfg: Dict[str, int], groups: List[WeightGroup], partial_groups: List[WeightGroup]) -> str:
    full = summarize(groups)
    partial = summarize(partial_groups)
    dim = cfg["dim"]
    inter_dim = cfg["inter_dim"]
    moe_inter_dim = cfg["moe_inter_dim"]
    n_layers = cfg["n_layers"]
    n_dense_layers = cfg["n_dense_layers"]
    n_moe_layers = n_layers - n_dense_layers
    n_heads = cfg["n_heads"]
    q_lora_rank = cfg["q_lora_rank"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_nope_head_dim = cfg["qk_nope_head_dim"]
    qk_rope_head_dim = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    n_routed_experts = cfg["n_routed_experts"]

    mla_block = summarize([
        WeightGroup("mla.wq_a", 1, dim, q_lora_rank),
        WeightGroup("mla.wq_b", 1, q_lora_rank, n_heads * qk_head_dim),
        WeightGroup("mla.wkv_a", 1, dim, kv_lora_rank + qk_rope_head_dim),
        WeightGroup("mla.wkv_b", 1, kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)),
        WeightGroup("mla.wo", 1, n_heads * v_head_dim, dim),
    ])
    dense_block = summarize([
        WeightGroup("mla.wq_a", 1, dim, q_lora_rank),
        WeightGroup("mla.wq_b", 1, q_lora_rank, n_heads * qk_head_dim),
        WeightGroup("mla.wkv_a", 1, dim, kv_lora_rank + qk_rope_head_dim),
        WeightGroup("mla.wkv_b", 1, kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)),
        WeightGroup("mla.wo", 1, n_heads * v_head_dim, dim),
        WeightGroup("dense_mlp.w1", 1, dim, inter_dim),
        WeightGroup("dense_mlp.w2", 1, inter_dim, dim),
        WeightGroup("dense_mlp.w3", 1, dim, inter_dim),
    ])
    moe_shared_block = summarize([
        WeightGroup("mla.wq_a", 1, dim, q_lora_rank),
        WeightGroup("mla.wq_b", 1, q_lora_rank, n_heads * qk_head_dim),
        WeightGroup("mla.wkv_a", 1, dim, kv_lora_rank + qk_rope_head_dim),
        WeightGroup("mla.wkv_b", 1, kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)),
        WeightGroup("mla.wo", 1, n_heads * v_head_dim, dim),
        WeightGroup("moe.shared_experts.w1", 1, dim, moe_inter_dim),
        WeightGroup("moe.shared_experts.w2", 1, moe_inter_dim, dim),
        WeightGroup("moe.shared_experts.w3", 1, dim, moe_inter_dim),
    ])
    one_expert = summarize([
        WeightGroup("moe.experts.w1", 1, dim, moe_inter_dim),
        WeightGroup("moe.experts.w2", 1, moe_inter_dim, dim),
        WeightGroup("moe.experts.w3", 1, dim, moe_inter_dim),
    ])
    routed_bank = summarize([
        WeightGroup("moe.experts.w1", n_routed_experts, dim, moe_inter_dim),
        WeightGroup("moe.experts.w2", n_routed_experts, moe_inter_dim, dim),
        WeightGroup("moe.experts.w3", n_routed_experts, dim, moe_inter_dim),
    ])
    lines = []
    lines.append("# Weight Dequant Cache Feasibility")
    lines.append("")
    lines.append("This report estimates the memory cost of caching dequantized fp8 weights for the current DeepSeek V3.2 config.")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Full fp8 weight set in bf16 cache form: {format_size(full['bf16_cache_bytes'])}")
    lines.append(f"- Full fp8 weight set in fp32 cache form: {format_size(full['fp32_cache_bytes'])}")
    lines.append(f"- Non-routed fp8 weights only in bf16 cache form: {format_size(partial['bf16_cache_bytes'])}")
    lines.append(f"- Non-routed fp8 weights only in fp32 cache form: {format_size(partial['fp32_cache_bytes'])}")
    lines.append(f"- One dense transformer block in bf16 cache form: {format_size(dense_block['bf16_cache_bytes'])} ({budget_flag(dense_block['bf16_cache_bytes'])} on a 24 GiB GPU)")
    lines.append(f"- One MoE routed-expert bank in bf16 cache form: {format_size(routed_bank['bf16_cache_bytes'])} ({budget_flag(routed_bank['bf16_cache_bytes'])} on a 24 GiB GPU)")
    lines.append("")
    lines.append("## Exact Config")
    for key in ["n_layers", "n_dense_layers", "n_heads", "n_routed_experts", "n_shared_experts", "n_activated_experts", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "inter_dim", "moe_inter_dim", "dim"]:
        lines.append(f"- `{key}` = {cfg[key]}")
    lines.append("")
    lines.append("## Per-Group Costs")
    lines.append("| group | instances | shape | fp8 weight | scale | bf16 cache | fp32 cache |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for g in groups:
        lines.append(
            f"| `{g.name}` | {g.instances} | `{g.rows} x {g.cols}` | {format_size(g.fp8_weight_bytes)} | "
            f"{format_size(g.scale_bytes)} | {format_size(g.dequant_bytes(2))} | {format_size(g.dequant_bytes(4))} |"
        )
    lines.append("")
    lines.append("## Conclusions")
    lines.append("- Caching every routed expert weight is not feasible on a 24 GiB GPU.")
    lines.append("- A single dense transformer block can fit, but only as a narrowly scoped layer-local cache.")
    lines.append("- The dense non-expert fp8 weights are too large to keep fully resident as dequantized bf16/fp32 copies across the full model.")
    lines.append("- Any useful cache policy must be selective, layer-local, or windowed rather than global.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- JSON summary: `{DEFAULT_JSON.name}`")
    lines.append(f"- This markdown report: `{DEFAULT_REPORT.name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    groups, partial_groups = build_groups(cfg)
    full_summary = summarize(groups)
    partial_summary = summarize(partial_groups)

    payload = {
        "config": args.config.name,
        "block_size": BLOCK_SIZE,
        "groups": [asdict(g) for g in groups],
        "summary": {
            "full": full_summary,
            "non_routed_only": partial_summary,
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    report = render_report(cfg, groups, partial_groups)
    with open(args.report, "w") as f:
        f.write(report)

    if args.stdout:
        print(report, end="")
    else:
        print(args.report)


if __name__ == "__main__":
    main()
