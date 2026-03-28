import json
import math
import statistics
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tilelang

if not hasattr(torch, "bfloat"):
    torch.bfloat = torch.bfloat16

tilelang.env.disable_cache()

from kernel import USE_TORCH_FP8_FALLBACK, act_quant, fp8_gemm, fp8_index
from model import ModelArgs, block_size, weight_dequant


FP8_MAX = 448.0


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_summary(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    equal = torch.equal(a, b)
    out: Dict[str, Any] = {
        "exact": bool(equal),
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
    }
    if equal:
        out["max_abs_diff"] = 0.0
        out["mean_abs_diff"] = 0.0
        return out
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    out["max_abs_diff"] = float(diff.max().item())
    out["mean_abs_diff"] = float(diff.mean().item())
    return out


def ceil_pow2_scale_cpu(x: torch.Tensor) -> torch.Tensor:
    x_np = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32, copy=False))
    bits = x_np.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    man_bits = bits & ((1 << 23) - 1)
    ceil_log2 = exp - 127 + (man_bits != 0)
    out_bits = ((ceil_log2 + 127).astype(np.uint32)) << 23
    out_np = out_bits.view(np.float32)
    return torch.from_numpy(out_np).to(device=x.device, dtype=torch.float32)


def act_quant_reference(
    x: torch.Tensor,
    block: int = block_size,
    scale_fmt: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block == 0
    n = x.size(-1)
    x_blocks = x.view(-1, n // block, block).float()
    amax = x_blocks.abs().amax(dim=-1).clamp_min_(1e-4)
    scales = amax / FP8_MAX
    if scale_fmt is not None:
        scales = ceil_pow2_scale_cpu(scales)
    y = torch.clamp(x_blocks / scales.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y.view_as(x), scales.view(*x.shape[:-1], n // block)


def fp8_gemm_reference(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block: int = block_size,
) -> torch.Tensor:
    assert a.is_contiguous() and a_s.is_contiguous()
    assert b.is_contiguous() and b_s.is_contiguous()
    k = a.size(-1)
    m = a.numel() // k
    a_deq = (
        a.view(m, k // block, block).float()
        * a_s.view(m, k // block, 1).float()
    ).reshape(m, k)
    b_deq = weight_dequant(b, b_s)
    c = a_deq @ b_deq.t().float()
    return c.view(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def fp8_index_reference(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    if q_s.dim() == 4 and q_s.size(-1) == 1:
        q_s = q_s.squeeze(-1)
    if k_s.dim() == 3 and k_s.size(-1) == 1:
        k_s = k_s.squeeze(-1)
    logits = torch.einsum("bnd,bmhd->bmnh", k.float(), q.float())
    logits = logits.clamp_min_(0) * q_s.float().unsqueeze(2)
    return logits.sum(dim=-1) * k_s.float().unsqueeze(1)


def benchmark_cuda(fn, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    times_ms.sort()
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "p95_ms": times_ms[min(len(times_ms) - 1, math.ceil(len(times_ms) * 0.95) - 1)],
    }


def make_model_args(config_path: str) -> ModelArgs:
    with open(config_path) as f:
        return ModelArgs(**json.load(f))


def preflight_hardware(model_args: ModelArgs) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")


def make_act_quant_cases(args: ModelArgs, batch_size: int, prefill_len: int) -> Dict[str, Tuple[int, ...]]:
    return {
        "index_q": (batch_size, prefill_len, args.index_n_heads, args.index_head_dim),
        "kv_cache": (batch_size, prefill_len, args.kv_lora_rank),
    }


def make_fp8_gemm_cases(args: ModelArgs, batch_size: int, prefill_len: int) -> Dict[str, Tuple[int, int, int]]:
    qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    return {
        "wq_b_prefill": (batch_size * prefill_len, args.q_lora_rank, args.n_heads * qk_head_dim),
        "wkv_b_prefill": (
            batch_size * prefill_len,
            args.kv_lora_rank,
            args.n_heads * (args.qk_nope_head_dim + args.v_head_dim),
        ),
    }


def run_act_quant_suite(
    args: ModelArgs,
    batch_size: int,
    prefill_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    results: Dict[str, Any] = {}
    for name, shape in make_act_quant_cases(args, batch_size, prefill_len).items():
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()
        y, s = act_quant(x, block_size, args.scale_fmt)
        y_ref, s_ref = act_quant_reference(x, block_size, args.scale_fmt)
        bench = benchmark_cuda(lambda: act_quant(x, block_size, args.scale_fmt), warmup, iters)
        results[name] = {
            "shape": list(shape),
            "benchmark": bench,
            "y_check": tensor_summary(y, y_ref),
            "scale_check": tensor_summary(s, s_ref),
        }
    return results


def run_fp8_gemm_suite(
    args: ModelArgs,
    batch_size: int,
    prefill_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed + 1)
    results: Dict[str, Any] = {}
    for name, (m, k, n) in make_fp8_gemm_cases(args, batch_size, prefill_len).items():
        a_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).contiguous()
        a_fp8, a_s = act_quant(a_bf16, block_size, args.scale_fmt)
        b_fp32 = torch.randn(n, k, device="cuda", dtype=torch.float32).clamp_(-FP8_MAX, FP8_MAX)
        b_fp8 = b_fp32.to(torch.float8_e4m3fn).contiguous()
        b_s = torch.rand(
            n // block_size, k // block_size, device="cuda", dtype=torch.float32
        ).mul_(0.05).add_(1e-4).contiguous()
        out = fp8_gemm(a_fp8, a_s, b_fp8, b_s)
        out_ref = fp8_gemm_reference(a_fp8, a_s, b_fp8, b_s)
        bench = benchmark_cuda(lambda: fp8_gemm(a_fp8, a_s, b_fp8, b_s), warmup, iters)
        results[name] = {
            "shape": {"m": m, "k": k, "n": n},
            "benchmark": bench,
            "out_check": tensor_summary(out, out_ref),
        }
    return results


def run_fp8_index_suite(
    args: ModelArgs,
    batch_size: int,
    prefill_len: int,
    decode_context: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed + 2)
    cases = {
        "decode": (batch_size, 1, args.index_n_heads, args.index_head_dim, decode_context),
        "prefill_small": (
            batch_size,
            min(prefill_len, 64),
            args.index_n_heads,
            args.index_head_dim,
            min(prefill_len, 512),
        ),
    }
    results: Dict[str, Any] = {}
    for name, (b, m, h, d, n) in cases.items():
        q_bf16 = torch.randn(b, m, h, d, device="cuda", dtype=torch.bfloat16).contiguous()
        k_bf16 = torch.randn(b, n, d, device="cuda", dtype=torch.bfloat16).contiguous()
        q_fp8, q_s = act_quant(q_bf16, block_size, args.scale_fmt)
        k_fp8, k_s = act_quant(k_bf16, block_size, args.scale_fmt)
        q_s = q_s.contiguous()
        k_s = k_s.squeeze(-1).contiguous()
        out = fp8_index(q_fp8, q_s, k_fp8, k_s)
        out_ref = fp8_index_reference(q_fp8, q_s, k_fp8, k_s)
        bench = benchmark_cuda(lambda: fp8_index(q_fp8, q_s, k_fp8, k_s), warmup, iters)
        results[name] = {
            "shape": {"b": b, "m": m, "h": h, "d": d, "n": n},
            "benchmark": bench,
            "out_check": tensor_summary(out, out_ref),
        }
    return results


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config_671B_v3.2.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-len", type=int, default=256)
    parser.add_argument("--decode-context", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=str, default="")
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)

    model_args = make_model_args(cli.config)
    preflight_hardware(model_args)
    results = {
        "config": cli.config,
        "batch_size": cli.batch_size,
        "prefill_len": cli.prefill_len,
        "decode_context": cli.decode_context,
        "warmup": cli.warmup,
        "iters": cli.iters,
        "seed": cli.seed,
        "device": torch.cuda.get_device_name(0),
        "torch_fp8_fallback": USE_TORCH_FP8_FALLBACK,
        "act_quant": run_act_quant_suite(
            model_args, cli.batch_size, cli.prefill_len, cli.warmup, cli.iters, cli.seed
        ),
        "fp8_gemm": run_fp8_gemm_suite(
            model_args, cli.batch_size, cli.prefill_len, cli.warmup, cli.iters, cli.seed
        ),
        "fp8_index": run_fp8_index_suite(
            model_args,
            cli.batch_size,
            cli.prefill_len,
            cli.decode_context,
            cli.warmup,
            cli.iters,
            cli.seed,
        ),
    }

    print(json.dumps(results, indent=2))
    if cli.json_out:
        out_path = Path(cli.json_out)
        out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
