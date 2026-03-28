import json
import statistics
from argparse import ArgumentParser
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from kernel import (
    _act_quant_torch,
    _fp8_index_torch,
    _weight_dequant_torch,
    act_quant,
    fp8_index,
)
from model import ModelArgs, block_size


def benchmark_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
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
    }


def tensor_check(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    exact = torch.equal(a, b)
    out = {"exact": bool(exact)}
    if exact:
        out["max_abs_diff"] = 0.0
        return out
    diff = (a.float() - b.float()).abs()
    out["max_abs_diff"] = float(diff.max().item())
    out["mean_abs_diff"] = float(diff.mean().item())
    return out


def _a_dequant(a_fp8: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    k = a_fp8.size(-1)
    m = a_fp8.numel() // k
    return (
        a_fp8.view(m, k // block_size, block_size).float()
        * a_s.view(m, k // block_size, 1).float()
    ).reshape(m, k)


def _b_dequant(b_fp8: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    return _weight_dequant_torch(b_fp8, b_s, block_size).float().contiguous()


def _reshape_gemm_out(out: torch.Tensor, a_fp8: torch.Tensor, n: int) -> torch.Tensor:
    return out.view(*a_fp8.shape[:-1], n).to(torch.get_default_dtype())


def fp8_gemm_exact_linear(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_fp8: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    n = b_fp8.size(0)
    a_deq = _a_dequant(a_fp8, a_s)
    b_deq = _b_dequant(b_fp8, b_s)
    return _reshape_gemm_out(F.linear(a_deq, b_deq), a_fp8, n)


def fp8_gemm_exact_linear_cached(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_deq: torch.Tensor
) -> torch.Tensor:
    n = b_deq.size(0)
    a_deq = _a_dequant(a_fp8, a_s)
    return _reshape_gemm_out(F.linear(a_deq, b_deq), a_fp8, n)


def fp8_gemm_exact_mm_cached(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_deq_t: torch.Tensor
) -> torch.Tensor:
    n = b_deq_t.size(1)
    a_deq = _a_dequant(a_fp8, a_s)
    return _reshape_gemm_out(a_deq @ b_deq_t, a_fp8, n)


def fp8_gemm_exact_addmm_cached(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_deq_t: torch.Tensor
) -> torch.Tensor:
    n = b_deq_t.size(1)
    a_deq = _a_dequant(a_fp8, a_s)
    zero = torch.zeros((a_deq.size(0), n), device=a_deq.device, dtype=a_deq.dtype)
    return _reshape_gemm_out(torch.addmm(zero, a_deq, b_deq_t), a_fp8, n)


def fp8_gemm_exact_matmul_cached(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_deq_t: torch.Tensor
) -> torch.Tensor:
    n = b_deq_t.size(1)
    a_deq = _a_dequant(a_fp8, a_s)
    return _reshape_gemm_out(torch.matmul(a_deq, b_deq_t), a_fp8, n)


def fp8_gemm_exact_einsum_cached(
    a_fp8: torch.Tensor, a_s: torch.Tensor, b_deq_t: torch.Tensor
) -> torch.Tensor:
    n = b_deq_t.size(1)
    a_deq = _a_dequant(a_fp8, a_s)
    return _reshape_gemm_out(torch.einsum("mk,kn->mn", a_deq, b_deq_t), a_fp8, n)


def benchmark_gemm_components(
    a_fp8: torch.Tensor,
    a_s: torch.Tensor,
    b_fp8: torch.Tensor,
    b_s: torch.Tensor,
    b_deq: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, Dict[str, float]]:
    n = b_fp8.size(0)
    a_deq = _a_dequant(a_fp8, a_s)
    b_deq_t = b_deq.t().contiguous()
    zero = torch.zeros((a_deq.size(0), n), device=a_deq.device, dtype=a_deq.dtype)

    return {
        "a_dequant": benchmark_cuda(lambda: _a_dequant(a_fp8, a_s), warmup, iters),
        "b_dequant": benchmark_cuda(lambda: _b_dequant(b_fp8, b_s), warmup, iters),
        "linear_flinear": benchmark_cuda(lambda: F.linear(a_deq, b_deq), warmup, iters),
        "linear_mm": benchmark_cuda(lambda: a_deq @ b_deq_t, warmup, iters),
        "linear_addmm": benchmark_cuda(lambda: torch.addmm(zero, a_deq, b_deq_t), warmup, iters),
    }


def fp8_index_exact_einsum(
    q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor
) -> torch.Tensor:
    if q_s.dim() == 4 and q_s.size(-1) == 1:
        q_s = q_s.squeeze(-1)
    if k_s.dim() == 3 and k_s.size(-1) == 1:
        k_s = k_s.squeeze(-1)
    logits = torch.einsum("bnd,bmhd->bmnh", k.float(), q.float())
    logits = logits.clamp_min_(0) * q_s.float().unsqueeze(2)
    return logits.sum(dim=-1) * k_s.float().unsqueeze(1)


def fp8_index_exact_matmul(
    q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor
) -> torch.Tensor:
    if q_s.dim() == 4 and q_s.size(-1) == 1:
        q_s = q_s.squeeze(-1)
    if k_s.dim() == 3 and k_s.size(-1) == 1:
        k_s = k_s.squeeze(-1)
    q_deq = q.float() * q_s.float().unsqueeze(-1)
    k_deq = k.float() * k_s.float().unsqueeze(-1)
    logits = torch.matmul(k_deq.unsqueeze(1), q_deq.permute(0, 1, 3, 2))
    return logits.clamp_min_(0).sum(dim=-1, dtype=torch.float32)


def load_args(config_path: str) -> ModelArgs:
    with open(config_path) as f:
        return ModelArgs(**json.load(f))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config_671B_v3.2.json")
    parser.add_argument("--prefill-len", type=int, default=256)
    parser.add_argument("--decode-context", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    model_args = load_args(args.config)

    m = args.prefill_len
    qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
    gemm_m = args.prefill_len
    gemm_k = model_args.q_lora_rank
    gemm_n = model_args.n_heads * qk_head_dim

    a_bf16 = torch.randn(gemm_m, gemm_k, device="cuda", dtype=torch.bfloat16).contiguous()
    a_fp8, a_s = act_quant(a_bf16, block_size, model_args.scale_fmt)
    b_fp32 = torch.randn(gemm_n, gemm_k, device="cuda", dtype=torch.float32).clamp_(-448, 448)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn).contiguous()
    b_s = torch.rand(
        gemm_n // block_size, gemm_k // block_size, device="cuda", dtype=torch.float32
    ).mul_(0.05).add_(1e-4).contiguous()
    b_deq = _weight_dequant_torch(b_fp8, b_s, block_size).float().contiguous()
    b_deq_t = b_deq.t().contiguous()

    q_bf16 = torch.randn(1, m, model_args.index_n_heads, model_args.index_head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    k_bf16 = torch.randn(1, args.decode_context, model_args.index_head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    q_fp8, q_s = _act_quant_torch(q_bf16, block_size, model_args.scale_fmt)
    k_fp8, k_s = _act_quant_torch(k_bf16, block_size, model_args.scale_fmt)
    q_s = q_s.contiguous()
    k_s = k_s.squeeze(-1).contiguous()

    gemm_ref = fp8_gemm_exact_linear(a_fp8, a_s, b_fp8, b_s)
    index_ref = fp8_index_exact_einsum(q_fp8, q_s, k_fp8, k_s)

    results = {
        "device": torch.cuda.get_device_name(0),
        "gemm": {
            "exact_linear": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_linear(a_fp8, a_s, b_fp8, b_s), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_linear(a_fp8, a_s, b_fp8, b_s), gemm_ref),
            },
            "exact_linear_cached_weight": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_linear_cached(a_fp8, a_s, b_deq), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_linear_cached(a_fp8, a_s, b_deq), gemm_ref),
            },
            "exact_mm_cached_weight": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_mm_cached(a_fp8, a_s, b_deq_t), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_mm_cached(a_fp8, a_s, b_deq_t), gemm_ref),
            },
            "exact_addmm_cached_weight": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_addmm_cached(a_fp8, a_s, b_deq_t), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_addmm_cached(a_fp8, a_s, b_deq_t), gemm_ref),
            },
            "exact_matmul_cached_weight": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_matmul_cached(a_fp8, a_s, b_deq_t), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_matmul_cached(a_fp8, a_s, b_deq_t), gemm_ref),
            },
            "exact_einsum_cached_weight": {
                "benchmark": benchmark_cuda(lambda: fp8_gemm_exact_einsum_cached(a_fp8, a_s, b_deq_t), args.warmup, args.iters),
                "check": tensor_check(fp8_gemm_exact_einsum_cached(a_fp8, a_s, b_deq_t), gemm_ref),
            },
        },
        "gemm_components": benchmark_gemm_components(
            a_fp8, a_s, b_fp8, b_s, b_deq, args.warmup, args.iters
        ),
        "index": {
            "exact_einsum": {
                "benchmark": benchmark_cuda(lambda: fp8_index_exact_einsum(q_fp8, q_s, k_fp8, k_s), args.warmup, args.iters),
                "check": tensor_check(fp8_index_exact_einsum(q_fp8, q_s, k_fp8, k_s), index_ref),
            },
            "exact_matmul": {
                "benchmark": benchmark_cuda(lambda: fp8_index_exact_matmul(q_fp8, q_s, k_fp8, k_s), args.warmup, args.iters),
                "check": tensor_check(fp8_index_exact_matmul(q_fp8, q_s, k_fp8, k_s), index_ref),
            },
            "current_kernel_api": {
                "benchmark": benchmark_cuda(lambda: fp8_index(q_fp8, q_s, k_fp8, k_s), args.warmup, args.iters),
                "check": tensor_check(fp8_index(q_fp8, q_s, k_fp8, k_s), index_ref),
            },
            "current_kernel_api_torch_path": {
                "note": "On sm_86 this is the exact torch fallback path.",
            },
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
