import os

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from typing import Tuple, Optional


tilelang.set_log_level("WARNING")

TileLangTarget = tilelang.jit.__globals__["Target"]

if torch.cuda.is_available():
    cc_major, cc_minor = torch.cuda.get_device_capability()
    tilelang_target = TileLangTarget(f"cuda -arch=sm_{cc_major}{cc_minor}")
else:
    tilelang_target = "cuda"

pass_configs = {}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_WARP_SPECIALIZED"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_TMA_LOWER"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = True
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
FP8_MAX = 448.0
FALLBACK_BLOCK_SIZE = 128

# Ampere-class GPUs do not expose the FP8 e8m0 intrinsics used by the TileLang
# path in this repo. Keep the public API the same and fall back to an equivalent
# PyTorch implementation on those devices.
USE_TORCH_FP8_FALLBACK = (
    os.getenv("DEEPSEEK_FORCE_TORCH_FP8_FALLBACK", "0") == "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9
)


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


def _round_scale_pow2(x: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(x)
    rounded_exponent = exponent - mantissa.eq(0.5).to(exponent.dtype)
    return torch.ldexp(torch.ones_like(x), rounded_exponent)


def _pick_torch_fallback_dtype(override_name: str, default_dtype: torch.dtype) -> torch.dtype:
    override = os.getenv(override_name, os.getenv("DEEPSEEK_TORCH_FP8_FALLBACK_DTYPE", "")).lower()
    if override == "fp16":
        return torch.float16
    if override == "bf16":
        return torch.bfloat16
    return default_dtype


def _torch_fallback_gemm_dtype() -> torch.dtype:
    return _pick_torch_fallback_dtype("DEEPSEEK_TORCH_FP8_FALLBACK_GEMM_DTYPE", torch.float32)


def _torch_fallback_index_dtype() -> torch.dtype:
    return _pick_torch_fallback_dtype("DEEPSEEK_TORCH_FP8_FALLBACK_INDEX_DTYPE", torch.float32)


def _run_cached_weight_op(
    a_deq: torch.Tensor, b_deq: torch.Tensor, op: str, transposed_cache: bool = False
) -> torch.Tensor:
    if transposed_cache:
        if op == "flinear":
            return F.linear(a_deq, b_deq.t())
        if op == "mm":
            return a_deq @ b_deq
        if op == "matmul":
            return torch.matmul(a_deq, b_deq)
    else:
        if op == "flinear":
            return F.linear(a_deq, b_deq)
        if op == "mm":
            return a_deq @ b_deq.t()
        if op == "matmul":
            return torch.matmul(a_deq, b_deq.t())
    raise ValueError(f"unknown cached-weight op: {op}")


def _select_cached_weight_plan(target_hint: Optional[str], m: int) -> tuple[str, str]:
    if target_hint == "mla_wkv_b":
        # The exact fallback search has shown the transposed cached layout to be
        # the best fit for the MLA wkv_b projection. Keep the fast path simple
        # and route all MLA wkv_b cached-weight calls through the transposed mm
        # plan so the model can reuse the pre-transposed cache directly.
        return "t", "mm"
    if target_hint == "indexer_wq_b":
        if m == 1:
            return "row", "matmul"
        return "row", "flinear"
    if target_hint == "mla_wq_b":
        if m in {2, 32}:
            return "row", "matmul"
        if m >= 1024 or m == 4:
            return "row", "mm"
        return "row", "flinear"
    if target_hint == "indexer_wk":
        if m == 1:
            return "row", "matmul"
        if m in {4, 8, 32, 1024}:
            return "t", "mm"
        if m in {2, 16, 256}:
            return "row", "flinear"
        if m == 128:
            return "row", "mm"
        if m == 512:
            return "row", "matmul"
        return "row", "flinear"
    return "row", "flinear"


def _act_quant_torch(
    x: torch.Tensor, block_size: int = FALLBACK_BLOCK_SIZE, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.size(-1)
    x_blocks = x.view(-1, n // block_size, block_size).float()
    amax = x_blocks.abs().amax(dim=-1).clamp_min_(1e-4)
    scales = amax / FP8_MAX
    if scale_fmt is not None:
        scales = _round_scale_pow2(scales)
    y = torch.clamp(x_blocks / scales.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y.view_as(x), scales.view(*x.shape[:-1], n // block_size)


def _weight_dequant_torch(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int = FALLBACK_BLOCK_SIZE
) -> torch.Tensor:
    n, k = weight.shape
    if n % block_size == 0 and k % block_size == 0:
        n_blocks = n // block_size
        k_blocks = k // block_size
        return (
            weight.view(n_blocks, block_size, k_blocks, block_size).float()
            * scale.view(n_blocks, 1, k_blocks, 1).float()
        ).to(torch.get_default_dtype()).view(n, k)

    scale_full = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return (weight.float() * scale_full[:n, :k].float()).to(torch.get_default_dtype())


def _fp8_gemm_torch(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = FALLBACK_BLOCK_SIZE,
) -> torch.Tensor:
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    compute_dtype = _torch_fallback_gemm_dtype()
    a_deq = fp8_dequant_input(a, a_s, block_size).view(m, k).to(compute_dtype)
    b_deq = _weight_dequant_torch(b, b_s, block_size).to(compute_dtype)
    c = F.linear(a_deq, b_deq)
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp8_dequant_input(
    a: torch.Tensor, a_s: torch.Tensor, block_size: int = FALLBACK_BLOCK_SIZE
) -> torch.Tensor:
    k = a.size(-1)
    m = a.numel() // k
    return (
        a.view(m, k // block_size, block_size).float()
        * a_s.view(m, k // block_size, 1).float()
    ).reshape(*a.shape)


def fp8_gemm_cached_weight(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b_deq: torch.Tensor,
    block_size: int = FALLBACK_BLOCK_SIZE,
    target_hint: Optional[str] = None,
    b_deq_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k = a.size(-1)
    m = a.numel() // k
    n = b_deq.size(0)
    compute_dtype = _torch_fallback_gemm_dtype()
    a_deq = fp8_dequant_input(a, a_s, block_size).view(m, k).to(compute_dtype)
    layout, op = _select_cached_weight_plan(target_hint, m)
    use_transposed_cache = layout == "t" and b_deq_t is not None
    weight = b_deq_t if use_transposed_cache else b_deq
    weight = weight.to(compute_dtype)
    c = _run_cached_weight_op(a_deq, weight, op, transposed_cache=use_transposed_cache)
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


def _fp8_index_torch(
    q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor
) -> torch.Tensor:
    compute_dtype = _torch_fallback_index_dtype()
    if q_s.dim() == 4 and q_s.size(-1) == 1:
        q_s = q_s.squeeze(-1)
    if k_s.dim() == 3 and k_s.size(-1) == 1:
        k_s = k_s.squeeze(-1)
    q_deq = (q.float() * q_s.float().unsqueeze(-1)).to(compute_dtype).contiguous()
    k_deq = (k.float() * k_s.float().unsqueeze(-1)).to(compute_dtype).contiguous()
    k_t = k_deq.unsqueeze(1).transpose(-1, -2)
    logits = torch.matmul(q_deq, k_t).permute(0, 1, 3, 2)
    return logits.clamp_min_(0).sum(dim=-1, dtype=torch.float32)


@tilelang.jit(target=tilelang_target, pass_configs=pass_configs)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    if USE_TORCH_FP8_FALLBACK:
        return _act_quant_torch(x, block_size, scale_fmt)
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


@tilelang.jit(target=tilelang_target, pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):
    assert out_dtype in [BF16, "float32"]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FP32],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    if USE_TORCH_FP8_FALLBACK:
        return _fp8_gemm_torch(a, a_s, b, b_s)
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_kernel(N, K)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


@tilelang.jit(target=tilelang_target, out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    if USE_TORCH_FP8_FALLBACK:
        return _fp8_index_torch(q, q_s, k, k_s)
    return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)
