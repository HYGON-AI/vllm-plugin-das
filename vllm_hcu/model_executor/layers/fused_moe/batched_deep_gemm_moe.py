# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv, round_up

from vllm.utils.import_utils import has_deep_gemm
from vllm.model_executor.layers.activation import SiluAndMul
from lightop import fuse_silu_mul_quant_ep
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
if has_deep_gemm():
    from deepgemm import m_grouped_w8a8_gemm_nt_masked
else:
    from lightop import m_grouped_w8a8_gemm_nt_masked

logger = init_logger(__name__)


# ==============================================
# MOE Grouped GEMM Triton内核 (int8量化 + 专家并行)
# 输入布局：All2All后 -> [E, M, K] / [E, N, K]
# 输出：[E, M, N] 直接写入传入的output张量
# ==============================================
@triton.jit
def moe_grouped_gemm_kernel(
    # 指针
    A_ptr, B_ptr,
    A_scale_ptr, B_scale_ptr,
    token_counts_ptr,
    output_ptr,

    # 维度步长 (Batch/E维度步长, M/Token步长, N/Out通道步长, K/特征步长)
    stride_A_E, stride_A_M, stride_A_K,
    stride_B_E, stride_B_N, stride_B_K,
    stride_A_scale_E, stride_A_scale_M,
    stride_B_scale_E, stride_B_scale_N,
    stride_out_E, stride_out_M, stride_out_N,

    # 固定维度
    E: tl.constexpr,  # 专家总数
    M: tl.constexpr,  # 每个专家最大Token数
    N: tl.constexpr,  # 每个专家输出维度
    K: tl.constexpr,  # 输入特征维度

    # 分块参数 (T自动调优)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ===================== 1. 专家ID + 计算坐标 =====================
    # 程序ID对应：专家ID(E) + Token分块(M) + 输出分块(N)
    pid_e = tl.program_id(0)    # 专家维度 (0~E-1)
    pid_m = tl.program_id(1)    # Token分块维度
    pid_n = tl.program_id(2)    # 输出分块维度

    # 当前专家实际需要计算的Token数量
    token_cnt = tl.load(token_counts_ptr + pid_e)
    # 超出实际Token数直接退出 (动态Token数)
    if pid_m * BLOCK_M >= token_cnt:
        return

    # ===================== 2. 计算当前分块的内存偏移 =====================
    # 输入A [E, M, K]
    A_base = A_ptr + pid_e * stride_A_E
    # 权重B [E, N, K]
    B_base = B_ptr + pid_e * stride_B_E
    # Scale
    A_scale_base = A_scale_ptr + pid_e * stride_A_scale_E
    B_scale_base = B_scale_ptr + pid_e * stride_B_scale_E
    # 输出 [E, M, N]
    out_base = output_ptr + pid_e * stride_out_E

    # 分块坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 内存索引
    a_ptrs = A_base + (offs_m[:, None] * stride_A_M + offs_k[None, :] * stride_A_K)
    b_ptrs = B_base + (offs_n[:, None] * stride_B_N + offs_k[None, :] * stride_B_K)

    # ===================== 3. 初始化累加器 =====================
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ===================== 4. K维度循环计算GEMM (int8矩阵乘) =====================
    for k in range(0, K, BLOCK_K):
        # 加载int8数据 (保持int8精度)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        # 矩阵乘累加
        acc += tl.dot(a, tl.trans(b))  # B: [N,K] -> 转置为[K,N]
        # 指针步进
        a_ptrs += BLOCK_K * stride_A_K
        b_ptrs += BLOCK_K * stride_B_K

    # ===================== 5. int8反量化 (Per-Token + Per-Output Channel) =====================
    # 加载当前专家的scale
    a_scale = tl.load(A_scale_base + offs_m * stride_A_scale_M)  # [BLOCK_M]
    b_scale = tl.load(B_scale_base + offs_n * stride_B_scale_N)  # [BLOCK_N]
    # 反量化：out = (int8_mm) * A_scale * B_scale
    result = acc * a_scale[:, None] * b_scale[None, :]

    # ===================== 6. 写入输出 [E, M, N] =====================
    out_ptrs = out_base + (offs_m[:, None] * stride_out_M + offs_n[None, :] * stride_out_N)
    # 掩码：只写有效Token + 有效输出通道
    mask_m = offs_m < token_cnt
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, result, mask=mask)


# ==============================================
# 包装函数 (对外调用接口，自动处理步长/启动网格)
# ==============================================
def moe_grouped_gemm(
    A: torch.Tensor,        # [E, M, K]
    B: torch.Tensor,        # [E, N, K] int8
    A_scale: torch.Tensor,  # [E, M, 1]
    B_scale: torch.Tensor,  # [E, N, 1]
    token_counts: torch.Tensor,  # [E]
    output: torch.Tensor,   # [E, M, N] (传入，直接写入)
):
    # 维度校验
    E, M, K = A.shape
    _, N, _ = B.shape
    assert B.shape == (E, N, K)
    assert A_scale.shape == (E, M, 1)
    assert B_scale.shape == (E, N, 1)
    assert token_counts.shape == (E,)
    assert output.shape == (E, M, N)

    # 设备统一
    assert A.device == B.device == A_scale.device == B_scale.device == token_counts.device == output.device
    assert A.is_cuda

    # 自动分块大小 (适配主流GPU)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # 计算网格：[E, ceil(M/BLOCK_M), ceil(N/BLOCK_N)]
    grid = (
        E,
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    # 启动内核
    moe_grouped_gemm_kernel[grid](
        # 数据指针
        A, B,
        A_scale, B_scale,
        token_counts,
        output,

        # 步长 (按最后一维连续的张量自动计算)
        stride_A_E=A.stride(0), stride_A_M=A.stride(1), stride_A_K=A.stride(2),
        stride_B_E=B.stride(0), stride_B_N=B.stride(1), stride_B_K=B.stride(2),
        stride_A_scale_E=A_scale.stride(0), stride_A_scale_M=A_scale.stride(1),
        stride_B_scale_E=B_scale.stride(0), stride_B_scale_N=B_scale.stride(1),
        stride_out_E=output.stride(0), stride_out_M=output.stride(1), stride_out_N=output.stride(2),

        # 固定维度
        E=E, M=M, N=N, K=K,

        # 分块参数
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return output
# ==============================================
# MOE Grouped GEMM Triton内核 (int8量化 + 专家并行)
# 输入布局：All2All后 -> [E, M, K] / [E, N, K]
# 输出：[E, M, N] 直接写入传入的output张量
# ==============================================
@triton.jit
def moe_grouped_gemm_kernel(
    # 指针
    A_ptr, B_ptr,
    A_scale_ptr, B_scale_ptr,
    token_counts_ptr,
    output_ptr,

    # 维度步长 (Batch/E维度步长, M/Token步长, N/Out通道步长, K/特征步长)
    stride_A_E, stride_A_M, stride_A_K,
    stride_B_E, stride_B_N, stride_B_K,
    stride_A_scale_E, stride_A_scale_M,
    stride_B_scale_E, stride_B_scale_N,
    stride_out_E, stride_out_M, stride_out_N,

    # 固定维度
    E: tl.constexpr,  # 专家总数
    M: tl.constexpr,  # 每个专家最大Token数
    N: tl.constexpr,  # 每个专家输出维度
    K: tl.constexpr,  # 输入特征维度

    # 分块参数 (T自动调优)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ===================== 1. 专家ID + 计算坐标 =====================
    # 程序ID对应：专家ID(E) + Token分块(M) + 输出分块(N)
    pid_e = tl.program_id(0)    # 专家维度 (0~E-1)
    pid_m = tl.program_id(1)    # Token分块维度
    pid_n = tl.program_id(2)    # 输出分块维度

    # 当前专家实际需要计算的Token数量
    token_cnt = tl.load(token_counts_ptr + pid_e)
    # 超出实际Token数直接退出 (动态Token数)
    if pid_m * BLOCK_M >= token_cnt:
        return

    # ===================== 2. 计算当前分块的内存偏移 =====================
    # 输入A [E, M, K]
    A_base = A_ptr + pid_e * stride_A_E
    # 权重B [E, N, K]
    B_base = B_ptr + pid_e * stride_B_E
    # Scale
    A_scale_base = A_scale_ptr + pid_e * stride_A_scale_E
    B_scale_base = B_scale_ptr + pid_e * stride_B_scale_E
    # 输出 [E, M, N]
    out_base = output_ptr + pid_e * stride_out_E

    # 分块坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 内存索引
    a_ptrs = A_base + (offs_m[:, None] * stride_A_M + offs_k[None, :] * stride_A_K)
    b_ptrs = B_base + (offs_n[:, None] * stride_B_N + offs_k[None, :] * stride_B_K)

    # ===================== 3. 初始化累加器 =====================
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ===================== 4. K维度循环计算GEMM (int8矩阵乘) =====================
    for k in range(0, K, BLOCK_K):
        # 加载int8数据 (保持int8精度)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        # 矩阵乘累加
        acc += tl.dot(a, tl.trans(b))  # B: [N,K] -> 转置为[K,N]
        # 指针步进
        a_ptrs += BLOCK_K * stride_A_K
        b_ptrs += BLOCK_K * stride_B_K

    # ===================== 5. int8反量化 (Per-Token + Per-Output Channel) =====================
    # 加载当前专家的scale
    a_scale = tl.load(A_scale_base + offs_m * stride_A_scale_M)  # [BLOCK_M]
    b_scale = tl.load(B_scale_base + offs_n * stride_B_scale_N)  # [BLOCK_N]
    # 反量化：out = (int8_mm) * A_scale * B_scale
    result = acc * a_scale[:, None] * b_scale[None, :]

    # ===================== 6. 写入输出 [E, M, N] =====================
    out_ptrs = out_base + (offs_m[:, None] * stride_out_M + offs_n[None, :] * stride_out_N)
    # 掩码：只写有效Token + 有效输出通道
    mask_m = offs_m < token_cnt
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, result, mask=mask)


# ==============================================
# 包装函数 (对外调用接口，自动处理步长/启动网格)
# ==============================================
def moe_grouped_gemm(
    A: torch.Tensor,        # [E, M, K]
    B: torch.Tensor,        # [E, N, K] int8
    A_scale: torch.Tensor,  # [E, M, 1]
    B_scale: torch.Tensor,  # [E, N, 1]
    token_counts: torch.Tensor,  # [E]
    output: torch.Tensor,   # [E, M, N] (传入，直接写入)
):
    # 维度校验
    E, M, K = A.shape
    _, N, _ = B.shape
    assert B.shape == (E, N, K)
    assert A_scale.shape == (E, M, 1)
    assert B_scale.shape == (E, N, 1)
    assert token_counts.shape == (E,)
    assert output.shape == (E, M, N)

    # 设备统一
    assert A.device == B.device == A_scale.device == B_scale.device == token_counts.device == output.device
    assert A.is_cuda

    # 自动分块大小 (适配主流GPU)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # 计算网格：[E, ceil(M/BLOCK_M), ceil(N/BLOCK_N)]
    grid = (
        E,
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    # 启动内核
    moe_grouped_gemm_kernel[grid](
        # 数据指针
        A, B,
        A_scale, B_scale,
        token_counts,
        output,

        # 步长 (按最后一维连续的张量自动计算)
        stride_A_E=A.stride(0), stride_A_M=A.stride(1), stride_A_K=A.stride(2),
        stride_B_E=B.stride(0), stride_B_N=B.stride(1), stride_B_K=B.stride(2),
        stride_A_scale_E=A_scale.stride(0), stride_A_scale_M=A_scale.stride(1),
        stride_B_scale_E=B_scale.stride(0), stride_B_scale_N=B_scale.stride(1),
        stride_out_E=output.stride(0), stride_out_M=output.stride(1), stride_out_N=output.stride(2),

        # 固定维度
        E=E, M=M, N=N, K=K,

        # 分块参数
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return output

def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    shape = (E, T, G)
    strides = (T * G, 1, T)
    if quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]:
        return shape, strides, torch.float32

    assert quant_scale_fmt == DeepGemmQuantScaleFMT.UE8M0
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    We launch a fixed grid of threads to accommodate CUDA graphs. Let `P2`
    be a parallelization factor for persistent_masked_m_silu_mul_quant over the
    hidden dimension.

    Let `expert_offsets = [0] + [num_tokens.cumsum()]` and
    `total_tokens = expert_offsets[-1]`.
    persistent_masked_m_silu_mul_quant launches `total_tokens x P2` number of
    thread blocks. Each thread block contains `NUM_WARPS` warps.

    Every thread block needs to find it's corresponding expert by warp-parallel scanning
    over the `expert_offsets` array.

    The i-th warp in the first thread block processes
    `[i * warp_chunk_size, (i + 1) * warp_chunk_size]` groups
    sequentially, where `warp_chunk_size = ((H / GROUP_SIZE) / P2) / NUM_WARPS`,
    pipelining loads and computes.

    The shared memory layout for 4 warps with a 2-stage pipeline for SiLU V2
    can is visualized like so:

                         stage0                              stage1
    ┌─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┐
    │gate0│up0│gate1│up1│gate2│up2│gate3│up3│gate0│up0│gate1│up1│gate2│up2│gate3│up3│
    └─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┘

    with the main difference between V1 and V2 being the global load
    stride between warps, and between half-warps. Regarding the latter stride,
    we assign the first half warp of every warp for `gate` loads and the second
    half-warp to `up` loads.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s` depends on quant_scale_fmt,
      - quant_scale_fmt == FLOAT32,
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0,
         `y_s`: Int32 tensor, shape (E, T, H // group_size // 4), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0_FLOAT32_SPARSE
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    Let NUM_WARPS be the number of warps in a single thread block and
    `GROUP_SIZE = 128` be the size of the quantization group.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    cuda_arch = current_platform.get_device_capability(
        device_id=y.device.index
    ).to_int()

    if cuda_arch >= 80:
        torch.ops._C.persistent_masked_m_silu_mul_quant(
            y, tokens_per_expert, y_q, y_s, ceil_ue8m0
        )
    else:
        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        f_info = torch.finfo(fp8_dtype)
        fp8_max = f_info.max
        fp8_min = f_info.min
        eps: float = 1e-10
        assert y_s.dtype == torch.float32, (
            "_silu_mul_fp8_quant_deep_gemm does"
            "not support {y_s.dtype} scales. Only torch.float32 supported."
        )
        _silu_mul_fp8_quant_deep_gemm[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            ys_strides[0],
            ys_strides[1],
            ys_strides[2],
            stride_cnt_e,
            eps,
            fp8_min,
            fp8_max,
            ceil_ue8m0,
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


class BatchedDeepGemmExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        N: int = -1,
        K: int = -1,
    ):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        num_dispatchers: The number of DP dispatchers.
        quant_config: Quantization configuration
        """
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        if quant_config.use_fp8_w8a8:
            assert self.block_shape == get_mk_alignment_for_contiguous_layout()

        self.N = N
        self.K = K
        self.act_fn = SiluAndMul()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [(kFp8Static128BlockSym, kFp8Dynamic128Sym)]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        DeepGemm supports packed ue8m0 activation scales format in devices == sm100
        """
        return (
            is_deep_gemm_e8m0_used()
            and current_platform.is_device_capability_family(100)
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
                scope="local",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        # clamp estimate
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
        use_nn_moe: bool | None = False,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.block_shape is not None

        a1q = hidden_states
        _, N, K = w1.size()

        #assert w2.size(1) == K

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if self.N > 0:
            N = self.N
            
        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )
        #expected_m = self.get_expected_m()

        if self.quant_config.use_fp8_w8a16 or self.quant_config.use_fp8_w8a8:
            fp8_m_grouped_gemm_nt_masked(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                workspace1,
                expert_num_tokens,
                expected_m,
            )

            quant_scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
            a2q, a2q_scale = persistent_masked_m_silu_mul_quant(
                workspace1,
                expert_num_tokens,
                quant_scale_fmt=quant_scale_fmt,
            )

            fp8_m_grouped_gemm_nt_masked(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                output,
                expert_num_tokens,
                expected_m,
            )
        elif self.quant_config.use_int8_w8a8:
            m_grouped_w8a8_gemm_nt_masked((a1q, a1q_scale), 
                                  (w1, self.w1_scale),
                                    workspace1,
                                    expert_num_tokens, 
                                    expected_m,
                                    )

            assert expert_num_tokens is not None

            a2q, a2q_scale = fuse_silu_mul_quant_ep(workspace1, expert_num_tokens)
            m_grouped_w8a8_gemm_nt_masked((a2q, a2q_scale),
                                          (w2, self.w2_scale),
                                          output,
                                          expert_num_tokens,
                                          expected_m)
                        
            # moe_grouped_gemm(a1q, w1, a1q_scale, self.w1_scale, expert_num_tokens, workspace1)
            # act_out = self.act_fn(workspace1)
            # a2q, a2q_scale = per_token_quant_int8(act_out)
            # moe_grouped_gemm(a2q, w2, a2q_scale, self.w2_scale, expert_num_tokens, output)
            
        else:
            raise ValueError(f"Unsupported dtype {self.quant_config.quant_dtype}")

