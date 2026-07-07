# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import math
from importlib.util import find_spec

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942
else:
    _ON_GFX942 = False
    
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerPrefillMetadata
import vllm_hcu.platforms.envs as henvs 
from vllm_hcu.platforms.hcu import on_gfx938
import lightop
from lightop import op, gemmopt


_GLOBAL_LOGITS_BUFFERS = {}

# mqa_logits分块全局缓存大小，避免大输入打开pc时OOM
MAX_ELEMENTS = 16384 * 16384


def get_logits_buffer(device):
    global _GLOBAL_LOGITS_BUFFERS

    if device not in _GLOBAL_LOGITS_BUFFERS or _GLOBAL_LOGITS_BUFFERS[device].numel() < MAX_ELEMENTS:
        _GLOBAL_LOGITS_BUFFERS[device] = torch.empty(
            MAX_ELEMENTS,
            dtype=torch.float32,
            device=device
        )
    return _GLOBAL_LOGITS_BUFFERS[device]


def mqa_logits_inner_chunked(
        chunk: DeepseekV32IndexerPrefillMetadata,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_scale: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
        topk_tokens: int):
    """
    Chunked Impl of mqa_logits for avoiding oom When prefix cache is heavily hit.
    """
    q_all = q_fp8[chunk.token_start:chunk.token_end]
    weights_all = weights[chunk.token_start:chunk.token_end]
    ks_all = chunk.cu_seqlen_ks
    ke_all = chunk.cu_seqlen_ke
    
    num_q = q_all.shape[0]
    num_k = k_fp8.shape[0]

    is_q_fp16_bf16 = q_all.dtype in (torch.float16, torch.bfloat16)
    align_size = 128 if is_q_fp16_bf16 else 1
    
    kv_seq_len_aligned = (num_k + align_size - 1) // align_size * align_size

    logits_buffer = get_logits_buffer(q_fp8.device)
    current_capacity = logits_buffer.numel()
    max_q_chunk_num = current_capacity // max(1, kv_seq_len_aligned)
    if align_size > 1:
        max_q_chunk_num = (max_q_chunk_num // align_size) * align_size
    max_q_chunk_num = max(1, max_q_chunk_num)

    slices = []

    for start_idx in range(0, num_q, max_q_chunk_num):
        end_idx = min(start_idx + max_q_chunk_num, num_q)
        slices.append((start_idx, end_idx))

    for q_start, q_end in slices:
        if q_end <= q_start:
            continue
            
        q_slice = q_all[q_start:q_end]
        weights_slice = weights_all[q_start:q_end]

        ks_slice = ks_all[q_start:q_end]
        ke_slice = ke_all[q_start:q_end]

        q_len = q_end - q_start
        q_seq_len_aligned = (q_len + align_size - 1) // align_size * align_size

        required_size = q_seq_len_aligned * kv_seq_len_aligned
        logits_slice_view = logits_buffer[:required_size].view(q_seq_len_aligned, kv_seq_len_aligned)

        if not on_gfx938():
            weights_slice = weights_slice.to(torch.float32)

        chunk_k_scale = k_scale.view(torch.float32).flatten() if on_gfx938() else None

        op.mqa_logits(
            q_slice,  
            k_fp8, 
            weights_slice, 
            ks_slice, 
            ke_slice,
            q_slice.shape[0], # logical lengths
            k_fp8.shape[0],
            q_slice.shape[1],
            q_slice.shape[2],
            chunk_k_scale,
            True,
            logits_slice_view # padded properly out of box for hardware requirements
        )

        # Extract the exact logical valid window for downstream topk
        logits_slice = logits_slice_view[:q_len, :num_k]

        num_rows_slice = logits_slice.shape[0]
                
        topk_indices_slice = topk_indices_buffer[
            chunk.token_start + q_start : chunk.token_start + q_end, :topk_tokens
        ]
        
        top_k_per_row_prefill_impl = op.top_k_per_row_prefill if \
            henvs.VLLM_HCU_USE_LIGHTOP_TOPK and \
            henvs.VLLM_HCU_USE_CUSTOM_OPS \
            else torch.ops._C.top_k_per_row_prefill
        
        top_k_per_row_prefill_impl(
            logits_slice,
            ks_slice,
            ke_slice,
            topk_indices_slice,
            num_rows_slice,
            logits_slice.stride(0), # Automatically fetches kv_seq_len_aligned stride
            logits_slice.stride(1),
            topk_tokens,)


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "SHUFFLE",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            block_id * kv_cache_stride
            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "SHUFFLE",
        head_dim,
        block_tile_size,
        head_tile_size,
    )


@triton.jit
def _indexer_k_bf16_cache_kernel(
    k_ptr,  # [num_tokens, head_dim] (bf16)
    kv_cache_ptr,  # [n_blks, block_size, head_dim] (bf16)
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_stride,  # KV Cache 第一维的stride
    block_size: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    """
    Triton 核函数：将 BF16 类型的 K 张量写入 KV Cache（
    """
    tid = tl.program_id(0)
    
    # 边界检查：超出 token 范围直接返回
    if tid >= num_tokens:
        return
    
    # 定义头维度索引偏移（覆盖整个 head_dim）
    offset = tl.arange(0, head_dim)
    
    # 计算输入 K 张量的源指针偏移
    src_ptr = k_ptr + tid * head_dim
    
    # 加载当前 token 对应的 cache slot ID
    slot_id = tl.load(slot_mapping_ptr + tid)
    
    # 无效 slot（-1）直接返回
    if slot_id < 0:
        return
    
    # 计算 block ID 和块内偏移
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    
    # 分块相关的偏移计算（兼容 SHUFFLE 布局）
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    
    # 根据布局计算 KV Cache 的目标指针偏移
    if LAYOUT == "SHUFFLE":
        # SHUFFLE 布局的偏移计算
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        # NHD 标准布局
        tile_offset = offset
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_stride + block_offset * head_dim
        )
    
    val = tl.load(src_ptr + offset)
    
    tl.store(dst_ptr + tile_offset, val)

def indexer_k_bf16_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim] (bf16)
    slot_mapping: torch.Tensor,
    block_tile_size=16,
    head_tile_size=16,
):
    """
    将 BF16 类型的 K 张量写入 BF16 类型的 KV Cache
    Args:
        k: 输入 K 张量 [num_tokens, head_dim] (bf16)
        kv_cache: KV Cache 张量 [num_blocks, block_size, head_dim] (bf16)
        slot_mapping: token 到 cache slot 的映射 [num_tokens]
        block_tile_size: 块分块大小
        head_tile_size: 头维度分块大小
    """
    # 输入类型校验
    assert k.dtype == torch.bfloat16, "k 必须是 bf16 类型"
    assert kv_cache.dtype == torch.bfloat16, "kv_cache 必须是 bf16 类型"
    
    # 解析张量维度
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    
    # 验证维度合法性
    assert kv_cache.shape[2] == head_dim, "kv_cache 的 head_dim 必须与 k 一致"
    
    # 重塑 KV Cache 为二维（便于指针计算）
    kv_cache_2d = kv_cache.view(num_blocks, -1)  # [num_blocks, block_size * head_dim]
    
    # 调整 head_tile_size（兼容原逻辑，按字节数归一化）
    head_tile_size = head_tile_size // kv_cache.element_size()
    
    # 配置 Triton 核函数的 grid（每个 token 一个 program）
    grid = (num_tokens,)
    

    _indexer_k_bf16_cache_kernel[grid](
        k,
        kv_cache_2d,
        slot_mapping,
        kv_cache_2d.stride(0), 
        block_size,
        num_tokens,
        head_dim,
        "NHD",  # 布局类型
        block_tile_size,
        head_tile_size,
    )

@triton.jit
def _cp_gather_indexer_k_bf16_cache_kernel(
    kv_cache_ptr,  # [num_blocks, block_size * head_dim] (bf16)
    k_bf16_ptr,    # [num_tokens, head_dim] (bf16)
    block_table_ptr,
    cu_seq_lens_ptr,
    block_size: tl.constexpr,
    batch_size: tl.constexpr,
    num_blocks_per_seq: tl.constexpr,
    kv_cache_stride: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    """
    Triton 核函数 BF16 K Cache 收集
    """
    token_idx = tl.program_id(0)
    
    # 边界检查：超出 token 范围直接返回
    if token_idx >= num_tokens:
        return
    
    # 定义头维度索引偏移（覆盖整个 head_dim）
    head_offset = tl.arange(0, head_dim)
    
    batch_idx = tl.full((), -1, dtype=tl.int32)
    
    # 遍历所有 batch（Triton 支持有限循环，需固定循环次数）
    for b in tl.static_range(batch_size):
        # 加载当前 batch 的序列起始/结束位置
        seq_start = tl.load(cu_seq_lens_ptr + b)
        seq_end = tl.load(cu_seq_lens_ptr + b + 1)
        
        # 条件判断：当前 token 是否属于该 batch
        is_in_batch = (token_idx >= seq_start) & (token_idx < seq_end)
        # 条件赋值：如果属于该 batch，更新 batch_idx（替代 break）
        batch_idx = tl.where(is_in_batch, b, batch_idx)
    
    # 无效的 batch ID（token 不在任何序列中），直接返回
    if batch_idx == -1:
        return
    
    # --------------------------
    # 计算序列内偏移和 block 索引
    # --------------------------
    # token 在所属序列内的相对偏移
    seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    inbatch_seq_idx = token_idx - seq_start
    
    # 计算该 token 对应的 block 索引（block_table 中的位置）
    block_table_id = inbatch_seq_idx // block_size
    
    # 边界检查：block 索引超出范围则返回
    if block_table_id >= num_blocks_per_seq:
        return
    
    # 计算 block_table 中的内存偏移并加载 block ID
    block_table_offset = batch_idx * num_blocks_per_seq + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    
    # 计算 token 在 block 内的偏移
    block_offset = inbatch_seq_idx % block_size
    
    # --------------------------
    # 计算内存偏移
    # --------------------------
    # KV Cache 源偏移：block_id * 块步长 + 块内偏移 * head_dim
    src_block_offset = block_id * kv_cache_stride
    src_inblock_offset = src_block_offset + block_offset * head_dim
    
    # 输出张量目标偏移
    dst_inblock_offset = token_idx * head_dim

    src_ptr = kv_cache_ptr + src_inblock_offset + head_offset
    val = tl.load(src_ptr)
    
    dst_ptr = k_bf16_ptr + dst_inblock_offset + head_offset
    tl.store(dst_ptr, val)

def cp_gather_indexer_k_bf16_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim] (bf16)
    k_bf16: torch.Tensor,   # [num_tokens, head_dim] (bf16)
    block_table: torch.Tensor,  # [batch_size, num_blocks_per_seq]
    cu_seq_lens: torch.Tensor,  # [batch_size + 1]
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    """
    BF16 K Cache 收集算子
    Args:
        k_cache: K缓存张量 [num_blocks, block_size, head_dim] (bf16)
        k_bf16: 输出张量 [num_tokens, head_dim] (bf16)
        block_table: 块表 [batch_size, num_blocks_per_seq]
        cu_seq_lens: 序列长度累积数组 [batch_size + 1]
        block_tile_size: 块分块大小
        head_tile_size: 头维度分块大小
    """
    # 输入类型校验
    assert k_cache.dtype == torch.bfloat16, "k_cache 必须是 bf16 类型"
    assert k_bf16.dtype == torch.bfloat16, "k_bf16 必须是 bf16 类型"
    
    # 解析维度参数
    num_tokens = k_bf16.size(0)
    block_size = k_cache.size(1)
    head_dim = k_bf16.shape[-1]
    num_blocks = k_cache.shape[0]
    batch_size = block_table.size(0)
    num_blocks_per_seq = block_table.size(1)
    
    # 重塑缓存张量（便于指针计算）
    k_cache_2d = k_cache.view(num_blocks, -1)  # [num_blocks, block_size * head_dim]
    
    # 配置 Triton 核函数的 grid（每个 token 一个 program）
    grid = (num_tokens,)
    
    _cp_gather_indexer_k_bf16_cache_kernel[grid](
        k_cache_2d,
        k_bf16,
        block_table,
        cu_seq_lens,
        block_size,
        batch_size,
        num_blocks_per_seq,
        k_cache_2d.stride(0),  # kv_cache stride (block维度)
        head_dim,
        num_tokens,
        block_tile_size,
        head_tile_size,
    )
    
# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            scale_offset = block_size * dim
            cache_value = (
                cache[..., :scale_offset].view(dtype=fp8_dtype).to(torch.float32)
            )
            cache_scale = (
                cache[..., scale_offset:].view(dtype=torch.float32).contiguous()
            )
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.view(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n, heads, head_dim = q_fp8.shape
    num_blocks, block_size, _, _ = kv_cache_fp8.shape

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            out_logits = torch.full(
                [batch_size * next_n, max_model_len],
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size == 64,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    elif current_platform.is_rocm():
        return gemmopt.paged_mqa_logits(
            q_fp8, 
            kv_cache_fp8, 
            weights,
            context_lens, 
            block_tables,
            None, 
            max_model_len, 
            False,
        )
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        k_fp8, scale = kv
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    elif current_platform.is_rocm():
        k_fp8, scale = kv
        return lightop.mqa_logits(q, k_fp8, weights, cu_seqlen_ks, cu_seqlen_ke, scale)
        # mqa_logits_inner_chunked(
        #     chunk,
        #     q_fp8,
        #     k_fp8,
        #     weights,
        #     scale,
        #     topk_indices_buffer,
        #     topk_tokens,
        # )
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def _topk_indices_torch(logits: torch.Tensor, topk_tokens: int) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def _decode_row_ends_from_seq_lens(
    seq_lens: torch.Tensor,
    next_n: int,
    num_rows: int,
) -> torch.Tensor:
    """Match top_k_per_row_decode's effective row length calculation."""
    if seq_lens.dim() == 2:
        return torch.clamp(seq_lens.reshape(-1)[:num_rows], min=0)

    flat_seq_lens = seq_lens.reshape(-1)
    if flat_seq_lens.numel() * next_n == num_rows and next_n > 1:
        offsets = torch.arange(
            next_n, dtype=flat_seq_lens.dtype, device=flat_seq_lens.device
        )
        row_ends = flat_seq_lens.unsqueeze(1) - next_n + offsets + 1
        return torch.clamp(row_ends.reshape(-1)[:num_rows], min=0)
    return torch.clamp(flat_seq_lens[:num_rows], min=0)


def _use_lightop_sparse_mla_topk() -> bool:
    return (
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK
    )


def _lightop_topk_indices_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    num_rows = topk_indices.shape[0]
    if logits.dim() != 2:
        raise RuntimeError(f"Prefill topk expects 2D logits, got {logits.shape}")
    if logits.shape[0] != num_rows:
        if logits.shape[1] == num_rows:
            logits = logits.transpose(0, 1)
        else:
            raise RuntimeError(
                "Prefill topk logits/query row mismatch: "
                f"logits={tuple(logits.shape)}, "
                f"topk_indices={tuple(topk_indices.shape)}"
            )

    max_seq_len = logits.shape[1]
    logits = logits.contiguous()
    row_starts_i32 = (
        row_starts.to(device=logits.device, dtype=torch.int32)
        .reshape(-1)[:num_rows]
        .clamp(0, max_seq_len)
    )
    row_ends_i32 = (
        row_ends.to(device=logits.device, dtype=torch.int32)
        .reshape(-1)[:num_rows]
        .clamp(0, max_seq_len)
    )
    row_ends_i32 = torch.maximum(row_ends_i32, row_starts_i32)
    topk_out = (
        topk_indices
        if topk_indices.is_contiguous()
        else torch.empty(
            topk_indices.shape,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
    )
    op.top_k_per_row_prefill(
        logits,
        row_starts_i32,
        row_ends_i32,
        topk_out,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    if topk_out is not topk_indices:
        topk_indices.copy_(topk_out)


def _lightop_topk_indices_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    row_ends = _decode_row_ends_from_seq_lens(seq_lens, next_n, logits.shape[0])
    op.top_k_per_row_decode(
        logits,
        1,
        row_ends.to(device=logits.device, dtype=torch.int32),
        topk_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    device = hidden_states.device if k is None else k.device
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=device, dtype=torch.uint8
    )
    fp8_dtype = current_platform.fp8_dtype()
    _k_fp8 = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer_native(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype() if not current_platform.is_rocm() or on_gfx938() else k.dtype
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping[:layer_attn_metadata.num_kv_actual_tokens]
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens
    device = hidden_states.device if k is None else k.device

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        if not current_platform.is_rocm() or on_gfx938():
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
        else:
            indexer_k_bf16_cache_triton(
                k,
                kv_cache,
                slot_mapping,
            )
            # indexer_k_quant_and_cache_triton(
            #     k,
            #     kv_cache,
            #     slot_mapping,
            #     quant_block_size,
            #     scale_fmt,
            # )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=device,
                dtype=torch.uint8,
            )
            if not current_platform.is_rocm() or on_gfx938():
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                cp_gather_indexer_k_bf16_cache_triton(
                    kv_cache,
                    k_fp8,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
                # cp_gather_indexer_k_quant_cache_triton(
                #     kv_cache,
                #     k_fp8,
                #     k_scale,
                #     chunk.block_table,
                #     chunk.cu_seq_lens,
                #     token_to_seq=chunk.token_to_seq,
                # )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if _use_lightop_sparse_mla_topk():
                _lightop_topk_indices_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    topk_tokens,
                )
            else:
                topk_indices.copy_(_topk_indices_torch(logits, topk_tokens))

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        use_lightop_sparse_mla_topk = _use_lightop_sparse_mla_topk()
        seq_lens = (
            decode_metadata.seq_lens[:batch_size]
            if use_lightop_sparse_mla_topk
            else decode_metadata.seq_lens
        )

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        if use_lightop_sparse_mla_topk:
            topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
            _lightop_topk_indices_decode(
                logits, seq_lens, next_n, topk_indices, topk_tokens
            )
        else:
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            topk_indices.copy_(_topk_indices_torch(logits, topk_tokens)[:num_decode_tokens]
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return rocm_aiter_sparse_attn_indexer_native(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_fp8,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert=False,
    )


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _apply_gptj_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def _apply_inv_rope_ref(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if hasattr(rotary_emb, "forward_native"):
        try:
            query, _ = rotary_emb.forward_native(
                positions,
                x.clone(),
                None,
                inverse=True,
            )
            return query
        except TypeError:
            pass
    return _apply_gptj_inv_rope_ref(x, positions, rotary_emb.cos_sin_cache, rope_dim)


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Reference inverse-RoPE + WO_A einsum path used on ROCm."""
    o_ref = _apply_inv_rope_ref(rotary_emb, o, positions, rope_head_dim).to(
        torch.bfloat16
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    hidden_dim = o_ref.shape[-1]
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        wo_a_weight = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)


def rocm_ref_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    indices = indices.clone().squeeze(1)
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    if topk_length is not None:
        mask = torch.arange(topk, device=indices.device).unsqueeze(
            0
        ) >= topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)
    indices[invalid_mask] = 0

    qf = q.float()
    gathered_kv = kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
    scores = qf @ gathered_kv.transpose(1, 2)
    scores *= scale
    scores[invalid_mask.unsqueeze(1).expand_as(scores)] = float("-inf")

    orig_lse = torch.logsumexp(scores, dim=-1)
    lse_for_o = orig_lse
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack(
                [orig_lse, attn_sink[:h_q].view(1, h_q).expand_as(orig_lse)],
                dim=0,
            ),
            dim=0,
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
    out = probs @ gathered_kv[..., :head_dim]
    lonely_q_mask = orig_lse == float("-inf")
    out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16)


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    output_chunk = rocm_ref_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        scale=scale,
        head_dim=head_dim,
        attn_sink=attn_sink,
    )
    output.copy_(output_chunk.to(output.dtype))


def rocm_dequantize_blocked_k_cache(
    quant_k_cache: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    fp8_dtype = current_platform.fp8_dtype()
    tile_size = 64
    num_tiles = nope_head_dim // tile_size

    num_blocks, block_size, _ = quant_k_cache.shape
    quant_k_cache = quant_k_cache.view(num_blocks, -1)
    input_nope_rope = quant_k_cache[
        :, : block_size * (nope_head_dim + 2 * rope_head_dim)
    ].view(num_blocks, block_size, nope_head_dim + 2 * rope_head_dim)
    input_nope = input_nope_rope[:, :, :nope_head_dim].view(fp8_dtype)
    input_rope = input_nope_rope[:, :, nope_head_dim:].view(torch.bfloat16)
    input_scale = (
        quant_k_cache[:, block_size * (nope_head_dim + 2 * rope_head_dim) :]
        .view(num_blocks, block_size, 8)[:, :, :num_tiles]
        .view(torch.float8_e8m0fnu)
    )

    result = torch.empty(
        (num_blocks, block_size, 1, head_dim),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    result[..., nope_head_dim:] = input_rope.unsqueeze(2)
    for tile_idx in range(num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.bfloat16)
        cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        ).unsqueeze(2)
    return result


def rocm_ref_sparse_attn_decode(
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    extra_blocked_k: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    b, s_q, h_q, d_qk = q.shape

    def process_scope(
        cur_blocked_k: torch.Tensor,
        cur_indices: torch.Tensor,
        cur_topk_length: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cur_indices = cur_indices.reshape(b, s_q, -1)
        topk = cur_indices.size(-1)
        fixed_indices = torch.clamp_min(cur_indices, 0)
        gathered_kv = (
            cur_blocked_k.view(-1, d_qk)
            .index_select(0, fixed_indices.view(-1))
            .view(b, s_q, topk, d_qk)
        )
        invalid_mask = cur_indices == -1
        if cur_topk_length is not None:
            cur_topk_length = cur_topk_length.reshape(b)
            invalid_mask |= torch.arange(0, topk, device=invalid_mask.device).view(
                1, 1, topk
            ) >= cur_topk_length.view(b, 1, 1)
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_scope(
        blocked_k, indices_in_kvcache, topk_length
    )
    if extra_blocked_k is not None:
        assert extra_indices_in_kvcache is not None
        gathered_kv1, invalid_mask1 = process_scope(
            extra_blocked_k, extra_indices_in_kvcache, extra_topk_length
        )
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)

    gathered_kv = gathered_kv.view(b * s_q, -1, d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    qf = q.float().view(b * s_q, h_q, d_qk)
    attn_weight = qf @ gathered_kv.transpose(-1, -2)
    attn_weight *= scale
    attn_weight[
        invalid_mask.view(b * s_q, 1, -1).expand(b * s_q, h_q, invalid_mask.size(-1))
    ] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :head_dim]
    output = output.view(b, s_q, h_q, head_dim)
    lse = lse.view(b, s_q, h_q)

    if attn_sink is not None:
        output *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(
            -1
        )

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).expand_as(output)] = 0.0
    return output.squeeze(1).to(torch.bfloat16)


def rocm_forward_decode_fallback(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    blocked_swa = rocm_dequantize_blocked_k_cache(
        swa_k_cache,
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )
    blocked_extra = None
    if not swa_only:
        assert kv_cache is not None
        blocked_extra = rocm_dequantize_blocked_k_cache(
            kv_cache,
            head_dim=head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )
    attn_out = rocm_ref_sparse_attn_decode(
        q=q.unsqueeze(1),
        blocked_k=blocked_swa,
        indices_in_kvcache=swa_indices.unsqueeze(1),
        topk_length=swa_lens,
        scale=scale,
        head_dim=head_dim,
        attn_sink=attn_sink[: q.shape[1]] if attn_sink is not None else None,
        extra_blocked_k=blocked_extra,
        extra_indices_in_kvcache=topk_indices,
        extra_topk_length=topk_lens,
    )
    output.copy_(attn_out.to(output.dtype))