# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_mqa_logits,
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_torch,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerPrefillMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops

import vllm_hcu.platforms.envs as henvs 
from vllm_hcu.platforms.hcu import on_gfx938
from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_bf16_cache_triton, 
    cp_gather_indexer_k_bf16_cache_triton,
)
from lightop import op, gemmopt

logger = init_logger(__name__)


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


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
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
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn if not current_platform.is_rocm() or on_gfx938() else k.dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        return sparse_attn_indexer_fake(
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
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping[:attn_metadata.num_kv_actual_tokens]
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

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

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype if not current_platform.is_rocm() or on_gfx938() else k.dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]  
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

            if is_deep_gemm_supported():
                logits = fp8_mqa_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32).flatten()),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            else:
                mqa_logits_inner_chunked(chunk,
                                        q_fp8,
                                        k_fp8,
                                        weights,
                                        k_scale,
                                        topk_indices_buffer,
                                        topk_tokens)

    if has_decode:
        decode_metadata = attn_metadata.decode
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
        if is_deep_gemm_supported():
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            logits = gemmopt.paged_mqa_logits(
                padded_q_fp8_decode_tokens, 
                kv_cache, 
                weights[:num_padded_tokens] if on_gfx938() else weights[:num_padded_tokens].to(torch.float32), 
                decode_metadata.seq_lens, 
                decode_metadata.block_table, 
                decode_metadata.schedule_metadata, 
                max_model_len,
            )

        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if decode_metadata.use_large_context_topk:
            if next_n == 1:
                lengths = decode_metadata.seq_lens
            else:
                # (bs,) -> (bs, 1) + (next_n,) -> (bs, next_n) -> (bs * next_n,)
                lengths = (
                    decode_metadata.seq_lens.unsqueeze(1)
                    - next_n
                    + 1
                    + decode_metadata.offsets
                ).flatten()

            torch.ops._C.large_context_topk(
                logits,
                topk_indices,
                lengths,
                None,
            )
        else:
            top_k_per_row_decode_impl = op.top_k_per_row_decode \
                if henvs.VLLM_HCU_USE_LIGHTOP_TOPK \
                and henvs.VLLM_HCU_USE_CUSTOM_OPS \
                else torch.ops._C.top_k_per_row_decode
            top_k_per_row_decode_impl(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
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


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
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
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM is not supported or available. SparseAttnIndexer will use a "
                "less efficient PyTorch implementation. "
                "Please make sure you have the required hardware and software setup "
                "for DeepGEMM to achieve optimal performance."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            # raise RuntimeError(
            #     "Sparse attention indexer ROCm custom op requires ROCm "
            #     "Aiter ops to be enabled."
            # )
            return torch.ops.vllm.sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )