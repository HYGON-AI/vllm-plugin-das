# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
import vllm_hcu.platforms.envs as henvs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_attn_window,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_window,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn_window,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    query_start_loc: torch.Tensor | None = None  # [num_reqs + 1]


_DEEPSEEK_V4_CACHE_WINDOW_BLOCK_THRESHOLD = 30000
_DEEPSEEK_V4_CACHE_WINDOW_BYTE_LIMIT = 1 << 30


def _enable_deepseek_v4_cache_window(*num_blocks: int) -> bool:
    if (
        not henvs.VLLM_HCU_USE_CUSTOM_OPS
        or not henvs.VLLM_HCU_ENABLE_DEEPSEEK_V4_CACHE_WINDOW
    ):
        return False
    return any(n > _DEEPSEEK_V4_CACHE_WINDOW_BLOCK_THRESHOLD for n in num_blocks)


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            query_start_loc=common_attn_metadata.query_start_loc,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._fused_kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            self._fused_window_kernel = (
                _fused_kv_compress_norm_rope_insert_sparse_attn_window
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
            self._num_warps = 4
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._fused_kernel = (
                    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
                )
                self._fused_window_kernel = (
                    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_window
                )
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._fused_kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
                self._fused_window_kernel = (
                    _fused_kv_compress_norm_rope_insert_indexer_attn_window
                )
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
            self._num_warps = 1
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}

        # Fused C128A path: save current state, then only boundary tokens
        # perform compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # uses register values from this program; older tokens are read from
        # state_cache, avoiding same-grid read-after-write dependence.
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_ENABLE_DEEPSEEK_V4_C128_COMPRESSOR
            and self.compress_ratio == 128
            and self.head_dim == 512
            and token_to_req_indices is not None
            and state_metadata.query_start_loc is not None
        ):
            cos_sin_cache = rotary_emb.cos_sin_cache
            k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
            kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
            compressed_kv = torch.empty(
                (num_actual, self.head_dim),
                dtype=torch.float32,
                device=kv_score.device,
            )
            _save_and_compress_c128_split_kernel[(
                num_actual,
                triton.cdiv(self.head_dim, 64),
            )](
                kv,
                kv.stride(0),
                score,
                score.stride(0),
                self.ape,
                self.ape.stride(0),
                positions,
                state_cache,
                state_cache.stride(0),
                state_cache.stride(1),
                slot_mapping,
                token_to_req_indices,
                state_metadata.query_start_loc,
                block_table,
                block_table.stride(0),
                block_size,
                compressed_kv,
                compressed_kv.stride(0),
                HEAD_SIZE=self.head_dim,
                SPLIT_BLOCK_SIZE=64,
                STATE_WIDTH=state_width,
                **pdl_kwargs,
            )
            _norm_rope_fp8_store_c128_kernel[(num_actual,)](
                compressed_kv,
                compressed_kv.stride(0),
                positions,
                slot_mapping,
                self.norm.weight,
                self.rms_norm_eps,
                cos_sin_cache,
                cos_sin_cache.stride(0),
                kv_cache,
                k_cache_metadata.slot_mapping,
                kv_cache.shape[1],
                HEAD_SIZE=self.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
                ROPE_HEAD_DIM=self.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=self._quant_block,
                TOKEN_STRIDE=self._token_stride,
                SCALE_DIM=self._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                num_warps=self._num_warps,
                **pdl_kwargs,
            )
            return

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and _fused_kernel below
        # depend on preceding kernel outputs (kv/score from the cublas GEMM;
        # state_cache from this kernel) but neither emits/waits on PDL grid
        # dependency primitives, so launch_pdl=True caused a read-after-write
        # race and non-deterministic output.
        _save_partial_states_kernel[(num_actual,)](
            kv,
            kv.stride(0),
            score,
            score.stride(0),
            self.ape,
            self.ape.stride(0),
            positions,
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            slot_mapping,
            block_size,
            HEAD_SIZE=kv.shape[-1],
            TRITON_BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            **pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

        state_block_stride = state_cache.stride(0)
        kv_block_stride = kv_cache.stride(0)
        if _enable_deepseek_v4_cache_window(
            state_cache.shape[0],
            kv_cache.shape[0],
        ):
            state_window_blocks = max(
                1, _DEEPSEEK_V4_CACHE_WINDOW_BYTE_LIMIT // state_block_stride
            )
            kv_window_blocks = max(
                1, _DEEPSEEK_V4_CACHE_WINDOW_BYTE_LIMIT // kv_block_stride
            )
            state_read_margin = 1 + self.overlap

            for state_core_start in range(
                0,
                state_cache.shape[0],
                state_window_blocks,
            ):
                state_core_end = min(
                    state_core_start + state_window_blocks,
                    state_cache.shape[0],
                )
                state_window_start = max(0, state_core_start - state_read_margin)
                state_window_end = min(
                    state_core_end + state_read_margin,
                    state_cache.shape[0],
                )
                window_state_cache = state_cache[state_window_start:state_window_end]
                for kv_window_start in range(
                    0,
                    kv_cache.shape[0],
                    kv_window_blocks,
                ):
                    kv_window_end = min(
                        kv_window_start + kv_window_blocks,
                        kv_cache.shape[0],
                    )
                    window_kv_cache = kv_cache[kv_window_start:kv_window_end]
                    self._fused_window_kernel[(num_actual,)](
                        # state cache
                        window_state_cache,
                        state_block_stride,
                        state_cache.stride(1),
                        # metadata
                        token_to_req_indices,
                        positions,
                        slot_mapping,
                        block_table,
                        block_table.stride(0),
                        block_size,
                        # RMSNorm
                        self.norm.weight,
                        self.rms_norm_eps,
                        # RoPE
                        cos_sin_cache,
                        cos_sin_cache.stride(0),
                        # KV cache
                        window_kv_cache,
                        k_cache_metadata.slot_mapping,
                        kv_cache.shape[1],
                        # constexprs
                        HEAD_SIZE=self.head_dim,
                        TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
                        STATE_WIDTH=state_width,
                        COMPRESS_RATIO=self.compress_ratio,
                        OVERLAP=self.overlap,
                        ROPE_HEAD_DIM=self.rope_head_dim,
                        FP8_MAX=448.0,
                        QUANT_BLOCK=self._quant_block,
                        TOKEN_STRIDE=self._token_stride,
                        SCALE_DIM=self._scale_dim,
                        KV_BLOCK_STRIDE=kv_block_stride,
                        STATE_WINDOW_START_BLOCK=state_window_start,
                        STATE_WINDOW_BLOCKS=state_window_end - state_window_start,
                        STATE_CORE_START_BLOCK=state_core_start,
                        STATE_CORE_END_BLOCK=state_core_end,
                        KV_WINDOW_START_BLOCK=kv_window_start,
                        KV_WINDOW_BLOCKS=kv_window_end - kv_window_start,
                        num_warps=self._num_warps,
                        **pdl_kwargs,
                    )
            return

        self._fused_kernel[(num_actual,)](
            # state cache
            state_cache,
            state_block_stride,
            state_cache.stride(1),
            # metadata
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            block_size,
            # RMSNorm
            self.norm.weight,
            self.rms_norm_eps,
            # RoPE
            cos_sin_cache,
            cos_sin_cache.stride(0),
            # KV cache
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size (tokens per block)
            # constexprs
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            ROPE_HEAD_DIM=self.rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=self._quant_block,
            TOKEN_STRIDE=self._token_stride,
            SCALE_DIM=self._scale_dim,
            KV_BLOCK_STRIDE=kv_block_stride,
            num_warps=self._num_warps,
            **pdl_kwargs,
        )


@triton.jit
def _save_and_compress_c128_split_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    token_to_req_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    compressed_kv_ptr,
    compressed_kv_stride,
    HEAD_SIZE: tl.constexpr,
    SPLIT_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
):
    token_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    offs = split_idx * SPLIT_BLOCK_SIZE + tl.arange(0, SPLIT_BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    position = tl.load(positions_ptr + token_idx)

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = state_cache_ptr + block_idx * state_cache_stride0 + pos_in_block * state_cache_stride1

    cur_kv = tl.load(kv_ptr + token_idx * kv_stride + offs, mask=mask, other=0.0)
    tl.store(base_ptr + offs, cur_kv, mask=mask)
    ape_row = position % 128
    cur_ape = tl.load(ape_ptr + ape_row * ape_stride + offs, mask=mask, other=0.0)
    cur_score = tl.load(score_ptr + token_idx * score_stride + offs, mask=mask, other=0.0)
    cur_score = cur_score + cur_ape
    tl.store(base_ptr + STATE_WIDTH + offs, cur_score, mask=mask)

    if (position + 1) % 128 != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    req_start = tl.load(query_start_loc_ptr + req_idx)
    local_idx = token_idx - req_start
    prefix_pos = position - local_idx

    tokens = tl.arange(0, 128)
    pos = position - 127 + tokens
    mask_pos = pos >= 0
    candidate_token_idx = req_start + pos - prefix_pos
    in_current_chunk = (candidate_token_idx >= req_start) & (candidate_token_idx <= token_idx)
    safe_candidate_token_idx = tl.maximum(candidate_token_idx, 0)

    direct_mask = in_current_chunk[:, None] & mask[None, :]
    direct_kv = tl.load(
        kv_ptr + safe_candidate_token_idx[:, None] * kv_stride + offs[None, :],
        mask=direct_mask,
        other=0.0,
    )
    direct_score = tl.load(
        score_ptr + safe_candidate_token_idx[:, None] * score_stride + offs[None, :],
        mask=direct_mask,
        other=0.0,
    )
    direct_ape = tl.load(
        ape_ptr + (pos % 128)[:, None] * ape_stride + offs[None, :],
        mask=direct_mask,
        other=0.0,
    )
    direct_score = direct_score + direct_ape

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + block_offsets * state_cache_stride1
    )
    combined_mask = mask_pos[:, None] & mask[None, :]
    state_mask = combined_mask & (~in_current_chunk[:, None])

    state_score = tl.load(
        row_base[:, None] + STATE_WIDTH + offs[None, :],
        mask=state_mask,
        other=-3.4028234663852886e38,
    )
    state_kv = tl.load(
        row_base[:, None] + offs[None, :],
        mask=state_mask,
        other=0.0,
    )

    score = tl.where(direct_mask, direct_score, state_score)
    kv = tl.where(direct_mask, direct_kv, state_kv)
    score = tl.softmax(score, dim=0)
    compressed_kv = tl.sum(kv * score, axis=0)
    tl.store(compressed_kv_ptr + token_idx * compressed_kv_stride + offs, compressed_kv, mask=mask)


@triton.jit
def _norm_rope_fp8_store_c128_kernel(
    compressed_kv_ptr,
    compressed_kv_stride,
    positions_ptr,
    slot_mapping_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % 128 != 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    compressed_kv = tl.load(
        compressed_kv_ptr + token_idx * compressed_kv_stride + block,
        mask=mask,
        other=0.0,
    )

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = cache_block_ptr + kv_cache_block_size * TOKEN_STRIDE + kv_pos_in_block * SCALE_DIM

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)
    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))
    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(scale_ptr + scale_idx, encoded.to(tl.uint8), mask=scale_idx < N_NOPE_BLOCKS)
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // 128) * 128
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)
    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )
