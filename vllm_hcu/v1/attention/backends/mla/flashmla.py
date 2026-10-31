# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm_hcu.model_executor.layers.attention.lightop_concat_runtime import (
    concat_mla_decode,
)
from vllm_hcu.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_fp8,
    flash_mla_with_kvcache_fp8_with_cat,
    get_mla_metadata,
    get_mla_metadata_dense_fp8,
    is_flashmla_dense_supported,
)
import vllm_hcu.platforms.envs as henvs
from vllm_hcu.platforms.hcu import on_gfx938

logger = init_logger(__name__)

_FP8_E4M3 = torch.float8_e4m3fn
# The vendor fp8 kernel consumes 64-token pages (kBlockN). Manager blocks may
# be any whole multiple; non-causal groups expand via the registry buffer.
_NC_PAGE_BLOCK = 64

# TASK2: process-lifetime storage for the non-causal expanded block table.
# Same FIX13 contract as the Triton fast-path registry it replaces: rebuilds
# re-bind the SAME tensor so captured graphs never see a moved base address.
_NC_BT_REGISTRY: dict = {}


def _is_flash_nc_kv_dtype(kv_cache_dtype) -> bool:
    """The nc kernel consumes e4m3-encoded bytes only.

    Mirrors the Triton fast-path dtype guard: e5m2 and nvfp4 keep the
    stock path; the platform's internal fp8_ds_mla remap is byte-
    compatible e4m3 storage and stays on the nc path.
    """
    return (
        isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_e5m2"
        and not kv_cache_dtype.endswith("per_token_head")
        and kv_cache_dtype != "nvfp4"
    )



class HcuFlashMLABackend(MLACommonBackend):
    # TASK2: flattened non-causal DSpark draft groups (uniform query_len>1)
    # are served on the vendor fp8 kernel through registry-stable expanded
    # block tables -- the same FIX13 buffer contract as the Triton fast path
    # this backend replaces as the managed dense-MLA selection winner.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return FlashMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @classmethod
    def supports_non_causal(cls) -> bool:
        # Non-causal multi-token decode rides the vendor kernel with
        # causal=False over the committed-prefix expanded block table.
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if use_sparse:
            from vllm_hcu.v1.attention.ops.flashmla import is_flashmla_sparse_supported

            return is_flashmla_sparse_supported()[1]
        else:
            from vllm_hcu.v1.attention.ops.flashmla import is_flashmla_dense_supported

            return is_flashmla_dense_supported()[1]


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    scheduler_metadata: FlashMLASchedMeta


@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    pass


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # TASK2: the builder (not only the backend) must own this ClassVar --
    # the MLACommon build() flatten gate reads it off the builder class,
    # mirroring upstream TritonMLAMetadataBuilder.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # ^ TODO(matt): tune this

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None
        self.is_fp8_kvcache = is_quantized_kv_cache(
            vllm_config.cache_config.cache_dtype
        )

        num_sms = num_compute_units(self.device.index)

        # TASK2: non-causal (DSpark draft) groups expand manager blocks to
        # 64-token kernel pages through registry-persistent buffers; every
        # rebuild re-binds the same storage (FIX13 contract).
        self._nc_ready = False
        spec_nc = getattr(kv_cache_spec, "non_causal_multi_token_decode", False)
        page = getattr(kv_cache_spec, "block_size", None)
        subpage = (
            page // _NC_PAGE_BLOCK
            if isinstance(page, int) and page > 0 and page % _NC_PAGE_BLOCK == 0
            else None
        )
        if spec_nc and subpage and is_quantized_kv_cache(
            vllm_config.cache_config.cache_dtype
        ):
            nc_max_seqs = vllm_config.scheduler_config.max_num_seqs + 1
            nc_cols = -(-vllm_config.model_config.max_model_len // page) + 1
            nc_key = (
                str(device), tuple(sorted(layer_names)), page,
                nc_max_seqs, nc_cols * subpage,
            )
            reg = _NC_BT_REGISTRY.get(nc_key)
            if reg is None:
                reg = {
                    "bt": torch.zeros(
                        (nc_max_seqs, nc_cols * subpage),
                        dtype=torch.int32, device=device,
                    ),
                    "arange": torch.arange(
                        subpage, dtype=torch.int32, device=device
                    ).view(1, 1, subpage),
                    "stamps": {},
                }
                _NC_BT_REGISTRY[nc_key] = reg
            self._nc_bt_buf = reg["bt"]
            self._nc_bt_arange = reg["arange"]
            self._nc_stamps = reg["stamps"]
            self._nc_bt_cols = nc_cols
            self._nc_subpage = subpage
            self._nc_ready = True

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.cg_buf_tile_scheduler_metadata = torch.zeros(
                # Fixed 4x upper bound: HCU dense-FP8 metadata rows can
                # exceed #SMs (they grow with num_q_tokens_per_head_k, e.g.
                # 128 for 7x16 on a 64-CU part), and growing later would
                # free storage that captured graphs still point at.
                # TileSchedulerMetaDataSize = 8.
                (num_sms * 4, 8),
                device=self.device,
                dtype=torch.int32,
            )
            self.cg_buf_num_splits = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:
        # TASK2: non-causal DSpark draft groups flatten to request-major
        # decode rows with a uniform query_len>1. The vendor fp8 kernel
        # walks 64-token pages, so the manager block table is expanded
        # into a registry-persistent buffer (FIX13 contract: rebuilds
        # rebind the SAME storage; captured graphs keep stable
        # addresses) with an unconditional in-pool clamp.
        if getattr(self, "_nc_ready", False) and (
            self.non_causal_multi_token_decode
        ):
            return self._build_nc_decode(
                block_table_tensor,
                seq_lens_device,
                max_seq_len,
                query_start_loc_cpu,
                query_start_loc_device,
                num_decode_tokens,
                dcp_tot_seq_lens_device,
            )
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # we use the max but all should be the same due to uniform length requirement
        max_query_len = query_lens_cpu.max().item()
        num_q_tokens_per_head_k = max_query_len * self.num_q_heads // 1
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,  # MQA for the decode path
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        if self.is_fp8_kvcache:
            tile_scheduler_metadata, num_splits = get_mla_metadata_dense_fp8(
                seq_lens_device,
                num_q_tokens_per_head_k,
                1,  # MQA for the decode path
            )
            # HCU dense-FP8 metadata can require more rows than the initial
            # one-row-per-CU graph buffer. NEVER grow after startup: a
            # reallocated buffer frees the old storage while graphs already
            # captured around the flash kernel still hold its data_ptr
            # (use-after-free at replay). Size to a fixed upper bound
            # (4x num_sms like the triton-MLA builder) and raise instead
            # if a step ever demands more.
            if self.cg_buf_tile_scheduler_metadata is not None:
                assert tile_scheduler_metadata.shape[0] <= (
                    self.cg_buf_tile_scheduler_metadata.shape[0]
                ), (
                    "hcu flash tile metadata rows "
                    f"{tile_scheduler_metadata.shape[0]} exceed graph buffer"
                )
            if self.cg_buf_num_splits is not None:
                assert num_splits.shape[0] <= self.cg_buf_num_splits.shape[0], (
                    f"hcu flash nsplits rows {num_splits.shape[0]} exceed "
                    "graph buffer"
                )

            # Copy FP8 metadata into persistent CUDA graph buffers
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                assert self.cg_buf_tile_scheduler_metadata is not None
                assert self.cg_buf_num_splits is not None
                n = tile_scheduler_metadata.size(0)
                assert n <= self.cg_buf_tile_scheduler_metadata.size(0)
                self.cg_buf_tile_scheduler_metadata[:n].copy_(tile_scheduler_metadata)
                tile_scheduler_metadata = self.cg_buf_tile_scheduler_metadata[:n]

                n = num_splits.size(0)
                assert n <= self.cg_buf_num_splits.size(0)
                self.cg_buf_num_splits[:n].copy_(num_splits)
                num_splits = self.cg_buf_num_splits[:n]

            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata
            scheduler_metadata.num_splits = num_splits

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            scheduler_metadata=scheduler_metadata,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


    def _fill_nc_bt(
        self,
        rows: int,
        cols: int,
        demand_cols: int,
        pool_pages: int,
        block_table_tensor: torch.Tensor,
    ) -> None:
        # Expand manager blocks into kernel pages inside the persistent
        # buffer. Every check happens BEFORE any write; the write section
        # is no-throw so a captured graph can never observe a half-full
        # table. Out-of-pool sentinel ids that vLLM pads past a short
        # row's allocation are clamped the same way the Triton fast path
        # clamps them (both bounds).
        bt3 = self._nc_bt_buf[:rows].view(rows, self._nc_bt_cols, self._nc_subpage)
        bt3[:, :demand_cols].copy_(
            block_table_tensor[:, :demand_cols].unsqueeze(-1) * self._nc_subpage
        )
        bt3[:, :demand_cols].add_(self._nc_bt_arange)
        bt3[:, :demand_cols].clamp_(min=0, max=pool_pages - 1)

    def _build_nc_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:
        num_decodes = int(seq_lens_device.shape[0])
        if num_decodes <= 0 or num_decode_tokens % num_decodes != 0:
            raise RuntimeError(
                f"nc scheduler shape mismatch: {num_decode_tokens} tokens "
                f"over {num_decodes} decode rows"
            )
        query_len = num_decode_tokens // num_decodes
        subpage = self._nc_subpage
        rows, cols = block_table_tensor.shape
        if rows > self._nc_bt_buf.shape[0]:
            raise RuntimeError(
                f"nc block table rows {rows} exceed flash buffer "
                f"{self._nc_bt_buf.shape[0]}"
            )
        demand_cols = (
            -(-max_seq_len // _NC_PAGE_BLOCK) + subpage - 1
        ) // subpage
        if demand_cols > self._nc_bt_cols:
            raise RuntimeError(
                f"nc walk window {demand_cols} exceeds buffer width "
                f"{self._nc_bt_cols}"
            )
        if demand_cols > cols:
            raise RuntimeError(
                f"nc walk window {demand_cols} exceeds table width {cols}"
            )

        num_q_tokens_per_head_k = query_len * self.num_q_heads
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,  # MQA for the decode path
            is_fp8_kvcache=True,
        )
        tile, nsplits = get_mla_metadata_dense_fp8(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,
        )
        n = int(tile.shape[0])
        m = int(nsplits.shape[0])
        if m != num_decodes + 1:
            raise RuntimeError(
                f"nc plan mismatch: nsplits rows {m} != decodes+1 "
                f"{num_decodes + 1}"
            )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            assert self.cg_buf_tile_scheduler_metadata is not None
            assert self.cg_buf_num_splits is not None
            if n > self.cg_buf_tile_scheduler_metadata.shape[0]:
                raise RuntimeError(
                    f"nc tile rows {n} exceed graph buffer "
                    f"{self.cg_buf_tile_scheduler_metadata.shape[0]}"
                )
            if m > self.cg_buf_num_splits.shape[0]:
                raise RuntimeError(
                    f"nc nsplits rows {m} exceed graph buffer "
                    f"{self.cg_buf_num_splits.shape[0]}"
                )
            # fix13 plan/graph stamping: per captured shape, the row
            # counts are pinned and every later identical-shape fill must
            # reproduce them so replay keeps captured launch geometry.
            shape_key = (num_decodes, query_len)
            stamp = self._nc_stamps
            if shape_key not in stamp:
                stamp[shape_key] = (n, m)
            elif (n, m) != stamp[shape_key]:
                raise RuntimeError(
                    f"nc plan mismatch: tile {n}/nsplits {m} != captured "
                    f"{stamp[shape_key]} for shape {shape_key}"
                )
            self.cg_buf_tile_scheduler_metadata[:n].copy_(tile)
            self.cg_buf_num_splits[:m].copy_(nsplits)
            tile_view = self.cg_buf_tile_scheduler_metadata[:n]
            ns_view = self.cg_buf_num_splits[:m]
        else:
            tile_view = tile
            ns_view = nsplits

        # Live pool bound: num_gpu_blocks is assigned after profiling, so
        # read it per fill and refuse without a pool (an unchecked clamp
        # would let out-of-pool sentinels reach the kernel).
        # The kernel indexes the KV cache in 64-token kernel pages, while
        # cache_config.num_gpu_blocks counts MANAGER blocks (e.g. 1536
        # tokens).  One manager block spans manager/kernel pages, so the
        # legal kernel-page bound is num_gpu_blocks * ratio.  Using
        # num_gpu_blocks * subpage clamps real page ids down to the manager
        # count, which made the NC attention read empty pages and emit a
        # constant junk token.
        _mgr_block = int(
            getattr(self.vllm_config.cache_config, "block_size", 0) or 0
        )
        _pages_per_block = (
            _mgr_block // _NC_PAGE_BLOCK
            if _mgr_block >= _NC_PAGE_BLOCK
            and _mgr_block % _NC_PAGE_BLOCK == 0
            else 1
        )
        pool_pages = int(
            getattr(self.vllm_config.cache_config, "num_gpu_blocks", 0) or 0
        ) * _pages_per_block
        if pool_pages <= 0:
            raise RuntimeError(
                "nc flash fill refused: KV pool size not yet assigned"
            )
        self._fill_nc_bt(
            rows, cols, demand_cols, pool_pages, block_table_tensor
        )
        scheduler_metadata.tile_scheduler_metadata = tile_view
        scheduler_metadata.num_splits = ns_view
        return FlashMLADecodeMetadata(
            # Exchange the stock (manager-block) table for the expanded
            # page-granular view; the impl forwards it to the kernel
            # verbatim. Same data_ptr every step within a shape.
            block_table=self._nc_bt_buf[:rows],
            seq_lens=seq_lens_device,
            scheduler_metadata=scheduler_metadata,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    supports_pcp: bool = True
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        # The gfx938 CAT kernel consumes split query components; the regular
        # gfx938 FP8 path consumes vLLM's pre-quantized concatenated query.
        self.supports_quant_query_input = (
            on_gfx938() and not henvs.VLLM_HCU_USE_CAT_MLA
        )

        is_supported, reason = is_flashmla_dense_supported()
        assert is_supported, reason

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashMLAImpl"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: (zyongye) decode function for mla here
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        # TASK2: non-causal multi-token (DSpark draft) decode on the
        # vendor fp8 kernel. The builder has already expanded the block
        # table into registry-stable pages and pinned tile/plan buffers;
        # this path just quantizes the query and launches.
        if (
            not attn_metadata.causal
            and attn_metadata.decode is not None
            and _is_flash_nc_kv_dtype(self.kv_cache_dtype)
        ):
            num_decodes = attn_metadata.num_decodes
            query_len = attn_metadata.num_decode_tokens // num_decodes
            if query_len > 1:
                return self._forward_nc_fp8(
                    q, kv_c_and_k_pe_cache, attn_metadata, layer,
                    num_decodes, query_len,
                )

        # if type(q) is tuple:
        #     q = torch.cat(q, dim=-1)
        
        if isinstance(q, tuple):
            q_nope, q_pe = q
            if henvs.VLLM_HCU_USE_CAT_MLA and  on_gfx938() and (self.kv_cache_dtype == "fp8_e4m3" or self.kv_cache_dtype == "fp8_ds_mla"):
                assert isinstance(q_nope, torch.Tensor)
                assert isinstance(q_pe, torch.Tensor)
                
                num_decodes = attn_metadata.num_decodes
                q_nope = reshape_query_for_spec_decode(q_nope, num_decodes)
                q_pe = reshape_query_for_spec_decode(q_pe, num_decodes)
                scheduler_metadata = attn_metadata.decode.scheduler_metadata
                
                assert q_nope.shape[0] == num_decodes  
                o, lse = flash_mla_with_kvcache_fp8_with_cat(
                    q_nope=q_nope,
                    q_pe=q_pe,
                    k_cache=kv_c_and_k_pe_cache.unsqueeze(-2).view(torch.float8_e4m3fn),  # Add head dim of 1
                    block_table=attn_metadata.decode.block_table,
                    cache_seqlens=attn_metadata.decode.seq_lens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=scheduler_metadata.tile_scheduler_metadata,
                    num_splits=scheduler_metadata.num_splits,
                    softmax_scale=self.scale,
                    causal=True,
                    descale_q=layer._q_scale.reshape(1),
                    descale_k=layer._k_scale.reshape(1),          
                )                
                o = reshape_attn_output_for_spec_decode(o)
                return o, lse 
            q = concat_mla_decode(q_nope, q_pe, dim=-1)
        
        # mypy assertion: q is now always a tensor
        assert isinstance(q, torch.Tensor)

        num_decodes = attn_metadata.num_decodes
        q = reshape_query_for_spec_decode(q, num_decodes)

        scheduler_metadata = attn_metadata.decode.scheduler_metadata
        if envs.VLLM_BATCH_INVARIANT and not is_quantized_kv_cache(self.kv_cache_dtype):
            device = q.device
            dtype = torch.int32

            B = q.shape[0]
            # block_table shape: [batch_size, max_num_blocks_per_seq]
            # The number of blocks per sequence is in the second dimension
            topk = attn_metadata.decode.block_table.shape[-1]
            B_TOPK = 64
            assert topk % B_TOPK == 0, f"topk ({topk}) must be divisible by {B_TOPK}"
            end_block_idx = topk // B_TOPK

            # Single partition => num_sm_parts = 1
            # TileSchedulerMetaDataSize = 8, layout:
            # [begin_idx, begin_block_idx, end_idx, end_block_idx,
            #  begin_n_split_idx, _, _, _]
            tile_scheduler_metadata = torch.zeros((1, 8), dtype=dtype, device=device)
            tile_scheduler_metadata[0, 0] = 0  # begin_idx
            tile_scheduler_metadata[0, 1] = 0  # sched_begin_block_idx
            tile_scheduler_metadata[0, 2] = B - 1  # end_idx
            tile_scheduler_metadata[0, 3] = end_block_idx
            tile_scheduler_metadata[0, 4] = 0  # begin_n_split_idx
            # fields [5..7] stay 0

            # Non-split path ignores num_splits, but the API requires it:
            # zeros of length B+1
            num_splits = torch.zeros((B + 1,), dtype=dtype, device=device)
            scheduler_metadata.tile_scheduler_metadata = tile_scheduler_metadata
            scheduler_metadata.num_splits = num_splits

        if self.kv_cache_dtype.startswith("fp8"):
            o, lse = flash_mla_with_kvcache_fp8(
                q=q,
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler_metadata.tile_scheduler_metadata,
                num_splits=scheduler_metadata.num_splits,
                softmax_scale=self.scale,
                causal=True,
                descale_q=layer._q_scale.reshape(1),
                descale_k=layer._k_scale.reshape(1),
                kv_cache_dtype=self.kv_cache_dtype,
            )
        else:
            o, lse = flash_mla_with_kvcache(
                q=q,
                k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=scheduler_metadata,
                softmax_scale=self.scale,
                causal=True,
                is_fp8_kvcache=False,
            )

        o = reshape_attn_output_for_spec_decode(o)

        return o, lse

    def _forward_nc_fp8(
        self, q, kv_c_and_k_pe_cache, attn_metadata, layer,
        num_decodes: int, query_len: int,
    ):
        decode = attn_metadata.decode
        seq_lens = decode.seq_lens
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        # [B*L, H, D] -> [B, L, H, D]; matches the stock flatten's
        # request-major/token-minor row order.
        q_spec = reshape_query_for_spec_decode(q, num_decodes)
        subpage = kv_c_and_k_pe_cache.shape[1] // _NC_PAGE_BLOCK
        blocks = kv_c_and_k_pe_cache.shape[0]
        # Kernel contract: [pages, page_tokens, 1, D] over 64-token pages.
        # A stored-view cache reshapes without a copy; a padded/
        # non-contiguous one raises and surfaces loudly instead of
        # cloning gigabytes inside a graph.
        k_cache = kv_c_and_k_pe_cache.view(
            blocks * subpage, _NC_PAGE_BLOCK, kv_c_and_k_pe_cache.shape[2]
        ).unsqueeze(-2)
        # Final safety bound: the computed pool bound assumes the standard
        # uncompressed storage layout, while k_cache.shape[0] is the actual
        # number of kernel pages the launch will index.  Clamp our own
        # expanded table to it before the kernel sees it.
        decode.block_table[:num_decodes].clamp_(
            min=0, max=k_cache.shape[0] - 1
        )
        if k_cache.dtype != _FP8_E4M3:
            k_cache = k_cache.view(_FP8_E4M3)
        # gfx938 fp8 kernel consumes the pre-quantized query (same
        # contract as the plugin's dense fp8 path): static per-tensor
        # quant with the layer's q scale, descale fused via descale_q.
        q_fp8 = (
            (q_spec.to(torch.float32) / layer._q_scale.to(torch.float32))
            .clamp(-448.0, 448.0)
            .to(_FP8_E4M3)
        )
        o, lse = flash_mla_with_kvcache_fp8(
            q=q_fp8,
            k_cache=k_cache,
            block_table=decode.block_table[:num_decodes],
            cache_seqlens=seq_lens[:num_decodes],
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=decode.scheduler_metadata.tile_scheduler_metadata,
            num_splits=decode.scheduler_metadata.num_splits,
            softmax_scale=self.scale,
            causal=False,
            descale_q=layer._q_scale.reshape(1),
            descale_k=layer._k_scale.reshape(1),
            kv_cache_dtype="fp8_e4m3",
        )
        o = reshape_attn_output_for_spec_decode(o)
        # Stock returns lse as [rows, H] in q-dtype; the dense kernel
        # returns fp32 [B, H, L] -> req-major/token-minor rows.
        lse = lse.permute(0, 2, 1).reshape(-1, q_spec.shape[2]).to(q.dtype)
        return o, lse
