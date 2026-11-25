# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU kernel leaves on the actual Core033 sink-aware sparse MLA backend."""

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseBackend as CoreSinkBackend,
    HYV4FlashMLASparseImpl as CoreSinkImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata

from .hisparse_runtime import (
    HiSparseBindingError,
    HiSparseRuntimeBinding,
    HiSparseReaderBinding,
    bind_hisparse_reader,
    install_hisparse_prewrite_hook,
    same_cache_layout,
)
from .flashmla_ops import get_flashmla_ops
from .runtime import is_hcu_platform


class FlashSparseImpl(CoreSinkImpl):
    def __init__(self, *args, **kwargs):
        parallel = get_current_vllm_config().parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise NotImplementedError("HYV4 Flash sparse attention requires PCP=DCP=1")
        super().__init__(*args, **kwargs)
        self._hcu_ops = get_flashmla_ops() if is_hcu_platform() else None
        # Qualify the current worker device once, outside compilation/capture.
        # The native 512-column layout is qualified only on gfx938.
        self._hcu_window_device = None
        if self._hcu_ops is not None and torch.cuda.is_available():
            device = torch.cuda.current_device()
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            if arch == "gfx938":
                self._hcu_window_device = torch.device("cuda", device)
        self._install_hisparse_hook()

    def _install_hisparse_hook(self):
        # Core prepares the handle and selects write_target BEFORE calling our
        # writer. Install at construction, not lazily inside the writer.
        if self._hcu_ops is None:
            return
        from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

        if isinstance(self.index_group, HiSparseMLAIndexGroup):
            install_hisparse_prewrite_hook(
                self, self.index_group.cache(self.index_group_index)
            )

    def _prepare_hisparse_layer(self, metadata):
        call = self._checked_hisparse_call()
        if call is None:
            return
        binding, selected = call
        if selected is not metadata:
            raise HiSparseBindingError("HiSparse prepare metadata owner changed")
        if isinstance(binding, HiSparseReaderBinding):
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            if not isinstance(context.slot_mapping, dict):
                raise HiSparseBindingError("reader needs per-layer source slots")
            binding.prepare_query(
                metadata, context.slot_mapping.get(binding.layer.layer_name)
            )

    def _bind_hisparse_aliases(self, context, layer, cache, stream):
        binding = getattr(self, "_hcu_hisparse_binding", None)
        if binding is not None:
            if binding.cache is not cache:
                raise HiSparseBindingError("HiSparse IndexGroup cache changed")
            return binding
        layers = tuple(context.no_compile_layers.values())
        if getattr(cache, "view", None) is None:
            # DFly's allocation alias intentionally has no separate synthetic
            # resident/hot allocation. Find exactly ONE non-draft Core owner;
            # the source pointer alone never establishes a valid binding.
            owners = [candidate for candidate in layers
                      if candidate is not layer
                      and not getattr(candidate, "is_draft_layer", False)
                      and getattr(candidate, "hisparse_cache", None) is not None
                      and getattr(candidate.hisparse_cache, "view", None) is not None
                      and same_cache_layout(candidate.kv_cache, layer.kv_cache)]
            if len(owners) != 1:
                raise HiSparseBindingError("reader needs one fully allocated target")
            owner = owners[0]
            if (not getattr(layer, "is_draft_layer", False)
                    or not isinstance(owner.impl, FlashSparseImpl)):
                raise HiSparseBindingError("source-only alias is not a DFly reader")
            owner.impl._bind_hisparse_aliases(
                context, owner, owner.hisparse_cache, stream
            )
            binding = getattr(self, "_hcu_hisparse_binding", None)
            if binding is None:
                raise HiSparseBindingError("DFly reader binding was not published")
            return binding
        binding = HiSparseRuntimeBinding(cache, stream=stream)
        self._hcu_hisparse_binding = binding
        try:
            for candidate in layers:
                tensor = getattr(candidate, "kv_cache", None)
                if (candidate is layer or tensor is None or not tensor.numel()
                        or tensor.data_ptr() != layer.kv_cache.data_ptr()):
                    continue
                impl = getattr(candidate, "impl", None)
                if (not getattr(candidate, "is_draft_layer", False)
                        or not isinstance(impl, FlashSparseImpl)
                        or impl._hcu_ops is None
                        or impl.index_group.cache(impl.index_group_index)
                        is not candidate.hisparse_cache):
                    raise HiSparseBindingError("source-only alias lacks a DFly backend")
                if getattr(impl, "_hcu_hisparse_binding", None) is not None:
                    raise HiSparseBindingError("reader has another live binding")
                reader = bind_hisparse_reader(
                    binding, candidate.hisparse_cache, reader_layer=candidate,
                    stream=stream, torch=torch,
                )
                impl._hcu_hisparse_binding = reader
            return binding
        except Exception as error:
            # An incomplete discovery must not become a partially usable lease.
            binding._poison(error)
            raise

    def _checked_hisparse_call(self, layer=None):
        # No global patch or allocation hook. Ordinary KV and non-HCU execution
        # keep the inherited path exactly; Core allocates all HiSparse storage.
        if self._hcu_ops is None:
            return None
        from vllm.forward_context import get_forward_context
        from vllm.utils.torch_utils import current_stream
        from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

        group = self.index_group
        if not isinstance(group, HiSparseMLAIndexGroup):
            return None
        context = get_forward_context()
        mode = context.cudagraph_runtime_mode
        if getattr(mode, "name", None) != "NONE":
            raise HiSparseBindingError("HCU HiSparse binding is eager-only")
        if not isinstance(context.attn_metadata, dict):
            raise HiSparseBindingError("HiSparse needs per-layer metadata; no DBO")
        cache = group.cache(self.index_group_index)
        if layer is None:
            matches = [candidate for candidate in context.no_compile_layers.values()
                       if getattr(candidate, "impl", None) is self]
            if len(matches) != 1:
                raise HiSparseBindingError(
                    "HiSparse writer layer ownership is ambiguous"
                )
            layer = matches[0]
        if (getattr(layer, "impl", None) is not self
                or getattr(layer, "hisparse_cache", None) is not cache):
            raise HiSparseBindingError("layer/IndexGroup cache ownership mismatch")
        metadata = context.attn_metadata.get(layer.layer_name)
        binding = self._bind_hisparse_aliases(context, layer, cache, current_stream)
        binding.check(source=layer.kv_cache, layers=context.no_compile_layers.values())
        return binding, metadata

    def do_kv_cache_update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping,
        kv_cache_dtype, k_scale, skip_scale=False,
    ):
        writer = super().do_kv_cache_update
        call = self._checked_hisparse_call()
        args = (kv_c_normed, k_pe, kv_cache, slot_mapping,
                kv_cache_dtype, k_scale, skip_scale)
        if call is None:
            return writer(*args)
        binding, metadata = call
        binding.validate_batch(
            metadata, query_rows=kv_c_normed.shape[0],
            topk_width=self.topk_indices_buffer.shape[1],
        )
        return binding.write(
            lambda: writer(*args), kv_cache=kv_cache,
            slot_mapping=slot_mapping, rows=kv_c_normed.shape[0],
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        call = self._checked_hisparse_call(layer)
        if call is not None:
            binding, metadata = call
            if metadata is not attn_metadata:
                raise HiSparseBindingError("HiSparse attention metadata owner changed")
            if not same_cache_layout(kv_c_and_k_pe_cache, layer.kv_cache):
                raise HiSparseBindingError("attention source differs from layer source")
            binding.validate_batch(
                metadata, query_rows=(q[0] if isinstance(q, tuple) else q).shape[0],
                topk_width=self.topk_indices_buffer.shape[1],
                allow_decode_prefix=True,
            )
        # Core still owns logical->physical conversion, swap-in, request unions,
        # spec-query reshape and BF16/FP8 selection. Our existing kernel leaves
        # consume the physical hot cache returned by that real IndexGroup path.
        reader = super().forward_mqa
        if call is None:
            return reader(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        return binding.read(
            lambda: reader(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        )

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hcu_ops is None:
            return super()._bf16_flash_mla_kernel(
                q, kv_c_and_k_pe_cache, topk_indices,
                topk_length, actual_num_heads, out,
            )
        heads = q.shape[1] if actual_num_heads is None else actual_num_heads
        if not 0 < heads <= q.shape[1]:
            raise ValueError("actual_num_heads must fit the query head layout")
        kernel_heads = (heads + self.prefill_padding - 1) // self.prefill_padding * self.prefill_padding
        sinks = self._sinks_for_query(q[:, :heads], 1, kernel_heads)
        # Never expose uninitialized workspace padding to the vendor kernel.
        q = q[:, :heads]
        if heads < kernel_heads:
            q = F.pad(q, (0, 0, 0, kernel_heads - heads))
        kv = kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1])
        if out is not None and (
            out.shape != (q.shape[0], kernel_heads, 512)
            or out.device != q.device or out.dtype != q.dtype
        ):
            raise ValueError("FlashMLA out must match the padded query output layout")
        # Some vendor shapes sort indices in-place. Keep Core-owned shared
        # top-k buffers immutable at this explicit ABI boundary.
        indices = topk_indices.view(q.shape[0], 1, -1).clone()
        result, _, lse = self._hcu_ops.sparse_fwd(
            q, kv, indices, self.softmax_scale,
            attn_sink=sinks, topk_length=topk_length,
        )
        if out is not None:
            out.copy_(result)
            result = out
        return result[:, :heads], lse[:, :heads]

    def _use_window_fp8_prefix(self, q, cache, indices):
        # Both window and compressed attention inherit this implementation.
        # Classify the live producer: DFly may replace its shared buffer after
        # construction, and an explicit Core buffer setter clears _indexer.
        from .swa import FlashWindowIndexer

        owner = getattr(self, "_indexer", None)
        if (self._hcu_window_device is None
                or type(owner) is not FlashWindowIndexer
                or type(owner.window) is not int or owner.window != 512):
            return False
        source = owner.topk_indices_buffer
        return (
            q.device == self._hcu_window_device
            and q.ndim == 4 and q.shape[0] == 1
            and q.shape[2] in (8, 16) and q.shape[3] == 576
            and q.dtype == torch.bfloat16
            and self.fp8_decode_padded_heads == 64
            and cache.device == q.device and cache.dtype == torch.uint8
            and cache.ndim == 3 and cache.shape[-1] == 656
            and source.ndim == 2 and source.shape[-1] == 2048
            and source.dtype == torch.int32 and source.device == q.device
            and indices.ndim == 3
            and indices.shape == (1, q.shape[1], 2048)
            and indices.dtype == torch.int32 and indices.device == q.device
            and indices.stride(-1) == 1
        )

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hcu_ops is None:
            return super()._fp8_flash_mla_kernel(
                q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata
            )
        use_window_prefix = self._use_window_fp8_prefix(
            q, kv_c_and_k_pe_cache, topk_indices
        )
        heads = q.shape[2]
        padded = self.fp8_decode_padded_heads
        sinks = self._sinks_for_query(q, 2, padded)
        if heads < padded:
            q = F.pad(q, (0, 0, 0, padded - heads))
        # IndexGroup paths may deliberately omit kernel metadata. A fresh
        # vendor scheduler is valid for sparse decode (no dummy tables needed).
        if kernel_metadata is None:
            scheduler, _ = self._hcu_ops.get_metadata()
            table = lengths = None
        else:
            scheduler = kernel_metadata.scheduler_metadata
            table = kernel_metadata.dummy_block_table
            lengths = kernel_metadata.cache_lens
        if use_window_prefix:
            # Core has already remapped physical indices. Window producers
            # populate only the first 512 columns; preserve the shared 2048
            # source and its scheduler for compressed attention.
            topk_indices = topk_indices[..., :512]
            if kernel_metadata is not None:
                scheduler, _ = self._hcu_ops.get_metadata()
        result, lse = self._hcu_ops.with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=table,
            cache_seqlens=lengths,
            head_dim_v=512,
            tile_scheduler_metadata=scheduler,
            softmax_scale=self.softmax_scale,
            is_fp8_kvcache=True,
            indices=topk_indices.clone(),
            attn_sink=sinks,
        )
        return result[:, :, :heads], lse[:, :heads]


class FlashSparseKernelBackend(CoreSinkBackend):
    @staticmethod
    def get_impl_cls():
        return FlashSparseImpl

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False
