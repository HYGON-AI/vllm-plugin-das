# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add HCU compressed-tensors INT8 support for Qwen4Exp PLE embeddings.

The vLLM snapshot used by HCU still constructs the AMD Qwen4Exp PLE table as
an unquantized ``PLEVocabParallelEmbedding``.  This patch keeps that model
implementation intact and replaces only the module-local PLE storage class
and its shard loader at worker startup.

When VLLM_HCU_PLE_CPU_OFFLOAD=1 is set, the INT8 PLE weights are allocated
in CPU pinned memory instead of GPU, and the embedding lookup uses an explicit
H2D fallback path (HCU lacks native UVA operators). This reduces GPU memory
usage at the cost of PCIe transfer latency during lookup.
"""

from __future__ import annotations

import functools
import os
import re
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

logger = init_logger(__name__)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ple_layer"
PATCH_ID = "worker.core_fix.qwen4_exp.ple_int8"
TARGETS = (
    f"{TARGET_MODULE}.PLEVocabParallelEmbedding",
    f"{TARGET_MODULE}.Qwen4ExpNGramEmbedding.load_weights",
)
_MODULE_MARKER = "_vllm_hcu_qwen4_exp_ple_int8_applied"
_CLASS_MARKER = "_vllm_hcu_qwen4_exp_ple_int8_storage"
_LOAD_MARKER = "_vllm_hcu_qwen4_exp_ple_int8_loader"


def _is_compressed_tensors_int8(quant_config, prefix: str) -> bool:
    if quant_config is None or quant_config.get_name() != "compressed-tensors":
        return False
    target_map = getattr(quant_config, "target_scheme_map", {})
    for target, scheme in target_map.items():
        if target.startswith("re:"):
            matched = re.search(target[3:], prefix) is not None
        else:
            matched = target == prefix
        if not matched or "ngram_embedding" not in target:
            continue
        weight_quant = scheme.get("weights") if scheme else None
        if weight_quant is None:
            continue
        quant_type = str(getattr(weight_quant, "type", "")).lower()
        strategy = str(getattr(weight_quant, "strategy", "")).lower()
        if (
            quant_type.endswith("int")
            and int(getattr(weight_quant, "num_bits", 0)) == 8
            and strategy.endswith("channel")
        ):
            return True
    return False


def _should_offload_ple_to_cpu() -> bool:
    """Check if PLE CPU offload is enabled via VLLM_HCU_PLE_CPU_OFFLOAD."""
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_PLE_CPU_OFFLOAD)
    except (ImportError, AttributeError):
        return os.environ.get("VLLM_HCU_PLE_CPU_OFFLOAD", "0").lower() in (
            "true",
            "1",
        )


def _is_pin_memory_available() -> bool:
    """Check if pinned memory allocation is available on this platform."""
    try:
        from vllm.utils.platform_utils import is_pin_memory_available
        return is_pin_memory_available()
    except (ImportError, AttributeError):
        # Fallback: try to allocate a small pinned tensor
        try:
            test = torch.empty(1, device="cpu", pin_memory=True)
            del test
            return True
        except (RuntimeError, TypeError):
            return False


def _is_uva_available() -> bool:
    """Check if UVA zero-copy operator is available at runtime."""
    return hasattr(torch.ops._C, 'get_cuda_view_from_cpu_tensor')


class HcuQwen4ExpPLEInt8EmbeddingMethod(QuantizeMethodBase):
    """INT8 PLE lookup with one BF16 scale for every vocabulary row."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(rows, input_size_per_partition, dtype=torch.int8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ChannelQuantScaleParameter(
                data=torch.empty(rows, 1, dtype=torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        raise NotImplementedError("PLE INT8 weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # ``VocabParallelEmbedding.forward`` invokes this method before its
        # tensor-parallel all-reduce.  Return a supported floating-point dtype
        # here; returning INT8 would make the communicator reject the tensor.
        # ``input_`` is already masked to local indices by the base forward
        # path for TP > 1, and that path clears non-owner rows immediately
        # after this lookup.
        embeddings = F.embedding(input_, layer.weight)
        scales = F.embedding(input_, layer.weight_scale)
        output_dtype = layer.params_dtype
        return embeddings.to(output_dtype) * scales.to(output_dtype)


def _pinned_int8_lookup(
    weight_cpu: torch.Tensor,
    weight_scale_cpu: torch.Tensor,
    ids: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """INT8 pinned-host lookup with explicit H2D staging (Phase 2/3 fallback).

    Gathers the requested rows on the host side from the CPU pinned INT8 table
    and BF16 per-row scales, stages the *narrow* gathered slices to the
    accelerator, and dequantizes there:

        output[row, d] = float(weight[id, d]) * float(weight_scale[id, 0])

    Only the looked-up rows cross the PCIe bus, not the whole table.  The base
    ``VocabParallelEmbedding.forward`` has already clamped ``ids`` into the
    local TP range and will zero non-owner rows *after* this call, so no bounds
    handling or TP communication happens here.

    ``ids`` may be N-D (e.g. ``[num_tokens, ngram_heads]``); the row axis is
    flattened for the gather and restored on the output.
    """
    device = ids.device
    id_shape = tuple(ids.shape)
    flat_ids = ids.reshape(-1)

    # Host-side gather. ``weight_cpu`` is pinned CPU memory; indexing it
    # requires host-resident indices, so move ids to CPU first. This D2H sync
    # is the source of the fallback's added latency and blocks CUDA graph
    # capture (tracked for the native HCU UVA bridge in a later phase).
    host_ids = flat_ids.to(device="cpu", dtype=torch.long)
    gathered_weight = weight_cpu.index_select(0, host_ids)
    gathered_scale = weight_scale_cpu.index_select(0, host_ids)

    # Stage the narrow gathered slices to the accelerator. Pin the staging
    # source when possible so the H2D copy can overlap.
    if gathered_weight.device != device:
        gathered_weight = gathered_weight.to(device=device, non_blocking=True)
        gathered_scale = gathered_scale.to(device=device, non_blocking=True)

    embedding_dim = weight_cpu.shape[1]
    output = gathered_weight.to(output_dtype) * gathered_scale.to(output_dtype)
    return output.reshape(*id_shape, embedding_dim)


class HcuQwen4ExpPLEInt8UVAEmbeddingMethod(HcuQwen4ExpPLEInt8EmbeddingMethod):
    """INT8 PLE lookup with UVA zero-copy from CPU pinned memory.

    Uses ``get_cuda_view_from_cpu_tensor`` to create device views of the host
    pinned tables, allowing Triton kernels to directly read host memory without
    explicit H2D copies. Supports CUDA graph capture and prefetch streams.
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        pin = _is_pin_memory_available()
        # Allocate on CPU pinned memory, then wrap with UVA views in
        # process_weights_after_loading
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    rows,
                    input_size_per_partition,
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ChannelQuantScaleParameter(
                data=torch.empty(
                    rows,
                    1,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=pin,
                ),
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # NOTE: This runs inside ``device_loading_context`` where params are
        # transiently on the accelerator, then restored to CPU pinned memory.
        # ``get_cuda_view_from_cpu_tensor`` only accepts CPU pinned input, so we
        # cannot create the UVA view here. The view is created lazily on the
        # first ``embedding`` call, when ``layer.weight`` is back on CPU pinned.
        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight is None or weight_scale is None:
            return
        offloaded_bytes = (
            weight.numel() * weight.element_size()
            + weight_scale.numel() * weight_scale.element_size()
        )
        logger.info_once(
            "Qwen4Exp PLE INT8 UVA offload active: %.3f MiB per TP rank "
            "(weight=%s, weight_scale=%s, zero-copy device views)",
            offloaded_bytes / (1024**2),
            tuple(weight.shape),
            tuple(weight_scale.shape),
        )

    @staticmethod
    def _uva_view(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazily create and cache device views over the CPU pinned tables.
        # ``layer.weight``/``weight_scale`` are CPU pinned at inference time.
        cached = getattr(layer, "_hcu_uva_views", None)
        if cached is not None:
            return cached
        op = torch.ops._C.get_cuda_view_from_cpu_tensor
        views = (op(layer.weight.data), op(layer.weight_scale.data))
        layer._hcu_uva_views = views
        return views

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # Zero-copy lookup: ``F.embedding`` gathers rows from the UVA device
        # view, which reads host pinned memory directly over the bus. No D2H
        # sync, so the path is CUDA-graph compatible.
        weight_view, scale_view = self._uva_view(layer)
        embeddings = F.embedding(input_, weight_view)
        scales = F.embedding(input_, scale_view)
        output_dtype = layer.params_dtype
        return embeddings.to(output_dtype) * scales.to(output_dtype)


class HcuQwen4ExpPLEInt8OffloadEmbeddingMethod(HcuQwen4ExpPLEInt8EmbeddingMethod):
    """INT8 PLE lookup with the weight table offloaded to CPU pinned memory.

    Identical numerics to :class:`HcuQwen4ExpPLEInt8EmbeddingMethod`; the only
    differences are (1) ``weight``/``weight_scale`` are allocated on CPU pinned
    memory rather than the ambient accelerator device, and (2) ``embedding``
    gathers rows on the host and stages them to the accelerator.

    This is the H2D fallback variant used when UVA is unavailable.
    """

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        pin = _is_pin_memory_available()
        # Explicitly allocate on CPU (pinned when possible), overriding the
        # ambient ``with target_device:`` context used during model init. This
        # keeps the table off the accelerator; ``device_loading_context`` moves
        # it to the device only transiently for post-load processing and then
        # restores it to CPU pinned memory.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    rows,
                    input_size_per_partition,
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ChannelQuantScaleParameter(
                data=torch.empty(
                    rows,
                    1,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=pin,
                ),
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Runs inside ``device_loading_context``: params are temporarily on the
        # accelerator here, then restored to CPU pinned memory afterwards. Log
        # the offloaded footprint for verification (Phase 8 memory accounting).
        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight is None or weight_scale is None:
            return
        offloaded_bytes = (
            weight.numel() * weight.element_size()
            + weight_scale.numel() * weight_scale.element_size()
        )
        logger.info_once(
            "Qwen4Exp PLE INT8 CPU offload active: %.3f MiB per TP rank "
            "(weight=%s, weight_scale=%s)",
            offloaded_bytes / (1024**2),
            tuple(weight.shape),
            tuple(weight_scale.shape),
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # ``layer.weight``/``layer.weight_scale`` live on CPU pinned memory.
        # Gather-on-host, stage the narrow slices to the accelerator, then
        # dequantize to BF16 there so the base forward's ``masked_fill_`` and
        # TP all-reduce receive a device-resident floating-point tensor.
        return _pinned_int8_lookup(
            layer.weight,
            layer.weight_scale,
            input_,
            layer.params_dtype,
        )


def _make_storage_class(module: ModuleType, quant_config):
    base = module.PLEVocabParallelEmbedding
    if getattr(base, _CLASS_MARKER, False):
        return base

    class HcuPLEVocabParallelEmbedding(base):
        _vllm_hcu_qwen4_exp_ple_int8_storage = True

        def __init__(self, *args, **kwargs):
            prefix = kwargs.get("prefix", "")
            effective_quant_config = quant_config
            try:
                from vllm.config import get_current_vllm_config

                effective_quant_config = getattr(
                    get_current_vllm_config(), "quant_config", None
                ) or effective_quant_config
            except (AssertionError, AttributeError, RuntimeError):
                pass
            use_int8 = _is_compressed_tensors_int8(
                effective_quant_config, prefix
            )
            if use_int8:
                if _should_offload_ple_to_cpu():
                    # Prefer UVA zero-copy when available (§11.6 validation)
                    if _is_uva_available():
                        kwargs["quant_method"] = (
                            HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
                        )
                    else:
                        # Fallback to explicit H2D when UVA op is missing
                        logger.warning_once(
                            "UVA zero-copy unavailable, using H2D fallback "
                            "(blocks CUDA graph capture)"
                        )
                        kwargs["quant_method"] = (
                            HcuQwen4ExpPLEInt8OffloadEmbeddingMethod()
                        )
                else:
                    kwargs["quant_method"] = HcuQwen4ExpPLEInt8EmbeddingMethod()
                if kwargs.get("params_dtype") is None:
                    try:
                        from vllm.config import get_current_vllm_config

                        kwargs["params_dtype"] = (
                            get_current_vllm_config().model_config.dtype
                        )
                    except (AssertionError, AttributeError, RuntimeError):
                        pass
            super().__init__(*args, **kwargs)

    HcuPLEVocabParallelEmbedding.__name__ = "HcuPLEVocabParallelEmbedding"
    HcuPLEVocabParallelEmbedding.__qualname__ = "HcuPLEVocabParallelEmbedding"
    setattr(HcuPLEVocabParallelEmbedding, _CLASS_MARKER, True)
    return HcuPLEVocabParallelEmbedding


def _patch_ngram_loader(module: ModuleType) -> bool:
    ngram_class = require_class(
        module,
        "Qwen4ExpNGramEmbedding",
        f"{TARGET_MODULE}.Qwen4ExpNGramEmbedding",
    )
    original = require_callable(
        ngram_class,
        "load_weights",
        TARGETS[1],
    )
    require_exact_signature(original, TARGETS[1], positional=("self", "weights"))
    if getattr(original, _LOAD_MARKER, False):
        return False

    @functools.wraps(original)
    def hcu_load_weights(self, weights):
        items = list(weights)
        if not isinstance(
            getattr(self.ngram_embedding, "quant_method", None),
            HcuQwen4ExpPLEInt8EmbeddingMethod,
        ):
            return original(self, items)

        regular = []
        scale_shards = []
        for name, loaded_weight in items:
            match = re.search(
                r"(?:^|\.)ngram_embedding\.shard_(\d+)\.weight_scale$",
                name,
            )
            if match is None:
                regular.append((name, loaded_weight))
            else:
                scale_shards.append((int(match.group(1)), loaded_weight))

        loaded = set(original(self, regular))
        embedding = self.ngram_embedding
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        for shard_index, loaded_weight in scale_shards:
            if shard_index >= self.split_ngram_parts:
                raise ValueError(
                    f"PLE embedding scale shard index {shard_index} exceeds "
                    f"split_ngram_parts={self.split_ngram_parts}"
                )
            checkpoint_start = shard_index * shard_size
            expected_rows = max(
                0,
                min(shard_size, embedding.org_vocab_size - checkpoint_start),
            )
            expected_shape = (expected_rows, 1)
            if tuple(loaded_weight.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for PLE embedding scale shard {shard_index}: "
                    f"expected {expected_shape}, got {tuple(loaded_weight.shape)}"
                )
            from vllm.models.qwen4_exp.common.ple import copy_ple_embedding_shard_

            copy_ple_embedding_shard_(
                embedding.weight_scale,
                loaded_weight,
                checkpoint_start=checkpoint_start,
                tp_start=embedding.shard_indices.org_vocab_start_index,
                tp_end=embedding.shard_indices.org_vocab_end_index,
            )
            loaded.add("ngram_embedding.weight_scale")
        return loaded

    setattr(hcu_load_weights, _LOAD_MARKER, True)
    setattr(ngram_class, "_vllm_hcu_original_load_weights", original)
    setattr(ngram_class, "load_weights", hcu_load_weights)
    return True


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, _MODULE_MARKER, False):
        return False
    base = require_class(
        module,
        "PLEVocabParallelEmbedding",
        f"{TARGET_MODULE}.PLEVocabParallelEmbedding",
    )
    if not callable(getattr(base, "forward", None)):
        raise PatchCompatibilityError(
            f"required target {TARGET_MODULE}.PLEVocabParallelEmbedding.forward "
            "is missing"
        )
    try:
        from vllm.config import get_current_vllm_config

        quant_config = get_current_vllm_config().quant_config
    except (AssertionError, AttributeError, RuntimeError):
        quant_config = None
    module.PLEVocabParallelEmbedding = _make_storage_class(module, quant_config)
    _patch_ngram_loader(module)
    setattr(module, _MODULE_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
