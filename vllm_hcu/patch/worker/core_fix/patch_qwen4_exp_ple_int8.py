# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add HCU compressed-tensors INT8 support for Qwen4Exp PLE embeddings.

The vLLM snapshot used by HCU still constructs the AMD Qwen4Exp PLE table as
an unquantized ``PLEVocabParallelEmbedding``.  This patch keeps that model
implementation intact and replaces only the module-local PLE storage class
and its shard loader at worker startup.

When EngramConfig.cpu_offload is enabled, the INT8 PLE weights are allocated
in CPU pinned memory instead of GPU. The legacy VLLM_HCU_PLE_CPU_OFFLOAD
environment variable remains a fallback when EngramConfig is omitted. The
offload path requires the HCU UVA bridge and fails during model initialization
when that bridge is unavailable.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

from vllm_hcu.models.qwen4_exp.engram import cpu_offload_enabled

logger = init_logger(__name__)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ple_layer"
REPLACEMENT_MODULE = "vllm_hcu.models.qwen4_exp.amd.ple_layer"
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
    """Resolve PLE CPU offload from EngramConfig or the HCU legacy flag."""
    return cpu_offload_enabled()


def _should_prefetch_ple() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_PLE_PREFETCH_STREAM)
    except (ImportError, AttributeError):
        return (
            os.environ.get("VLLM_HCU_USE_CUSTOM_OPS", "1").lower()
            in ("true", "1")
            and os.environ.get("VLLM_HCU_PLE_PREFETCH_STREAM", "0").lower()
            in ("true", "1")
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
    return hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor")


class HcuQwen4ExpPLEInt8EmbeddingMethod(QuantizeMethodBase):
    """INT8 PLE lookup with one BF16 scale for every vocabulary row."""

    supports_prefetch = False

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
        # embedding-parallel all-reduce. Return a supported floating-point dtype
        # here; returning INT8 would make the communicator reject the tensor.
        # ``input_`` is already masked to local indices by the base forward
        # path for TP > 1, and that path clears non-owner rows immediately
        # after this lookup.
        embeddings = F.embedding(input_, layer.weight)
        scales = F.embedding(input_, layer.weight_scale)
        output_dtype = layer.params_dtype
        return embeddings.to(output_dtype) * scales.to(output_dtype)


class HcuQwen4ExpPLEInt8UVAEmbeddingMethod(HcuQwen4ExpPLEInt8EmbeddingMethod):
    """INT8 PLE lookup with UVA zero-copy from CPU pinned memory.

    Uses ``get_cuda_view_from_cpu_tensor`` to create device views of the host
    pinned tables, allowing Triton kernels to directly read host memory without
    explicit H2D copies. Supports CUDA graph capture and prefetch streams.
    """

    supports_prefetch = True
    # The PLE tables are already in their final pinned-host representation.
    # Post-load only invalidates cached UVA views and records diagnostics, so
    # staging the complete tables on the accelerator is unnecessary.
    requires_device_loading = False

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
        # Allocate directly in CPU pinned memory. UVA views are created lazily
        # after weight loading, either by prepare_prefetch or the first lookup.
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
        # This runs in place while the parameters remain in pinned CPU memory.
        # Recreate the view after every load/reload so it cannot reference stale
        # storage if an external loader replaced either parameter.
        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight is None or weight_scale is None:
            return
        layer._hcu_uva_views = None
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

    def prefetch_output_dtype(self, layer: nn.Module) -> torch.dtype:
        if layer.params_dtype is not torch.bfloat16:
            raise RuntimeError(
                "Qwen4Exp PLE prefetch currently supports BF16 output only; "
                f"got {layer.params_dtype}"
            )
        return torch.bfloat16

    def prepare_prefetch(self, layer: nn.Module) -> None:
        self._uva_view(layer)

    @staticmethod
    def is_prefetch_prepared(layer: nn.Module) -> bool:
        return getattr(layer, "_hcu_uva_views", None) is not None

    def prefetch_lookup_into(
        self,
        layer: nn.Module,
        local_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        weight_view, scale_view = self._uva_view(layer)
        embeddings = F.embedding(local_ids, weight_view)
        scales = F.embedding(local_ids, scale_view)
        output.copy_(
            embeddings.to(torch.bfloat16) * scales.to(torch.bfloat16)
        )

    def finalize_prefetched(
        self,
        layer: nn.Module,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        if layer.tp_size == 1:
            return rows
        return tensor_model_parallel_all_reduce(rows)

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
                    if not _is_uva_available():
                        raise RuntimeError(
                            "Qwen4Exp PLE INT8 CPU offload requires the HCU UVA "
                            "operator torch.ops._C.get_cuda_view_from_cpu_tensor, "
                            "but it is unavailable. Install a compatible "
                            "vLLM/vllm-plugin-das build or disable PLE CPU offload."
                        )
                    kwargs["quant_method"] = HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
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
            if _should_prefetch_ple():
                method = self.quant_method
                if not _should_offload_ple_to_cpu():
                    logger.warning_once(
                        "VLLM_HCU_PLE_PREFETCH_STREAM=1 requires PLE CPU "
                        "offload via --engram-config.cpu_offload=true or "
                        "VLLM_HCU_PLE_CPU_OFFLOAD=1; using inline PLE lookup"
                    )
                elif not getattr(method, "supports_prefetch", False):
                    logger.warning_once(
                        "Qwen4Exp PLE prefetch is unavailable for %s; using its "
                        "existing inline behavior. This release supports INT8 "
                        "UVA only; FP8 prefetch is future work.",
                        type(method).__name__,
                    )

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
    if module is None:
        module = importlib.import_module(TARGET_MODULE)
    if module.__name__ not in (TARGET_MODULE, REPLACEMENT_MODULE):
        raise PatchCompatibilityError(
            f"expected module {TARGET_MODULE} or {REPLACEMENT_MODULE}, "
            f"got {module.__name__}"
        )
    return apply_to_module(module)


__all__ = [
    "PATCH_ID",
    "REPLACEMENT_MODULE",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
