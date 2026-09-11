# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add HCU compressed-tensors INT8 support for Qwen4Exp PLE embeddings.

The vLLM snapshot used by HCU still constructs the AMD Qwen4Exp PLE table as
an unquantized ``PLEVocabParallelEmbedding``.  This patch keeps that model
implementation intact and replaces only the module-local PLE storage class
and its shard loader at worker startup.
"""

from __future__ import annotations

import functools
import re
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

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
        return F.embedding(input_, layer.weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        input_: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = input_.long()
        shard = layer.shard_indices
        local = ids - shard.org_vocab_start_index
        owned = (ids >= shard.org_vocab_start_index) & (
            ids < shard.org_vocab_end_index
        )
        local = local.clamp(0, layer.weight_scale.shape[0] - 1)
        scale = F.embedding(local, layer.weight_scale)
        scale.masked_fill_(~owned.unsqueeze(-1), 0)
        if layer.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            scale = tensor_model_parallel_all_reduce(scale)
        # The checkpoint stores one scalar scale for each vocabulary row.  The
        # lookup result is ``[..., embedding_dim]`` and the ``[..., 1]`` scale
        # broadcasts over the embedding dimension, including batched lookups.
        if embeddings.shape[:-1] != scale.shape[:-1]:
            raise ValueError(
                "PLE INT8 scale lookup shape does not match embedding output: "
                f"{tuple(scale.shape)} vs {tuple(embeddings.shape)}"
            )
        return embeddings.to(output_dtype) * scale.to(output_dtype)


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

        def forward(self, input_):
            raw = super().forward(input_)
            method = self.quant_method
            if not isinstance(method, HcuQwen4ExpPLEInt8EmbeddingMethod):
                return raw
            return method.dequantize(self, raw, input_, self.params_dtype)

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
