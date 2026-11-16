# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optionally build QSA full-attention layers as plain dense attention.

``VLLM_HCU_DISABLE_QSA=1`` makes every ``Qwen4ExpDecoderLayer`` take its
native non-QSA branch (``Qwen3NextAttention``), so full-attention layers run
the platform's standard FlashAttention path over a regular KV cache with no
indexer or top-k selection.  Backbone and MTP drafter layers share
``Qwen4ExpDecoderLayer``, so both are covered.  The same switch also enables
the FlashAttention backend's mixed decode/prefill batch split (see
``_SPLIT_PD`` in ``vllm_hcu/v1/attention/backends/flash_attn.py``), so dense
replacement and the split ship as one option.

The QSA owner registers itself and its indexer side caches in the global
``static_forward_context`` while it is constructed, so it must never be built:
``config.indexer_n_heads`` is hidden only for the duration of the layer's
``__init__`` and restored afterwards.  ``Qwen4ExpModel`` computes
``_qsa_layer_ids`` from the unmodified config before any layer exists, so
``load_weights`` is wrapped to recompute that set from the modules that were
actually built: a dense replacement keeps its checkpoint cache scales on the
native ``self_attn.attn`` path instead of having them remapped onto buffers
the QSA owner would have owned (and then silently dropped).  The checkpoint's
``self_attn.indexer.*`` weights then have no module and are ignored through
the shared Qwen4Exp ignored-suffix list used by the backbone and MTP loaders.
Any other value (or unset) keeps the unmodified QSA behavior.
"""

from __future__ import annotations

import functools
import os
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

_DISABLE_QSA = os.environ.get("VLLM_HCU_DISABLE_QSA", "0") == "1"

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.core_fix.qwen4_exp.disable_qsa"
_LAYER_INIT_TARGET = f"{TARGET_MODULE}.Qwen4ExpDecoderLayer.__init__"
_LOAD_WEIGHTS_TARGET = f"{TARGET_MODULE}.Qwen4ExpModel.load_weights"
_IGNORED_SUFFIXES_TARGET = f"{TARGET_MODULE}._QWEN4_EXP_IGNORED_MISSING_SUFFIXES"
TARGETS = (_LAYER_INIT_TARGET, _LOAD_WEIGHTS_TARGET, _IGNORED_SUFFIXES_TARGET)
_MARKER = "_vllm_hcu_qwen4_exp_disable_qsa_applied"
_LAYER_INIT_WRAPPER = "_vllm_hcu_disable_qsa_layer_init"
_LOAD_WEIGHTS_WRAPPER = "_vllm_hcu_disable_qsa_load_weights"
_QSA_CONFIG_ATTR = "indexer_n_heads"
_INDEXER_WEIGHT_SUFFIX = ".self_attn.indexer."


def _already_applied(owner: object, wrapped: tuple[tuple, ...]) -> bool:
    if not getattr(owner, _MARKER, False):
        return False
    for target_owner, name, target, wrapper_marker in wrapped:
        function = require_callable(target_owner, name, target)
        marker_owner = getattr(function, "__func__", function)
        if not getattr(marker_owner, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {target} is stale; restart the process"
            )
    return True


def apply_to_module(module: ModuleType) -> bool:
    owner = load_exact_module(TARGET_MODULE, module)
    layer_cls = require_class(owner, "Qwen4ExpDecoderLayer", _LAYER_INIT_TARGET)
    model_cls = require_class(owner, "Qwen4ExpModel", _LOAD_WEIGHTS_TARGET)
    ignored_suffixes = getattr(owner, "_QWEN4_EXP_IGNORED_MISSING_SUFFIXES", None)
    if not isinstance(ignored_suffixes, list):
        raise PatchCompatibilityError(
            f"required HCU patch target {_IGNORED_SUFFIXES_TARGET} is missing"
        )
    wrapped = (
        (layer_cls, "__init__", _LAYER_INIT_TARGET, _LAYER_INIT_WRAPPER),
        (model_cls, "load_weights", _LOAD_WEIGHTS_TARGET, _LOAD_WEIGHTS_WRAPPER),
    )
    if _already_applied(owner, wrapped):
        return False

    original_layer_init = require_callable(
        layer_cls, "__init__", _LAYER_INIT_TARGET
    )
    require_exact_signature(
        original_layer_init,
        _LAYER_INIT_TARGET,
        positional=("self", "vllm_config", "layer_type", "prefix"),
        defaults={"prefix": ""},
    )
    original_load_weights = require_callable(
        model_cls, "load_weights", _LOAD_WEIGHTS_TARGET
    )
    require_exact_signature(
        original_load_weights,
        _LOAD_WEIGHTS_TARGET,
        positional=("self", "weights"),
    )

    @functools.wraps(original_layer_init)
    def hcu_disable_qsa_layer_init(self, vllm_config, layer_type, prefix=""):
        config = vllm_config.model_config.hf_text_config
        if (
            not _DISABLE_QSA
            or layer_type != "full_attention"
            or getattr(config, _QSA_CONFIG_ATTR, None) is None
        ):
            return original_layer_init(self, vllm_config, layer_type, prefix)
        logger.info_once(
            "[HCU QSA PATH] VLLM_HCU_DISABLE_QSA=1: full-attention layers "
            "run dense Qwen3NextAttention instead of QSA"
        )
        # Hide the QSA switch only while the layer picks its attention class,
        # so the native non-QSA branch builds Qwen3NextAttention and no QSA
        # owner/indexer is ever registered in the forward context.
        qsa_heads = getattr(config, _QSA_CONFIG_ATTR)
        setattr(config, _QSA_CONFIG_ATTR, None)
        try:
            original_layer_init(self, vllm_config, layer_type, prefix)
        finally:
            setattr(config, _QSA_CONFIG_ATTR, qsa_heads)

    setattr(hcu_disable_qsa_layer_init, _LAYER_INIT_WRAPPER, True)

    @functools.wraps(original_load_weights)
    def hcu_disable_qsa_load_weights(self, weights):
        if _DISABLE_QSA:
            # The native __init__ derived _qsa_layer_ids from the unmodified
            # config before any layer existed, so dense replacements were
            # still treated as QSA scale owners by the remap below. Recompute
            # the set from the modules that were actually built: a QSA owner
            # is exactly an attention that still has its indexer. The remap
            # generator reads the attribute lazily and picks this set up.
            self._qsa_layer_ids = frozenset(
                layer_idx
                for layer_idx, layer in enumerate(self.layers)
                if getattr(getattr(layer, "self_attn", None), "indexer", None)
                is not None
            )
        return original_load_weights(self, weights)

    setattr(hcu_disable_qsa_load_weights, _LOAD_WEIGHTS_WRAPPER, True)

    added_suffix = _DISABLE_QSA and _INDEXER_WEIGHT_SUFFIX not in ignored_suffixes
    try:
        if added_suffix:
            # The backbone and MTP loaders copy this shared list per load, so
            # the unused checkpoint indexer weights are skipped on both paths.
            ignored_suffixes.append(_INDEXER_WEIGHT_SUFFIX)
        setattr(layer_cls, "__init__", hcu_disable_qsa_layer_init)
        setattr(model_cls, "load_weights", hcu_disable_qsa_load_weights)
        setattr(owner, _MARKER, True)
    except BaseException:
        if added_suffix and _INDEXER_WEIGHT_SUFFIX in ignored_suffixes:
            ignored_suffixes.remove(_INDEXER_WEIGHT_SUFFIX)
        setattr(layer_cls, "__init__", original_layer_init)
        setattr(model_cls, "load_weights", original_load_weights)
        if hasattr(owner, _MARKER):
            delattr(owner, _MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
