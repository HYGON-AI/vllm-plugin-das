# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt an ordinary BF16 ParallelLMHead into post-load ConvRot INT8."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.vocab_parallel_embedding"
PATCH_ID = "worker.op_opt.lm_head.convrot_int8"
TARGETS = (f"{TARGET_MODULE}.ParallelLMHead.__init__",)
_CLASS_MARKER = "_vllm_hcu_convrot_int8_applied"
_WRAPPER_MARKER = "_vllm_hcu_convrot_int8_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    head_class = require_class(target, "ParallelLMHead", f"{TARGET_MODULE}.ParallelLMHead")
    if already_applied(
        head_class, _CLASS_MARKER,
        ((head_class, "__init__", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False
    original = require_callable(head_class, "__init__", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0],
        positional=(
            "self", "num_embeddings", "embedding_dim", "bias", "params_dtype",
            "org_num_embeddings", "padding_size", "quant_config", "prefix",
        ),
        keyword_only=("disable_tp",),
        defaults={
            "bias": False, "params_dtype": None, "org_num_embeddings": None,
            "padding_size": target.DEFAULT_VOCAB_PADDING_SIZE,
            "quant_config": None, "prefix": "", "disable_tp": False,
        },
    )

    @functools.wraps(original)
    def hcu_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if not get_hcu_config(config).lm_head_convrot_int8:
            return
        # Default-off model construction never imports LightOp or this runtime.
        from vllm_hcu.model_executor.layers.quantization.lm_head_convrot_int8 import (
            install_convrot_lm_head,
        )

        install_convrot_lm_head(self, config)

    setattr(hcu_init, _WRAPPER_MARKER, True)
    head_class._vllm_hcu_original_convrot_init = original
    head_class.__init__ = hcu_init
    setattr(head_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
