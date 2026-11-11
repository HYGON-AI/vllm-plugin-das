# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Give the Kimi-K3/K3.1 DSpark draft a dense HCU MLA attention path.

Exact-name post-import hook on ``vllm.models.kimi_k3.nvidia.dspark_mla``:

* ``K3DSparkDecoderLayer.__init__`` -- the draft decoder layer builds a dense
  ``MultiHeadLatentAttention`` from the run-level ``cache_config``.  Under
  Kimi-K3.1 that is the sparse-only ``fp8_ds_mla`` record, which no dense HCU
  MLA backend accepts (``No valid attention backend found for hip``).  While
  the stock constructor runs, the shared ``cache_config.cache_dtype`` is
  presented as the draft KV dtype (``VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE``,
  default BF16) and, when ``VLLM_HCU_K3_DSPARK_DRAFT_MLA_BACKEND`` pins
  FLASHMLA or TRITON_MLA, the K3 MLA module's ``get_attn_backend`` resolves
  the pinned backend.  Both are restored unconditionally afterwards (the
  qwen4-exp QSA FP8 construction-scope precedent), so the target model, the
  KV-cache grouping and every later reader keep the run dtype; the draft
  layer keeps its own ``kv_cache_dtype`` and publishes it in its
  ``MLAAttentionSpec``.

Run dtypes other than ``fp8_ds_mla`` with ``auto`` backend are passed
through untouched, and ``VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE=inherit`` restores
the pre-adapter construction exactly.
"""

from __future__ import annotations

import functools
import importlib
from inspect import signature
from types import ModuleType
from typing import Any

from vllm.logger import init_logger

from ._common import PatchCompatibilityError, load_exact_module, require_class

logger = init_logger(__name__)

TARGET_MODULE = "vllm.models.kimi_k3.nvidia.dspark_mla"
MLA_MODULE = "vllm.models.kimi_k3.nvidia.mla"
PATCH_ID = "worker.op_opt.kimi_k3.dspark_draft_attention"
TARGETS = (f"{TARGET_MODULE}.K3DSparkDecoderLayer.__init__",)
_MARKER = "_vllm_hcu_k3_dspark_draft_attention_applied"


def _load_mla_module(target: ModuleType) -> ModuleType:
    # The draft resolves its attention class from the K3 MLA module, whose
    # globals are where ``MultiHeadLatentAttention.__init__`` looks up
    # ``get_attn_backend``; pin through exactly that module.
    attention_cls = require_class(
        target,
        "MultiHeadLatentAttention",
        f"{TARGET_MODULE}.MultiHeadLatentAttention",
    )
    if attention_cls.__module__ != MLA_MODULE:
        raise PatchCompatibilityError(
            f"{TARGET_MODULE}.MultiHeadLatentAttention comes from "
            f"{attention_cls.__module__!r}, expected {MLA_MODULE!r}"
        )
    mla_module = importlib.import_module(MLA_MODULE)
    if not callable(getattr(mla_module, "get_attn_backend", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {MLA_MODULE}.get_attn_backend is missing"
        )
    return mla_module


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    from vllm_hcu.models.kimi_k3_dspark import (
        draft_attention_scope,
        resolve_draft_kv_cache_dtype,
    )
    from vllm_hcu.platforms import envs as henvs

    layer_cls = require_class(
        target, "K3DSparkDecoderLayer", TARGETS[0].rsplit(".", 1)[0]
    )
    mla_module = _load_mla_module(target)
    original_init = layer_cls.__init__
    init_signature = signature(original_init)
    if "vllm_config" not in init_signature.parameters:
        raise PatchCompatibilityError(
            f"{TARGETS[0]} no longer takes vllm_config"
        )

    @functools.wraps(original_init)
    def layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            bound = init_signature.bind(self, *args, **kwargs)
        except TypeError:
            # Let the stock constructor raise its own argument error.
            original_init(self, *args, **kwargs)
            return
        cache_config = getattr(
            bound.arguments.get("vllm_config"), "cache_config", None
        )
        run_dtype = getattr(cache_config, "cache_dtype", None)
        draft_dtype = resolve_draft_kv_cache_dtype(
            run_dtype, henvs.VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE
        )
        backend = henvs.VLLM_HCU_K3_DSPARK_DRAFT_MLA_BACKEND
        if draft_dtype is None and backend == "auto":
            original_init(self, *args, **kwargs)
            return
        logger.info_once(
            "Kimi-K3 DSpark draft attention: KV dtype %s (run %s), dense MLA "
            "backend %s.",
            draft_dtype or run_dtype,
            run_dtype,
            backend,
        )
        with draft_attention_scope(cache_config, draft_dtype, mla_module, backend):
            original_init(self, *args, **kwargs)

    layer_cls.__init__ = layer_init
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
