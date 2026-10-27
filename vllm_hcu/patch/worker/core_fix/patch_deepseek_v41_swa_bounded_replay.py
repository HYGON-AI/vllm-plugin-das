# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Wire DeepSeek-V4.1 SWA ``bounded_replay`` for encoder-side APC on 0.29.

Pinned ``DeepseekV4Attention`` constructs ``DeepseekV4SWACache`` without
``bounded_replay=...``. Upstream #56227 / #58132 pass
``CacheConfig.swa_bounded_replay`` (gated on MRV2 / no PCP) so
``get_kv_cache_spec`` can mark SWA groups non-prefix-cacheable.

This adapter sets ``swa_cache_layer.bounded_replay`` after ``__init__``. The
exchanged HCU ``sparse_swa.DeepseekV4SWACache.get_kv_cache_spec`` then returns
``BoundedReplaySlidingWindowMLASpec`` when the flag is on.

Pairs with ``patch_swa_bounded_replay_apc`` (scheduler rewind /
``replay_start``) and ``patch_deepseek_v41_decoder_replay_state``
(``pad_replayed_slots`` + ReplayAttnMetadata).
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.swa_bounded_replay"
TARGETS = (f"{TARGET_MODULE}.DeepseekV4Attention.__init__",)
_MARKER = "_vllm_hcu_dsv41_swa_bounded_replay_applied"
_WRAPPER = "_vllm_hcu_dsv41_swa_bounded_replay_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "DeepseekV4Attention", f"{TARGET_MODULE}.DeepseekV4Attention"
    )
    if getattr(cls, _MARKER, False):
        current = vars(cls).get("__init__")
        if not getattr(current, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False

    original_init = require_callable(cls, "__init__", TARGETS[0])
    # 0.29: all params are POSITIONAL_OR_KEYWORD (optional buffers default None).
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=(
            "self",
            "vllm_config",
            "prefix",
            "topk_indices_buffer",
            "aux_stream_list",
            "candidate_block_buffer",
        ),
        defaults={
            "topk_indices_buffer": None,
            "aux_stream_list": None,
            "candidate_block_buffer": None,
        },
    )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
        candidate_block_buffer=None,
    ):
        original_init(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer,
            aux_stream_list,
            candidate_block_buffer,
        )
        swa = getattr(self, "swa_cache_layer", None)
        if swa is None:
            raise PatchCompatibilityError(
                f"{TARGET_MODULE}.DeepseekV4Attention.swa_cache_layer is missing "
                "after __init__"
            )
        # Import the HCU module directly: exchange maps
        # ``vllm.v1...sparse_swa`` onto it when plugins are loaded, but this
        # patch must work even when tests disable plugins.
        from vllm_hcu.v1.attention.backends.mla.sparse_swa import (
            resolve_swa_bounded_replay,
        )

        swa.bounded_replay = resolve_swa_bounded_replay(vllm_config)

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "__init__", hcu_init)
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
