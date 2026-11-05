# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Batch SWA decode-meta Triton launches across KV-cache groups.

``DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build`` runs once per SWA group
(~6 target + ~3 draft). Even with fused ``_decode_swa_meta_kernel``, serial
Python between launches left ~300us+ GPU idle gaps on HCU. Wrap
``build_attn_metadata`` so every eligible group enqueues into
``swa_meta_launch_batch`` and flushes as one multi-program grid.

Non-SWA builders (MLA, …) flush pending SWA jobs first so the batch grid can
overlap with their host-side prep instead of sitting until prepare_attn ends.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.v1.worker.gpu.attn_utils"
PATCH_ID = "worker.op_opt.deepseek_v41.swa_meta_batch"
TARGETS = (f"{TARGET_MODULE}.build_attn_metadata",)
_MARKER = "_vllm_hcu_dsv41_swa_meta_batch_applied"
_WRAPPER = "_vllm_hcu_dsv41_swa_meta_batch_wrapper"

_REBIND_MODULES = (
    "vllm.v1.worker.gpu.attn_utils",
    "vllm.v1.worker.gpu.model_states.default",
    "vllm.v1.worker.gpu.model_states.mamba_hybrid",
    "vllm.v1.worker.gpu.model_states.encoder_decoder",
    "vllm_hcu.patch.worker.core_fix.patch_deepseek_v41_decoder_replay_state",
)


def _rebind(name: str, value) -> None:
    for mod_name in _REBIND_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, name):
            setattr(mod, name, value)


def _wrap_builders(attn_groups) -> None:
    from vllm_hcu.v1.attention.ops.deepseek_v41_swa_metadata import (
        ensure_builder_flush_wrap,
    )

    if not attn_groups:
        return
    for group_list in attn_groups:
        for attn_group in group_list or ():
            builders = getattr(attn_group, "metadata_builders", None)
            if builders:
                for builder in builders:
                    ensure_builder_flush_wrap(builder)
                continue
            getter = getattr(attn_group, "get_metadata_builder", None)
            if getter is None:
                continue
            try:
                ensure_builder_flush_wrap(getter(0))
            except Exception:
                continue


def apply_to_module(module: ModuleType) -> bool:
    utils = load_exact_module(TARGET_MODULE, module)
    wrapped = ((utils, "build_attn_metadata", TARGETS[0], _WRAPPER),)
    if already_applied(utils, _MARKER, wrapped):
        return False

    original = require_callable(utils, "build_attn_metadata", TARGETS[0])

    from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import (
        decode_metadata_build_scope,
    )

    @functools.wraps(original)
    def build_attn_metadata(*args, **kwargs):
        from vllm_hcu.v1.attention.ops.deepseek_v41_swa_metadata import (
            swa_meta_launch_batch,
        )

        attn_groups = args[0] if args else kwargs.get("attn_groups")
        _wrap_builders(attn_groups)
        with decode_metadata_build_scope(), swa_meta_launch_batch():
            return original(*args, **kwargs)

    setattr(build_attn_metadata, _WRAPPER, True)
    utils.build_attn_metadata = build_attn_metadata
    _rebind("build_attn_metadata", build_attn_metadata)
    setattr(utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
