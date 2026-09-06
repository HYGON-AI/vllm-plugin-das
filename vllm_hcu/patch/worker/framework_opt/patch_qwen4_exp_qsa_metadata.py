# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable replay-safe fused MTP metadata refresh for Qwen4Exp QSA caches."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)


TARGET_MODULE = "vllm.models.qwen4_exp.common.qsa_cache"
PATCH_ID = "worker.framework_opt.spec_decode.qwen4_exp_qsa_metadata"
TARGETS = (
    f"{TARGET_MODULE}.QSAMetadataBuilder.build",
    f"{TARGET_MODULE}.QSAMetadataBuilder.update_draft_decode_metadata",
)
_MARKER = "_vllm_hcu_qwen4_exp_qsa_metadata_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_qsa_metadata_wrapper"
_COMMON_METADATA = "_vllm_hcu_qsa_draft_common_attn_metadata"


def apply_to_module(module: ModuleType) -> bool:
    qsa_cache = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(
        qsa_cache,
        "QSAMetadataBuilder",
        f"{TARGET_MODULE}.QSAMetadataBuilder",
    )
    wrapped = (
        (builder_cls, "build", TARGETS[0], _WRAPPER),
        (
            builder_cls,
            "update_draft_decode_metadata",
            TARGETS[1],
            _WRAPPER,
        ),
    )
    if already_applied(qsa_cache, _MARKER, wrapped):
        if not builder_cls.supports_draft_decode_metadata_update:
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[1]} is stale"
            )
        return False

    # A newer official wheel may provide this hook itself. Do not shadow it.
    if bool(getattr(builder_cls, "supports_draft_decode_metadata_update", False)):
        return False
    if "update_draft_decode_metadata" in vars(builder_cls):
        raise PatchCompatibilityError(
            f"audited target {TARGETS[1]} unexpectedly has a local implementation"
        )

    original_build = require_callable(builder_cls, "build", TARGETS[0])
    require_exact_signature(
        original_build,
        TARGETS[0],
        positional=(
            "self",
            "common_prefix_len",
            "common_attn_metadata",
            "fast_build",
        ),
        defaults={"fast_build": False},
    )
    original_update = require_callable(
        builder_cls, "update_draft_decode_metadata", TARGETS[1]
    )
    require_exact_signature(
        original_update,
        TARGETS[1],
        positional=("self", "metadata"),
    )

    @functools.wraps(original_build)
    def hcu_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
    ):
        metadata = original_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        # The fused loop mutates this common metadata's persistent tensors
        # between steps. Retain the object so the official QSA builder can
        # refresh every derived view/buffer without duplicating its formulas.
        setattr(self, _COMMON_METADATA, common_attn_metadata)
        return metadata

    def hcu_update_draft_decode_metadata(self, metadata) -> None:
        common_attn_metadata = getattr(self, _COMMON_METADATA, None)
        if common_attn_metadata is None:
            raise RuntimeError(
                "QSA draft metadata update was called before metadata build"
            )

        refreshed = original_build(
            self,
            0,
            common_attn_metadata,
            True,
        )
        if type(refreshed) is not type(metadata) or not hasattr(metadata, "__dict__"):
            raise RuntimeError("QSA metadata build returned an incompatible object")

        # Preserve the metadata object's identity held by captured forwards,
        # while adopting the official builder's refreshed persistent views.
        vars(metadata).update(vars(refreshed))

    setattr(hcu_build, _WRAPPER, True)
    setattr(hcu_update_draft_decode_metadata, _WRAPPER, True)
    setattr(builder_cls, "_vllm_hcu_original_build", original_build)
    setattr(builder_cls, "_vllm_hcu_original_update_draft_metadata", original_update)
    setattr(builder_cls, "build", hcu_build)
    setattr(
        builder_cls,
        "update_draft_decode_metadata",
        hcu_update_draft_decode_metadata,
    )
    builder_cls.supports_draft_decode_metadata_update = True
    setattr(qsa_cache, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
