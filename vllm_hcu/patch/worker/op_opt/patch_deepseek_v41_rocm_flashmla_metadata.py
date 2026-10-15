# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attach isolated native decode schedulers to V4.1 ROCm SWA metadata.

Wrap the final ROCm builder, retaining V4.1/PCP indices and original ragged
fallback buffers. Do not bypass the inheritance chain or change V4 builders.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_exact_signature,
    require_unpatched,
)
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import (
    SCHEDULERS_ATTR,
    DecodeScheduler,
    decode_context_supported,
    decode_query_limit,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
PATCH_ID = "worker.op_opt.deepseek_v41.flashmla_decode_metadata"
_CLASS_TARGET = f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder"
_BUILD_TARGET = f"{_CLASS_TARGET}.build"
TARGETS = (_BUILD_TARGET,)
_MARKER = "_vllm_hcu_dsv41_flashmla_metadata_applied"
_WRAPPER = "_vllm_hcu_dsv41_flashmla_metadata_build_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "DeepseekV4ROCMAiterSparseSWAMetadataBuilder", _CLASS_TARGET,
    )
    wrapped = ((cls, "build", _BUILD_TARGET, _WRAPPER),)
    if already_applied(cls, _MARKER, wrapped):
        return False
    original = require_unpatched(cls, "build", _BUILD_TARGET, _WRAPPER)
    require_exact_signature(
        original, _BUILD_TARGET,
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )

    @functools.wraps(original)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original(self, common_prefix_len, common_attn_metadata, fast_build)
        # Always clear on build, even if the original builder returns a reused
        # Python object. Disabled/unsupported builds must retain no old leases.
        setattr(metadata, SCHEDULERS_ATTR, None)
        if (
            not henvs.dsv41_flashmla_decode_enabled()
            or not decode_context_supported(self.vllm_config)
            or metadata.num_decode_tokens == 0
            or not 1 <= metadata.max_decode_query_len <= decode_query_limit(self.vllm_config)
            or metadata.decode_swa_indices is None
            or not runtime._supported_device(metadata.decode_swa_indices)
        ):
            return metadata
        bindings = runtime._decode_bindings()
        if bindings is None:
            return metadata
        # layer_names are the model-owned SWA cache prefixes, including target
        # or draft namespace. A separate lease per name avoids unsafe same-ratio
        # sharing; the existing per-type holders remain untouched for fallback.
        schedulers = {
            name: DecodeScheduler(
                bindings[2], allow_graph=henvs.dsv41_flashmla_decode_graph_enabled(),
                query_limit=decode_query_limit(self.vllm_config),
                speculative=self.vllm_config.speculative_config is not None,
            )
            for name in self.layer_names
        }
        setattr(metadata, SCHEDULERS_ATTR, schedulers)
        return metadata

    setattr(build, _WRAPPER, True)
    cls._vllm_hcu_original_flashmla_metadata_build = original
    cls.build = build
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
