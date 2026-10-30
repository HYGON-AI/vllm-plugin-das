# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attach isolated native decode schedulers to V4.1 ROCm SWA metadata.

Wrap the final ROCm builder, retaining V4.1/PCP indices and original ragged
fallback buffers. Do not bypass the inheritance chain or change V4 builders.

Installed ROCm ``DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build`` may omit
``replay_start`` while its parent (plugin ``DeepseekSparseSWAMetadataBuilder`` /
``DeepseekV41SparseSWAMetadataBuilder``) already accepts it. In that case this
patch forwards ``replay_start`` through the parent, then reapplies the ROCm
ragged post-process so SWA bounded replay works without editing vLLM sources.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_exact_signature,
    require_unpatched,
    PatchCompatibilityError,
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


def _build_accepts_replay_start(fn) -> bool:
    try:
        return "replay_start" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _parent_build_with_replay(cls):
    """First MRO parent whose ``build`` accepts ``replay_start``."""
    for base in cls.__mro__[1:]:
        parent_build = getattr(base, "build", None)
        if parent_build is None or not callable(parent_build):
            continue
        if _build_accepts_replay_start(parent_build):
            return parent_build
    return None


def _enrich_rocm_ragged(self, base, rocm_module: ModuleType):
    """Mirror installed ROCm ``build`` post-``super()`` ragged packing."""
    build_ragged = getattr(rocm_module, "build_ragged_indices_from_dense", None)
    copy_ragged = getattr(rocm_module, "_copy_ragged_to_graph_buffers", None)
    meta_cls = getattr(rocm_module, "DeepseekV4ROCMAiterSparseSWAMetadata", None)
    if build_ragged is None or copy_ragged is None or meta_cls is None:
        raise TypeError(
            f"{_BUILD_TARGET}: cannot forward replay_start; ROCm module is "
            "missing ragged helpers or DeepseekV4ROCMAiterSparseSWAMetadata"
        )
    ragged_indices = None
    ragged_indptr = None
    if (
        getattr(base, "num_decode_tokens", 0) > 0
        and getattr(base, "decode_swa_indices", None) is not None
        and getattr(base, "decode_swa_lens", None) is not None
    ):
        ragged_indices, ragged_indptr = build_ragged(
            base.decode_swa_indices.reshape(
                base.num_decode_tokens, base.decode_swa_width
            ),
            base.decode_swa_lens,
        )
        ragged_indices, ragged_indptr = copy_ragged(
            ragged_indices,
            ragged_indptr,
            self.decode_swa_ragged_indices_buffer,
            self.decode_swa_ragged_indptr_buffer,
            base.num_decode_tokens,
            base.decode_swa_width,
        )
    return meta_cls(
        **vars(base),
        decode_swa_ragged_indices=ragged_indices,
        decode_swa_ragged_indptr=ragged_indptr,
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "DeepseekV4ROCMAiterSparseSWAMetadataBuilder", _CLASS_TARGET,
    )
    wrapped = ((cls, "build", _BUILD_TARGET, _WRAPPER),)
    if already_applied(cls, _MARKER, wrapped):
        return False
    original = require_unpatched(cls, "build", _BUILD_TARGET, _WRAPPER)
    # Prefer SWA bounded-replay signature; fall back to pre-port ROCm build.
    try:
        require_exact_signature(
            original,
            _BUILD_TARGET,
            positional=(
                "self",
                "common_prefix_len",
                "common_attn_metadata",
                "fast_build",
                "replay_start",
            ),
            defaults={"fast_build": False, "replay_start": None},
        )
        supports_replay_start = True
    except PatchCompatibilityError:
        require_exact_signature(
            original,
            _BUILD_TARGET,
            positional=(
                "self",
                "common_prefix_len",
                "common_attn_metadata",
                "fast_build",
            ),
            defaults={"fast_build": False},
        )
        supports_replay_start = False

    parent_with_replay = None if supports_replay_start else _parent_build_with_replay(cls)

    @functools.wraps(original)
    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        replay_start=None,
    ):
        if supports_replay_start:
            metadata = original(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                replay_start=replay_start,
            )
        elif replay_start is None:
            metadata = original(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
            )
        else:
            # Installed ROCm build drops replay_start; parent already has it.
            if parent_with_replay is None:
                raise TypeError(
                    f"{_BUILD_TARGET} does not accept replay_start and no MRO "
                    "parent forwards it; cannot enable SWA bounded replay."
                )
            base = parent_with_replay(
                self,
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                replay_start,
            )
            metadata = _enrich_rocm_ragged(self, base, target)
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
