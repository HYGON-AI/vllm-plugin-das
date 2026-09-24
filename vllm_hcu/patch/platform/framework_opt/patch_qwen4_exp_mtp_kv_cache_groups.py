# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Identify Qwen hybrid MTP KV groups without classifying target Mamba groups."""

from __future__ import annotations

import functools
import inspect
import logging
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)


TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.qwen4_exp_mtp_kv_cache_groups"
TARGETS = (
    f"{TARGET_MODULE}._annotate_eagle_groups",
    f"{TARGET_MODULE}.get_kv_cache_groups",
    f"{TARGET_MODULE}.get_kv_cache_config_from_groups",
    f"{TARGET_MODULE}._warn_if_unannotated_eagle_mamba",
)
_MARKER = "_vllm_hcu_qwen4_exp_mtp_kv_groups_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_kv_groups_wrapper"
_GROUPS_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_get_kv_groups_wrapper"
_CONFIG_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_get_kv_config_wrapper"
_WARN_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_warn_wrapper"
logger = logging.getLogger(__name__)
_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen4_exp",
    }
)


def _is_qwen_hybrid_mtp(vllm_config: object) -> bool:
    spec_config = getattr(vllm_config, "speculative_config", None)
    use_block_drop = getattr(spec_config, "use_eagle_block_drop", None)
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return (
        callable(use_block_drop)
        and use_block_drop()
        and getattr(hf_config, "model_type", None) in _SUPPORTED_MODEL_TYPES
    )


def _is_mtp_layer(name: object) -> bool:
    return isinstance(name, str) and "mtp" in name.lower().split(".")


def _group_contains_mtp_layer(group: object) -> bool:
    layer_names = getattr(group, "layer_names", ())
    if any(_is_mtp_layer(name) for name in layer_names):
        return True
    if layer_names:
        return False

    # PP projection empties layer_names on stages that do not own this group,
    # while UniformTypeKVCacheSpecs retains the global layer-to-spec mapping.
    group_spec = getattr(group, "kv_cache_spec", None)
    global_specs = getattr(group_spec, "kv_cache_specs", None)
    return isinstance(global_specs, dict) and any(
        _is_mtp_layer(name) for name in global_specs
    )


def _annotate_qwen_mtp_groups(vllm_config, kv_cache_groups) -> None:
    if not _is_qwen_hybrid_mtp(vllm_config):
        return

    # These Qwen architectures register their draft model below the stable
    # ``mtp`` prefix.  Mark every group containing such a layer and leave
    # target Mamba groups untouched so align-mode checkpoints remain reusable.
    for group in kv_cache_groups:
        if _group_contains_mtp_layer(group):
            group.is_eagle_group = True


def apply_to_module(module: ModuleType) -> bool:
    kv_cache_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        kv_cache_utils,
        _MARKER,
        (
            (kv_cache_utils, "_annotate_eagle_groups", _WRAPPER),
            (kv_cache_utils, "get_kv_cache_groups", _GROUPS_WRAPPER),
            (
                kv_cache_utils,
                "get_kv_cache_config_from_groups",
                _CONFIG_WRAPPER,
            ),
            (
                kv_cache_utils,
                "_warn_if_unannotated_eagle_mamba",
                _WARN_WRAPPER,
            ),
        ),
    ):
        return False

    original = require_callable(
        kv_cache_utils, "_annotate_eagle_groups", TARGETS[0]
    )
    signature = inspect.signature(original)
    if (
        tuple(signature.parameters)
        != (
            "vllm_config",
            "kv_cache_spec",
            "kv_cache_groups",
            "use_deepseek_v4_fallback",
        )
        or signature.parameters["use_deepseek_v4_fallback"].default is not False
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    original_get_groups = require_callable(
        kv_cache_utils, "get_kv_cache_groups", TARGETS[1]
    )
    get_groups_signature = inspect.signature(original_get_groups)
    if tuple(get_groups_signature.parameters) != ("vllm_config", "kv_cache_spec"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {get_groups_signature}"
        )
    original_get_config = require_callable(
        kv_cache_utils, "get_kv_cache_config_from_groups", TARGETS[2]
    )
    uniform_specs_type = require_class(
        kv_cache_utils,
        "UniformTypeKVCacheSpecs",
        f"{TARGET_MODULE}.UniformTypeKVCacheSpecs",
    )
    get_config_signature = inspect.signature(original_get_config)
    if tuple(get_config_signature.parameters) != (
        "vllm_config",
        "kv_cache_groups",
        "available_memory",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} has incompatible "
            f"signature {get_config_signature}"
        )
    original_warn = require_callable(
        kv_cache_utils,
        "_warn_if_unannotated_eagle_mamba",
        TARGETS[3],
    )
    warn_signature = inspect.signature(original_warn)
    if tuple(warn_signature.parameters) != (
        "vllm_config",
        "kv_cache_groups",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[3]} has incompatible "
            f"signature {warn_signature}"
        )

    @functools.wraps(original_warn)
    def hcu_warn_if_unannotated_eagle_mamba(
        vllm_config,
        kv_cache_groups,
    ) -> None:
        _annotate_qwen_mtp_groups(vllm_config, kv_cache_groups)
        return original_warn(vllm_config, kv_cache_groups)

    @functools.wraps(original)
    def hcu_annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ) -> None:
        original(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_deepseek_v4_fallback,
        )
        _annotate_qwen_mtp_groups(vllm_config, kv_cache_groups)

    @functools.wraps(original_get_groups)
    def hcu_get_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = original_get_groups(vllm_config, kv_cache_spec)
        _annotate_qwen_mtp_groups(vllm_config, groups)
        if _is_qwen_hybrid_mtp(vllm_config):
            logger.info(
                "Qwen hybrid MTP KV cache groups after Eagle annotation: %s",
                [
                    {
                        "index": index,
                        "layers": tuple(group.layer_names),
                        "spec": type(group.kv_cache_spec).__name__,
                        "is_eagle_group": group.is_eagle_group,
                    }
                    for index, group in enumerate(groups)
                ],
            )
        return groups

    @functools.wraps(original_get_config)
    def hcu_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        available_memory,
    ):
        config = original_get_config(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if (
            not _is_qwen_hybrid_mtp(vllm_config)
            or getattr(parallel_config, "pipeline_parallel_size", 1) <= 1
        ):
            return config

        # Upstream keeps the global UniformTypeKVCacheSpecs on a PP-projected
        # group even when that stage owns none of the group's layers.  The
        # config builder then emits tensors for those remote layers, while the
        # group's local layer_names is empty; allocate_kv_cache cannot map such
        # a tensor back to any group.  Preserve the empty group (the scheduler
        # needs a stable global group index) but remove only tensors proven to
        # belong exclusively to that empty projection.
        owned_layers = {
            layer_name
            for group in kv_cache_groups
            for layer_name in group.layer_names
        }
        remote_layers: set[str] = set()
        for group in kv_cache_groups:
            if group.layer_names:
                continue
            group_spec = group.kv_cache_spec
            if not isinstance(group_spec, uniform_specs_type):
                continue
            remote_spec = group_spec.kv_cache_specs
            if not isinstance(remote_spec, dict) or not remote_spec:
                raise RuntimeError(
                    "Qwen hybrid MTP PP found an incompatible empty "
                    "UniformType KV group"
                )
            # PP projection clears the Eagle marker when this worker owns no
            # layer in the group.  The scheduler builds its configuration from
            # the first worker, so restore the marker from the retained global
            # spec names even though this stage allocates no draft KV tensor.
            if any(_is_mtp_layer(layer_name) for layer_name in remote_spec):
                group.is_eagle_group = True
            remote_layers.update(remote_spec)

        duplicated_layers = owned_layers.intersection(remote_layers)
        if duplicated_layers:
            raise RuntimeError(
                "Qwen hybrid MTP PP KV layers are both locally owned and "
                f"remote: {sorted(duplicated_layers)}"
            )

        local_tensors = []
        for tensor in config.kv_cache_tensors:
            tensor_layers = set(tensor.layers)
            if not tensor_layers:
                raise RuntimeError(
                    "Qwen hybrid MTP PP found a KV tensor without layers"
                )
            local = tensor_layers.intersection(owned_layers)
            remote = tensor_layers.intersection(remote_layers)
            unknown = tensor_layers.difference(owned_layers, remote_layers)
            if unknown:
                raise RuntimeError(
                    "Qwen hybrid MTP PP found unowned KV tensor layers: "
                    f"{sorted(unknown)}"
                )
            if local and remote:
                raise RuntimeError(
                    "Qwen hybrid MTP PP KV tensor mixes local and remote "
                    f"layers: {tensor.layers}"
                )
            if local:
                local_tensors.append(tensor)
        config.kv_cache_tensors = local_tensors
        return config

    setattr(hcu_annotate_eagle_groups, _WRAPPER, True)
    setattr(hcu_get_kv_cache_groups, _GROUPS_WRAPPER, True)
    setattr(hcu_get_kv_cache_config_from_groups, _CONFIG_WRAPPER, True)
    setattr(hcu_warn_if_unannotated_eagle_mamba, _WARN_WRAPPER, True)
    setattr(kv_cache_utils, "_vllm_hcu_original_annotate_eagle_groups", original)
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_get_kv_cache_groups",
        original_get_groups,
    )
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_get_kv_cache_config_from_groups",
        original_get_config,
    )
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_warn_if_unannotated_eagle_mamba",
        original_warn,
    )
    setattr(kv_cache_utils, "_annotate_eagle_groups", hcu_annotate_eagle_groups)
    setattr(kv_cache_utils, "get_kv_cache_groups", hcu_get_kv_cache_groups)
    setattr(
        kv_cache_utils,
        "get_kv_cache_config_from_groups",
        hcu_get_kv_cache_config_from_groups,
    )
    setattr(
        kv_cache_utils,
        "_warn_if_unannotated_eagle_mamba",
        hcu_warn_if_unannotated_eagle_mamba,
    )
    setattr(kv_cache_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
