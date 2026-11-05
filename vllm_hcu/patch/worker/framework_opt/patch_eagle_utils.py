# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU compatibility fixes for the Eagle model loader."""

from __future__ import annotations

import functools
from copy import copy
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
PATCH_ID = "worker.framework_opt.spec_decode.eagle_topk_buffer"
TARGETS = (
    f"{TARGET_MODULE}.load_eagle_model",
    f"{TARGET_MODULE}._should_share",
)
_MARKER = "_vllm_hcu_eagle_topk_buffer_applied"
_WRAPPER = "_vllm_hcu_eagle_topk_buffer_wrapper"
_SHARE_WRAPPER = "_vllm_hcu_eagle_step_mtp_head_wrapper"


def _draft_uses_target_model_geometry(draft_model_config: object) -> bool:
    """Return whether the draft config is only an MTP architecture selector."""
    hf_config = getattr(draft_model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    return getattr(text_config, "num_hidden_layers", None) == 0


def _use_fixed_v4_draft_layout(config: object) -> bool:
    """Isolate native V4 MTP/DSpark from the target's static placement contract."""
    from vllm_hcu.deepseek_v4_runtime import config_field, model_architectures

    spec = config_field(config, "speculative_config")
    draft = config_field(spec, "draft_model_config")
    if draft is None:
        return False
    architectures = model_architectures(config, model_config=draft)
    feature = get_hcu_config(config)
    architecture = {
        "mtp": "DeepSeekV4MTPModel",
        "dspark": "DSparkDraftModel",
    }.get(config_field(spec, "method"))
    if (
        architecture is None
        or architecture not in architectures
        or not feature.expert_map_path
    ):
        return False
    config.parallel_config = copy(config.parallel_config)
    config.parallel_config.enable_eplb = False
    config.parallel_config.eplb_config = copy(config.parallel_config.eplb_config)
    config.parallel_config.eplb_config.num_redundant_experts = 0
    additional = dict(config.additional_config or {})
    additional["hcu"] = feature.with_updates(
        expert_map_path=None, expert_map_record_path=None,
        eplb_disable_rearrange=True,
    ).to_dict()
    config.additional_config = additional
    from vllm_hcu.patch.config import bind_hcu_eplb_config
    bind_hcu_eplb_config(config)
    return True


def _merge_draft_forward_context(
    target_context: dict,
    draft_context: dict,
) -> dict[str, str]:
    """Merge draft layers while keeping target and draft runtime names unique."""
    renamed: dict[str, str] = {}
    reserved = set(target_context)
    for name in draft_context:
        merged_name = name
        if merged_name in reserved:
            merged_name = f"eagle_head.{name}"
            suffix = 2
            while merged_name in reserved:
                merged_name = f"eagle_head.{suffix}.{name}"
                suffix += 1
        renamed[name] = merged_name
        reserved.add(merged_name)

    # Attention custom ops use layer_name to retrieve their metadata and KV
    # cache at runtime, so it must match the merged registry key.
    for name, layer in draft_context.items():
        merged_name = renamed[name]
        if merged_name != name:
            if getattr(layer, "layer_name", None) != name:
                raise RuntimeError(
                    "Cannot namespace colliding Eagle/MTP layer without a "
                    f"matching layer_name: {name}"
                )
            layer.layer_name = merged_name

        sharing_target = getattr(layer, "kv_sharing_target_layer_name", None)
        if sharing_target in renamed:
            layer.kv_sharing_target_layer_name = renamed[sharing_target]

    target_context.update(
        (renamed[name], layer) for name, layer in draft_context.items()
    )
    return renamed


def apply_to_module(module: ModuleType) -> bool:
    eagle = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (eagle, "load_eagle_model", TARGETS[0], _WRAPPER),
        (eagle, "_should_share", TARGETS[1], _SHARE_WRAPPER),
    )
    if already_applied(eagle, _MARKER, wrapped):
        return False
    original = require_callable(eagle, "load_eagle_model", TARGETS[0])
    original_should_share = require_callable(eagle, "_should_share", TARGETS[1])
    require_exact_signature(
        original, TARGETS[0], positional=("target_model", "vllm_config")
    )
    require_exact_signature(
        original_should_share,
        TARGETS[1],
        positional=("eagle", "flag", "draft", "target"),
    )

    @functools.wraps(original_should_share)
    def hcu_should_share(eagle_model, flag, draft, target):
        config = getattr(eagle_model, "config", None)
        if (
            flag == "has_own_lm_head"
            and getattr(config, "model_type", None) == "step3p5_mtp"
        ):
            # Step MTP checkpoints train a distinct output head for every
            # prediction layer. Replacing them with the target lm_head leaves
            # speculative decoding correct but collapses draft acceptance.
            return False
        return original_should_share(eagle_model, flag, draft, target)

    @functools.wraps(original)
    def hcu_load_eagle_model(target_model, vllm_config):
        speculative_config = vllm_config.speculative_config
        draft_model_config = speculative_config.draft_model_config
        # Keep initialized engine configs and worker runtime state intact while
        # presenting the draft model config to the upstream Eagle loader.
        # replace() reconstructs VllmConfig and reruns its validation hooks.
        draft_vllm_config = copy(vllm_config)
        if not _draft_uses_target_model_geometry(draft_model_config):
            draft_vllm_config.model_config = draft_model_config
        target_compilation_config = vllm_config.compilation_config
        draft_compilation_config = copy(target_compilation_config)
        draft_forward_context: dict = {}
        draft_compilation_config.static_forward_context = draft_forward_context
        target_all_moe_layers = getattr(
            target_compilation_config, "static_all_moe_layers", None
        )
        draft_all_moe_layers: list[str] | None = None
        if target_all_moe_layers is not None:
            draft_all_moe_layers = []
            draft_compilation_config.static_all_moe_layers = draft_all_moe_layers
        draft_vllm_config.compilation_config = draft_compilation_config
        fixed_draft = _use_fixed_v4_draft_layout(draft_vllm_config)
        eagle_model = original(target_model, draft_vllm_config)
        if fixed_draft:
            eagle_model._vllm_hcu_fixed_draft_layout = True
        renamed = _merge_draft_forward_context(
            target_compilation_config.static_forward_context,
            draft_forward_context,
        )
        if draft_all_moe_layers is not None:
            missing = [name for name in draft_all_moe_layers if name not in renamed]
            if missing:
                raise RuntimeError(
                    "Cannot merge Eagle/MTP MoE layer names missing from the "
                    f"draft forward context: {missing}"
                )
            target_all_moe_layers.extend(
                renamed[name] for name in draft_all_moe_layers
            )
        if not get_hcu_config(vllm_config).enable_multi_layers_mtp:
            return eagle_model
        from vllm_hcu.v1.worker_framework_runtime import share_eagle_topk_buffer

        return share_eagle_topk_buffer(target_model, eagle_model)

    setattr(hcu_load_eagle_model, _WRAPPER, True)
    setattr(hcu_should_share, _SHARE_WRAPPER, True)
    setattr(eagle, "_vllm_hcu_original_load_eagle_model", original)
    setattr(eagle, "_vllm_hcu_original_should_share", original_should_share)
    setattr(eagle, "_should_share", hcu_should_share)
    setattr(eagle, "load_eagle_model", hcu_load_eagle_model)
    setattr(eagle, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
