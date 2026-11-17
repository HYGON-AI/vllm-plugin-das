# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small runtime predicates shared by the HYV4 Flash plugin modules."""

from vllm.platforms import current_platform


def is_hcu_platform() -> bool:
    """Return whether vLLM is using the HCU out-of-tree platform."""
    return type(current_platform).__module__ == "vllm_hcu.platforms.hcu"


def get_paged_mqa_provider():
    """Select explicit HCU compatibility without rebinding vLLM/DeepGEMM."""
    if is_hcu_platform():
        try:
            from hcu_adapter.bridges.vllm_paged_mqa import get_paged_mqa_provider
        except ImportError as error:
            raise RuntimeError(
                "HYV4 Flash on HCU requires adapter-das with the non-MX "
                "FP8 paged-MQA bridge (hcu_adapter.bridges.vllm_paged_mqa)"
            ) from error
        return get_paged_mqa_provider()
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
        has_deep_gemm,
    )

    if not has_deep_gemm():
        raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
    return get_paged_mqa_logits_metadata, fp8_fp4_paged_mqa_logits


def validate_flash_platform() -> None:
    """Validate that the active platform has a Flash MLA implementation."""
    if is_hcu_platform():
        # HCU validates its actual GCN architecture in the registered sparse
        # MLA backend; CUDA compute capability numbers do not apply.
        return
    capability = current_platform.get_device_capability()
    if capability is None or capability.major not in (9, 10):
        raise NotImplementedError(
            "Flash MLA requires Hopper or Blackwell data-center GPUs"
        )


def _validate_block_fp8_config(config: object) -> None:
    """Validate the vLLM 0.33 compressed-tensors FP8 contract."""
    error = (
        "HYV4 Flash requires compressed-tensors W8A8 FP8 with 128x128 "
        "weight blocks, dynamic group-128 activations, and no KV-cache "
        "scheme or transforms"
    )
    if not isinstance(config, dict) or (
        config.get("kv_cache_scheme") is not None
        or config.get("transform_config") is not None
    ):
        raise ValueError(error)
    if config.get("quant_method") == "fp8":
        if (
            config.get("activation_scheme") != "dynamic"
            or config.get("weight_block_size") != [128, 128]
            or config.get("fmt") not in (None, "e4m3")
            or config.get("store_dtype") not in (None, "fp8")
            or config.get("expert_dtype") not in (None, "fp8")
        ):
            raise ValueError(error)
        return
    if (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "float-quantized"
    ):
        raise ValueError(error)
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    expected_weight = dict(
        num_bits=8, type="float", strategy="block",
        block_structure=[128, 128], symmetric=True, dynamic=False,
    )
    expected_activation = dict(
        num_bits=8, type="float", strategy="group", group_size=128,
        symmetric=True, dynamic=True,
    )
    for group in groups.values():
        if not isinstance(group, dict):
            raise ValueError(error)
        weights = group.get("weights")
        activations = group.get("input_activations")
        if (
            not isinstance(weights, dict)
            or not isinstance(activations, dict)
            or any(weights.get(k) != v for k, v in expected_weight.items())
            or any(activations.get(k) != v for k, v in expected_activation.items())
            or group.get("output_activations") is not None
            or group.get("format") not in (None, "float-quantized")
        ):
            raise ValueError(error)
