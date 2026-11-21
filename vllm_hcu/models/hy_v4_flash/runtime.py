# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small runtime predicates shared by the HYV4 Flash plugin modules."""

from vllm.platforms import current_platform


def is_hcu_platform() -> bool:
    """Return whether vLLM is using the HCU out-of-tree platform."""
    return type(current_platform).__module__ in {
        "vllm_hcu.platforms.hcu",
        "hcu_adapter.bridges.vllm",
    }


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


def _validate_fp8_config(config: object) -> None:
    """Validate supported vLLM FP8 contracts without changing checkpoint scales."""
    error = (
        "HYV4 Flash requires W8A8 FP8 with 128x128 weight blocks and "
        "dynamic group-128 activations, or channel weights and dynamic "
        "per-token activations, and no KV-cache "
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
    block_weight = dict(
        num_bits=8, type="float", strategy="block",
        block_structure=[128, 128], symmetric=True, dynamic=False,
    )
    block_activation = dict(
        num_bits=8, type="float", strategy="group", group_size=128,
        symmetric=True, dynamic=True,
    )
    channel_weight = dict(
        num_bits=8, type="float", strategy="channel",
        symmetric=True, dynamic=False,
    )
    token_activation = dict(
        num_bits=8, type="float", strategy="token",
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
            or not any(
                all(weights.get(k) == v for k, v in weight.items())
                and all(activations.get(k) == v for k, v in activation.items())
                and (
                    weight is block_weight
                    or (
                        weights.get("block_structure") is None
                        and weights.get("group_size") is None
                        and activations.get("block_structure") is None
                        and activations.get("group_size") is None
                    )
                )
                for weight, activation in (
                    (block_weight, block_activation),
                    (channel_weight, token_activation),
                )
            )
            or group.get("output_activations") is not None
            or group.get("format") not in (None, "float-quantized")
        ):
            raise ValueError(error)


def _validate_channel_w4a8_config(config: object) -> None:
    """Admit only CT channel INT4 experts plus channel INT8 linears.

    This does not remove activation metadata or select a W4A16 implementation.
    Runtime selection and model qualification are separate gates.
    """
    error = "HYV4 Flash W4A8 requires symmetric channel INT4/INT8 weights, dynamic token INT8 activations, and packed CT metadata"
    if not isinstance(config, dict) or (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "pack-quantized"
        or config.get("quantization_status") != "compressed"
        or config.get("kv_cache_scheme") is not None
        or config.get("transform_config") is not None
    ):
        raise ValueError(error)
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    bits_seen = set()
    for group in groups.values():
        if not isinstance(group, dict):
            raise ValueError(error)
        w, a = group.get("weights"), group.get("input_activations")
        if not isinstance(w, dict) or not isinstance(a, dict):
            raise ValueError(error)
        bits = w.get("num_bits")
        if type(bits) is not int or bits not in (4, 8):
            raise ValueError(error)
        expected_format = "pack-quantized" if bits == 4 else "int-quantized"
        if (
            group.get("format") != expected_format
            or group.get("output_activations") is not None
            or not isinstance(group.get("targets"), list)
            or not group["targets"]
            or any(not isinstance(t, str) or not t for t in group["targets"])
            or any(w.get(k) != v for k, v in dict(
                type="int", strategy="channel", symmetric=True, dynamic=False,
            ).items())
            or any(a.get(k) != v for k, v in dict(
                num_bits=8, type="int", strategy="token", symmetric=True,
                dynamic=True,
            ).items())
            or any(q.get(k) is not None for q in (w, a)
                   for k in ("group_size", "block_structure", "actorder", "scale_dtype", "zp_dtype"))
            or any(q.get("observer_kwargs") not in (None, {}) for q in (w, a))
        ):
            raise ValueError(error)
        bits_seen.add(bits)
    if bits_seen != {4, 8}:
        raise ValueError(error)


def validate_flash_quant_config(config: object) -> str:
    """Return the admitted storage family without changing the source config."""
    if isinstance(config, dict) and config.get("format") == "pack-quantized":
        _validate_channel_w4a8_config(config)
        return "channel_w4a8"
    _validate_fp8_config(config)
    return "fp8"


def get_engram_cpu_offload(storage: object | None) -> bool:
    """Absent Core Engram settings mean TP-local device tables for Flash."""
    if storage is None:
        return False
    if any(getattr(storage, name, False) for name in (
        "disk_offload", "embedding_across_dp", "dp_shared_memory",
    )):
        raise ValueError(
            "Flash Engram supports TP-local tables, not DP or disk offload"
        )
    return storage.cpu_offload
