# SPDX-License-Identifier: Apache-2.0
"""Dependency-light selection policy; unrelated models retain their old backend."""

import os
from vllm_hcu.runtime_compat.step5 import is_step5_config, is_step5_owner


def profile_for_model(config):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_config(config):
        return None
    if phase not in ("prefill", "decode"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill or decode")
    model = config.model_config
    hf = model.hf_config
    quantization = getattr(model, "quantization", None)
    if quantization is None:
        quantization = (getattr(hf, "quantization_config", None) or {}).get(
            "quant_method"
        )
    if quantization != "slimquant_w4a8":
        raise ValueError(
            "Step5 native W4A8 requires a SlimQuant packed W4A8 checkpoint"
        )
    parallel = config.parallel_config
    expected = (1, 8, 1) if phase == "prefill" else (4, 2, 1)
    if (
        (
            parallel.tensor_parallel_size,
            parallel.data_parallel_size,
            parallel.pipeline_parallel_size,
        )
        != expected
        or not parallel.enable_expert_parallel
        or parallel.enable_eplb
    ):
        raise ValueError(f"Step5 {phase} requires static TP/DP/PP={expected}, EP8")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("Step5 native W4A8 LoRA has not been enabled")
    return phase


def resolve_profile(moe):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_owner(moe):
        return None
    activation = getattr(moe.activation, "value", moe.activation)
    if activation != "swiglustep":
        return None
    if phase not in ("prefill", "decode"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill or decode")
    parallel = moe.moe_parallel_config
    if (
        parallel.enable_eplb
        or parallel.all2all_backend != "allgather_reducescatter"
        or parallel.ep_size != 8
        or parallel.dp_size != (8 if phase == "prefill" else 2)
    ):
        raise ValueError(
            "Step5 native W4A8 requires static EP8 AG/RS and matching P/D ranks"
        )
    return phase


def padded_route_rows(tokens, topk, experts, alignment=64):
    routes = tokens * topk
    return (
        (routes + min(routes, experts) * (alignment - 1) + alignment - 1)
        // alignment
        * alignment
    )
