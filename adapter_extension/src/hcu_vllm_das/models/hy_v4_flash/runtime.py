# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small runtime predicates shared by the HYV4 Flash plugin modules."""

from vllm.platforms import current_platform


def is_hcu_platform() -> bool:
    """Return whether vLLM is using the HCU out-of-tree platform."""
    return type(current_platform).__module__ == "hcu_adapter.bridges.vllm"


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
