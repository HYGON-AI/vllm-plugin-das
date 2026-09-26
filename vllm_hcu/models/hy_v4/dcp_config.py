# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed configuration contract for Hy4 decode context parallelism."""

from __future__ import annotations

import os

from vllm_hcu.patch.config import get_hcu_config


_HYV4_ARCHITECTURE = "HYV4ForCausalLM"
_SUPPORTED_TOPOLOGY = (8, 2, 1, 1, 1)
_SUPPORTED_CACHE_DTYPES = frozenset({"fp8_e4m3", "fp8_ds_mla"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def validate_hy4_dcp_config(vllm_config: object) -> bool:
    """Validate the first supported Hy4 FP8 DCP2 topology.

    Returns ``False`` for configurations outside Hy4 DCP, allowing their
    existing validation paths to remain authoritative. Invalid Hy4 DCP
    configurations raise before model weights are loaded.
    """

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if model_config.architectures != [_HYV4_ARCHITECTURE]:
        return False
    if parallel_config.decode_context_parallel_size == 1:
        return False

    topology = (
        parallel_config.tensor_parallel_size,
        parallel_config.decode_context_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.data_parallel_size,
    )
    if topology != _SUPPORTED_TOPOLOGY:
        raise ValueError(
            "HY V4 DCP topology is not validated: "
            f"TP/DCP/PCP/PP/DP={topology}; expected {_SUPPORTED_TOPOLOGY}."
        )

    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in _SUPPORTED_CACHE_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_CACHE_DTYPES))
        raise ValueError(
            "HY V4 DCP requires --kv-cache-dtype to be one of: "
            f"{supported}."
        )
    if not vllm_config.use_v2_model_runner or not model_config.use_mla:
        raise ValueError("HY V4 DCP requires MLA Model Runner V2.")

    num_heads = model_config.hf_config.num_attention_heads
    if num_heads != 64:
        raise ValueError("HY V4 DCP requires 64 attention heads.")
    local_heads = num_heads // parallel_config.tensor_parallel_size
    gathered_heads = local_heads * parallel_config.decode_context_parallel_size
    if (local_heads, gathered_heads) != (8, 16):
        raise ValueError(
            "HY V4 DCP FP8 head envelope requires 8 local and 16 gathered "
            "attention heads."
        )

    if parallel_config.dcp_comm_backend != "ag_rs":
        raise ValueError("HY V4 DCP requires dcp_comm_backend=ag_rs.")
    if parallel_config.cp_kv_cache_interleave_size != 1:
        raise ValueError("HY V4 DCP requires cache interleave size 1.")
    if not parallel_config.enable_expert_parallel:
        raise ValueError("HY V4 DCP requires expert parallelism.")
    if vllm_config.kernel_config.moe_backend != "aiter":
        raise ValueError("HY V4 DCP requires the AITER MoE backend.")

    qrep_env = os.getenv("VLLM_DCP_Q_REPLICATE", "0").strip().lower()
    if parallel_config.dcp_q_replicate or qrep_env in _TRUE_VALUES:
        raise ValueError("HY V4 DCP query replication is not validated.")

    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and (
        speculative_config.method != "mtp"
        or speculative_config.num_speculative_tokens != 3
    ):
        raise ValueError("HY V4 DCP supports only built-in MTP3.")

    if vllm_config.lora_config is not None:
        raise ValueError("HY V4 DCP does not support LoRA.")
    if model_config.is_multimodal_model:
        raise ValueError("HY V4 DCP does not support multimodal models.")
    if vllm_config.cache_config.kv_offloading_size is not None:
        raise ValueError("HY V4 DCP does not support KV offload.")
    kv_transfer_config = vllm_config.kv_transfer_config
    if (
        kv_transfer_config is not None
        and getattr(kv_transfer_config, "kv_connector", None) is not None
    ):
        raise ValueError("HY V4 DCP does not support P/D disaggregation.")

    features = get_hcu_config(vllm_config)
    if features.enable_multi_layers_mtp:
        raise ValueError("HY V4 DCP does not support HCU multi-layer MTP.")
    if features.enable_lightly_cp:
        raise ValueError("HY V4 DCP does not support lightly-CP.")
    return True
