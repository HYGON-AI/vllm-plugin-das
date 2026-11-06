# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light Step5 family predicate shared by lazy adapters."""


def is_step5_config(config):
    model = getattr(config, "model_config", config)
    hf = getattr(model, "hf_config", model)
    text = getattr(hf, "text_config", hf)
    return getattr(text, "model_type", "") in ("step5", "step5_mtp") or any(
        name.startswith("Step5") for name in (getattr(hf, "architectures", None) or ())
    )


def is_step5_owner(owner=None):
    """Resolve ownership at construction time; unknown owners never opt in."""
    marker = getattr(owner, "_hcu_step5_owner", None)
    if marker is not None:
        return marker is True
    config = getattr(owner, "_hcu_vllm_config", None)
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    return config is not None and is_step5_config(config)


def is_step5_pd_config(config):
    """Opt in only separate Step5 Mooncake producers and consumers."""
    transfer = getattr(config, "kv_transfer_config", None)
    return (
        config is not None
        and is_step5_config(config)
        and getattr(transfer, "kv_connector", None)
        in ("MooncakeConnector", "Step5MooncakeConnector")
        and getattr(transfer, "kv_role", None) in ("kv_producer", "kv_consumer")
    )


def configure_step5_pd_producer_compilation(config):
    """Keep fused producer execution free of runtime autotuning device syncs."""
    if (not is_step5_pd_config(config)
            or config.kv_transfer_config.kv_role != "kv_producer"
            or config.model_config.enforce_eager):
        return False
    compilation = config.compilation_config
    if getattr(compilation.cudagraph_mode, "name", None) != "NONE":
        raise ValueError("Compiled Step5 Mooncake producers require cudagraph_mode=NONE")
    from vllm import envs

    # vLLM overwrites these two Inductor options for single-size ranges from
    # its environment getters. Setting only inductor_compile_config is not
    # sufficient, including for the MTP model sharing this configuration.
    if (envs.VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
            or envs.VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING):
        raise ValueError(
            "Compiled Step5 Mooncake producers require "
            "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 and "
            "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0"
        )
    options = compilation.inductor_compile_config
    required = {
        "triton.autotune_at_compile_time": True,
        "coordinate_descent_tuning": False,
        "triton.autotune_pointwise": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "max_autotune_gemm": False,
        "combo_kernels": False,
        "benchmark_combo_kernel": False,
    }
    changed = any(options.get(key) is not value for key, value in required.items())
    options.update(required)
    return changed
