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
