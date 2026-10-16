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
