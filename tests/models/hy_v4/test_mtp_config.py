# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from typing import get_args

import vllm.config.speculative as speculative

from vllm_hcu.models.hy_v4.config import HYV4Config
from vllm_hcu.patch.platform.core_fix import patch_hy_v4_mtp_config


patch_hy_v4_mtp_config.apply_to_module(speculative)
SpeculativeConfig = speculative.SpeculativeConfig


def test_hy_v4_speculative_config_selects_native_mtp_architecture() -> None:
    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
        hc_mult=4,
    )

    actual = SpeculativeConfig.hf_config_override(config)

    assert actual is config
    assert actual.model_type == "hy_v4_mtp"
    assert actual.architectures == ["HYV4MTPModel"]
    assert actual.n_predict == 1
    assert actual.hc_mult == 1
    assert "hy_v4_mtp" in get_args(speculative.MTPModelTypes)


def test_hy_v4_speculative_config_adapter_is_idempotent() -> None:
    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
    )

    first = SpeculativeConfig.hf_config_override(config)
    second = SpeculativeConfig.hf_config_override(first)

    assert second.model_type == "hy_v4_mtp"
    assert second.architectures == ["HYV4MTPModel"]
    assert second.n_predict == 1
