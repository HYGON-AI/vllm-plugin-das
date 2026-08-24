# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_hcu.ops.rotary_embedding import HcuRotaryEmbedding


@pytest.mark.parametrize("is_neox_style", [False, True])
def test_hcu_rotary_uses_native_accuracy_path(is_neox_style: bool) -> None:
    with set_current_vllm_config(VllmConfig()):
        op = HcuRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=10_000,
            is_neox_style=is_neox_style,
            dtype=torch.float32,
        )
    positions = torch.tensor([0, 3, 7], dtype=torch.long)
    query = torch.randn(3, 2, 8, dtype=torch.float32)
    key = torch.randn(3, 1, 8, dtype=torch.float32)

    expected_query, expected_key = op.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )
    actual_query, actual_key = op.forward_hip(
        positions,
        query.clone(),
        key.clone(),
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
