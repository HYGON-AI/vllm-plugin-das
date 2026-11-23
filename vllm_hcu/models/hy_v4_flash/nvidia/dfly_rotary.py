# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from torch import nn


class HYV4DFlyRotaryEmbedding(nn.Module):
    """Adapt old HCU MLA's head views to Core's dense-row M-RoPE ABI.

    Keep the cached rotary module untouched: target layers may share it.
    reshape intentionally materializes non-contiguous q_pe head slices.
    """

    def __init__(self, rotary: nn.Module):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        query_shape, key_shape = query.shape, key.shape
        query, key = self.rotary(
            positions, query.reshape(query_shape[0], -1),
            key.reshape(key_shape[0], -1),
        )
        return query.reshape(query_shape), key.reshape(key_shape)
