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


class HYV4StandardRotaryEmbedding(HYV4DFlyRotaryEmbedding):
    """Keep stock 1-D semantics; adapt M-RoPE's dense-row kernel ABI.

    Native HY4 callers assign returned Q/K into their strided head views.
    No per-forward position cache is shared between target and draft layers.
    """

    def forward(self, positions, query, key):
        if positions.ndim == 1:
            return self.rotary(positions, query, key)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("HY4 M-RoPE positions must have shape [3, tokens]")
        return super().forward(positions, query, key)


def unwrap_dfly_rotary(rotary):
    while isinstance(rotary, HYV4DFlyRotaryEmbedding):
        rotary = rotary.rotary
    return rotary


def validate_dfly_rotary_pair(source, dest):
    source, dest = unwrap_dfly_rotary(source), unwrap_dfly_rotary(dest)
    if type(source) is not type(dest):
        raise ValueError("DFly target and draft use incompatible RoPE types")
    for name in ("head_size", "rotary_dim", "base", "is_neox_style", "mrope_section",
                 "mrope_interleaved", "scaling_factor", "beta_fast", "beta_slow",
                 "mscale", "truncate"):
        if getattr(source, name, None) != getattr(dest, name, None):
            raise ValueError(f"DFly target and draft use incompatible RoPE {name}")
