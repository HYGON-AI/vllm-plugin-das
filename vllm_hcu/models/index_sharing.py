# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared sparse-indexer configuration helpers."""

from transformers import PretrainedConfig


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return layers that reuse a preceding layer's sparse TopK indices."""
    if not hasattr(config, "index_topk") and not hasattr(config, "indexer_types"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                "indexer_types only supports 'full' and 'shared', "
                f"got {invalid_types}."
            )
        seen_full = False
        for layer_idx, indexer_type in enumerate(indexer_types):
            if indexer_type == "full":
                seen_full = True
            elif not seen_full:
                raise ValueError(
                    "A 'shared' indexer requires a preceding 'full' producer; "
                    f"layer {layer_idx} has none."
                )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


__all__ = ["compute_skip_topk_layers"]
