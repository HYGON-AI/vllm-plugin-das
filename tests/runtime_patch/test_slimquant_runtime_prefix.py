# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_marlin import (
    _add_runtime_prefix_ignore_aliases,
)


def test_slimquant_preserves_and_aliases_runtime_prefixed_ignore_rules() -> None:
    original_regex = r"re:^model\.layers\.[012]\..*"
    original_name = "model.embed_tokens"
    ignore = [original_regex, original_name]

    _add_runtime_prefix_ignore_aliases(ignore)
    once = tuple(ignore)
    _add_runtime_prefix_ignore_aliases(ignore)

    assert tuple(ignore) == once
    assert original_regex in ignore
    assert original_name in ignore
    assert r"re:^language_model\.model\.layers\.[012]\..*" in ignore
    assert "language_model.model.embed_tokens" in ignore
    assert should_ignore_layer(
        "language_model.model.layers.0.mlp.experts",
        ignore=ignore,
        fused_mapping={},
    )
