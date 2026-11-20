# SPDX-License-Identifier: Apache-2.0
"""Keep official AMD Qwen4 models; discard only disabled QSA indexer weights.

The public HF override selects dense attention in the official constructor.
No constructor, cache-scale mapping, forward, attention or MTP code is copied.
"""
from __future__ import annotations

import re

from vllm.models.qwen4_exp.amd.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP
from vllm.transformers_utils.config import get_hf_text_config

from ..qwen4_dense_activation import check_health

# Accepted raw language-model/draft checkpoint namespaces, before the official
# mapper. Anchor the path so visual/indexer-like unknown modules remain errors.
_INDEXER = re.compile(
    r"(?:(?:model\.language_model\.|language_model\.)(?:model\.|mtp\.)?"
    r"|model\.(?:mtp\.)?|mtp\.)"
    r"layers\.[0-9]+\.self_attn\.indexer\..+"
)


def _dense_weights(config, weights):
    text = get_hf_text_config(config)
    if getattr(text, "indexer_n_heads", None) is not None:
        return weights
    return ((name, value) for name, value in weights
            if _INDEXER.fullmatch(name) is None)


class _DenseCheckpointLoader:
    def load_weights(self, weights):
        check_health()
        return super().load_weights(_dense_weights(self.config, weights))


class HcuQwen4ExpForCausalLM(_DenseCheckpointLoader, Qwen4ExpForCausalLM):
    pass


class HcuQwen4ExpForConditionalGeneration(
    _DenseCheckpointLoader, Qwen4ExpForConditionalGeneration
):
    pass


class HcuQwen4ExpMTP(_DenseCheckpointLoader, Qwen4ExpMTP):
    pass
