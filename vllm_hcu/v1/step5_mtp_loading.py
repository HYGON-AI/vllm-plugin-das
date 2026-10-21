# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    VllmConfig,
    _should_share,
    get_model,
    get_target_lm_head,
    maybe_share_target_embed,
    nn,
    replace,
    torch,
)


def _should_share_mtp_head(eagle: nn.Module, head: nn.Module, target) -> bool:
    """Share an MTP layer's shared_head.head unless the draft declares its own
    trained per-layer heads and this one is loaded (no NaN) and distinct."""

    w = getattr(head, "weight", None)
    if not isinstance(w, torch.Tensor):
        return True
    if getattr(eagle, "has_own_mtp_heads", False) and torch.isnan(w).any():
        return True
    return _should_share(eagle, "has_own_mtp_heads", head, target)


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    if speculative_config.kv_cache_dtype is not None:
        vllm_config = replace(
            vllm_config,
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            ),
        )
    if speculative_config.attention_backend is not None:
        # Before get_model(): the backend is read off the constructed layers.
        # Only when set, so the draft keeps a KV cache layout the target shares.
        vllm_config = replace(
            vllm_config,
            attention_config=replace(
                vllm_config.attention_config,
                backend=speculative_config.attention_backend,
            ),
        )
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    maybe_share_target_embed(eagle_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        # Drafts declaring has_own_mtp_heads (e.g. Step5 multi-layer MTP) keep
        # each independently trained head; missing/duplicate ones are shared.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    if not _should_share_mtp_head(eagle_model, sh.head, target_lm_head):
                        continue
                    del sh.head
                    sh.head = target_lm_head

    # MTP shares topk_indices_buffer with the target model. We update
    # every module in the draft that holds a buffer reference so that
    # the per-layer indexer and sparse-attention backends all point to
    # the target's buffer.
    if hasattr(target_inner, "topk_indices_buffer"):
        target_buffer = target_inner.topk_indices_buffer
        if target_buffer is not None:
            for _, module in draft_inner.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer

    return eagle_model
