# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression: the wheel DSpark lm_head share must survive K3 wrapper targets.

``load_dspark_model`` probes ``target_model.lm_head`` only; a
``*ForConditionalGeneration`` target keeps the head on ``get_language_model()``,
so without the fork-parity share the K3 draft stays headless until
``compute_logits`` raises "DSpark target LM head has not been attached".
"""

from __future__ import annotations

import torch

from vllm_hcu.patch.worker.framework_opt.patch_dspark_target_lm_head_share import (
    _resolve_target_lm_head_fork_parity,
    _share_target_lm_head_fork_parity,
)


class _Head:
    def __init__(self, value: float = 1.0) -> None:
        self.weight = torch.full((4,), value)


class _LanguageModel:
    def __init__(self) -> None:
        self.lm_head = _Head()


class _WrapperTarget:
    """*ForConditionalGeneration shape: head lives behind get_language_model."""

    def __init__(self) -> None:
        self._language_model = _LanguageModel()

    def get_language_model(self) -> _LanguageModel:
        return self._language_model


class _FlatTarget:
    def __init__(self) -> None:
        self.lm_head = _Head()


class _Draft(torch.nn.Module):
    has_own_lm_head = False

    def __init__(self) -> None:
        super().__init__()
        self.lm_head = None


def test_resolve_prefers_language_model_head():
    wrapper = _WrapperTarget()
    head = _resolve_target_lm_head_fork_parity(wrapper)
    assert head is wrapper._language_model.lm_head


def test_resolve_falls_back_to_top_level_head():
    flat = _FlatTarget()
    assert _resolve_target_lm_head_fork_parity(flat) is flat.lm_head


def test_share_attaches_head_for_wrapper_target():
    target = _WrapperTarget()
    draft = _Draft()
    _share_target_lm_head_fork_parity(draft, target)
    assert draft.lm_head is target._language_model.lm_head


def test_share_is_noop_without_any_target_head():
    draft = _Draft()
    _share_target_lm_head_fork_parity(draft, object())
    assert draft.lm_head is None


def test_share_keeps_divergent_own_head():
    class _OwnDraft(_Draft):
        has_own_lm_head = True

    target = _WrapperTarget()
    draft = _OwnDraft()
    own = _Head(value=9.0)
    draft.lm_head = own
    _share_target_lm_head_fork_parity(draft, target)
    assert draft.lm_head is own
