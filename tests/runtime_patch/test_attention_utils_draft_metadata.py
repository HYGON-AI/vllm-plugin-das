# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attention utility contracts for uniform multi-step draft metadata."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


TARGET_MODULE = "vllm.v1.attention.backends.utils"
ADAPTER_MODULE = (
    "vllm_hcu.patch.worker.op_opt.patch_attention_backend_utils"
)


def _patched_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[bool] = []
    target = ModuleType(TARGET_MODULE)

    def compute_causal_conv1d_metadata(query_start_loc_p_cpu, *, device):
        return ({}, object(), object())

    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        calls.append(treat_short_extends_as_decodes)
        if not treat_short_extends_as_decodes:
            assert common_attn_metadata.is_prefilling is not None
        return (2, 0, 2, 0)

    target.compute_causal_conv1d_metadata = compute_causal_conv1d_metadata
    target.split_decodes_and_prefills = split_decodes_and_prefills
    monkeypatch.setitem(sys.modules, TARGET_MODULE, target)
    monkeypatch.delitem(sys.modules, ADAPTER_MODULE, raising=False)
    adapter = importlib.import_module(ADAPTER_MODULE)
    adapter.apply_to_module(target)
    return target.split_decodes_and_prefills, calls


def test_uniform_draft_metadata_without_prefill_mask_is_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_batch, calls = _patched_split(monkeypatch)
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        is_prefilling=None,
        num_reqs=2,
        num_actual_tokens=2,
    )

    assert split_batch(common, 1, True, False) == (2, 0, 2, 0)
    assert calls == [True]
