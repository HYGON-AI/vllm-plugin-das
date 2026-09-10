# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import importlib
import importlib.util
import sys
from types import ModuleType

import torch


TARGET = "vllm_hcu.v1.attention.ops.lightop_kpool_topk_transform"


def _load_target():
    spec = importlib.util.find_spec(TARGET)
    assert spec is not None
    return importlib.import_module(TARGET)


def test_lightop_kpool_transform_uses_existing_sparse_topk_gate(monkeypatch):
    calls = []
    expected = torch.empty(2, 2051, dtype=torch.int32)

    def fused(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    fused_module = ModuleType("lightop.fuse_topk_transform")
    fused_module.fast_kpool_topk_transform_fused = fused
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.fuse_topk_transform", fused_module)

    target = _load_target()
    monkeypatch.setattr(target.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        target.henvs,
        "VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK",
        True,
    )
    target._get_lightop_kpool_transform.cache_clear()
    score = torch.empty(2, 8)
    lengths = torch.tensor([8, 7], dtype=torch.int32)
    row_starts = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([9, 10], dtype=torch.int32)

    result = target.lightop_kpool_topk_transform(
        score,
        lengths,
        pool_size=4,
        topk=2048,
        row_starts=row_starts,
        seq_lens=seq_lens,
        out_rows=2,
    )

    assert result is expected
    assert calls == [
        (
            (score, lengths, 4, 2048),
            {
                "page_table": None,
                "topk_indices_offset": None,
                "row_starts": row_starts,
                "seq_lens": seq_lens,
                "out_rows": 2,
                "page_table_row_index": None,
            },
        )
    ]


def test_lightop_kpool_transform_falls_back_for_unsupported_contract(monkeypatch):
    target = _load_target()
    monkeypatch.setattr(target.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        target.henvs,
        "VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK",
        True,
    )

    assert (
        target.lightop_kpool_topk_transform(
            torch.empty(2, 8),
            torch.tensor([8, 8], dtype=torch.int32),
            pool_size=8,
            topk=2048,
            seq_lens=torch.tensor([9, 10], dtype=torch.int32),
        )
        is None
    )


def test_lightop_kpool_transform_installs_through_vllm_hook():
    target = _load_target()
    callbacks = []
    kpool_module = ModuleType("vllm_kpool")
    kpool_module.register_kpool_topk_transform_fused = callbacks.append

    assert target.install_lightop_kpool_topk_transform(kpool_module) is True
    assert callbacks == [target.lightop_kpool_topk_transform]
