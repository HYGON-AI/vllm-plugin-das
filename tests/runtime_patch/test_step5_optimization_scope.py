# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regressions for the scope of Step5 optimizations."""
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "opt_in,has_bias,eligible,head_dtype,expected",
    [
        (False, False, True, None, "native"),
        (True, False, True, None, "skinny"),
        (True, True, True, None, "native"),
        (True, False, False, None, "native"),
        (True, False, True, torch.float32, "native"),
    ],
)
def test_skinny_head_requires_step5_opt_in(
    monkeypatch, opt_in, has_bias, eligible, head_dtype, expected
):
    from vllm_hcu.patch.worker.framework_opt import patch_step5_skinny_lm_head as a
    from vllm_hcu.model_executor.layers import skinny_lm_head

    calls = []

    class Processor:
        def __init__(self):
            self.head_dtype = head_dtype
            if opt_in:
                self._step5_skinny_lm_head = True

        def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
            calls.append("native")
            return "native"

    class Unquantized:
        pass

    target = ModuleType(a.TARGET_MODULE)
    target.LogitsProcessor = Processor
    target.UnquantizedEmbeddingMethod = Unquantized
    target.UnquantizedLinearMethod = Unquantized
    target.logger = SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setattr(skinny_lm_head, "usable", lambda *args: eligible)
    monkeypatch.setattr(
        skinny_lm_head, "skinny_linear",
        lambda *args: calls.append("skinny") or "skinny",
    )
    a.apply_to_module(target)
    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(4, 4))
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    bias = torch.zeros(4) if has_bias else None
    assert Processor()._apply_head(head, hidden, bias) == expected
    assert calls == [expected]


def test_skinny_head_master_off_uses_native_without_probing_provider(monkeypatch):
    """Master-off must branch before the optional skinny kernel is probed."""
    from vllm_hcu.patch.worker.framework_opt import patch_step5_skinny_lm_head as a
    from vllm_hcu.model_executor.layers import skinny_lm_head

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("STEP5_SKINNY_LM_HEAD", "1")

    class Processor:
        head_dtype = None
        _step5_skinny_lm_head = True

        def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
            return "native"

    class Unquantized:
        pass

    target = ModuleType(a.TARGET_MODULE)
    target.LogitsProcessor = Processor
    target.UnquantizedEmbeddingMethod = Unquantized
    target.UnquantizedLinearMethod = Unquantized
    target.logger = SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setattr(
        skinny_lm_head,
        "usable",
        lambda *args: pytest.fail("master-off probed the skinny provider"),
    )
    a.apply_to_module(target)

    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(4, 4))
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    assert Processor()._apply_head(head, hidden) == "native"


def test_eplb_native_metric_preserves_cross_layer_cancellation():
    from vllm_hcu.distributed.eplb_runtime import _rank_load_imbalance

    mapping = torch.tensor([[0, 1], [0, 1]])
    # Equal totals across ranks, but opposite bottlenecks in the two layers.
    loads = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=False) == 1.0
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=True) == pytest.approx(1.8)


@pytest.mark.parametrize("layerwise", [False, True])
def test_eplb_metric_handles_replicas_and_zero_load(layerwise):
    from vllm_hcu.distributed.eplb_runtime import _rank_load_imbalance

    mapping = torch.tensor([[0, 1, 0, 1]])
    loads = torch.tensor([[8.0, 4.0]])
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=layerwise) == 1.0
    assert _rank_load_imbalance(mapping, loads * 0, 2, layerwise=layerwise) == 1.0
