# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for trimming only finite-window, actually executed MTP layers."""
from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.models.step5_config import Step5Config
from vllm_hcu.patch.worker.framework_opt import patch_mtp_prefill_tail as tail


ROOT = Path(__file__).parents[2]


def _mtp_layer_selector():
    """Use the model's real selector without constructing GPU model weights."""
    tree = ast.parse((ROOT / "vllm_hcu/models/step5_mtp.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Step5MultiTokenPredictor")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name == "_layer")
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), fn,
    ], type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(module), "<Step5 MTP selector>", "exec"), scope)
    return scope["_layer"]


def _owner(monkeypatch, mtp_types, *, steps=2, window=4, windows=None,
           tail_tokens=7, prefix_caching=False):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(tail, "_TAIL", tail_tokens)
    config = Step5Config(
        num_hidden_layers=2, num_nextn_predict_layers=len(mtp_types),
        layer_types=["full_attention", "sliding_attention", *mtp_types],
        sliding_window=window,
    )
    if windows is None:
        windows = [window if kind == "sliding_attention" else None for kind in mtp_types]
    layers = {
        str(2 + idx): SimpleNamespace(mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(attn=SimpleNamespace(sliding_window=value))))
        for idx, value in enumerate(windows)
    }
    predictor = SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=len(mtp_types), layers=layers)
    predictor._layer = MethodType(_mtp_layer_selector(), predictor)
    return SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config),
                                    cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching)),
        draft_model_config=SimpleNamespace(hf_text_config=config),
        num_speculative_steps=steps, model=SimpleNamespace(model=predictor),
        dp_size=1, device=torch.device("cpu"),
    )


def _propose(monkeypatch, owner):
    """Run the actual trimming wrapper, stopping at the native propose boundary."""
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    module = ModuleType(tail.TARGET_MODULE)
    seen = []

    class MultiModuleMTPSpeculator:
        def propose(self, input_batch, attn_metadata, slot_mappings,
                    last_hidden_states, aux_hidden_states, *args, **kwargs):
            seen.append((input_batch, last_hidden_states))
            return input_batch

    module.MultiModuleMTPSpeculator = MultiModuleMTPSpeculator
    tail.apply_to_module(module)
    runner = MultiModuleMTPSpeculator()
    runner.__dict__.update(vars(owner))
    batch = InputBatch.make_dummy(1, 12, InputBuffers(1, 12, torch.device("cpu")))
    batch.is_prefilling_np[:] = True
    batch.has_prefill = True
    batch.input_ids.copy_(torch.arange(12))
    batch.positions.copy_(torch.arange(12))
    hidden = torch.arange(24).reshape(12, 2)
    result = runner.propose(batch, None, None, hidden, None)
    assert len(seen) == 1
    return batch, result, hidden, seen[0][1]


def test_actual_step5_mixed_mtp_attention_keeps_complete_prefill(monkeypatch):
    owner = _owner(monkeypatch, ["full_attention", "sliding_attention"])
    assert owner.draft_model_config.hf_text_config.layer_types == ["full_attention", "sliding_attention"]
    assert not tail._tail_enabled(owner)
    original, result, hidden, forwarded_hidden = _propose(monkeypatch, owner)
    assert result is original and forwarded_hidden is hidden
    assert result.positions.tolist() == list(range(12))
    assert tail._mtp_tail_required_tokens(owner) is None


def test_actual_step5_all_sliding_mtp_trims_only_safe_tail(monkeypatch):
    owner = _owner(monkeypatch, ["sliding_attention", "sliding_attention"])
    assert tail._mtp_tail_required_tokens(owner) == 7
    assert tail._tail_enabled(owner)
    original, result, hidden, forwarded_hidden = _propose(monkeypatch, owner)
    assert result is not original
    assert result.positions.tolist() == list(range(5, 12))
    torch.testing.assert_close(forwarded_hidden, hidden[5:])


def test_unexecuted_full_attention_module_does_not_disable_safe_prefix(monkeypatch):
    owner = _owner(monkeypatch, ["sliding_attention", "full_attention"],
                   steps=1, tail_tokens=4)
    assert owner.model.model._layer(0) is owner.model.model.layers["2"]
    assert tail._mtp_tail_required_tokens(owner) == 4
    assert tail._tail_enabled(owner)


@pytest.mark.parametrize("tail_tokens,expected", [(9, False), (10, True)])
def test_reused_mtp_module_counts_each_execution_dependency(monkeypatch, tail_tokens, expected):
    owner = _owner(monkeypatch, ["sliding_attention"], steps=3, tail_tokens=tail_tokens)
    assert all(owner.model.model._layer(step) is owner.model.model.layers["2"] for step in range(3))
    assert tail._mtp_tail_required_tokens(owner) == 10
    assert tail._tail_enabled(owner) is expected


def test_actual_per_layer_windows_are_summed_in_execution_order(monkeypatch):
    owner = _owner(monkeypatch, ["sliding_attention", "sliding_attention"],
                   steps=3, windows=[2, 5], tail_tokens=7)
    assert [owner.model.model._layer(step).mtp_block.self_attn.attn.sliding_window
            for step in range(3)] == [2, 5, 2]
    assert tail._mtp_tail_required_tokens(owner) == 7
    assert tail._tail_enabled(owner)


@pytest.mark.parametrize("window", [None, 0, -1, True, 4.0])
def test_actual_attention_must_have_known_positive_finite_window(monkeypatch, window):
    owner = _owner(monkeypatch, ["sliding_attention"], windows=[window])
    assert tail._mtp_tail_required_tokens(owner) is None
    assert not tail._tail_enabled(owner)


@pytest.mark.parametrize("missing", ["model", "types", "layer", "count", "start"])
def test_unknown_or_inconsistent_executed_layer_geometry_falls_back(monkeypatch, missing):
    owner = _owner(monkeypatch, ["sliding_attention", "sliding_attention"])
    if missing == "model":
        owner.model = None
    elif missing == "types":
        owner.draft_model_config.hf_text_config.layer_types_with_mtp = None
    elif missing == "layer":
        del owner.model.model.layers["3"]
    elif missing == "count":
        owner.model.model.num_mtp_layers = 1
    else:
        owner.model.model.mtp_start_layer_idx = 0
    assert tail._mtp_tail_required_tokens(owner) is None
    assert not tail._tail_enabled(owner)


@pytest.mark.parametrize("steps,window,tail_tokens,expected", [
    (1, 1, 1, True), (2, 4, 6, False), (2, 4, 7, True),
    (0, 4, 7, False), (2, 4, 0, False),
])
def test_tail_dependency_boundaries(monkeypatch, steps, window, tail_tokens, expected):
    owner = _owner(monkeypatch, ["sliding_attention"], steps=steps,
                   window=window, tail_tokens=tail_tokens)
    assert tail._tail_enabled(owner) is expected


def test_prefix_caching_disables_otherwise_safe_tail(monkeypatch):
    owner = _owner(monkeypatch, ["sliding_attention", "sliding_attention"], prefix_caching=True)
    assert not tail._tail_enabled(owner)
