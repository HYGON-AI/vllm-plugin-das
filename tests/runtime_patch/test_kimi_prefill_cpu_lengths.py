"""Kimi convolution must reuse the scheduler's CPU sequence lengths."""
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_kimi_prefill_metadata as patch
from vllm_hcu.models.kimi_k3.amd.ops.prefill_metadata import (
    LENGTHS_KEY, prefill_sequence_lengths,
)


def target():
    module = ModuleType(patch.TARGET_MODULE)
    def compute_causal_conv1d_metadata(query_start_loc_p_cpu, *, device):
        return {8: {"tot": 2}}, object(), object()
    module.compute_causal_conv1d_metadata = compute_causal_conv1d_metadata
    return module


@pytest.mark.parametrize("master,leaf,enabled", [
    ("1", "1", True), ("true", "true", True),
    ("0", "1", False), ("false", "true", False),
    ("1", "0", False), ("true", "false", False),
])
def test_controls_and_exact_fallback(monkeypatch, master, leaf, enabled):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", leaf)
    module = target()
    assert patch.apply(module)
    assert not patch.apply(module)
    # This is already the non-spec subset; retain decode and padded zero rows.
    result = module.compute_causal_conv1d_metadata(
        torch.tensor([0, 1, 9, 9, 26]), device="cpu")
    assert result[0][8] == {"tot": 2}
    assert (LENGTHS_KEY in result[0]) is enabled
    metadata = SimpleNamespace(nums_dict=result[0])
    class DeviceBoundary:
        def diff(self):
            if enabled:
                raise AssertionError("unexpected device read")
            return torch.tensor([1, 8, 0, 17])
    assert prefill_sequence_lengths(metadata, DeviceBoundary()) == [1, 8, 0, 17]


def test_absent_leaf_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", raising=False)
    module = target()
    patch.apply(module)
    assert LENGTHS_KEY not in module.compute_causal_conv1d_metadata(
        torch.tensor([0, 3]), device="cpu")[0]


def test_snapshot_is_not_stale_when_scheduler_reuses_storage(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", "1")
    module = target()
    patch.apply(module)
    starts = torch.tensor([0, 4, 9])
    a = module.compute_causal_conv1d_metadata(starts, device="cpu")[0]
    starts.copy_(torch.tensor([0, 2, 8]))
    b = module.compute_causal_conv1d_metadata(starts, device="cpu")[0]
    assert a[LENGTHS_KEY] == [4, 5]
    assert b[LENGTHS_KEY] == [2, 6]


def test_stale_marker_signature_and_missing_target_fail():
    module = target()
    original = module.compute_causal_conv1d_metadata
    patch.apply(module)
    module.compute_causal_conv1d_metadata = original
    with pytest.raises(patch.PatchCompatibilityError, match="stale"):
        patch.apply(module)
    module = target()
    module.compute_causal_conv1d_metadata = lambda unexpected: None
    with pytest.raises(patch.PatchCompatibilityError):
        patch.apply(module)
    module.compute_causal_conv1d_metadata = None
    with pytest.raises(patch.PatchCompatibilityError):
        patch.apply(module)


def test_selected_metadata_failure_propagates(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", "1")
    module = target()
    def broken(query_start_loc_p_cpu, *, device):
        raise RuntimeError("metadata failure")
    module.compute_causal_conv1d_metadata = broken
    patch.apply(module)
    with pytest.raises(RuntimeError, match="metadata failure"):
        module.compute_causal_conv1d_metadata(torch.tensor([0, 1]), device="cpu")


def test_worker_registration_and_late_captured_binding():
    from vllm_hcu.patch.worker import worker_callback_names, _patch_features
    assert (patch.PATCH_ID, patch.TARGET_MODULE) in worker_callback_names()
    assert _patch_features()[patch.PATCH_ID] == "always"
    module = target()
    # The builder resolves its helper in its own module on each build.
    exec("def build(starts):\n    return compute_causal_conv1d_metadata(starts, device='cpu')", module.__dict__)
    build = module.build
    assert patch.apply(module)
    assert build.__globals__["compute_causal_conv1d_metadata"] is module.compute_causal_conv1d_metadata


def test_missing_metadata_keeps_original_device_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", "1")
    assert prefill_sequence_lengths(SimpleNamespace(nums_dict=None), torch.tensor([0, 2, 9])) == [2, 7]
