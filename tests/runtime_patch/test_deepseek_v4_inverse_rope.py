# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_inverse_rope as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def original(rotary_emb, o, positions, rope_head_dim, n_local_groups,
             o_lora_rank, wo_a):
    return ("original", o, positions)


def target(function=original):
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_inv_rope_einsum = function
    return module


def test_registration_and_fallback(monkeypatch):
    module = target()
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: False)
    o, positions = torch.ones(3, 2, 512), torch.arange(3)
    result = module.rocm_inv_rope_einsum(None, o, positions, 64, 2, 7, None)
    assert result[0] == "original"
    assert result[1] is o
    assert result[2] is positions


def test_stale_marker_and_signature():
    module = target()
    patch.apply_to_module(module)
    module.rocm_inv_rope_einsum = lambda *args: None
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(module)

    module = target(lambda rotary_emb, o: None)
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(module)


@pytest.mark.parametrize(
    "flag", ["enabled", "master", "child", "grad", "cpu", "short", "fp32"]
)
def test_guards_keep_reference(monkeypatch, flag):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0" if flag == "master" else "1")
    monkeypatch.setenv(
        "VLLM_HCU_INVERSE_ROPE_SUFFIX", "0" if flag == "child" else "1"
    )
    monkeypatch.setattr(torch.version, "hip", "test")
    rows = 17 if flag == "short" else 512
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(1024, 64))
    o = SimpleNamespace(
        is_cuda=flag != "cpu",
        dtype=torch.float32 if flag == "fp32" else torch.bfloat16,
        ndim=3,
        shape=(rows, 2, 512),
        device=torch.device("cpu"),
        is_contiguous=lambda: True,
    )
    scope = torch.enable_grad() if flag == "grad" else torch.no_grad()
    with scope:
        assert patch._eligible(rotary, o, torch.arange(rows), 64, 2) == (
            flag == "enabled"
        )


def test_optimized_helper_uses_native_op_and_cached_weight(monkeypatch):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    module = target()
    patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    weight = torch.randn(2, 7, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(ops, "_get_cached_wo_a_bf16", lambda *args: weight)
    calls = []

    def inverse_rope(rope, positions, cache):
        calls.append((positions, cache))
        rope.mul_(-1)

    monkeypatch.setattr(
        torch.ops.hcu_ops, "deepseek_v4_inv_rope", inverse_rope, raising=False
    )
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(10, 64))
    o = torch.randn(3, 2, 512, dtype=torch.bfloat16)
    expected_o = o.clone()
    expected_o[..., -64:] *= -1
    expected = torch.einsum("tgd,grd->tgr", expected_o, weight)

    result = module.rocm_inv_rope_einsum(
        rotary, o, torch.arange(3), 64, 2, 7, object()
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(o, expected_o)
    assert len(calls) == 1
