import builtins
import functools
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix.patch_glm5next_channel_fp8 import (
    _boltops_mhc_pre,
    _bind_glm5next_boltops_mhc,
    _bind_glm5next_native_mhc,
    _native_mhc_pre,
    _patch_glm5next_boltops_mhc,
)


class _FakeOp:
    def __init__(self, result):
        self.result = result
        self._forward_method = lambda *args, **kwargs: None

    def forward_native(self, *args, **kwargs):
        return self.result


def _fake_glm_decoder_module() -> ModuleType:
    module = ModuleType("fake_glm5next_model")

    class Glm5NextDecoderLayer:
        def __init__(
            self,
            vllm_config,
            config,
            layer_idx,
            prefix="",
            topk_indices_buffer=None,
            is_mtp_layer=False,
            **kwargs,
        ):
            del vllm_config, config, layer_idx, prefix, topk_indices_buffer, kwargs
            self.mhc = True
            self.is_mtp_layer = is_mtp_layer
            self.mhc_pre_op = _FakeOp(None)
            self.mhc_post_op = _FakeOp(None)
            self.mhc_fused_post_pre_op = _FakeOp(None)

    module.Glm5NextDecoderLayer = Glm5NextDecoderLayer
    return module


def test_glm5next_decoder_selects_native_mhc_when_master_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MHC", True)
    original_import = builtins.__import__

    def reject_hcu_mhc_import(name, *args, **kwargs):
        if name == "vllm_hcu.model_executor.layers":
            pytest.fail("master-off decoder construction imported HCU mHC")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_hcu_mhc_import)
    module = _fake_glm_decoder_module()
    _patch_glm5next_boltops_mhc(module)

    layer = module.Glm5NextDecoderLayer(None, None, 0)

    assert isinstance(layer.mhc_pre_op._forward_method, functools.partial)
    assert layer.mhc_pre_op._forward_method.func is _native_mhc_pre


def test_glm5next_decoder_selects_boltops_mhc_when_master_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor import layers as hcu_layers
    from vllm_hcu.platforms import envs as henvs

    backend = ModuleType("fake_boltops_mhc")
    backend.mhc_pre = lambda *args, **kwargs: None
    backend.mhc_post = lambda *args, **kwargs: None
    backend.mhc_fused_post_pre = lambda *args, **kwargs: None
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MHC", True)
    monkeypatch.setattr(hcu_layers, "mhc", backend, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_hcu.model_executor.layers.mhc", backend)
    module = _fake_glm_decoder_module()
    _patch_glm5next_boltops_mhc(module)

    layer = module.Glm5NextDecoderLayer(None, None, 0)

    assert isinstance(layer.mhc_pre_op._forward_method, functools.partial)
    assert layer.mhc_pre_op._forward_method.func is _boltops_mhc_pre
    assert layer.mhc_pre_op._forward_method.args[0] is backend


def test_mhc_post_master_off_ignores_materialized_true_attribute(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers import mhc as backend
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MHC", True)
    monkeypatch.setattr(
        backend,
        "_boltops_mhc_post_fwd",
        lambda *args, **kwargs: pytest.fail(
            "master-off MHC must use the native fallback"
        ),
    )
    residual = torch.arange(6, dtype=torch.bfloat16).reshape(1, 2, 3)
    x = torch.ones((1, 3), dtype=torch.bfloat16)
    post_mix = torch.tensor([[[0.25], [0.5]]], dtype=torch.float32)
    comb_mix = torch.eye(2, dtype=torch.float32).unsqueeze(0)

    actual = backend.mhc_post(x, residual, post_mix, comb_mix)
    expected = (
        residual.to(torch.float32)
        + post_mix * x.unsqueeze(-2).to(torch.float32)
    ).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected)


def test_bind_glm5next_native_mhc_preserves_norm_semantics():
    one = torch.tensor(1.0)
    two = torch.tensor(2.0)
    three = torch.tensor(3.0)
    four = torch.tensor(4.0)
    layer = SimpleNamespace(
        mhc_pre_op=_FakeOp((one, two, three)),
        mhc_post_op=_FakeOp(four),
        mhc_fused_post_pre_op=_FakeOp((one, two, three, four)),
    )
    mhc = SimpleNamespace(
        _apply_mhc_norm=lambda value, weight, eps: value + weight + eps
    )

    _bind_glm5next_native_mhc(layer, mhc)

    pre = layer.mhc_pre_op._forward_method(
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )
    assert pre[:2] == (one, two)
    assert torch.equal(pre[2], torch.tensor(5.5))

    post = layer.mhc_post_op._forward_method(one, one, one, one)
    assert torch.equal(post, four)

    fused = layer.mhc_fused_post_pre_op._forward_method(
        one,
        one,
        one,
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )
    assert fused[:3] == (one, two, three)
    assert torch.equal(fused[3], torch.tensor(6.5))


def test_bind_glm5next_boltops_mhc_preserves_official_norm_semantics():
    one = torch.tensor(1.0)
    two = torch.tensor(2.0)
    three = torch.tensor(3.0)
    four = torch.tensor(4.0)
    calls = []

    class Backend:
        @staticmethod
        def mhc_pre(*args, **kwargs):
            calls.append(("pre", args, kwargs))
            return one, two, three

        @staticmethod
        def mhc_post(*args, **kwargs):
            calls.append(("post", args, kwargs))
            return four

        @staticmethod
        def mhc_fused_post_pre(*args, **kwargs):
            calls.append(("fused", args, kwargs))
            return one, two, three, four

    layer = SimpleNamespace(
        mhc_pre_op=_FakeOp(None),
        mhc_post_op=_FakeOp(None),
        mhc_fused_post_pre_op=_FakeOp(None),
    )
    mhc = SimpleNamespace(
        _apply_mhc_norm=lambda value, weight, eps: value + weight + eps
    )

    _bind_glm5next_boltops_mhc(layer, mhc, Backend)

    pre = layer.mhc_pre_op._forward_method(
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )
    post = layer.mhc_post_op._forward_method(one, one, one, one)
    fused = layer.mhc_fused_post_pre_op._forward_method(
        one,
        one,
        one,
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )

    assert pre[:2] == (one, two)
    assert torch.equal(pre[2], torch.tensor(5.5))
    assert post is four
    assert fused[:3] == (one, two, three)
    assert torch.equal(fused[3], torch.tensor(6.5))
    assert [call[0] for call in calls] == ["pre", "post", "fused"]
    assert all("norm_weight" not in call[2] for call in calls)
    assert all("norm_eps" not in call[2] for call in calls)


def test_hcu_mhc_backend_symbols_are_bound_to_boltops():
    from vllm_hcu.model_executor.layers import mhc as backend

    assert backend._boltops_mhc_fused_tilelang.__module__.startswith("boltops.mhc")
    assert backend._boltops_mhc_post_fwd.__module__.startswith("boltops.mhc")
    assert backend._boltops_mhc_pre_big_fuse.__module__.startswith("boltops.mhc")
    assert backend._boltops_pre_big_fuse_tilelang.__module__.startswith(
        "boltops.mhc"
    )
