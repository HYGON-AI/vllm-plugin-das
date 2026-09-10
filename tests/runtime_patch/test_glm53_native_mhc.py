from types import SimpleNamespace

import torch

from vllm_hcu.patch.worker.core_fix.patch_glm5next_channel_fp8 import (
    _bind_glm5next_boltops_mhc,
    _bind_glm5next_native_mhc,
)


class _FakeOp:
    def __init__(self, result):
        self.result = result
        self._forward_method = lambda *args, **kwargs: None

    def forward_native(self, *args, **kwargs):
        return self.result


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
