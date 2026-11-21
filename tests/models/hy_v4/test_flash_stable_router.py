# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm_hcu.models.hy_v4_flash.nvidia.router import (
    HYV4FlashStableRouter, stable_sigmoid_topk,
)


def make_router(**overrides):
    args = dict(top_k=2, global_num_experts=4, num_expert_group=1, topk_group=1,
                renormalize=True, scoring_func="sigmoid", routed_scaling_factor=2.0,
                e_score_correction_bias=torch.zeros(4), skip_padding=False)
    args.update(overrides)
    return GroupedTopKRouter(**args)


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.827])
def test_bias_only_selects_and_normalization_uses_deterministic_order(renormalize, scale):
    logits = torch.tensor([[3., 1., -2., 0.], [1., 1., 1., 1.]])
    bias = torch.tensor([-5., -4., 9., 8.])
    weights, ids = stable_sigmoid_topk(logits, bias, 2, renormalize, scale)
    assert ids.tolist() == [[2, 3], [2, 3]]
    expected = logits.sigmoid().gather(1, ids.long())
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected * scale, rtol=0, atol=0)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32


def test_ties_use_logical_expert_id_before_normalization():
    logits = torch.zeros(3, 8)
    w, ids = stable_sigmoid_topk(logits, torch.zeros(8), 4, True, 1.)
    assert ids.tolist() == [[0, 1, 2, 3]] * 3
    assert torch.equal(w, torch.full((3, 4), .25))


def test_empty_and_strided_inputs():
    w, ids = stable_sigmoid_topk(torch.empty(0, 4), torch.zeros(4), 2, True, 1.)
    assert w.shape == ids.shape == (0, 2)
    logits = torch.randn(6, 8)[:, ::2]
    w, ids = stable_sigmoid_topk(logits, torch.zeros(8)[::2], 2, True, 1.)
    w2, ids2 = stable_sigmoid_topk(logits.contiguous(), torch.zeros(4), 2, True, 1.)
    assert torch.equal(ids, ids2) and torch.equal(w, w2)


@pytest.mark.parametrize("case", ["dtype", "matrix", "bias", "topk0", "topk5"])
def test_bad_math_contract_rejected(case):
    logits, bias, k = torch.zeros(2, 4), torch.zeros(4), 2
    if case == "dtype": logits = logits.bfloat16()
    if case == "matrix": logits = logits[0]
    if case == "bias": bias = torch.zeros(3)
    if case == "topk0": k = 0
    if case == "topk5": k = 5
    with pytest.raises(ValueError): stable_sigmoid_topk(logits, bias, k, True, 1.)


@pytest.mark.parametrize("change", [dict(num_expert_group=2), dict(topk_group=2),
                                    dict(scoring_func="softmax"),
                                    dict(e_score_correction_bias=None),
                                    dict(num_fused_shared_experts=1)])
def test_router_rejects_unqualified_algorithms(change):
    with pytest.raises(ValueError): HYV4FlashStableRouter.from_router(make_router(**change))


def test_inherited_capture_mapping_and_index_dtype_lifecycle(monkeypatch):
    old = make_router()
    captured = []
    old.set_capture_fn(lambda x: captured.append(x.clone()))
    router = HYV4FlashStableRouter.from_router(old)
    assert router.e_score_correction_bias is old.e_score_correction_bias
    assert router.capture_fn is old.capture_fn and router.routing_method_type == old.routing_method_type
    monkeypatch.setattr(router, "_apply_eplb_mapping", lambda ids: ids + 10)
    weights, ids = router.select_experts(torch.zeros(1, 7), torch.tensor([[0., 2., 1., -1.]]),
                                         topk_indices_dtype=torch.int64)
    assert captured[0].tolist() == [[1, 2]] and ids.tolist() == [[11, 12]]
    assert ids.dtype == torch.int64 and weights.sum().item() == pytest.approx(2.)


def test_padding_keeps_core_contract(monkeypatch):
    import vllm_hcu.models.hy_v4_flash.nvidia.router as module
    router = HYV4FlashStableRouter.from_router(make_router(skip_padding=True))
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(is_padding=torch.tensor([False, True])))
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    w, ids = router.select_experts(torch.zeros(2, 7), torch.zeros(2, 4), topk_indices_dtype=torch.int32)
    assert ids.tolist() == [[0, 1], [-1, -1]] and w[1].tolist() == [0., 0.]


@pytest.mark.parametrize("hcu", [False, True])
def test_only_flash_hcu_constructor_replaces_router(monkeypatch, hcu):
    import vllm_hcu.models.hy_v4_flash.nvidia.moe as module
    old = make_router()
    def base_init(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.experts = SimpleNamespace(router=old)
    monkeypatch.setattr(module.HYV4MoEFused, "__init__", base_init)
    monkeypatch.setattr(module, "is_hcu_platform", lambda: hcu)
    config = SimpleNamespace(routed_expert_hidden_size=None, sandwich_norm=False)
    layer = module.HYV4FlashMoE(config, None, "model.layers.1.mlp")
    if hcu: assert isinstance(layer.experts.router, HYV4FlashStableRouter)
    else: assert layer.experts.router is old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires accelerator")
@pytest.mark.parametrize("rows", [1, 2, 6, 12, 335])
@pytest.mark.parametrize("renormalize", [False, True])
def test_real_device_repeats_and_batch_slicing(rows, renormalize):
    torch.manual_seed(910)
    logits = torch.randn(rows, 512, device="cuda", dtype=torch.float32)
    bias = torch.randn(512, device="cuda", dtype=torch.float32)
    first_w, first_ids = stable_sigmoid_topk(logits, bias, 16, renormalize, 2.827)
    for _ in range(20):
        w, ids = stable_sigmoid_topk(logits, bias, 16, renormalize, 2.827)
        assert torch.equal(ids, first_ids) and torch.equal(w, first_w)
    one_w, one_ids = stable_sigmoid_topk(logits[:1], bias, 16, renormalize, 2.827)
    assert torch.equal(one_ids, first_ids[:1]) and torch.equal(one_w, first_w[:1])
    scores = logits.cpu().sigmoid()
    reference_ids = torch.argsort(scores + bias.cpu(), dim=-1, descending=True, stable=True)[:, :16]
    reference_w = scores.gather(1, reference_ids)
    if renormalize: reference_w /= reference_w.sum(-1, keepdim=True)
    assert torch.equal(first_ids.cpu().long(), reference_ids)
    torch.testing.assert_close(first_w.cpu(), reference_w * 2.827, atol=3e-7, rtol=3e-6)
