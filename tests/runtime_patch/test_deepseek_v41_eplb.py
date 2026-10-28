# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_eplb as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

NUM_LOGICAL = 16
NUM_LAYERS = 2


def _vllm_config(*, enable_eplb=True, redundant=0, speculative=None,
                 all2all="deepep_high_throughput"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=False,
            all2all_backend=all2all,
            eplb_config=SimpleNamespace(num_redundant_experts=redundant),
        ),
        speculative_config=speculative,
        additional_config={},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS)
        ),
    )


def _make_modules(monkeypatch, *, ep_size=4, local_override=None):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        enable = kwargs.get("enable_eplb", False)
        global_num = NUM_LOGICAL + kwargs.get("num_redundant_experts", 0)
        local = local_override or global_num // ep_size
        routed = SimpleNamespace(
            global_num_experts=global_num,
            local_num_experts=local,
            moe_config=SimpleNamespace(
                moe_parallel_config=SimpleNamespace(ep_size=ep_size)
            ),
        )
        return SimpleNamespace(
            routed_experts=routed,
            eplb_state=object() if enable else None,
        )

    base = ModuleType(patch.BASE_MODULE)
    base.FusedMoEFactory = factory
    monkeypatch.setitem(sys.modules, patch.BASE_MODULE, base)

    class DeepseekV4MoE:
        def __init__(self, vllm_config, prefix="", use_sequence_parallel=False):
            self.prefix = prefix
            self.experts = base.FusedMoEFactory(num_experts=NUM_LOGICAL)
            self.n_routed_experts = NUM_LOGICAL
            self.n_logical_experts = NUM_LOGICAL
            self.n_physical_experts = NUM_LOGICAL
            self.n_local_physical_experts = NUM_LOGICAL
            self.n_redundant_experts = 0

    class DeepseekV4Model:
        def __init__(self, layers):
            self.layers = layers
            self.config = SimpleNamespace(n_routed_experts=NUM_LOGICAL)

        def get_expert_mapping(self):
            return "official"

    def make_mapping(model, **kwargs):
        return kwargs

    class DeepseekV41LLMForCausalLM:
        def set_eplb_state(
            self, expert_load_view, logical_to_physical_map, logical_replica_count
        ):
            self.expert_weights = [
                list(layer.routed_experts.get_expert_weights())
                for layer in self.moe_layers
            ]

    target = ModuleType(patch.TARGET_MODULE)
    target.DeepseekV4MoE = DeepseekV4MoE
    target.DeepseekV4Model = DeepseekV4Model
    target.DeepseekV41LLMForCausalLM = DeepseekV41LLMForCausalLM
    target.fused_moe_make_expert_params_mapping = make_mapping
    target.extract_layer_index = lambda prefix: int(prefix.split(".")[1])
    assert patch.apply_to_module(target)
    return target, base, factory, calls


def test_disabled_eplb_keeps_official_construction(monkeypatch) -> None:
    target, base, factory, calls = _make_modules(monkeypatch)

    moe = target.DeepseekV4MoE(_vllm_config(enable_eplb=False), "layers.0.ffn")

    assert calls == [{"num_experts": NUM_LOGICAL}]
    assert base.FusedMoEFactory is factory
    assert moe.n_local_physical_experts == NUM_LOGICAL
    model = target.DeepseekV4Model([SimpleNamespace(ffn=moe)])
    assert model.get_expert_mapping() == "official"


def test_enabled_eplb_reaches_routed_experts_and_publishes_ep_layout(
    monkeypatch,
) -> None:
    target, base, factory, calls = _make_modules(monkeypatch)

    moe = target.DeepseekV4MoE(_vllm_config(redundant=4), "layers.1.ffn")

    assert calls == [
        {"num_experts": NUM_LOGICAL, "enable_eplb": True, "num_redundant_experts": 4}
    ]
    assert base.FusedMoEFactory is factory
    assert (moe.n_physical_experts, moe.n_local_physical_experts) == (20, 5)
    assert moe.n_redundant_experts == 4
    model = target.DeepseekV4Model([SimpleNamespace(), SimpleNamespace(ffn=moe)])
    assert model.get_expert_mapping() == {
        "ckpt_gate_proj_name": "w1",
        "ckpt_down_proj_name": "w2",
        "ckpt_up_proj_name": "w3",
        "num_experts": NUM_LOGICAL,
        "num_redundant_experts": 4,
    }


def test_zero_redundancy_keeps_official_mapping(monkeypatch) -> None:
    target, _, _, _ = _make_modules(monkeypatch)

    moe = target.DeepseekV4MoE(_vllm_config(redundant=0), "layers.0.ffn")

    assert moe.n_local_physical_experts == NUM_LOGICAL // 4
    model = target.DeepseekV4Model([SimpleNamespace(ffn=moe)])
    assert model.get_expert_mapping() == "official"


def test_draft_layers_do_not_receive_eplb(monkeypatch) -> None:
    target, _, _, calls = _make_modules(monkeypatch)

    target.DeepseekV4MoE(_vllm_config(redundant=4), f"layers.{NUM_LAYERS}.ffn")

    assert calls == [{"num_experts": NUM_LOGICAL}]


@pytest.mark.parametrize(
    "config",
    (
        _vllm_config(speculative=object()),
        _vllm_config(all2all="deepep_auto"),
        _vllm_config(all2all="deepep_low_latency"),
    ),
)
def test_unvalidated_eplb_paths_fail_closed(monkeypatch, config) -> None:
    target, base, factory, calls = _make_modules(monkeypatch)

    with pytest.raises(NotImplementedError):
        target.DeepseekV4MoE(config, "layers.0.ffn")

    assert calls == []
    assert base.FusedMoEFactory is factory


def test_layout_mismatch_is_rejected_and_factory_restored(monkeypatch) -> None:
    target, base, factory, _ = _make_modules(monkeypatch, local_override=3)

    with pytest.raises(RuntimeError, match="layout mismatch"):
        target.DeepseekV4MoE(_vllm_config(redundant=4), "layers.0.ffn")

    assert base.FusedMoEFactory is factory


def test_reapply_is_idempotent_and_detects_stale_marker(monkeypatch) -> None:
    target, _, _, _ = _make_modules(monkeypatch)

    assert patch.apply_to_module(target) is False
    target.DeepseekV4Model.get_expert_mapping = lambda self: None
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(target)


def _deepgemm_layer(*, local=4, copy_scale=False, kernel=True):
    w13 = torch.zeros(local, 2, 3, 4, 5, 6)
    w2 = torch.zeros(local, 3, 3, 4, 5, 6)
    w13_scale = torch.ones(local, 8, 1)
    w2_scale = torch.ones(local, 4, 1)
    names = ("_deepgemm_w13", "_deepgemm_w2", "w1_scale", "w2_scale")
    tensors = (w13, w2, w13_scale.clone() if copy_scale else w13_scale, w2_scale)
    experts = SimpleNamespace(**dict(zip(names, tensors))) if kernel else object()
    routed = SimpleNamespace(
        local_num_experts=local,
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(fused_experts=experts)
        ),
        get_expert_weights=lambda: [
            t.view(local, -1) for t in (w13, w2, w13_scale, w2_scale)
        ],
    )
    return SimpleNamespace(routed_experts=routed)


def _register(target, layers):
    model = target.DeepseekV41LLMForCausalLM()
    model.moe_layers = layers
    model.set_eplb_state(None, None, None)
    return model


def test_eplb_registration_accepts_aliased_deepgemm_tensors(monkeypatch) -> None:
    target, _, _, _ = _make_modules(monkeypatch)

    model = _register(target, [_deepgemm_layer(), _deepgemm_layer()])

    assert len(model.expert_weights) == 2


def test_eplb_registration_rejects_unmoved_kernel_scale(monkeypatch) -> None:
    target, _, _, _ = _make_modules(monkeypatch)

    with pytest.raises(RuntimeError, match="w1_scale"):
        _register(target, [_deepgemm_layer(copy_scale=True)])


def test_eplb_registration_rejects_unvalidated_kernel(monkeypatch) -> None:
    target, _, _, _ = _make_modules(monkeypatch)

    with pytest.raises(NotImplementedError, match="DeepGEMM"):
        _register(target, [_deepgemm_layer(kernel=False)])


def test_redundant_mapping_loads_replicas_from_logical_checkpoint() -> None:
    pytest.importorskip("vllm.model_executor.layers.fused_moe")
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )

    mapping = fused_moe_make_expert_params_mapping(
        SimpleNamespace(named_parameters=lambda: iter(())),
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=384,
        num_redundant_experts=8,
    )

    physical_to_ckpt = {
        expert_id: weight_name
        for _, weight_name, expert_id, shard in mapping
        if shard == "w1"
    }
    assert len(physical_to_ckpt) == 392
    assert physical_to_ckpt[384] == "experts.0.w1."
    assert physical_to_ckpt[391] == "experts.7.w1."
    assert physical_to_ckpt[383] == "experts.383.w1."


def test_real_target_signatures_match() -> None:
    target = pytest.importorskip(patch.TARGET_MODULE)
    for name in ("DeepseekV4MoE", "DeepseekV4Model", "DeepseekV41LLMForCausalLM"):
        assert hasattr(target, name)
    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV4MoE = type(
        "DeepseekV4MoE", (), {"__init__": target.DeepseekV4MoE.__init__}
    )
    module.DeepseekV4Model = type(
        "DeepseekV4Model",
        (),
        {"get_expert_mapping": target.DeepseekV4Model.get_expert_mapping},
    )
    module.DeepseekV41LLMForCausalLM = type(
        "DeepseekV41LLMForCausalLM",
        (),
        {"set_eplb_state": target.DeepseekV41LLMForCausalLM.set_eplb_state},
    )
    module.fused_moe_make_expert_params_mapping = (
        target.fused_moe_make_expert_params_mapping
    )
    module.extract_layer_index = target.extract_layer_index

    assert patch.apply_to_module(module)


@pytest.mark.parametrize("path_field", ["expert_map_path", "expert_map_record_path"])
def test_dspark_offline_eplb_only_changes_target_experts(monkeypatch, path_field):
    target, base, factory, calls = _make_modules(monkeypatch)
    config = _vllm_config(speculative=SimpleNamespace(method="dspark"))
    config.additional_config = {"hcu": {
        path_field: "/target-map.json", "eplb_disable_rearrange": True,
    }}

    target.DeepseekV4MoE(config, "layers.0.ffn")
    target.DeepseekV4MoE(config, f"layers.{NUM_LAYERS}.ffn")

    assert calls == [
        {"num_experts": NUM_LOGICAL, "enable_eplb": True, "num_redundant_experts": 0},
        {"num_experts": NUM_LOGICAL},
    ]
    assert base.FusedMoEFactory is factory
    assert config.parallel_config.enable_eplb is True


@pytest.mark.parametrize("method", ["eagle3", "dflash", "mtp"])
def test_other_speculative_methods_still_rejected(monkeypatch, method):
    target, _, _, calls = _make_modules(monkeypatch)
    config = _vllm_config(speculative=SimpleNamespace(method=method))
    config.additional_config = {"hcu": {
        "expert_map_path": "/target-map.json", "eplb_disable_rearrange": True,
    }}
    with pytest.raises(NotImplementedError, match="DSpark"):
        target.DeepseekV4MoE(config, "layers.0.ffn")
    assert not calls


def test_dspark_online_rearrangement_is_not_enabled(monkeypatch):
    target, _, _, calls = _make_modules(monkeypatch)
    config = _vllm_config(speculative=SimpleNamespace(method="dspark"))
    with pytest.raises(NotImplementedError, match="offline"):
        target.DeepseekV4MoE(config, "layers.0.ffn")
    assert not calls
