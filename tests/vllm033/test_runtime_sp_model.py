# SPDX-License-Identifier: Apache-2.0
"""Real official MLP/loaders and early factory; explicit CPU GEMM doubles."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adapter_extension/src"))
from hcu_vllm_das.parallel import runtime_sp_attention as sp
from hcu_vllm_das.parallel import runtime_sp_moe as moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers import linear
from vllm.model_executor import parameter
from vllm.model_executor.models import glm4_moe, hy_v3
from vllm.model_executor.layers.fused_moe import layer as official_moe


@pytest.fixture
def core(monkeypatch):
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    monkeypatch.setattr(sp, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: group.rank_in_group)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: group.rank_in_group)
    monkeypatch.setattr(sp, "_publication_lifecycle", sp.ProcessLocalLifecycle(
        runtime_action=sp._PUBLICATION_ACTION, error_type=sp.ExtensionCommitError))
    for base, _ in sp._BINDINGS:
        monkeypatch.delitem(op_registry_oot, base.__name__, raising=False)
    def no_collective(*args, **kwargs):
        pytest.fail("replicated MLP must not mix different ranks' token shards")
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", no_collective)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", no_collective)
    sp.install_attention_layers()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        yield group
    for base, _ in sp._BINDINGS:
        op_registry_oot.pop(base.__name__, None)


def values(rows, columns, epoch=0):
    return (((torch.arange(rows * columns).reshape(rows, columns) + epoch) % 13) - 6).float() / 16


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("family,shared", [("glm4", False), ("glm4", True), ("hy3", False), ("hy3", True)])
def test_official_dense_shared_mlp_load_reload_and_local_tokens(core, rank, family, shared):
    core.rank_in_group = rank
    # Shared weights may be nondivisible; dense intermediate weights retain TP.
    hidden, middle, rows = 13, 19 if shared else 20, 3
    cls = glm4_moe.Glm4MoeMLP if family == "glm4" else hy_v3.HYV3FeedForward
    prefix = "model.layers.0.mlp"
    if shared:
        prefix += ".shared_experts" if family == "glm4" else ".shared_mlp"
    with sp.runtime_sp_model_construction():
        module = cls(hidden, middle, "silu", reduce_results=not shared, prefix=prefix)
    up, down = module.gate_up_proj, module.down_proj
    assert type(up) is sp.RuntimeSPMergedColumnParallelLinear
    assert type(down) is sp.RuntimeSPRowParallelLinear
    partitions = 1 if shared else 4
    assert up.tp_size == down.tp_size == partitions
    assert up.tp_rank == down.tp_rank == (0 if shared else rank)
    assert up.weight.shape == (2 * middle // partitions, hidden)
    assert down.weight.shape == (hidden, middle // partitions)
    assert up.weight_loader_v2.__func__ is linear.MergedColumnParallelLinear.weight_loader_v2
    assert down.weight_loader_v2.__func__ is linear.RowParallelLinear.weight_loader_v2
    assert not down._runtime_sp and down._runtime_sp_mlp
    assert not down.reduce_results
    assert up._runtime_sp_mlp_gather is (not shared)
    assert down._runtime_sp_mlp_gather is (not shared)
    for layer in (up, down):
        layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    module.act_fn._forward_method = module.act_fn.forward_native
    for epoch in (0, 2):
        wup, wdown = values(2 * middle, hidden, epoch), values(hidden, middle, epoch + 1)
        for index in (0, 1):
            up.weight.weight_loader(up.weight, wup[index * middle:(index + 1) * middle], index)
        down.weight.weight_loader(down.weight, wdown)
        for layer in (up, down):
            layer.update_param_tp_status()
        x = values(4 * rows, hidden, epoch + 3)
        local = x[rank * rows:(rank + 1) * rows]
        if shared:
            torch.testing.assert_close(up.weight, wup, rtol=0, atol=0)
            torch.testing.assert_close(down.weight, wdown, rtol=0, atol=0)
            gate, value = F.linear(local, wup).chunk(2, dim=-1)
            expected = F.linear(F.silu(gate) * value, wdown)
        else:
            width = middle // 4
            shards = [torch.cat([part[i * width:(i + 1) * width]
                                 for part in wup.chunk(2)], dim=0) for i in range(4)]
            torch.testing.assert_close(up.weight, shards[rank], rtol=0, atol=0)
            torch.testing.assert_close(down.weight, wdown[:, rank * width:(rank + 1) * width], rtol=0, atol=0)
            partials = []
            for i in range(4):
                gate, value = F.linear(x, shards[i]).chunk(2, dim=-1)
                partials.append(F.linear(F.silu(gate) * value, wdown[:, i * width:(i + 1) * width]))
            total = torch.stack(partials).sum(0)
            calls = []
            def gather(actual, dim):
                assert dim == 0
                torch.testing.assert_close(actual, local, rtol=0, atol=0)
                calls.append("gather")
                return x
            def scatter(actual, dim):
                assert dim == 0
                torch.testing.assert_close(actual, partials[rank], rtol=0, atol=0)
                calls.append("scatter")
                return total[rank * rows:(rank + 1) * rows]
            core.all_gather, core.reduce_scatter = gather, scatter
            expected = total[rank * rows:(rank + 1) * rows]
        torch.testing.assert_close(module(local), expected, rtol=0, atol=0)
        if not shared:
            assert calls == ["gather", "scatter"]


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_dense_mlp_refuses_replicated_allocation_before_constructor(core, projection):
    cls, args = ((linear.MergedColumnParallelLinear, (16, [20, 20]))
                 if projection == "gate_up_proj" else (linear.RowParallelLinear, (20, 16)))
    with sp.runtime_sp_model_construction(), pytest.raises(ValueError, match="TP weight allocation"):
        cls(*args, disable_tp=True, bias=False, prefix="model.layers.0.mlp." + projection)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("family", ["glm4", "hy3"])
def test_dense_mlp_padding_mask_changes_keep_local_sum_and_finite_values(core, monkeypatch, rank, family):
    from hcu_vllm_das.parallel import runtime_sp_batch as layout
    from vllm.forward_context import ForwardContext, override_forward_context

    core.rank_in_group = rank
    hidden, middle, rows = 13, 20, 2
    cls = glm4_moe.Glm4MoeMLP if family == "glm4" else hy_v3.HYV3FeedForward
    with sp.runtime_sp_model_construction():
        module = cls(hidden, middle, "silu", prefix="model.layers.0.mlp")
    up, down = module.gate_up_proj, module.down_proj
    wup, wdown = values(2 * middle, hidden), values(hidden, middle, 2)
    width = middle // 4
    shards = [torch.cat([part[i * width:(i + 1) * width] for part in wup.chunk(2)]) for i in range(4)]
    for i, part in enumerate(wup.chunk(2)):
        up.weight.weight_loader(up.weight, part, i)
    down.weight.weight_loader(down.weight, wdown)
    for layer in (up, down):
        layer.quant_method._gemm_impl = lambda layer, x, w, b: F.linear(x, w, b)
    module.act_fn._forward_method = module.act_fn.forward_native
    padding = torch.zeros(7, dtype=torch.bool)
    batch = layout.RuntimeSPBatch.create(4, rank, 7, padding, padding.device)
    original = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
                              is_padding=batch.local_padding,
                              additional_kwargs={layout._BATCH_KEY: batch})
    pointers = batch.global_padding.data_ptr(), batch.local_padding.data_ptr()
    for epoch, active in enumerate((5, 7, 3)):
        batch.global_padding.copy_(torch.arange(8) >= active)
        x = values(8, hidden, epoch + 3).masked_fill(batch.global_padding[:, None], 0)
        local = x[rank * rows:(rank + 1) * rows].clone()
        local.masked_fill_(batch.local_padding[:, None], float("nan"))
        partials = []
        for i in range(4):
            gate, value = F.linear(x, shards[i]).chunk(2, dim=-1)
            partials.append(F.linear(F.silu(gate) * value, wdown[:, i * width:(i + 1) * width]))
        total = torch.stack(partials).sum(0)
        calls = []
        def gather(actual, dim):
            assert dim == 0 and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, x[rank * rows:(rank + 1) * rows], rtol=0, atol=0)
            calls.append("gather")
            return x
        def scatter(actual, dim):
            assert dim == 0 and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, partials[rank], rtol=0, atol=0)
            calls.append("scatter")
            return total[rank * rows:(rank + 1) * rows]
        core.all_gather, core.reduce_scatter = gather, scatter
        with override_forward_context(original):
            actual = module(local)
        expected = total[rank * rows:(rank + 1) * rows].masked_fill(batch.local_padding[:, None], 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all() and calls == ["gather", "scatter"]
        assert pointers == (batch.global_padding.data_ptr(), batch.local_padding.data_ptr())


def test_mlp_scope_does_not_change_attention_only_or_unrelated_layers(core):
    prefix = "model.layers.0.mlp.gate_up_proj"
    for context in (sp.runtime_sp_attention_construction,):
        with context():
            layer = linear.MergedColumnParallelLinear(16, [20, 20], bias=False, prefix=prefix)
        assert not layer._runtime_sp_mlp and layer.tp_size == 4
    with pytest.raises(RuntimeError, match="unwind"):
        with sp.runtime_sp_model_construction():
            other = linear.MergedColumnParallelLinear(16, [20, 20], bias=False,
                prefix="unrelated.gate_up_proj")
            assert not other._runtime_sp_mlp and other.tp_size == 4
            raise RuntimeError("unwind")
    assert not sp.model_construction_active()
    layer = linear.MergedColumnParallelLinear(16, [20, 20], bias=False, prefix=prefix)
    assert not layer._runtime_sp_mlp and layer.tp_size == 4


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize("drift", ["pid", "group", "rank"])
def test_mlp_owner_drift_refuses_before_gemm(core, monkeypatch, projection, drift):
    with sp.runtime_sp_model_construction():
        cls, args = ((linear.MergedColumnParallelLinear, (16, [20, 20]))
                     if projection == "gate_up_proj" else (linear.RowParallelLinear, (20, 16)))
        layer = cls(*args, bias=False, prefix="model.layers.0.mlp." + projection)
    layer.quant_method.apply = lambda *a: pytest.fail("no GEMM after ownership drift")
    if drift == "pid":
        layer._runtime_sp_pid -= 1
    elif drift == "rank":
        core.rank_in_group += 1
    else:
        monkeypatch.setattr(sp, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=0))
    with pytest.raises(RuntimeError):
        layer(torch.zeros(3, 16 if projection == "gate_up_proj" else 20))


@pytest.fixture
def factory(monkeypatch):
    original = official_moe.FusedMoEFactory
    for module in (glm4_moe, hy_v3):
        monkeypatch.setattr(module, "FusedMoEFactory", original)
    monkeypatch.setattr(moe, "_targets", ())
    monkeypatch.setattr(moe, "_factory", None)
    monkeypatch.setattr(moe, "_official_module", None)
    monkeypatch.setattr(moe, "_original_factory", None)
    monkeypatch.setattr(moe, "_lifecycle", moe.ProcessLocalLifecycle(
        runtime_action=moe._ACTION, error_type=moe.ExtensionCommitError))
    return original


@pytest.mark.parametrize("family", ["glm4", "hy3"])
@pytest.mark.parametrize("scoped,prefix,expected", [(False, "model.layers.0.mlp.experts", False),
    (True, "model.layers.0.mlp.experts", True), (True, "other.experts", False)])
def test_real_factory_gets_sp_before_parallel_config_router_or_weights(core, factory,
        monkeypatch, family, scoped, prefix, expected):
    moe.install_moe_factory_policy()
    seen = []
    class BeforeAllocation(Exception):
        pass
    def before(**kwargs):
        seen.append(kwargs)
        raise BeforeAllocation
    monkeypatch.setattr(official_moe, "make_parallel_config", before)
    monkeypatch.setattr(official_moe, "create_fused_moe_router",
                        lambda **kw: pytest.fail("router must not precede parallel config"))
    module = glm4_moe if family == "glm4" else hy_v3
    assert official_moe.FusedMoEFactory is factory
    args = (8, 2, 16, 32)
    with pytest.raises(BeforeAllocation):
        if scoped:
            with sp.runtime_sp_model_construction():
                module.FusedMoEFactory(*args, prefix=prefix)
        else:
            module.FusedMoEFactory(*args, prefix=prefix)
    assert len(seen) == 1 and seen[0]["is_sequence_parallel"] is expected
    assert not sp.model_construction_active()


def test_scoped_factory_does_not_overwrite_a_foreign_runner(core, factory):
    moe.install_moe_factory_policy()
    class ForeignRunner:
        pass
    with pytest.raises(RuntimeError, match="foreign runtime SP MoE runner"), sp.runtime_sp_model_construction():
        glm4_moe.FusedMoEFactory(8, 2, 16, 32, prefix="model.layers.0.mlp.experts",
                                runner_cls=ForeignRunner)


@pytest.mark.parametrize("family", ["glm4", "hy3"])
@pytest.mark.parametrize("shared", [False, True])
def test_official_moe_model_constructor_reaches_early_sp_policy(core, factory,
        monkeypatch, family, shared):
    moe.install_moe_factory_policy()
    module = glm4_moe if family == "glm4" else hy_v3
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "get_ep_group", lambda: SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 4)))
    monkeypatch.setattr(glm4_moe.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(glm4_moe, "resolve_layer_fused_shared_expert", lambda *a: False)
    config = SimpleNamespace(hidden_size=16, hidden_act="silu", n_routed_experts=8,
        n_shared_experts=1 if shared else None, routed_scaling_factor=1.,
        moe_intermediate_size=32, num_experts_per_tok=2, norm_topk_prob=True,
        n_group=1, topk_group=1, num_experts=8, num_shared_experts=int(shared),
        expert_hidden_dim=32, route_norm=True)
    seen = []
    class BeforeAllocation(Exception):
        pass
    def before(**kwargs):
        seen.append(kwargs["is_sequence_parallel"])
        raise BeforeAllocation
    monkeypatch.setattr(official_moe, "make_parallel_config", before)
    cls = module.Glm4MoE if family == "glm4" else module.HYV3MoEFused
    with pytest.raises(BeforeAllocation), sp.runtime_sp_model_construction():
        cls(config, prefix="model.layers.0.mlp")
    assert seen == [True]


def test_factory_foreign_owner_preflight_and_runtime_health(core, factory, monkeypatch):
    foreign = lambda *args, **kwargs: None
    monkeypatch.setattr(hy_v3, "FusedMoEFactory", foreign)
    with pytest.raises(RuntimeError, match="foreign"):
        moe.install_moe_factory_policy()
    assert glm4_moe.FusedMoEFactory is factory


def test_factory_ownership_change_refuses_before_original(core, factory, monkeypatch):
    moe.install_moe_factory_policy()
    cached = glm4_moe.FusedMoEFactory
    monkeypatch.setattr(hy_v3, "FusedMoEFactory", lambda *a, **kw: pytest.fail("foreign invoked"))
    with pytest.raises(RuntimeError, match="ownership"):
        cached(8, 2, 16, 32, prefix="model.layers.0.mlp.experts")


def test_shared_factory_replacement_is_not_silently_ignored(core, factory, monkeypatch):
    moe.install_moe_factory_policy()
    cached = glm4_moe.FusedMoEFactory
    monkeypatch.setattr(official_moe, "FusedMoEFactory", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="ownership"):
        cached(8, 2, 16, 32, prefix="model.layers.0.mlp.experts")


def test_partial_model_factory_publication_stays_failed(core, factory, monkeypatch):
    first = SimpleNamespace(FusedMoEFactory=factory)
    class Second:
        FusedMoEFactory = staticmethod(factory)
        reject = True
        def __setattr__(self, name, value):
            if name == "FusedMoEFactory" and self.reject:
                raise RuntimeError("injected second alias write failure")
            object.__setattr__(self, name, value)
    second = Second()
    targets = {"vllm.model_executor.layers.fused_moe.layer": official_moe,
               "vllm.model_executor.models.glm4_moe": first,
               "vllm.model_executor.models.hy_v3": second}
    monkeypatch.setattr(moe, "import_module", targets.__getitem__)
    with pytest.raises(moe.ExtensionCommitError, match="incomplete"):
        moe.install_moe_factory_policy()
    assert first.FusedMoEFactory is not factory
    second.reject = False
    second.FusedMoEFactory = factory
    with pytest.raises(RuntimeError):
        moe.install_moe_factory_policy()
    with pytest.raises(RuntimeError):
        first.FusedMoEFactory(8, 2, 16, 32)
