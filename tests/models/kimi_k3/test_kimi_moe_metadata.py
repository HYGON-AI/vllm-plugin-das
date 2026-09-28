"""Kimi's real model/protocol seam, without model loading or device allocation."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import KimiK3W4A8MoEMethod
from vllm_hcu.models.kimi_k3.amd import linear


WEIGHTS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def make_runner(logical=4, physical=4, local=2):
    owner = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(owner)
    owner.local_num_experts = local
    owner.moe_config = SimpleNamespace(num_experts=physical, num_logical_experts=logical)
    for name in WEIGHTS:
        dtype = torch.float32 if "scale" in name else torch.int8
        owner.register_parameter(name, torch.nn.Parameter(
            torch.ones(local, 32, 16, dtype=dtype), requires_grad=False))
    owner.e_score_correction_bias = torch.nn.Parameter(torch.ones(logical))
    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = owner
    runner.router = SimpleNamespace(eplb_state=EplbLayerState())
    return runner


def make_model(monkeypatch, *, start=1, end=5, mismatch=False):
    inner = torch.nn.Module()
    inner.start_layer, inner.end_layer = start, end
    inner.layers = torch.nn.ModuleList()
    for index in range(6):
        layer = torch.nn.Module()
        if index == 2:
            layer.mlp = torch.nn.Linear(2, 2)
        else:
            layer.mlp = linear.KimiMoE.__new__(linear.KimiMoE)
            torch.nn.Module.__init__(layer.mlp)
            layer.mlp.experts = make_runner(local=1 if mismatch and index == 3 else 2)
        inner.layers.append(layer)
    monkeypatch.setattr(linear, "KimiLinearModel", lambda **kwargs: inner)
    monkeypatch.setattr(linear, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    monkeypatch.setattr(linear, "LogitsProcessor", lambda *args, **kwargs: torch.nn.Identity())
    config = SimpleNamespace(num_experts=4, num_expert_group=2,
                             num_shared_experts=1, vocab_size=16)
    model = linear.KimiLinearForCausalLM(vllm_config=SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config), quant_config=None))
    return model


def test_constructor_registers_only_pipeline_local_moe_layers(monkeypatch):
    model = make_model(monkeypatch)
    assert is_mixture_of_experts(model)
    assert model.num_moe_layers == 3
    assert model.moe_layers == [model.model.layers[i].mlp.experts for i in (1, 3, 4)]
    assert (model.num_logical_experts, model.num_routed_experts,
            model.num_physical_experts, model.num_local_physical_experts,
            model.num_redundant_experts, model.num_shared_experts,
            model.num_expert_groups) == (4, 4, 4, 2, 0, 1, 2)
    assert model.expert_weights == []  # no pre-repack Parameter snapshots
    assert not any(name.startswith("moe_layers.") for name, _ in
                   model.named_parameters(remove_duplicate=False))


def test_protocol_reads_current_packed_parameters_and_scales(monkeypatch):
    model = make_model(monkeypatch)
    owner = model.moe_layers[0].routed_experts
    old = owner.w13_weight
    owner.w13_weight = torch.nn.Parameter(torch.zeros_like(old), requires_grad=False)
    loads = torch.zeros(3, 4, dtype=torch.int32)
    maps = torch.arange(4).expand(3, 4).unsqueeze(-1).clone()
    replicas = torch.ones(3, 4, dtype=torch.int32)
    model.set_eplb_state(loads, maps, replicas)
    assert len(model.expert_weights) == 3
    for index, runner in enumerate(model.moe_layers):
        assert [w.data_ptr() for w in model.expert_weights[index]] == [
            getattr(runner.routed_experts, name).data_ptr() for name in WEIGHTS]
        assert runner.eplb_state.expert_load_view.data_ptr() == loads[index].data_ptr()
        assert runner.eplb_state.logical_to_physical_map.data_ptr() == maps[index].data_ptr()
    assert model.expert_weights[0][0].data_ptr() != old.data_ptr()
    # Register again after a second post-load replacement: no stale cached views.
    owner.w2_weight_scale = torch.nn.Parameter(torch.full_like(owner.w2_weight_scale, 3),
                                             requires_grad=False)
    model.set_eplb_state(loads, maps, replicas)
    assert model.expert_weights[0][3].data_ptr() == owner.w2_weight_scale.data_ptr()


def test_empty_pipeline_range_is_not_an_moe_model(monkeypatch):
    model = make_model(monkeypatch, start=2, end=3)
    assert model.num_moe_layers == 0
    assert not is_mixture_of_experts(model)
    assert model.expert_weights == []


def test_multimodal_wrapper_resolves_the_text_moe_protocol(monkeypatch):
    from vllm_hcu.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration
    from vllm.v1.worker.gpu.eplb_utils import _unwrap_moe
    wrapper = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = make_model(monkeypatch)
    wrapper._language_model_names = ["language_model"]
    text = _unwrap_moe(wrapper)
    assert text is wrapper.language_model
    assert is_mixture_of_experts(text)


def test_layer_count_mismatch_fails_before_metadata_is_published(monkeypatch):
    with pytest.raises(ValueError, match="expert counts"):
        make_model(monkeypatch, mismatch=True)


def test_metadata_update_cannot_silently_resize_packed_weights(monkeypatch):
    model = make_model(monkeypatch)
    model.update_physical_experts_metadata(4, 2)
    with pytest.raises(ValueError, match="resize"):
        model.update_physical_experts_metadata(8, 4)
    assert (model.num_physical_experts, model.num_local_physical_experts) == (4, 2)


def test_actual_runner_registers_multimodal_text_before_forward(monkeypatch):
    import ast
    from pathlib import Path
    from vllm.model_executor.models.interfaces import SupportsMultiModal
    from vllm_hcu.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration
    path = Path(__file__).resolve().parents[3] / 'vllm_hcu/v1/hcu_model_runner.py'
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_register_eplb_model')
    context = dict(SupportsMultiModal=SupportsMultiModal,
                   is_mixture_of_experts=is_mixture_of_experts,
                   logger=SimpleNamespace(info_once=lambda *a: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), 'exec'), context)
    wrapper = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = make_model(monkeypatch)
    wrapper._language_model_names = ['language_model']
    calls = []
    runner = SimpleNamespace(model=wrapper, parallel_config=SimpleNamespace(enable_eplb=True),
        model_config=SimpleNamespace(model='kimi'),
        eplb_state=SimpleNamespace(add_model=lambda *a: calls.append(a)))
    register = context['_register_eplb_model']
    assert register(runner, False) == 1
    assert calls == [(wrapper.language_model, runner.model_config)]
    assert register(runner, True) == 0
    runner.parallel_config.enable_eplb = False
    assert register(runner, False) == 0
    assert len(calls) == 1
    step_method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and node.name == 'eplb_step')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step_method], type_ignores=[])),
                 str(path), 'exec'), context)
    runner.parallel_config.enable_eplb = True
    runner.parallel_config.eplb_config = SimpleNamespace(log_balancedness=True)
    runner.eep_eplb_suppressed = False
    runner.get_model = lambda: wrapper
    steps = []
    runner.eplb_state.step = lambda *a, **kw: steps.append((a, kw))
    context['eplb_step'](runner, is_dummy=True, is_profile=True)
    assert steps == [((True, True), {'log_stats': True})]


@pytest.mark.parametrize('tokens', [0, 1, 257])
def test_runner_updates_unpadded_eplb_count_before_forward(tokens):
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / 'vllm_hcu/v1/hcu_model_runner.py'
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'execute_model')
    block = next(node for node in method.body if isinstance(node, ast.If)
                 and any(isinstance(call, ast.Attribute) and call.attr == 'prepare_forward'
                         for call in ast.walk(node)))
    forwards = [node.lineno for node in ast.walk(method) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == '_model_forward']
    assert forwards and block.lineno < min(forwards)
    calls = []
    runner = SimpleNamespace(eplb_state=SimpleNamespace(
        prepare_forward=lambda *args: calls.append(args)), model_config=object())
    slices = object()
    context = dict(self=runner, num_tokens_unpadded=tokens, ubatch_slices_padded=slices)
    code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                   str(path), 'exec')
    exec(code, context)
    assert calls == [(runner.model_config, tokens, slices)]
    runner.eplb_state = None
    exec(code, context)
    assert len(calls) == 1


@pytest.mark.parametrize("ht,ll", [(False, False), (True, False), (False, True)])
def test_metadata_does_not_enable_unvalidated_online_w4a8_eplb(ht, ll):
    config = SimpleNamespace(activation=SimpleNamespace(value="situ"), swiglu_beta=None,
        moe_parallel_config=SimpleNamespace(use_deepep_ht_kernels=ht, use_deepep_ll_kernels=ll))
    assert not KimiK3W4A8MoEMethod(config).supports_eplb
