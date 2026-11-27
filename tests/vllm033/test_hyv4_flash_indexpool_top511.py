# SPDX-License-Identifier: Apache-2.0
"""Portable active-model dispatch; native GPU execution is qualified separately."""
import pytest
import torch


class CpuHcuTensor(torch.Tensor):
    @property
    def is_cuda(self):return True


@pytest.fixture
def api(monkeypatch):
    from hcu_vllm_das.models.hy_v4_flash import indexpool_ops
    from hcu_adapter.kernels import indexer_topk_lightop
    monkeypatch.setattr(indexpool_ops,'is_hcu_platform',lambda:True)
    monkeypatch.delenv('VLLM_HCU_USE_CUSTOM_OPS',raising=False)
    return indexpool_ops,indexer_topk_lightop


def reference(scores,upper,budget):
    masked=scores.masked_fill(torch.arange(scores.shape[1])[None,:]>=upper[:,None],-torch.inf)
    values,ids=masked.topk(min(budget,scores.shape[1]),dim=-1)
    sentinel=torch.iinfo(torch.int32).max
    out=torch.where(values!=-torch.inf,ids,sentinel).int().sort(-1).values
    return torch.nn.functional.pad(out.masked_fill(out==sentinel,-1),(0,budget-out.shape[1]),value=-1)


def test_active_model_dispatches_original_logits_and_lengths_without_full_mask(api,monkeypatch):
    model,vendor=api;calls=[]
    scores=torch.randn(2,1024).as_subclass(CpuHcuTensor);upper=torch.tensor([512,0])
    expected=torch.full((2,511),-1,dtype=torch.int32)
    def select(x,lengths,budget):
        calls.append((x,lengths,budget));return expected
    monkeypatch.setattr(vendor,'select_indexpool_groups',select,raising=False)
    actual=model.select_groups(scores,upper,511)
    assert actual is expected
    assert len(calls)==1 and calls[0][0] is scores and calls[0][1] is upper and calls[0][2]==511


def test_native_none_keeps_original_torch_selection(api,monkeypatch):
    model,vendor=api
    monkeypatch.setattr(vendor,'select_indexpool_groups',lambda *a:None,raising=False)
    scores=torch.linspace(-2,2,2048).reshape(2,1024).as_subclass(CpuHcuTensor)
    upper=torch.tensor([513,0])
    assert torch.equal(model.select_groups(scores,upper,511),reference(scores,upper,511))


def test_missing_new_adapter_api_keeps_original_selection(api,monkeypatch):
    model,vendor=api;monkeypatch.delattr(vendor,'select_indexpool_groups',raising=False)
    scores=torch.arange(1024).float()[None,:].as_subclass(CpuHcuTensor);upper=torch.tensor([512])
    assert torch.equal(model.select_groups(scores,upper,511),reference(scores,upper,511))


def test_real_native_failure_is_not_hidden(api,monkeypatch):
    model,vendor=api
    def fail(*a):raise RuntimeError('native launch failed')
    monkeypatch.setattr(vendor,'select_indexpool_groups',fail,raising=False)
    with pytest.raises(RuntimeError,match='native launch failed'):
        model.select_groups(torch.ones(1,1024).as_subclass(CpuHcuTensor),torch.tensor([1024]),511)


def test_other_budget_preserves_existing_path(api,monkeypatch):
    model,vendor=api
    def fail(*a):raise AssertionError('unsupported budget must not dispatch')
    monkeypatch.setattr(vendor,'select_indexpool_groups',fail,raising=False)
    scores=torch.arange(32).float()[None,:].as_subclass(CpuHcuTensor);upper=torch.tensor([19])
    assert torch.equal(model.select_groups(scores,upper,7),reference(scores,upper,7))


@pytest.mark.parametrize('off',['0','false','off','no'])
def test_explicit_master_off_retains_original_selection(api,monkeypatch,off):
    model,vendor=api
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS',off)
    def fail(*a):raise AssertionError('master OFF must bypass native')
    monkeypatch.setattr(vendor,'select_indexpool_groups',fail,raising=False)
    scores=torch.arange(512).float()[None,:].as_subclass(CpuHcuTensor);upper=torch.tensor([512])
    assert torch.equal(model.select_groups(scores,upper,511),reference(scores,upper,511))


def test_dispatch_reuses_the_registered_default_on_master_query(api,monkeypatch):
    from types import SimpleNamespace
    from hcu_vllm_das import environment
    model,vendor=api
    registry=SimpleNamespace(environment_variables={},compile_factors=lambda:None)
    environment.register_compile_factors(registry)
    assert registry.environment_variables['VLLM_HCU_DAS_FP8_FUSIONS_ENABLED'] is environment.fp8_fusions_enabled
    assert environment.fp8_fusions_enabled() is True
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','false')
    assert registry.environment_variables['VLLM_HCU_DAS_FP8_FUSIONS_ENABLED']() is False
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','1')
    expected=torch.full((1,511),-1,dtype=torch.int32)
    monkeypatch.setattr(vendor,'select_indexpool_groups',lambda *a:expected,raising=False)
    assert model.select_groups(torch.ones(1,512).as_subclass(CpuHcuTensor),torch.tensor([512]),511) is expected
