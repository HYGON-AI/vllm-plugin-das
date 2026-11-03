"""PCP provides the expert-parallel world for static-only HYV4 W4A8."""
from types import SimpleNamespace
import pytest

from tests.runtime_patch.test_glm52_pcp_config import _make_pcp_config, _native_hyv4_mtp_config
from vllm_hcu.patch.platform.core_fix import patch_vllm_config


def config():
    value=_native_hyv4_mtp_config(_make_pcp_config,architecture='HYV4ForCausalLM',pp=1,tp=1,pcp=16)
    value.parallel_config.enable_eplb=True
    value.parallel_config.eplb_config=SimpleNamespace(num_redundant_experts=0)
    value.model_config.quantization='compressed-tensors'
    value.additional_config['hcu']['expert_map_path']='/tmp/static.json'
    return value


def test_exact_static_pcp16_native_mtp_contract_is_admitted():
    assert patch_vllm_config._validate_hcu_pcp_scope(config()) is True


@pytest.mark.parametrize('path,value',[
    ('parallel_config.prefill_context_parallel_size',8),
    ('parallel_config.tensor_parallel_size',2),
    ('parallel_config.enable_eplb',False),
    ('parallel_config.eplb_config.num_redundant_experts',16),
    ('model_config.quantization','fp8')])
def test_other_static_modes_stay_closed(path,value):
    current=config();owner=current;fields=path.split('.')
    for name in fields[:-1]:owner=getattr(owner,name)
    setattr(owner,fields[-1],value)
    with pytest.raises(ValueError):patch_vllm_config._validate_hcu_pcp_scope(current)


def test_dynamic_or_recording_pcp_eplb_is_still_rejected():
    for change in [dict(expert_map_path=None),dict(expert_map_record_path='/tmp/record.json')]:
        current=config();current.additional_config['hcu'].update(change)
        with pytest.raises(ValueError):patch_vllm_config._validate_hcu_pcp_scope(current)


def test_upstream_parallel_config_counts_pcp_for_eplb(monkeypatch):
    import vllm.config.parallel as module
    monkeypatch.setattr(module.current_platform,'is_cuda_alike',lambda:True)
    current=module.ParallelConfig(tensor_parallel_size=1,data_parallel_size=1,
        prefill_context_parallel_size=16,enable_expert_parallel=True,enable_eplb=True,
        all2all_backend='deepep_high_throughput',distributed_executor_backend='mp')
    assert current.enable_eplb and current.tensor_parallel_size==1 and current.data_parallel_size==1
    assert current.prefill_context_parallel_size==16


def test_upstream_single_rank_eplb_is_still_rejected(monkeypatch):
    import vllm.config.parallel as module
    monkeypatch.setattr(module.current_platform,'is_cuda_alike',lambda:True)
    with pytest.raises(ValueError,match='EPLB requires'):
        module.ParallelConfig(tensor_parallel_size=1,data_parallel_size=1,
            prefill_context_parallel_size=1,enable_expert_parallel=True,enable_eplb=True,
            distributed_executor_backend='mp')
