"""CPU policy/lifecycle checks; real device mapping is a separate GPU gate."""
import importlib
from types import SimpleNamespace as NS

import pytest
import torch


def config():
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4',hidden_size=6144,
        n_routed_experts=256,num_experts_per_tok=8),dtype=torch.bfloat16,
        is_multimodal_model=False,enforce_eager=True),lora_config=None,
        use_v2_model_runner=True,compilation_config=NS(mode=0),
        scheduler_config=NS(async_scheduling=False),kernel_config=NS(moe_backend='deep_gemm'),
        additional_config={'hcu_static_eplb':{'expert_map_path':'r16.json',
            'num_redundant_experts':16,'static_dispatch_policy':'nearest'}},
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            prefill_context_parallel_size=16,decode_context_parallel_size=1,data_parallel_size=1,
            enable_expert_parallel=True,enable_eplb=False,enable_elastic_ep=False,
            enable_dbo=False,ubatch_size=0,all2all_backend='deepep_high_throughput'))


@pytest.fixture
def policy(monkeypatch):
    m=importlib.import_module('hcu_vllm_das.hy4_fused_routing')
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','1')
    monkeypatch.setenv('HCU_VLLM_HY4_FUSED_ROUTING','1')
    monkeypatch.setattr(m,'_device_arch',lambda:'gfx938')
    monkeypatch.setattr(m.rocm_aiter_ops,'is_fused_moe_enabled',lambda:False)
    monkeypatch.setattr(m.envs,'VLLM_BATCH_INVARIANT',False)
    return m


def tree():
    from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    layer=HYV4MoEFused.__new__(HYV4MoEFused)
    torch.nn.Module.__init__(layer)
    layer.enable_eplb=True
    layer.n_redundant_experts=16
    layer.n_logical_experts=256
    layer.n_physical_experts=272
    layer.expert_bias=torch.nn.Parameter(torch.zeros(256,dtype=torch.float32))
    state=NS(expert_load_view=object(),logical_to_physical_map=object(),
        logical_replica_count=object(),should_record_tensor=object(),
        num_unpadded_tokens_tensors=object())
    router=GroupedTopKRouter(top_k=8,global_num_experts=272,num_expert_group=1,topk_group=1,
        renormalize=True,scoring_func='sigmoid',routed_scaling_factor=2.827,
        e_score_correction_bias=layer.expert_bias,eplb_state=state,skip_padding=False)
    layer.experts=NS(router=router,moe_config=NS(num_experts=272,num_logical_experts=256,
        moe_parallel_config=NS(all2all_backend='deepep_high_throughput',
            tp_size=1,dp_size=1,pcp_size=16,ep_size=16,use_ep=True)))
    return torch.nn.ModuleList([layer]),layer,router


def test_deployed_static_p_is_admitted_only_when_requested(policy,monkeypatch):
    assert policy.enabled(config())
    monkeypatch.setenv('HCU_VLLM_HY4_FUSED_ROUTING','0')
    assert not policy.enabled(config())


@pytest.mark.parametrize('mode',['no_map','record','redundancy','async','graph','backend','pcp','dp','dynamic'])
def test_p_configuration_refuses_unvalidated_modes(policy,mode):
    cfg=config()
    if mode=='no_map':cfg.additional_config={}
    elif mode=='record':cfg.additional_config={'hcu_static_eplb':{
        'expert_map_record_path':'record.json','num_redundant_experts':16}}
    elif mode=='redundancy':cfg.additional_config['hcu_static_eplb']['num_redundant_experts']=8
    elif mode=='async':cfg.scheduler_config.async_scheduling=True
    elif mode=='graph':cfg.model_config.enforce_eager=False
    elif mode=='backend':cfg.parallel_config.all2all_backend='deepep_low_latency'
    elif mode=='pcp':cfg.parallel_config.prefill_context_parallel_size=8
    elif mode=='dp':cfg.parallel_config.data_parallel_size=2
    elif mode=='dynamic':cfg.parallel_config.enable_eplb=True
    assert not policy.enabled(cfg)


def test_static_p_retains_core_pipeline_and_private_padding_context(policy,monkeypatch):
    from vllm.forward_context import ForwardContext,get_forward_context,override_forward_context
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    root,layer,router=tree()
    original_select=router.select_experts.__func__
    original_class_compute=GroupedTopKRouter._compute_routing
    original_state=router.eplb_state
    original_type=router.routing_method_type
    weights=torch.ones(2,8,dtype=torch.float32)
    logical=torch.arange(8,dtype=torch.int64).expand(2,-1)
    captured=[]
    router.set_capture_fn(lambda ids:captured.append(ids.clone()))
    def map_after_capture(ids):
        assert len(captured)==1 and torch.equal(captured[0],logical)
        return ids+256
    monkeypatch.setattr(router,'_apply_eplb_mapping',map_after_capture)
    monkeypatch.setattr(policy,'_device_supported',lambda device:device.type=='cpu')
    mask=torch.tensor([False,True])
    context=ForwardContext(no_compile_layers={},attn_metadata=None,slot_mapping={},is_padding=mask)
    calls=[]
    def fused(**kwargs):
        assert get_forward_context().is_padding is None
        calls.append(kwargs)
        return weights,logical
    monkeypatch.setattr(policy,'fused_topk_bias',fused)
    assert policy.attach(root,config()) is root
    assert router._compute_routing.__func__ is not original_class_compute
    assert router.select_experts.__func__ is original_select
    assert GroupedTopKRouter._compute_routing is original_class_compute
    assert router.eplb_state is original_state and router.routing_method_type==original_type
    with override_forward_context(context):
        actual=router.select_experts(torch.ones(2,6144,dtype=torch.bfloat16),
            torch.zeros(2,256),torch.int32)
        assert get_forward_context() is context and context.is_padding is mask
    assert actual[0] is weights and actual[1].dtype==torch.int32
    assert torch.equal(actual[1],(logical+256).int())
    assert calls[0]['e_score_correction_bias'] is layer.expert_bias
    assert policy.attach(root,config()) is root


@pytest.mark.parametrize('mode',['no_state','logical','physical','slots','backend','padding'])
def test_p_layer_boundaries_keep_original_router(policy,mode):
    root,layer,router=tree()
    original=router._compute_routing.__func__
    if mode=='no_state':router.eplb_state=None
    elif mode=='logical':layer.n_logical_experts=128
    elif mode=='physical':router.global_num_experts=256
    elif mode=='slots':layer.n_redundant_experts=8
    elif mode=='backend':layer.experts.moe_config.moe_parallel_config.all2all_backend='deepep_low_latency'
    elif mode=='padding':router.skip_padding=True
    assert policy.attach(root,config()) is root
    assert router._compute_routing.__func__ is original


def test_static_mtp_attaches_routing_after_native_and_dense(monkeypatch):
    from hcu_vllm_das.models import hy4_static as m
    seen=[]
    def native(owner,**kwargs):
        torch.nn.Module.__init__(owner)
        owner.model=torch.nn.Module()
        owner.model.layers=torch.nn.ModuleDict()
        seen.append('native')
    monkeypatch.setattr(m,'_preflight',lambda *args,**kwargs:object())
    monkeypatch.setattr(m,'_MTP_INIT',native)
    monkeypatch.setattr(m.dense_fusion,'attach',lambda model,cfg:seen.append('dense') or model)
    monkeypatch.setattr(m.fused_routing,'attach',lambda model,cfg:seen.append('routing') or model)
    monkeypatch.setattr(m,'_publish_protocol',lambda *args:seen.append('metadata'))
    m.HcuHY4StaticMTP(vllm_config=config())
    assert seen==['native','dense','routing','metadata']
