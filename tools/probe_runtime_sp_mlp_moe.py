"""Installed official dense/shared MLP and routed MoE components, TP4/SP4/EP4.

Synthetic BF16/FP16 components with changed-input Graph replay. This is not a
checkpoint/model/HumanEval run. Use an external owned-process controller.
"""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import sys
import traceback

OUT = Path(sys.argv[1])
PREFIX = Path(sys.prefix).resolve()


def worker(rank):
    import torch
    import torch.nn.functional as F
    torch.cuda.set_device(rank)
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, CacheConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, cleanup_dist_env_and_memory, get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from hcu_adapter.bridges.vllm import require_extension_healthy
    for capability in ['runtime-sp-communication', 'hcu-flash-attention']:
        require_extension_healthy(capability)
    from hcu_vllm_das.parallel import runtime_sp_attention as sp
    from vllm.model_executor.models.glm4_moe import Glm4MoeAttention
    from vllm.model_executor.models.hy_v3 import HYV3Attention
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    sp.install_attention_layers()
    result = dict(rank=rank, architecture=torch.cuda.get_device_properties(rank).gcnArchName, cases=[])
    parallel = ParallelConfig(tensor_parallel_size=4, disable_custom_all_reduce=True, enable_expert_parallel=True, all2all_backend="allgather_reducescatter")
    cache = CacheConfig(block_size=64, cache_dtype='auto')
    cache.kv_cache_layout = 'LBHNC'
    config = VllmConfig(parallel_config=parallel, cache_config=cache, device_config=DeviceConfig('cuda'),
        additional_config={'hcu': {'enable_custom_sp': True}})
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    hidden, heads, hd, tokens = 128, 8, 64, 8

    def values(shape, epoch, factor):
        count = 1
        for dim in shape:count *= dim
        return ((((torch.arange(count).reshape(shape) + epoch * 7) % 31) - 15).float() * factor)

    try:
        with set_current_vllm_config(config):
            init_distributed_environment(world_size=4, rank=rank, local_rank=rank,
                distributed_init_method='file://' + str(OUT/'rendezvous'), backend='nccl', timeout=timedelta(seconds=120))
            initialize_model_parallel(4, 1)
            tp = get_tp_group()
            from vllm.v1.worker.workspace import init_workspace_manager, current_workspace_manager
            init_workspace_manager(torch.device("cuda", rank))
            assert type(tp.device_communicator).__name__ == 'RuntimeSPCommunicator'
            case_no = 0
            from vllm.model_executor.models import glm4_moe, hy_v3
            from hcu_vllm_das.parallel import runtime_sp_moe as moe
            moe.install_moe_factory_policy()
            # Real native MLP and routed-expert execution over unequal token shards.
            def mlp_reference(x, up, down, dtype):
                z=F.linear(x.float(),up.float()).to(dtype).float()
                gate,value=z.chunk(2,-1)
                activation=(F.silu(gate)*value).to(dtype).float()
                return F.linear(activation,down.float()).to(dtype)

            def check_native(run, update, label):
                expected=update(0)
                with torch.inference_mode():
                    for _ in range(3):out=run()
                    torch.cuda.synchronize()
                    torch.save(dict(actual=out.cpu(),reference=expected),OUT/f'{label}-rank{rank}-eager.pt')
                    torch.testing.assert_close(out.cpu(),expected[rank*2:(rank+1)*2],atol=.016,rtol=.016)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):out=run()
                    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                    if "moe-shared" in label:current_workspace_manager().lock()
                    graph=torch.cuda.CUDAGraph()
                    dist.barrier(group=tp.device_group)
                    with torch.cuda.graph(graph,stream=stream):out=run()
                    torch.cuda.current_stream().wait_stream(stream)
                    address=(local.data_ptr(),out.data_ptr());previous=out.clone();errors=[]
                    for epoch in [1,3]:
                        expected=update(epoch);graph.replay();torch.cuda.synchronize()
                        target=expected[rank*2:(rank+1)*2]
                        torch.save(dict(actual=out.cpu(),reference=target,epoch=epoch),OUT/f'{label}-rank{rank}-graph{epoch}.pt')
                        torch.testing.assert_close(out.cpu(),target,atol=.016,rtol=.016)
                        assert not torch.equal(previous,out)
                        assert address==(local.data_ptr(),out.data_ptr())
                        errors.append(float((out.cpu().float()-target.float()).abs().max()))
                        previous=out.clone()
                    graph.reset();del graph
                dist.barrier(group=tp.device_group)
                print('NATIVE',rank,label,errors,flush=True)
                return errors

            for family,mlp_cls,moe_cls in [('glm4',glm4_moe.Glm4MoeMLP,glm4_moe.Glm4MoE),
                                          ('hy3',hy_v3.HYV3FeedForward,hy_v3.HYV3MoEFused)]:
                for dtype in [torch.bfloat16,torch.float16]:
                    torch.set_default_dtype(dtype)
                    for shared in [False,True]:
                        prefix='model.layers.20.mlp'
                        if shared:prefix += '.shared_experts' if family=='glm4' else '.shared_mlp'
                        with torch.device('cuda'),sp.runtime_sp_model_construction():
                            module=mlp_cls(128,192,'silu',reduce_results=not shared,prefix=prefix)
                        up=values((384,128),3,1/64).to(dtype)
                        down=values((128,192),5,1/128).to(dtype)
                        for index in [0,1]:
                            module.gate_up_proj.weight.weight_loader(module.gate_up_proj.weight,up[index*192:(index+1)*192],index)
                        module.down_proj.weight.weight_loader(module.down_proj.weight,down)
                        for layer in [module.gate_up_proj,module.down_proj]:
                            layer.quant_method.process_weights_after_loading(layer)
                            layer.update_param_tp_status()
                            assert layer.tp_size==1 and layer.weight.tp_size==1
                        local=torch.empty(2,128,device='cuda',dtype=dtype)
                        def update(epoch):
                            x=values((8,128),epoch,1/8).to(dtype)
                            local.copy_(x[rank*2:(rank+1)*2]);return mlp_reference(x,up,down,dtype)
                        errors=check_native(lambda:module(local),update,f'{family}-{dtype}-shared{shared}')
                        result['cases'].append(dict(kind='mlp',family=family,dtype=str(dtype),shared=shared,
                            max_abs_errors=errors,graph_epochs=[1,3],stable_addresses=True,
                            native_execution=True,official_loaders=True,atol=.016,rtol=.016))
                        del module

                    case_no+=1
                    hf=SimpleNamespace(hidden_size=128,hidden_act='silu',n_routed_experts=8,
                        n_shared_experts=1,routed_scaling_factor=1.,moe_intermediate_size=128,
                        num_experts_per_tok=2,norm_topk_prob=True,n_group=1,topk_group=1,
                        num_experts=8,num_shared_experts=1,expert_hidden_dim=128,route_norm=True)
                    with torch.device('cuda'),sp.runtime_sp_model_construction():
                        module=moe_cls(hf,prefix=f'model.layers.{30+case_no}.mlp')
                    runner=module.experts;routed=runner.routed_experts
                    assert runner.moe_config.is_sequence_parallel and runner.moe_config.sp_size==4
                    assert runner.moe_config.tp_size==1 and runner.moe_config.ep_size==4
                    generator=torch.Generator().manual_seed(77)
                    gate_weight=torch.randn((8,128),generator=generator,dtype=torch.float32)*.05
                    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
                    getattr(module.gate.weight,'weight_loader',default_weight_loader)(module.gate.weight,gate_weight)
                    correction=module.gate.e_score_correction_bias if family=='glm4' else module.expert_bias
                    correction.data.zero_()
                    all_up=[];all_down=[]
                    for expert in range(8):
                        up=values((256,128),expert*2+1,1/128).to(dtype)
                        down=values((128,128),expert*3+2,1/128).to(dtype)
                        all_up.append(up);all_down.append(down)
                        for tensor,param,name,shard in [(up[:128],routed.w13_weight,'gate_proj.weight','w1'),
                                (up[128:],routed.w13_weight,'up_proj.weight','w3'),
                                (down,routed.w2_weight,'down_proj.weight','w2')]:
                            routed.weight_loader(param,tensor,name,shard,expert)
                    shared=module.shared_experts if family=='glm4' else module.shared_mlp
                    shared_up=values((256,128),8,1/128).to(dtype)
                    shared_down=values((128,128),9,1/128).to(dtype)
                    for index in [0,1]:
                        shared.gate_up_proj.weight.weight_loader(shared.gate_up_proj.weight,shared_up[index*128:(index+1)*128],index)
                    shared.down_proj.weight.weight_loader(shared.down_proj.weight,shared_down)
                    for layer in [module.gate,shared.gate_up_proj,shared.down_proj,routed]:
                        layer.quant_method.process_weights_after_loading(layer)
                    local=torch.empty(2,128,device='cuda',dtype=dtype)
                    def update(epoch):
                        x=values((8,128),epoch,1/8).to(dtype)
                        local.copy_(x[rank*2:(rank+1)*2])
                        scores=F.linear(x.float(),gate_weight).sigmoid()
                        weights,ids=scores.topk(2,dim=-1);weights=weights/weights.sum(-1,keepdim=True)
                        total=torch.zeros((8,128),dtype=torch.float32)
                        for expert in range(8):
                            # Independent CPU implementation of the observed
                            # native activation/GEMM cast boundaries. Core's
                            # silu_kernel<T> returns T before multiplying the
                            # up half; Triton GEMM2 weights FP32 before casting.
                            z=F.linear(x.float(),all_up[expert].float()).to(dtype).float()
                            gate,value=z.chunk(2,-1)
                            activation=(F.silu(gate).to(dtype).float()*value).to(dtype).float()
                            y=F.linear(activation,all_down[expert].float())
                            coefficient=(weights*(ids==expert)).sum(-1)
                            total += (y*coefficient[:,None]).to(dtype).float()
                        return (total.to(dtype)+mlp_reference(x,shared_up,shared_down,dtype)).to(dtype)
                    def run_moe():
                        with set_forward_context(None,config,num_tokens=8):return module(local)
                    errors=check_native(run_moe,update,f'{family}-{dtype}-moe-shared')
                    result['cases'].append(dict(kind='moe',family=family,dtype=str(dtype),sp_size=4,ep_size=4,tp_size=1,
                        routed_weight_shape=list(routed.w13_weight.shape),quant_method=type(routed.quant_method).__name__,
                        max_abs_errors=errors,graph_epochs=[1,3],stable_addresses=True,
                        native_execution=True,official_loaders=True,atol=.016,rtol=.016))
                    del module,runner,routed,shared

            result['origins']={}
            for name,obj in [('layout',sp),('glm4',Glm4MoeAttention),('hy3',HYV3Attention)]:
                file=Path(inspect.getfile(obj)).resolve();assert file.is_relative_to(PREFIX)
                result['origins'][name]=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            result['success']=True
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        (OUT/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        cleanup_dist_env_and_memory()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(worker,nprocs=4,join=True)
