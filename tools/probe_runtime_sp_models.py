"""Synthetic complete official GLM4/HY3 models; not trained-checkpoint/HE32 acceptance."""
from datetime import timedelta
from pathlib import Path
import json, sys, traceback
OUT=Path(sys.argv[1])

def worker(rank):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    # Bound this tiny synthetic diagnostic's allocator; it is not a serving
    # memory/performance qualification or a production runtime policy.
    torch.cuda.set_per_process_memory_fraction(.05)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import (ParallelConfig,VllmConfig,CacheConfig,ModelConfig,
        CompilationConfig,CompilationMode,set_current_vllm_config)
    from vllm.config.device import DeviceConfig
    from vllm.distributed import (init_distributed_environment,initialize_model_parallel,
        cleanup_dist_env_and_memory,get_tp_group)
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager,current_workspace_manager
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.model_executor.models import glm4_moe,hy_v3
    from hcu_vllm_das.parallel.runtime_sp_models import RuntimeSPGlm4MoeForCausalLM,RuntimeSPHYV3ForCausalLM
    from transformers import Glm4MoeConfig,HYV3Config
    result=dict(rank=rank,architecture=torch.cuda.get_device_properties(rank).gcnArchName,cases=[], failures=[])
    def compare(label,actual,expected):
        difference=float((actual.float()-expected.float()).abs().max())
        result.setdefault('comparisons',[]).append(dict(gate=label,max_abs_error=difference))
        try:torch.testing.assert_close(actual,expected,atol=.016,rtol=.016)
        except AssertionError as error:
            result['failures'].append(dict(gate=label,error=str(error),max_abs_error=difference))
        return difference
    dtype=torch.bfloat16
    torch.set_default_dtype(dtype)
    def parallel():
        return ParallelConfig(tensor_parallel_size=4,disable_custom_all_reduce=True,
            enable_expert_parallel=True,all2all_backend="allgather_reducescatter")
    seed=VllmConfig(parallel_config=parallel(),device_config=DeviceConfig("cuda"),
        additional_config={"hcu":{"enable_custom_sp":True}})
    def values(shape,epoch,factor=1/512):
        n=1
        for x in shape:n*=x
        return (((torch.arange(n).reshape(shape)+epoch*7)%31)-15).float()*factor
    def fill(model,family):
        for p in model.parameters():p.data.zero_()
        for component in [model.model.embed_tokens,model.lm_head]:
            component.weight.weight_loader(component.weight,values((128,128),3).to(dtype))
        for i,layer in enumerate(model.model.layers):
            a=layer.self_attn
            offset=0
            for name,size in [('q',512),('k',128),('v',128)]:
                weight=values((size,128),i*3+offset)
                a.qkv_proj.weight.weight_loader(a.qkv_proj.weight,weight.to(dtype),name)
                offset+=1
            a.o_proj.weight.weight_loader(a.o_proj.weight,values((128,512),i+2).to(dtype))
            for norm in [layer.input_layernorm,layer.post_attention_layernorm,a.q_norm,a.k_norm]:norm.weight.data.fill_(1)
            mlp=layer.mlp
            def dense(module,epoch):
                middle=256 if i==0 else 128
                for index in [0,1]:module.gate_up_proj.weight.weight_loader(module.gate_up_proj.weight,
                    values((middle,128),epoch+index).to(dtype),index)
                module.down_proj.weight.weight_loader(module.down_proj.weight,values((128,middle),epoch+2).to(dtype))
            if i==0:dense(mlp,7)
            else:
                from vllm.model_executor.model_loader.weight_utils import default_weight_loader
                default_weight_loader(mlp.gate.weight,values((8,128),5).float())
                routed=mlp.experts.routed_experts
                for expert in range(8):
                    for shard,param,shape in [('w1',routed.w13_weight,(128,128)),
                        ('w3',routed.w13_weight,(128,128)),('w2',routed.w2_weight,(128,128))]:
                        name={'w1':'gate_proj.weight','w3':'up_proj.weight','w2':'down_proj.weight'}[shard]
                        routed.weight_loader(param,values(shape,expert*3+len(shard)).to(dtype),name,shard,expert)
                dense(mlp.shared_experts if family=='glm4' else mlp.shared_mlp,11)
        model.model.norm.weight.data.fill_(1)
        for module in model.modules():
            method=getattr(module,'quant_method',None)
            if method is not None:method.process_weights_after_loading(module)
            if callable(getattr(module,'update_param_tp_status',None)):module.update_param_tp_status()
            if type(module).__name__=='Attention':
                module.process_weights_after_loading(dtype)
                module.kv_cache=torch.zeros((1,1,64,128),device='cuda',dtype=dtype)
        model.eval()
    try:
        with set_current_vllm_config(seed):
            from vllm.distributed import set_custom_all_reduce
            # Match the stock worker initialization for the already selected policy.
            set_custom_all_reduce(not seed.parallel_config.disable_custom_all_reduce)
            init_distributed_environment(world_size=4,rank=rank,local_rank=rank,
                distributed_init_method='file://'+str(OUT/'rendezvous'),backend='nccl',timeout=timedelta(seconds=120))
            initialize_model_parallel(4,1)
            init_workspace_manager(torch.device('cuda',rank))
        tp=get_tp_group()
        assert tp.device_communicator.ca_comm is None
        result["custom_all_reduce_disabled"] = True
        for family,hfcls,base,candidate in [('glm4',Glm4MoeConfig,glm4_moe.Glm4MoeForCausalLM,RuntimeSPGlm4MoeForCausalLM),
                ('hy3',HYV3Config,hy_v3.HYV3ForCausalLM,RuntimeSPHYV3ForCausalLM)]:
            hf=hfcls(vocab_size=128,hidden_size=128,intermediate_size=256,
                num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=2,
                head_dim=64,moe_intermediate_size=128,n_routed_experts=8,n_shared_experts=1,
                num_experts=8,num_shared_experts=1,expert_hidden_dim=128,num_experts_per_tok=2,
                first_k_dense_replace=1,n_group=1,topk_group=1,norm_topk_prob=True,route_norm=True,
                use_qk_norm=True,qk_norm=True,rms_norm_eps=1e-5,attention_bias=False,
                hidden_act='silu',routed_scaling_factor=1.,max_position_embeddings=256,
                rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5},
                tie_word_embeddings=False)
            hf.architectures=[base.__name__]
            cfgpath=OUT/f'{family}-config-rank{rank}';hf.save_pretrained(cfgpath)
            arms=[]
            for name,cls in [('reference',base),('candidate',candidate)]:
                cache=CacheConfig(block_size=64,cache_dtype='auto');cache.kv_cache_layout='LBHNC'
                config=VllmConfig(model_config=ModelConfig(model=str(cfgpath),skip_tokenizer_init=True,
                    dtype=dtype,max_model_len=256,enforce_eager=True),parallel_config=parallel(),cache_config=cache,
                    compilation_config=CompilationConfig(mode=CompilationMode.NONE),device_config=DeviceConfig('cuda'),
                    additional_config={'hcu':{'enable_custom_sp':True}})
                config.attention_config.backend=AttentionBackendEnum.FLASH_ATTN
                with set_current_vllm_config(config),torch.device('cuda'):
                    model=cls(vllm_config=config,prefix=f'{family}.{name}')
                    fill(model,family)
                print('CONSTRUCTED',rank,family,name,'embedding_dtype',model.model.embed_tokens.weight.dtype,flush=True)
                attention=[m for m in model.modules() if type(m).__name__=='Attention']
                assert len(attention)==2 and all(type(a.impl).__name__=='HcuFlashAttentionImpl' for a in attention)
                slots=torch.arange(7,device='cuda',dtype=torch.long)
                metadata=FlashAttentionMetadata(num_actual_tokens=7,max_query_len=7,
                    query_start_loc=torch.tensor([0,7],device='cuda',dtype=torch.int32),max_seq_len=7,
                    seq_lens=torch.tensor([7],device='cuda',dtype=torch.int32),
                    block_table=torch.zeros((1,1),device='cuda',dtype=torch.int32),slot_mapping=slots)
                arms.append((model,config,attention,slots,metadata))
            ids=torch.arange(7,device='cuda',dtype=torch.long)
            positions=torch.arange(7,device='cuda',dtype=torch.long)
            padding=torch.zeros(7,device='cuda',dtype=torch.bool)
            def update(epoch):
                ids.copy_((torch.arange(7,device='cuda')+epoch*3)%128)
                positions.copy_(torch.arange(7,device='cuda')+epoch)
                active=5 if epoch==1 else 7
                padding.copy_(torch.arange(7,device='cuda')>=active)
                for model,config,attention,slots,meta in arms:
                    slots.copy_(torch.arange(7,device='cuda'));slots[active:]=-1
                    meta.query_start_loc[1]=active;meta.seq_lens[0]=active
                return active
            def run(arm):
                model,config,attention,slots,metadata=arms[arm]
                with set_current_vllm_config(config),set_forward_context({a.layer_name:metadata for a in attention},config,
                        num_tokens=7,slot_mapping={a.layer_name:slots for a in attention},is_padding=padding):
                    return model(ids,positions)
            with torch.inference_mode():
                active=update(0)
                for _ in range(3):expected=run(0);actual=run(1)
                torch.cuda.synchronize()
                torch.save(dict(actual=actual.cpu(),reference=expected.cpu()),OUT/f'{family}-rank{rank}-eager.pt')
                eager_error=compare(family+':eager-vs-tp-reference',actual[:active],expected[:active])
                print('EAGER',rank,family,flush=True)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):actual=run(1)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph();dist.barrier(group=tp.device_group)
                with torch.cuda.graph(graph,stream=stream):actual=run(1)
                torch.cuda.current_stream().wait_stream(stream)
                address=(ids.data_ptr(),positions.data_ptr(),padding.data_ptr(),actual.data_ptr())
                errors=[]
                for epoch in [1,3]:
                    active=update(epoch);expected=run(0);eager=run(1).clone();graph.replay();torch.cuda.synchronize()
                    torch.save(dict(actual=actual.cpu(),reference=expected.cpu()),OUT/f'{family}-rank{rank}-graph{epoch}.pt')
                    compare(family+':graph-vs-tp-reference:'+str(epoch),actual[:active],expected[:active])
                    compare(family+':graph-vs-sp-eager:'+str(epoch),actual[:active],eager[:active])
                    assert torch.isfinite(actual).all() and torch.all(actual[active:]==0)
                    assert address==(ids.data_ptr(),positions.data_ptr(),padding.data_ptr(),actual.data_ptr())
                    errors.append(float((actual[:active].float()-expected[:active].float()).abs().max()))
                graph.reset();del graph
            result['cases'].append(dict(family=family,dtype=str(dtype),full_official_constructor=True,
                official_weight_loaders=True,synthetic_parameters=True,graph_epochs=[1,3],active_tokens=[5,7],
                selected_tokens=7,sp_padded_tokens=8,max_abs_error=errors,atol=.016,rtol=.016,
                eager_vs_tp_error=eager_error,trained_checkpoint=False,humaneval=False))
            print('FULL_MODEL',rank,family,errors,flush=True)
            dist.barrier(group=tp.device_group)
        result['success']=not result['failures']
        # Keep numerical failures in the receipt, but let all ranks finish
        # cleanup. Raising here makes mp.spawn terminate still-exiting peers.
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        (OUT/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        cleanup_dist_env_and_memory()

if __name__=='__main__':
    import torch.multiprocessing as mp
    mp.spawn(worker,nprocs=4,join=True)
    reports=[json.loads((OUT/f'rank{rank}.json').read_text()) for rank in range(4)]
    if any(report['failures'] for report in reports):
        raise SystemExit(1)
