"""Installed official attention modules, native KV writer/reader, four TP ranks.

This is a synthetic attention component, not a checkpoint/model/HE32 run.
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
    parallel = ParallelConfig(tensor_parallel_size=4, disable_custom_all_reduce=True)
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

    def reference(x, w, b, outw, positions, kvheads, dtype):
        # All reference math is independent CPU FP32 over unsharded weights.
        qkv = F.linear(x.float(), w.float(), b.float()).to(dtype).float()
        q, k, v = qkv.split([heads * hd, kvheads * hd, kvheads * hd], -1)
        q, k, v = [a.reshape(tokens, n, hd) for a, n in [(q, heads), (k, kvheads), (v, kvheads)]]
        q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-5)).to(dtype).float()
        k = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-5)).to(dtype).float()
        rotary = hd // 2
        frequencies = 10000.0 ** (-torch.arange(0, rotary, 2).float() / rotary)
        angle = positions.float()[:, None] * frequencies[None]
        c, s = angle.cos()[:, None], angle.sin()[:, None]
        def rope(t):
            a, b = t[..., :rotary//2], t[..., rotary//2:rotary]
            return torch.cat((a*c-b*s, b*c+a*s, t[..., rotary:]), -1).to(dtype).float()
        q, k = rope(q), rope(k)
        k = k.repeat_interleave(heads // kvheads, dim=1)
        v = v.repeat_interleave(heads // kvheads, dim=1)
        scores = torch.einsum('qhd,khd->hqk', q, k) * hd ** -.5
        causal = torch.arange(tokens)[None, :] > torch.arange(tokens)[:, None]
        scores.masked_fill_(causal[None], -torch.inf)
        attention = torch.einsum('hqk,khd->qhd', scores.softmax(-1), v).to(dtype).float()
        return F.linear(attention.reshape(tokens, heads * hd), outw.float()).to(dtype)

    try:
        with set_current_vllm_config(config):
            init_distributed_environment(world_size=4, rank=rank, local_rank=rank,
                distributed_init_method='file://' + str(OUT/'rendezvous'), backend='nccl', timeout=timedelta(seconds=120))
            initialize_model_parallel(4, 1)
            tp = get_tp_group()
            assert type(tp.device_communicator).__name__ == 'RuntimeSPCommunicator'
            case_no = 0
            for family, cls in [('glm4', Glm4MoeAttention), ('hy3', HYV3Attention)]:
                for dtype in [torch.bfloat16, torch.float16]:
                    for kvheads in [2, 8]:
                        case_no += 1
                        torch.set_default_dtype(dtype)
                        prefix = f'model.layers.{case_no}.self_attn'
                        hf = SimpleNamespace(rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                            'partial_rotary_factor': .5}, head_dim=hd, qk_norm=True)
                        kwargs = dict(config=hf, hidden_size=hidden, num_heads=heads, num_kv_heads=kvheads,
                            max_position_embeddings=256, head_dim=hd, rms_norm_eps=1e-5,
                            cache_config=cache, prefix=prefix)
                        if family == 'glm4':kwargs.update(qkv_bias=True, use_qk_norm=True)
                        else:kwargs.update(rope_parameters=hf.rope_parameters)
                        with torch.device('cuda'), sp.runtime_sp_attention_construction():
                            module = cls(**kwargs)
                        assert type(module.qkv_proj) is sp.RuntimeSPQKVParallelLinear
                        assert type(module.o_proj) is sp.RuntimeSPRowParallelLinear
                        if family == 'hy3':assert module.hpc_rope_norm is None
                        module.eval()
                        width = (heads + 2 * kvheads) * hd
                        weight = values((width, hidden), 0, 1/256).to(dtype)
                        bias = values((width,), 0, 1/256).to(dtype)
                        if family == 'hy3':bias.zero_()
                        outw = values((hidden, heads * hd), 2, 1/512).to(dtype)
                        params = module.qkv_proj
                        offset = 0
                        for name, size in [('q', heads*hd), ('k', kvheads*hd), ('v', kvheads*hd)]:
                            params.weight.weight_loader(params.weight, weight[offset:offset+size], name)
                            if params.bias is not None:params.bias.weight_loader(params.bias, bias[offset:offset+size], name)
                            offset += size
                        module.o_proj.weight.weight_loader(module.o_proj.weight, outw)
                        for layer in [module.qkv_proj, module.o_proj]:
                            layer.quant_method.process_weights_after_loading(layer)
                            layer.update_param_tp_status()
                        assert params.weight.tp_rank == 0 and params.weight.tp_size == 1
                        assert module.o_proj.weight.tp_rank == 0 and module.o_proj.weight.tp_size == 1
                        module.q_norm.weight.data.fill_(1)
                        module.k_norm.weight.data.fill_(1)
                        attention = module.attn
                        attention.process_weights_after_loading(dtype)
                        assert type(attention.impl).__name__ == 'HcuFlashAttentionImpl'
                        attention.kv_cache = torch.zeros((1, max(1, kvheads//4), 64, 2*hd), device='cuda', dtype=dtype)
                        positions = torch.arange(tokens, device='cuda', dtype=torch.long)
                        local = torch.empty(tokens//4, hidden, device='cuda', dtype=dtype)
                        slots = torch.arange(tokens, device='cuda', dtype=torch.long)
                        metadata = FlashAttentionMetadata(num_actual_tokens=tokens, max_query_len=tokens,
                            query_start_loc=torch.tensor([0,tokens], device='cuda',dtype=torch.int32),
                            max_seq_len=tokens, seq_lens=torch.tensor([tokens],device='cuda',dtype=torch.int32),
                            block_table=torch.zeros((1,1),device='cuda',dtype=torch.int32),slot_mapping=slots)
                        def update(epoch):
                            x = values((tokens,hidden),epoch,1/64).to(dtype)
                            local.copy_(x[rank*2:(rank+1)*2])
                            positions.copy_(torch.arange(tokens,device='cuda') + epoch)
                            return reference(x,weight,bias,outw,torch.arange(tokens)+epoch,kvheads,dtype)
                        def run():
                            with set_forward_context({attention.layer_name:metadata}, config, num_tokens=tokens,
                                slot_mapping={attention.layer_name:slots}):
                                return module(positions, local)
                        expected = update(0)
                        with torch.inference_mode():
                            for _ in range(3):out=run()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(out.cpu(),expected[rank*2:(rank+1)*2],atol=1.6e-2,rtol=1.6e-2)
                            print('STAGE',rank,family,str(dtype),kvheads,'eager-valid',flush=True)
                            # Warm the exact capture stream before hipBLASLt resources
                            # may be initialized there. Keep kernels/assertions intact.
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(3):out=run()
                            torch.cuda.current_stream().wait_stream(stream)
                            torch.cuda.synchronize()
                            torch.testing.assert_close(out.cpu(),expected[rank*2:(rank+1)*2],atol=1.6e-2,rtol=1.6e-2)
                            print('STAGE',rank,family,str(dtype),kvheads,'capture-stream-warm',flush=True)
                            graph = torch.cuda.CUDAGraph()
                            dist.barrier(group=tp.device_group)
                            with torch.cuda.graph(graph,stream=stream):out=run()
                            print('STAGE',rank,family,str(dtype),kvheads,'graph-captured',flush=True)
                            torch.cuda.current_stream().wait_stream(stream)
                            addr=(local.data_ptr(),positions.data_ptr(),out.data_ptr(),attention.kv_cache.data_ptr())
                            previous = out.clone()
                            errors=[]
                            for epoch in [1,3]:
                                expected=update(epoch)
                                graph.replay();torch.cuda.synchronize()
                                target=expected[rank*2:(rank+1)*2]
                                torch.testing.assert_close(out.cpu(),target,atol=1.6e-2,rtol=1.6e-2)
                                assert not torch.equal(out, previous)
                                assert addr==(local.data_ptr(),positions.data_ptr(),out.data_ptr(),attention.kv_cache.data_ptr())
                                errors.append(float((out.cpu().float()-target.float()).abs().max()))
                                previous=out.clone()
                            graph.reset()
                            del graph
                        dist.barrier(group=tp.device_group)
                        result['cases'].append(dict(family=family,dtype=str(dtype),kv_heads=kvheads,
                            official_attention=True,official_weight_loaders=True,real_kv_write_read=True,
                            native_flash_unmodified=True,graph_epochs=[1,3],stable_addresses=True,
                            max_abs_errors=errors,atol=1.6e-2,rtol=1.6e-2))
                        print('CASE',rank,family,str(dtype),kvheads,errors,flush=True)
                        del module, attention, params, out
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
