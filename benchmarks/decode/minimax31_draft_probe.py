"""Real five-layer DSpark component with explicit synthetic target context.

Not a full decode cycle. Run under torchrun through the full decode loader to
qualify the KV-only view, actual draft path, Markov proposal and changed graph.
"""
import json
import os
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.distributed.parallel_state import get_world_group
from hcu_vllm_das.parallel.minimax31_dspark_decode import DSparkDecodeRuntime
from hcu_vllm_das.parallel.minimax31_markov import greedy_markov_tokens


def run(args,cfg,draft):
    rank=int(os.environ['LOCAL_RANK']);torch.manual_seed(1120+rank)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    runtime=DSparkDecodeRuntime(draft,fake_zero_history=True)
    markov_fused = os.environ.get("HCU_MINIMAX31_DSPARK_MARKOV_FUSED") == "1"
    aux=[(torch.randn((128,6144),device='cuda')*.1).bfloat16() for _ in range(5)]
    old_positions=(147456+torch.arange(128,device='cuda')%8).long()
    accepted=(torch.arange(16,device='cuda')%9).int()
    prefix=147456+accepted
    runtime.prepare(prefix)
    ids=torch.full((16,7),200058,device='cuda',dtype=torch.int64)
    seed=torch.full((16,),1000+rank,device='cuda',dtype=torch.int64)
    dp_tokens=torch.full((8,),112,dtype=torch.int32)
    checks=[]
    with torch.inference_mode(),set_current_vllm_config(cfg):
        context=draft.model.hidden_norm(draft.model.fc(torch.cat(aux,dim=-1)))
        for index,layer in enumerate(draft.model.layers):
            a=layer.self_attn
            full=a.qkv_proj(context)[0][:,a.q_size:]
            sliced=a.qkv_proj.scheme.kernel.apply_weights(runtime.projections[index],context)
            assert torch.equal(full,sliced),('KV-only projection differs',index)
            checks.append({'layer':index,'kv_only_matches_full_projection':True,
                'weight_pointer_is_channel_slice':runtime.projections[index].weight.data_ptr()==a.qkv_proj.weight.data_ptr()+a.q_size*6144})
    def forward():
        with set_current_vllm_config(cfg),set_forward_context(None,cfg,num_tokens=112,num_tokens_across_dp=dp_tokens):
            runtime.update_context(aux,old_positions,accepted)
            runtime.prepare(prefix)
            ids[:,0].copy_(seed)
            hidden=draft(ids.flatten(),runtime.positions)
            logits=draft.compute_draft_logits(hidden).view(16,7,-1)
            return hidden,greedy_markov_tokens(
                draft, logits, seed, fused=markov_fused)
    with torch.inference_mode():
        for _ in range(2):hidden,tokens=forward()
        assert bool(torch.isfinite(hidden).all())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):hidden,tokens=forward()
        for _ in range(5):graph.replay()
        durations=[]
        for _ in range(3):
            get_world_group().barrier()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(8):graph.replay()
            end.record();end.synchronize();durations.append(begin.elapsed_time(end)/8)
        for state in aux:state.mul_(.9)
        seed.add_(7);prefix.add_(13);old_positions.add_(13)
        graph.replay();graph_hidden=hidden.clone();graph_tokens=tokens.clone()
        eager_hidden,eager_tokens=forward()
        torch.testing.assert_close(graph_hidden,eager_hidden,rtol=.01,atol=.01)
        assert torch.equal(graph_tokens,eager_tokens)
    result={'scope':'real five-layer DRAFT component; synthetic target aux; not complete cycle',
        'rank':rank,'requests':16,'draft_width':7,'context_window':4096,
        'context_precision':'FP8 unit K/V; BF16 Q/fresh K/V','graph_ms':durations,
        'kv_only_checks':checks,'changed_input_graph_matches_eager':True,
        'markov_steps':7,'finite_hidden':True}
    (args.output/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_OFFLINE_DRAFT '+json.dumps(result),flush=True)
    get_world_group().barrier()
    runtime.unbind()
