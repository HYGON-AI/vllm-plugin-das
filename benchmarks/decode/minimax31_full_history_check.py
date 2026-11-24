"""Exercise every full host-history layer before qualifying graph execution."""
import json
import torch
from minimax31_validation import finite


def check(*,runtime,forward,destination,rank,world,compare):
    if (getattr(runtime.backing,'zero_history_layer_alias',False) or
            not runtime.backing.readonly_zero_history or len(runtime.states)!=60 or
            runtime.backing.registered_bytes!=146158387200):
        raise RuntimeError('full readonly leased history is required for the cold IO check')
    print('MINIMAX_FULL_HISTORY_COLD_BEGIN '+json.dumps({'rank':rank}),flush=True)
    pointers=tuple(t.data_ptr() for t in runtime.main+runtime.index)
    torch.cuda.synchronize();world.barrier()
    expected_hidden,expected_logits=forward();torch.cuda.synchronize()
    # Invalidate ownership metadata, forcing the real swap kernels to fetch
    # past pages from all 60 independent host layers. No weights/history change.
    for state in runtime.states:
        state.tags.fill_(-1);state.page_slots.fill_(-1)
    torch.cuda.synchronize();world.barrier()
    hidden,logits=forward();torch.cuda.synchronize()
    misses=torch.stack([state.miss_count.sum() for state in runtime.states]).cpu().tolist()
    if len(misses)!=60 or any(count<=0 for count in misses):
        raise RuntimeError('cold full-history check did not exercise all 60 host layers')
    if hidden.data_ptr()==expected_hidden.data_ptr() or logits.data_ptr()==expected_logits.data_ptr():
        raise RuntimeError('cold/warm comparison needs independent live outputs')
    if not finite(hidden) or not finite(logits):raise RuntimeError('nonfinite cold full-history outputs')
    hidden_error=compare(hidden,expected_hidden,.02,.02)
    logits_error=compare(logits,expected_logits,.02,.04)
    if tuple(t.data_ptr() for t in runtime.main+runtime.index)!=pointers:
        raise RuntimeError('cold history check changed GPU cache storage')
    result={'rank':rank,'scope':'untimed eager cold host IO correctness; not steady or graph latency',
            'registered_host_bytes':runtime.backing.registered_bytes,
            'shared_zero_host_history':False,'cold_host_miss_pages_per_layer':misses,
            'cold_host_swap_bytes':sum(misses)*2*128*128,
            'all_60_layers_exercised':True,'complete_output_comparison':True,
            'hidden_max_abs':hidden_error,'logits_max_abs':logits_error,'gpu_caches_preserved':True}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/('rank%d.cold_history.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_FULL_HISTORY_COLD_PASSED '+json.dumps(result),flush=True)
    world.barrier()
    return result
