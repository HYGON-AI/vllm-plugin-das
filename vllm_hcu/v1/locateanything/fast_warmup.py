# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fast-aware startup warmup; validates real KV progress before graph capture."""
import json
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, CachedRequestData, SchedulerOutput


def _warmup_prompts(max_num_reqs, max_num_tokens):
    """Keep the synthetic prefill and all decode steps within runner buffers."""
    minimum_prompt = 13
    physical_tail = 6
    if max_num_reqs < 1 or max_num_tokens < minimum_prompt + physical_tail:
        raise ValueError("Fast warmup requires a request and at least 19 tokens")
    n = min(max_num_reqs, max_num_tokens // (minimum_prompt + physical_tail))
    maximum_prompt = min(minimum_prompt + n - 1,
                         max_num_tokens // n - physical_tail)
    period = maximum_prompt - minimum_prompt + 1
    return [list(range(minimum_prompt + i % period)) for i in range(n)]


def warmup(runner, execute, sample):
    from vllm.v1.worker.gpu.warmup import _warmup_block_counter
    block_count=_warmup_block_counter(runner)
    specs=[g.kv_cache_spec for g in runner.kv_cache_config.kv_cache_groups]
    prompts=_warmup_prompts(runner.max_num_reqs, runner.max_num_tokens)
    n=len(prompts)
    ids=[f'_locate_fast_warmup_{i}' for i in range(n)]
    next_block=1
    def allocate(size):
        nonlocal next_block
        result=list(range(next_block,next_block+size));next_block+=size
        return result
    start=SchedulerOutput.make_empty()
    start.scheduled_new_reqs=[NewRequestData(
        req_id=rid,prompt_token_ids=prompt,mm_features=[],
        sampling_params=SamplingParams(max_tokens=128,temperature=.7,top_p=.9,repetition_penalty=1.1,seed=i),
        pooling_params=None,block_ids=tuple(allocate(block_count(len(prompt)+128+6,s)) for s in specs),
        num_computed_tokens=0,lora_request=None,prefill_token_ids=prompt)
        for i,(rid,prompt) in enumerate(zip(ids,prompts))]
    assert next_block < runner.kv_cache_config.num_blocks
    start.num_scheduled_tokens={rid:len(prompt)+6 for rid,prompt in zip(ids,prompts)}
    start.total_num_scheduled_tokens=sum(start.num_scheduled_tokens.values())
    assert start.total_num_scheduled_tokens <= runner.max_num_tokens
    start.num_common_prefix_blocks=[0]*len(specs)
    if runner.kv_block_zeroer is not None:runner.kv_block_zeroer.warmup(runner.kv_cache_config)
    runner.kv_connector.set_disabled(True)
    records=[]
    schedule=start
    try:
        for step in range(5):
            execute(schedule)
            batch=runner.execute_model_state.input_batch
            mapping=batch.idx_mapping.cpu().tolist()
            before=runner.req_states.num_computed_tokens.gpu[mapping].cpu().tolist()
            physical=[schedule.num_scheduled_tokens[rid] for rid in batch.req_ids]
            # Verify physical IDs / shifted RoPE before modifying real history.
            inputs=batch.input_ids[:batch.num_tokens].cpu().tolist()
            rope=runner.model_state.fast_positions[:batch.num_tokens].cpu().tolist()
            histories=runner.req_states.all_token_ids.gpu[mapping].cpu().tolist()
            offset=0
            for c,q,h in zip(before,physical,histories):
                l=c+q-6
                assert inputs[offset:offset+q]==h[c:l]+[h[l-1]]+[151676]*5
                assert rope[offset:offset+q]==list(range(c,l))+list(range(l-1,l+5))
                offset+=q
            sample(None)
            after=runner.req_states.num_computed_tokens.gpu[mapping].cpu().tolist()
            totals=runner.req_states.total_len.gpu[mapping].cpu().tolist()
            assert after==[c+q-6 for c,q in zip(before,physical)], (before,physical,after)
            assert all(1 <= t-c <= 6 for t,c in zip(totals,after)),(totals,after)
            records.append({'step':step,'requests':len(mapping),'query_lengths':physical,
                            'committed_kv':after,'new_output_counts':[t-c for t,c in zip(totals,after)]})
            # Shrink/reorder batches to exercise request-index independence.
            selected=list(reversed(range(len(mapping))))
            if step==1:selected=selected[::2]
            if step==3:selected=selected[:1]
            cached=CachedRequestData.make_empty()
            cached.req_ids=[batch.req_ids[i] for i in selected]
            cached.num_computed_tokens=[after[i] for i in selected]
            cached.num_output_tokens=[totals[i]-len(prompts[ids.index(batch.req_ids[i])]) for i in selected]
            cached.new_block_ids=[None]*len(selected)
            schedule=SchedulerOutput.make_empty()
            schedule.scheduled_cached_reqs=cached
            schedule.num_scheduled_tokens={batch.req_ids[i]:totals[i]-after[i]+6 for i in selected}
            schedule.total_num_scheduled_tokens=sum(schedule.num_scheduled_tokens.values())
            schedule.num_common_prefix_blocks=[0]*len(specs)
    finally:
        cleanup=SchedulerOutput.make_empty();cleanup.finished_req_ids=set(ids)
        execute(cleanup)
        runner.kv_connector.set_disabled(False)
        torch.cuda.synchronize()
    print('LOCATE_FAST_KV_WARMUP '+json.dumps({'passed':True,'capacity':runner.max_num_reqs,
        'token_budget':runner.max_num_tokens,'prefill_tokens':start.total_num_scheduled_tokens,
        'max_prompt_length':max(map(len,prompts)),'steps':records}),flush=True)
