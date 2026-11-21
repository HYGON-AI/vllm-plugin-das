# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused DFly input kernels must reproduce the torch preparation they
replace: query block, sampling state, query slots and sparse reuse indices."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm_hcu.models.hy_v4_flash.nvidia.dfly_inputs import (
    prepare_dfly_inputs, write_dfly_query_slots, build_dfly_reuse_indices,
)

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() or torch.version.hip), reason="GPU input kernels require CUDA/HIP"
)

QUERY_WIDTH = 4
MASK_TOKEN_ID = 99
MAX_MODEL_LEN = 106
BLOCK_SIZE = 4


def _reference_reuse_indices(snapshot_rows, anchors, local_window, active, width):
    """The torch `build_reuse_indices` the kernel replaced."""
    batch, topk = snapshot_rows.shape
    rows = torch.full(
        (batch, QUERY_WIDTH, width), -1, dtype=torch.int32, device=anchors.device
    )
    starts = anchors[:, None] - local_window
    history = torch.where(
        (snapshot_rows >= 0) & (snapshot_rows < starts), snapshot_rows, -1
    )
    rows[:, :, :topk] = history[:, None, :]
    window = starts + torch.arange(local_window, device=anchors.device)
    rows[:, :, topk : topk + local_window] = window.clamp(min=-1)[:, None, :]
    offsets = torch.arange(QUERY_WIDTH, device=anchors.device)
    block = anchors[:, None] + offsets
    causal = offsets[None, :] <= offsets[:, None]
    rows[:, :, topk + local_window : topk + local_window + QUERY_WIDTH] = torch.where(
        causal[None], block[:, None, :], -1
    )
    rows.masked_fill_(~active[:, None, None], -1)
    return rows.view(batch * QUERY_WIDTH, width)


@pytest.fixture
def prepared():
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_inputs import prepare_dfly_inputs

    device = torch.device("cuda")
    max_reqs, max_tokens, hidden_size, padded_batch = 8, 64, 16, 5
    # req0 drafts after one rejection; req1's block would pass max_model_len;
    # req2 is mid-prefill (nothing sampled). Rows 3 and 4 are graph padding.
    target_qsl = torch.tensor([0, 4, 8, 9], dtype=torch.int32, device=device)
    target_positions = torch.tensor(
        [10, 11, 12, 13, 100, 101, 102, 103, 50], dtype=torch.int64, device=device
    )
    num_rejected = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
    num_sampled = torch.tensor([2, 4, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([5, 1, 6], dtype=torch.int32, device=device)
    last_sampled = torch.arange(max_reqs * 2, device=device).view(max_reqs, 2) + 1000
    last_hidden = torch.randn(9, hidden_size, dtype=torch.bfloat16, device=device)
    mrope_delta = torch.arange(max_reqs, dtype=torch.int32, device=device) * 7
    temperature = torch.rand(max_reqs, device=device) + 0.5
    seeds = torch.arange(max_reqs, dtype=torch.int64, device=device) + 17

    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device=device)

    out = dict(
        input_ids=full((max_tokens,), -7, torch.int32),
        positions=full((max_tokens,), -7, torch.int64),
        is_padding=full((max_tokens,), False, torch.bool),
        query_start_loc=full((max_reqs + 1,), -7, torch.int32),
        seq_lens=full((max_reqs,), -7, torch.int32),
        active=full((max_reqs,), True, torch.bool),
        source_rows=full((max_reqs,), -7, torch.int64),
        anchors=full((max_reqs,), -7, torch.int64),
        target_hidden_out=torch.randn(
            max_reqs, hidden_size, dtype=torch.bfloat16, device=device
        ),
        mrope_positions=full((3, max_tokens), -7, torch.int64),
        draft_idx_mapping=full((max_reqs,), -7, torch.int64),
        out_temperature=full((max_reqs,), 0.0, torch.float32),
        out_seeds=full((max_reqs,), 0, torch.int64),
    )
    prepare_dfly_inputs(
        **out,
        target_query_start_loc=target_qsl,
        target_positions=target_positions,
        idx_mapping=idx_mapping,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        target_hidden=last_hidden,
        mrope_delta=mrope_delta,
        temperature=temperature,
        seeds=seeds,
        max_model_len=torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=device),
        mask_token_id=MASK_TOKEN_ID,
        query_width=QUERY_WIDTH,
        num_reqs=3,
        padded_batch=padded_batch,
    )
    inputs = dict(
        target_qsl=target_qsl,
        target_positions=target_positions,
        num_rejected=num_rejected,
        num_sampled=num_sampled,
        idx_mapping=idx_mapping,
        last_sampled=last_sampled,
        last_hidden=last_hidden,
        mrope_delta=mrope_delta,
        temperature=temperature,
        seeds=seeds,
        padded_batch=padded_batch,
    )
    return out, inputs


def test_prepare_inputs_matches_torch_reference(prepared):
    out, inp = prepared
    device = out["positions"].device
    batch, padded_batch = 3, inp["padded_batch"]
    count = padded_batch * QUERY_WIDTH
    offsets = torch.arange(QUERY_WIDTH, device=device)

    source_rows = (inp["target_qsl"][1:] - inp["num_rejected"] - 1).long()
    anchors = inp["target_positions"][source_rows] + 1
    active = (inp["num_sampled"] > 0) & (anchors + QUERY_WIDTH <= MAX_MODEL_LEN)
    assert active.tolist() == [True, False, False]
    positions = torch.where(active[:, None], anchors[:, None] + offsets, offsets)
    ids = torch.full((batch, QUERY_WIDTH), MASK_TOKEN_ID, device=device)
    ids[:, 0] = torch.where(active, inp["last_sampled"][inp["idx_mapping"], 0], 0)
    delta = inp["mrope_delta"][inp["idx_mapping"].long()]
    rope = torch.where(active[:, None], positions + delta[:, None], 0)

    pad = padded_batch - batch
    pad_ids = torch.full((pad, QUERY_WIDTH), MASK_TOKEN_ID, device=device)
    pad_ids[:, 0] = 0
    expect_ids = torch.cat([ids, pad_ids]).flatten()
    expect_positions = torch.cat([positions, offsets.expand(pad, -1)]).flatten()
    expect_padding = torch.cat(
        [
            (~active)[:, None].expand(-1, QUERY_WIDTH),
            torch.ones(pad, QUERY_WIDTH, device=device),
        ]
    ).flatten()
    expect_rope = torch.cat(
        [rope, torch.zeros(pad, QUERY_WIDTH, device=device)]
    ).flatten()

    torch.testing.assert_close(out["input_ids"][:count].long(), expect_ids.long())
    torch.testing.assert_close(out["positions"][:count], expect_positions)
    assert out["is_padding"][:count].tolist() == expect_padding.bool().tolist()
    for dim in range(3):
        torch.testing.assert_close(
            out["mrope_positions"][dim, :count], expect_rope.long()
        )
    assert out["seq_lens"][:padded_batch].tolist() == (
        (positions[:, -1] + 1).tolist() + [QUERY_WIDTH] * pad
    )
    assert out["query_start_loc"][: padded_batch + 1].tolist() == [
        i * QUERY_WIDTH for i in range(padded_batch + 1)
    ]
    assert out["active"][:padded_batch].tolist() == active.tolist() + [False] * pad
    torch.testing.assert_close(
        out["target_hidden_out"][:batch], inp["last_hidden"][source_rows]
    )
    assert not out["target_hidden_out"][batch:padded_batch].any()
    assert out["source_rows"][:batch].tolist() == source_rows.tolist()
    assert out["anchors"][:batch].tolist() == anchors.tolist()

    # Sampling state: only active rows sample, read from fixed buffers.
    assert out["draft_idx_mapping"][:padded_batch].tolist() == [5, -1, -1, -1, -1]
    states = inp["idx_mapping"].long()
    torch.testing.assert_close(
        out["out_temperature"][states], inp["temperature"][states]
    )
    torch.testing.assert_close(out["out_seeds"][states], inp["seeds"][states])


def test_query_slots_skip_inactive_and_null_blocks(prepared):
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_inputs import write_dfly_query_slots

    out, inp = prepared
    device = out["positions"].device
    padded_batch = inp["padded_batch"]
    # req0's block covers positions 13..16 across blocks 3 and 4; block 4 of its
    # table is the null block.
    block_table = torch.arange(1, 41, dtype=torch.int32, device=device).view(8, 5)
    block_table[0, 4] = 0
    slots = torch.full((64,), -7, dtype=torch.int64, device=device)
    write_dfly_query_slots(
        slots,
        out["positions"],
        out["active"],
        block_table,
        BLOCK_SIZE,
        torch.tensor(MAX_MODEL_LEN, dtype=torch.int32, device=device),
        QUERY_WIDTH,
        padded_batch,
    )
    req0 = [
        int(block_table[0, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE
        for p in (13, 14, 15)
    ] + [PAD_SLOT_ID]
    expected = req0 + [PAD_SLOT_ID] * (QUERY_WIDTH * (padded_batch - 1))
    assert slots[: padded_batch * QUERY_WIDTH].tolist() == expected


@pytest.mark.parametrize("layer_slots", [[0, 0], [1, 0]])
def test_reuse_indices_match_torch_reference(prepared, layer_slots):
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_inputs import build_dfly_reuse_indices

    out, inp = prepared
    device = out["positions"].device
    padded_batch = inp["padded_batch"]
    target_topk, local_window, width = 8, 3, 128
    snapshots = torch.randint(-1, 20, (2, 9, target_topk), device=device).int()
    indices = torch.full((2, 64, width), -7, dtype=torch.int32, device=device)
    slots = torch.tensor(layer_slots, dtype=torch.int32, device=device)
    build_dfly_reuse_indices(
        indices,
        snapshots,
        slots,
        out["active"],
        out["source_rows"],
        out["anchors"],
        target_topk,
        local_window,
        QUERY_WIDTH,
        padded_batch,
    )
    count = padded_batch * QUERY_WIDTH
    for layer, slot in enumerate(layer_slots):
        expected = _reference_reuse_indices(
            snapshots[slot][out["source_rows"][:padded_batch]],
            out["anchors"][:padded_batch],
            local_window,
            out["active"][:padded_batch],
            width,
        )
        torch.testing.assert_close(indices[layer, :count], expected)


# HY4 Flash HCU contracts: six queries, hidden3072, limit8192 and Graph replay.

def make_case(num_reqs=3, padded_batch=5, query_width=6, mrope=True):
    device = 'cuda'
    capacity, hidden = 8, 3072
    def full(shape, val, dtype):
        return torch.full(shape, val, device=device, dtype=dtype)
    result = dict(
        input_ids=full((capacity*query_width,), -7, torch.int32),
        positions=full((capacity*query_width,), -7, torch.int64),
        is_padding=full((capacity*query_width,), False, torch.bool),
        query_start_loc=full((capacity+1,), -7, torch.int32),
        seq_lens=full((capacity,), -7, torch.int32),
        active=full((capacity,), True, torch.bool),
        source_rows=full((capacity,), -7, torch.int64),
        anchors=full((capacity,), -7, torch.int64),
        target_hidden_out=full((capacity, hidden), -7, torch.bfloat16),
        mrope_positions=full((3,capacity*query_width), -7, torch.int64) if mrope else None,
        draft_idx_mapping=full((capacity,), -7, torch.int64),
        out_temperature=full((capacity,), -7, torch.float32),
        out_seeds=full((capacity,), -7, torch.int64),
    )
    inputs = dict(
        target_query_start_loc=torch.tensor([0,2,4,6], dtype=torch.int32, device=device),
        # source rows 1 and 3 have anchors 8186 and 8187: exact limit and overflow.
        target_positions=torch.tensor([8184,8185,8185,8186,18,19], dtype=torch.int64, device=device),
        idx_mapping=torch.tensor([5,1,6], dtype=torch.int32, device=device),
        num_sampled=torch.tensor([1,1,0], dtype=torch.int32, device=device),
        num_rejected=full((3,),0,torch.int32),
        last_sampled=torch.arange(capacity*2,device=device).view(capacity,2)+1000,
        target_hidden=torch.randn(6,hidden,device=device,dtype=torch.bfloat16),
        mrope_delta=torch.arange(capacity,device=device,dtype=torch.int32)*7 if mrope else None,
        temperature=torch.arange(capacity,device=device,dtype=torch.float32)+0.5,
        seeds=torch.arange(capacity,device=device,dtype=torch.int64)+17,
        max_model_len=torch.tensor(8192,device=device,dtype=torch.int32),
        mask_token_id=120009,query_width=query_width,num_reqs=num_reqs,padded_batch=padded_batch,
    )
    return result, inputs


def assert_prepared(out, inp):
    q, n, batch = inp['query_width'], inp['num_reqs'], inp['padded_batch']
    rows=(inp['target_query_start_loc'][1:n+1]-inp['num_rejected'][:n]-1).long()
    anchors=inp['target_positions'][rows]+1
    active=(inp['num_sampled'][:n]>0)&(anchors+q<=inp['max_model_len'])
    pad=batch-n
    expected_active=torch.cat((active,torch.zeros(pad,device='cuda',dtype=torch.bool)))
    expected_anchor=torch.cat((anchors,torch.zeros(pad,device='cuda',dtype=torch.int64)))
    offsets=torch.arange(q,device='cuda')
    positions=torch.where(expected_active[:,None],expected_anchor[:,None]+offsets,offsets)
    ids=torch.full((batch,q),120009,device='cuda',dtype=torch.int32)
    for i in range(n):
        ids[i,0]=torch.where(active[i],inp['last_sampled'][inp['idx_mapping'][i],0],0)
    ids[n:,0]=0
    torch.testing.assert_close(out['positions'][:batch*q],positions.flatten())
    torch.testing.assert_close(out['input_ids'][:batch*q],ids.flatten())
    torch.testing.assert_close(out['active'][:batch],expected_active)
    torch.testing.assert_close(out['is_padding'][:batch*q],(~expected_active[:,None]).expand(batch,q).flatten())
    torch.testing.assert_close(out['seq_lens'][:batch].long(),torch.where(expected_active,expected_anchor,0)+q)
    torch.testing.assert_close(out['query_start_loc'][:batch+1].long(),torch.arange(batch+1,device='cuda')*q)
    torch.testing.assert_close(out['target_hidden_out'][:n],inp['target_hidden'][rows])
    assert torch.count_nonzero(out['target_hidden_out'][n:batch])==0
    expected_mapping=torch.full((batch,),-1,device='cuda',dtype=torch.int64)
    expected_mapping[:n]=torch.where(active,inp['idx_mapping'][:n],-1).long()
    torch.testing.assert_close(out['draft_idx_mapping'][:batch],expected_mapping)
    states=inp['idx_mapping'][:n].long()
    torch.testing.assert_close(out['out_temperature'][states],inp['temperature'][states])
    torch.testing.assert_close(out['out_seeds'][states],inp['seeds'][states])
    if out['mrope_positions'] is not None:
        delta=torch.zeros(batch,device='cuda',dtype=torch.int64)
        delta[:n]=inp['mrope_delta'][states]
        rope=torch.where(expected_active[:,None],positions+delta[:,None],0)
        for d in range(3):torch.testing.assert_close(out['mrope_positions'][d,:batch*q],rope.flatten())
    assert (out['positions'][batch*q:]==-7).all()


@pytest.mark.parametrize('mrope',[False,True])
@pytest.mark.parametrize('num_reqs',[0,3])
def test_five_drafts_8192_boundary(mrope,num_reqs):
    out,inp=make_case(num_reqs=num_reqs,mrope=mrope)
    prepare_dfly_inputs(**out,**inp)
    assert_prepared(out,inp)
    if num_reqs:
        assert out['active'][:3].tolist()==[True,False,False]
        assert out['positions'][:6].tolist()==list(range(8186,8192))


@pytest.mark.parametrize('mrope',[False,True])
def test_changing_inputs_graph_replay(mrope):
    out,inp=make_case(mrope=mrope)
    prepare_dfly_inputs(**out,**inp)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):prepare_dfly_inputs(**out,**inp)
    for rejected in [0,1,0]:
        inp['target_positions'].sub_(13)
        inp['num_rejected'].fill_(rejected)
        inp['num_sampled'].copy_(torch.tensor([1,0,1],device='cuda',dtype=torch.int32))
        inp['target_hidden'].add_(1)
        inp['temperature'].add_(0.25)
        inp['seeds'].add_(3)
        inp['last_sampled'].add_(5)
        if mrope:inp['mrope_delta'].add_(2)
        graph.replay()
        assert_prepared(out,inp)


def test_five_drafts_query_slots_null_blocks_and_8192():
    out,inp=make_case()
    prepare_dfly_inputs(**out,**inp)
    # Only request0 is active. Its final two entries are in an explicitly null page.
    tables=torch.arange(1,5*128+1,device='cuda',dtype=torch.int32).view(5,128)
    tables[0,127]=0
    slots=torch.full((48,),-7,device='cuda',dtype=torch.int64)
    write_dfly_query_slots(slots,out['positions'],out['active'],tables,64,inp['max_model_len'],6,5)
    assert slots[:30].tolist()==[-1]*30
    assert (slots[30:]==-7).all()
    tables[0,127]=19
    write_dfly_query_slots(slots,out['positions'],out['active'],tables,64,inp['max_model_len'],6,5)
    assert slots[:6].tolist()==[19*64+p%64 for p in range(8186,8192)]
    assert slots[6:30].tolist()==[-1]*24


def reuse_reference(snapshots,slots,active,source_rows,anchors,topk,window,q,width,batch):
    result=torch.full((len(slots),batch*q,width),-1,device='cuda',dtype=torch.int32)
    for layer,slot in enumerate(slots.tolist()):
        for req in range(batch):
            if not active[req]:continue
            history=snapshots[slot,source_rows[req],:topk]
            start=anchors[req]-window
            history=torch.where((history>=0)&(history<start),history,-1)
            for query in range(q):
                row=result[layer,req*q+query]
                row[:topk]=history
                row[topk:topk+window]=(start+torch.arange(window,device='cuda')).clamp(min=-1)
                row[topk+window:topk+window+query+1]=anchors[req]+torch.arange(query+1,device='cuda')
    return result


@pytest.mark.parametrize('window',[0,128])
@pytest.mark.parametrize('graph_mode',[False,True])
def test_five_layer_topk2048_reuse(window,graph_mode):
    q,batch,topk,layers=6,4,2048,5
    width=((topk+window+q+127)//128)*128
    snapshots=torch.randint(-10,8193,(5,12,topk),device='cuda',dtype=torch.int32)
    slots=torch.tensor([0,2,4,1,3],device='cuda',dtype=torch.int32)
    active=torch.tensor([True,True,False,True],device='cuda')
    rows=torch.tensor([2,7,0,10],device='cuda',dtype=torch.int64)
    anchors=torch.tensor([8186,23,0,4],device='cuda',dtype=torch.int64)
    out=torch.full((layers,batch*q+3,width),-7,device='cuda',dtype=torch.int32)
    def run():build_dfly_reuse_indices(out,snapshots,slots,active,rows,anchors,topk,window,q,batch)
    run();torch.cuda.synchronize()
    if graph_mode:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
    for iteration in range(3):
        if iteration:
            snapshots.add_(1);rows.copy_(rows.roll(1));active.copy_(~active);anchors.add_(1)
        if graph_mode:graph.replay()
        else:run()
        expect=reuse_reference(snapshots,slots,active,rows,anchors,topk,window,q,width,batch)
        torch.testing.assert_close(out[:,:batch*q],expect,rtol=0,atol=0)
        assert (out[:,batch*q:]==-7).all()
