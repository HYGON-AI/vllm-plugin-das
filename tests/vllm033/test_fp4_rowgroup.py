# SPDX-License-Identifier: Apache-2.0
"""Request ownership and canonical-page reuse contracts; no GPU required."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das.fp4_rowgroup import uniform_decode_rows, stage_request_rows


def metadata(requests=7, group=6):
    rows = requests * group
    return NS(num_decodes=requests, num_decode_tokens=rows,
              decode=NS(write_max_decode_len=group, decode_is_uniform=True,
                        requires_padding=False, indices=None,
                        seq_lens=torch.ones((rows,1),dtype=torch.int32),
                        block_table=torch.ones((rows,8),dtype=torch.int32)))


@pytest.mark.parametrize('requests', [1,2,7,17])
@pytest.mark.parametrize('group', [1,2,3,5,6,7,8,9])
@pytest.mark.parametrize('topk', [512,1024,2048])
def test_request_width_follows_actual_official_metadata(requests,group,topk):
    assert uniform_decode_rows(metadata(requests,group),topk_tokens=topk) == group


@pytest.mark.parametrize('topk',[1,128,256,4096,None])
def test_other_consumers_retain_original_row_schedule(topk):
    assert uniform_decode_rows(metadata(),topk_tokens=topk) == 1


@pytest.mark.parametrize('mutation', [
    lambda m:setattr(m,'num_decode_tokens',m.num_decode_tokens-1),
    lambda m:setattr(m,'num_decodes',0),
    lambda m:setattr(m.decode,'decode_is_uniform',False),
    lambda m:setattr(m.decode,'requires_padding',True),
    lambda m:setattr(m.decode,'indices',torch.zeros(42,dtype=torch.int32)),
    lambda m:setattr(m.decode,'write_max_decode_len',True),
    lambda m:delattr(m.decode,'write_max_decode_len'),
    lambda m:delattr(m.decode,'decode_is_uniform'),
    lambda m:setattr(m.decode,'seq_lens',torch.ones((7,6),dtype=torch.int32)),
    lambda m:setattr(m.decode,'block_table',torch.ones((7,8),dtype=torch.int32)),
])
def test_ragged_padded_remapped_unknown_contracts_keep_native_schedule(mutation):
    m=metadata();mutation(m)
    assert uniform_decode_rows(m,topk_tokens=512) == 1


@pytest.mark.parametrize('records',[32,64,128])
@pytest.mark.parametrize('group',[1,2,3,6,9])
def test_staging_reuses_final_request_prefix_only_for_noncanonical_pages(records,group):
    requests=3
    table=torch.arange(requests*8,dtype=torch.int32).reshape(requests,8).repeat_interleave(group,0)
    lengths=torch.arange(requests*group,dtype=torch.int32)+65
    cache=object();pool=object();calls=[]
    def stage(c,t,l,w,g,*,live_only):
        calls.append((c,t.clone(),l.clone(),w,g,live_only))
        return pool,torch.arange(t.shape[0]*4,dtype=torch.int32).reshape(t.shape[0],4)
    result, actual=stage_request_rows(cache,table,lengths,512,None,
        rows_per_request=group,page_records=records,stage=stage)
    assert result is pool and len(calls)==1 and calls[0][0] is cache and calls[0][-1]
    if records==64 or group==1:
        assert torch.equal(calls[0][1],table) and torch.equal(calls[0][2],lengths)
        assert actual.shape==(requests*group,4)
    else:
        assert torch.equal(calls[0][1],table[group-1::group])
        assert torch.equal(calls[0][2],lengths[group-1::group])
        assert calls[0][2].is_contiguous()
        assert actual.shape==(requests*group,4)
        assert torch.equal(actual.reshape(requests,group,4),actual[::group,None,:].expand(requests,group,4))
