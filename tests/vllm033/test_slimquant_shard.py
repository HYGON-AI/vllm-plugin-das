# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from hcu_vllm_das.quantization.slimquant_shard import shard_expert


@pytest.mark.parametrize("pad", [4, 6])
def test_tp16_all_ranks_reassemble_and_zero_padding(pad):
    torch.manual_seed(8)
    inter, hidden, tp = 64, 12, 16
    w1 = torch.randint(-128,128,(inter,hidden//2),dtype=torch.int8)
    w3 = torch.randint(-128,128,(inter,hidden//2),dtype=torch.int8)
    w2 = torch.randint(-128,128,(hidden,inter//2),dtype=torch.int8)
    s1, s3, s2 = torch.rand(inter,1), torch.rand(inter,1), torch.rand(hidden,1)
    for rank in range(tp):
        a,b,sa,sb = shard_expert(w1,w3,w2,s1,s3,s2,rank=rank,tp_size=tp,padded_intermediate=pad)
        local=inter//tp; start=rank*local
        assert torch.equal(a[:local], w1[start:start+local])
        assert torch.equal(a[pad:pad+local], w3[start:start+local])
        assert torch.equal(b[:,:local//2], w2[:,start//2:(start+local)//2])
        assert torch.count_nonzero(a[local:pad]) == 0 and torch.count_nonzero(a[pad+local:]) == 0
        assert torch.count_nonzero(b[:,local//2:]) == 0
        assert torch.equal(sa[:local],s1[start:start+local])
        assert torch.equal(sa[pad:pad+local],s3[start:start+local])
        assert torch.equal(sb,s2) and sb.data_ptr() != s2.data_ptr()


def test_odd_shard_rejected():
    w = torch.zeros(6,2,dtype=torch.int8)
    with pytest.raises(ValueError,match="even unpadded"):
        shard_expert(w,w,torch.zeros(4,3,dtype=torch.int8),torch.ones(6,1),torch.ones(6,1),torch.ones(4,1),rank=0,tp_size=2,padded_intermediate=4)
