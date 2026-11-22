import numpy as np
import pytest
from types import SimpleNamespace
from hcu_vllm_das.parallel import minimax31_pcp as p
from vllm.v1.worker.gpu.pcp_manager import PCPManager

@pytest.mark.parametrize('budget,requests',[(1,1),(16,4),(257,80),(8192,4),(65536,1),(65536,80)])
def test_probe_budget_dominates_native_partition(budget,requests):
    manager=PCPManager.__new__(PCPManager)
    manager.pcp_world_size=8;manager.dcp_world_size=1;manager.shard_decode_requests=True
    rng=np.random.default_rng(20261115)
    bound=p.profile_local_token_budget(budget,requests)
    for _ in range(30):
        count=min(requests,budget)
        lengths=rng.multinomial(budget,np.full(count,1/count)).astype(np.int32)
        # No speculation: non-prefill queries have at most one token.
        prefill=lengths>1
        actual=manager.get_num_tokens_for_dispatch(lengths,prefill)
        assert actual<=bound,(budget,requests,lengths.tolist(),actual,bound)
    assert bound<=budget

def test_profile_only_changes_synthetic_local_rows(monkeypatch):
    calls=[]
    monkeypatch.setattr(p.GPUModelRunner,'_dummy_run',lambda self,n,*args,**kw:calls.append((n,kw)) or 'result')
    obj=p.MiniMax31PCPModelRunner.__new__(p.MiniMax31PCPModelRunner)
    obj.pcp_manager=SimpleNamespace(pcp_world_size=8,dcp_world_size=1,shard_decode_requests=True, _hcu_native_msa=False)
    obj.speculator=None;obj.max_num_reqs=80;obj.max_num_tokens=65536
    assert obj._dummy_run(65536,is_profile=True,skip_attn=True)=='result'
    assert calls[-1][0]==8448 and obj.max_num_tokens==65536
    assert obj._dummy_run(65536,is_profile=False,skip_attn=False)=='result'
    assert calls[-1][0]==65536
    obj.pcp_manager.dcp_world_size=2
    with pytest.raises(RuntimeError,match='synthetic memory probe'):obj._dummy_run(65536,is_profile=True,skip_attn=True)
