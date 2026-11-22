# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run destructive device-assert probes in an isolated process only."""
import os
import sys
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
from hcu_vllm_das.parallel.minimax31_prefill import bind_validated_scales, check_bound_scales


@triton.jit
def raw_store(P, X: tl.constexpr):
    tl.store(P, X)


def main(kind):
    obj=SimpleNamespace(kv_cache=torch.empty((1,4,128,256),device='cuda',dtype=torch.uint8),
        _k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
    bind_validated_scales(obj)
    version=obj._k_scale._version
    if kind=='valid':
        check_bound_scales(obj);torch.cuda.synchronize();print('GUARD_VALID',flush=True);return
    if kind=='fill':obj._k_scale.fill_(.5)
    elif kind=='zero':obj._k_scale.zero_()
    elif kind=='nan':obj._k_scale.fill_(float('nan'))
    elif kind in ('native','event'):
        if kind=='event':
            s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):raw_store[(1,)](obj._k_scale,.5)
            torch.cuda.current_stream().wait_stream(s)
        else:raw_store[(1,)](obj._k_scale,.5)
        assert obj._k_scale._version==version
    else:raise ValueError(kind)
    print('GUARD_EXPECT_REJECTION '+kind,flush=True)
    try:
        check_bound_scales(obj);torch.cuda.synchronize()
    except RuntimeError as e:
        print('GUARD_REJECTED '+kind+' '+str(e),flush=True)
        os._exit(23)
    print('GUARD_INCORRECTLY_ACCEPTED '+kind,flush=True)
    os._exit(24)


if __name__=='__main__':main(sys.argv[1])
