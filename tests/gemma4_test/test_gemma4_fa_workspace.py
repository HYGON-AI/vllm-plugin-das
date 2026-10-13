# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU native-call contracts, including replay capacity and output ownership."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.models.gemma4.fa_utils import gemma4_flash_attention


@pytest.mark.parametrize(
    "capturing,batch,max_k,used,capacity",
    [
        (False, 2, 19968, 64, 262144),
        (False, 24, 19968, 16384, 24 * 16384),
        (True, 24, 19968, 1, 24 * 19968),
        (True, 2, 19968, 1, 262144),
    ],
)
def test_native_gather_capacity_and_output(
    monkeypatch, capturing, batch, max_k, used, capacity
):
    calls = []
    interface = ModuleType("flash_attn.flash_attn_interface")

    def gather(*args):
        calls.append(("gather", args))

    def native(*args):
        calls.append(("native", args))
        args[3].fill_(3)

    interface.flash_attn_cuda = SimpleNamespace(
        pagedkv_to_contiguouskv=gather, varlen_fwd=native
    )
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    # Meta gather buffers check realistic capacity without allocating GiBs on CPU.
    q = torch.zeros((batch * 5, 16, 512), dtype=torch.bfloat16)
    k = torch.empty((2, 64, 2, 512), dtype=torch.bfloat16, device="meta")
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    cu_q = torch.arange(batch + 1, dtype=torch.int32) * 5
    seq = torch.full((batch,), used, dtype=torch.int32)
    if capturing:
        seq = seq.to("meta")  # A host read of dummy lengths would fail.
    block_table = torch.zeros((batch, 2), dtype=torch.int32)
    result = gemma4_flash_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=5,
        seqused_k=seq,
        max_seqlen_k=max_k,
        softmax_scale=1.0,
        block_table=block_table,
    )
    assert result is out
    assert torch.all(out == 3)
    gather_args = calls[0][1]
    assert gather_args[0].shape == gather_args[1].shape == (capacity, 2, 512)
    assert gather_args[2] is k and gather_args[3] is v
    assert gather_args[4] is block_table and gather_args[5] is seq
    assert gather_args[-1] == 2
    native_args = calls[1][1]
    assert native_args[1] is gather_args[0]
    assert native_args[2] is gather_args[1]
    assert native_args[3] is out
    assert native_args[5] is gather_args[6]
    assert native_args[6:10] == (None, None, None, None)
    assert native_args[10:20] == (5, max_k, 0.0, 1.0, False, True, -1, -1, 0.0, False)
