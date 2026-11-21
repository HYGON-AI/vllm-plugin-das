# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash orchestration must publish logical top-k to Core's side stream."""
import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _orchestrate():
    source = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/attention.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_indexer_and_attn")
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU streams")
@pytest.mark.parametrize("nondefault_stream", [False, True])
@pytest.mark.parametrize("external_topk", [False, True])
def test_flash_index_group_waits_for_logical_producer(nondefault_stream, external_topk):
    """Exercise the real Core remapper, not a same-stream stand-in.

    Leave a previous layer's front-packed indices in the shared buffer, then
    delay publication of this layer's right-aligned SWA indices. The ready event
    must make remapping observe the new contents without any host synchronize
    between the producer and consumer.
    """
    from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroup
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
    )
    run = _orchestrate()
    device = torch.device("cuda")
    tokens, width, window, block_size = 205, 2048, 512, 256
    rows = torch.arange(tokens, device=device)[:, None]
    columns = torch.arange(width, device=device)[None, :]
    logical = rows - window + 1 + columns
    wanted = torch.where((columns < window) & (logical >= 0), logical, -1).int()
    previous = torch.where(columns <= rows, columns.expand(tokens, -1), -1).int()
    table = torch.tensor([[7, 2, 13, 5]], dtype=torch.int32, device=device)
    req_ids = torch.zeros(tokens, dtype=torch.int32, device=device)
    expected = torch.where(wanted >= 0,
                           table[0, (wanted.clamp_min(0) // block_size).long()] * 3840
                           + wanted.clamp_min(0) % block_size, -1).int()
    # Compile before the delay, otherwise JIT latency can hide the missing event.
    triton_convert_req_index_to_global_index(
        req_ids, table, wanted, BLOCK_SIZE=block_size,
        BLOCK_STRIDE_ROWS=3840, NUM_TOPK_TOKENS=width,
    )
    metadata = SimpleNamespace(req_id_per_token=req_ids, block_table=table,
                               block_size=block_size)
    shared = previous.clone()
    group = SparseMLAIndexGroup(
        logical_topk_indices=shared, physical_topk_indices=torch.empty_like(shared),
        valid_topk_counts=torch.empty(tokens, dtype=torch.int32, device=device),
        side_stream=torch.Stream(device=device), logical_topk_ready=torch.Event(),
        physical_topk_ready=torch.Event(), has_indexer=True,
    )
    group.register_layer()
    output = torch.empty_like(shared)
    impl = SimpleNamespace(record_logical_topk_ready=partial(group.set_logical_topk_ready, 0))
    class Attention:
        def __init__(self):
            self.impl = impl
        def __call__(self, *args, **kwargs):
            return group.convert_logical_to_physical_topk(
                0, shared, metadata, block_stride_rows=3840, return_valid_counts=False,
            )
    def produce(*args):
        torch.cuda._sleep(20_000_000)
        shared.copy_(wanted)
    attention = Attention()
    model = SimpleNamespace(indexer=produce, is_sparse=True, skip_topk=external_topk,
                            indexer_rope_emb=None, mla_attn=attention, topk_snapshot=None)
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.default_stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        for _ in range(4):
            shared.copy_(previous)
            if external_topk:
                produce()
            run(model, None, None, None, None, None, None, output)
            # End-of-call checking only: no synchronization that could mask the race.
            assert torch.equal(output, expected)
