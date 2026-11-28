# SPDX-License-Identifier: Apache-2.0
"""Native gfx936 FP8 host-backed prefill parity against decoded KV rows."""

from types import SimpleNamespace

import pytest


@pytest.mark.hcu
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "graph"])
def test_fp8_prefill_gathers_ragged_chunks_from_host_and_pitched_gpu(replay):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if (
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]
        != "gfx936"
    ):
        pytest.skip("requires gfx936")

    from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
    from hcu_vllm_das.models.hy_v4_flash.hisparse_fp8_prefill import (
        hisparse_fp8_prefill_scope,
    )
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
    from vllm.v1.hisparse.runtime import (
        HiSparseRuntime,
        build_hisparse_prefill_staging_plan,
    )

    ensure_hisparse_ops()
    host_block, resident_block, width = 128, 64, 656
    nblocks = 8
    resident_pages = {
        1: (0, 7),
        2: (3, 0),
        3: (1, 0),
        4: (0, 0),
        5: (6, 8),
        6: (5, 0),
        7: (0, 0),
        0: (0, 0),
    }
    tables = [[4, 1, 3, 5], [2, 3, 0, 0], [1, 0, 0, 0], [5, 7, 6, 0]]
    lengths = [393, 137, 9, 263]
    rows = torch.arange(nblocks * host_block, dtype=torch.int32)[:, None]
    packed = torch.empty((nblocks * host_block, width), dtype=torch.uint8)
    packed[:, :512] = (rows * 53 + torch.arange(512, dtype=torch.int32)) % 126 + 1
    scales = (rows % 17 + torch.arange(4, dtype=torch.int32) + 1).float() / 16.0
    packed[:, 512:528] = scales.view(torch.uint8)
    rope = ((rows % 897 + torch.arange(64, dtype=torch.int32)).float() / 8).bfloat16()
    packed[:, 528:] = rope.view(torch.uint8)
    nope = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    expected = torch.cat(
        (
            (nope.view(-1, 4, 128) * scales[:, :, None]).flatten(1).bfloat16(),
            rope,
        ),
        dim=1,
    ).view(nblocks, host_block, 576)
    packed = packed.view(nblocks, host_block, width)

    # Pitch and offset are both nonzero. Poison the host rows that the plan must
    # read from their current resident copies.
    token_pitch, page_pitch, offset = 672, resident_block * 672 + 128, 16
    raw = torch.empty(9 * page_pitch + offset, dtype=torch.uint8, device="cuda")
    resident = raw.as_strided(
        (9, resident_block, width), (page_pitch, token_pitch, 1), offset
    )
    source = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
    source.copy_(packed)
    for host_page, gpu_pages in resident_pages.items():
        for half, gpu_page in enumerate(gpu_pages):
            if gpu_page:
                source[
                    host_page, half * resident_block : (half + 1) * resident_block
                ].zero_()

    class Runtime(HiSparseRuntime):
        copies = 0

        def gather_prefill_cache(self, *args, **kwargs):
            self.copies += 1
            return super().gather_prefill_cache(*args, **kwargs)

    runtime = Runtime.__new__(Runtime)
    source_table = torch.tensor(tables, dtype=torch.int32, device="cuda")
    true_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    resident_table = torch.zeros((8, 8), dtype=torch.int32, device="cuda")
    states = [4, 6, 2, 7]
    for state, table in zip(states, tables, strict=True):
        resident_table[state] = torch.tensor(
            [page for host_page in table for page in resident_pages[host_page]],
            dtype=torch.int32,
            device="cuda",
        )
    runtime.request_state_indices = torch.tensor([3, *states], device="cuda")
    cache = SimpleNamespace(
        runtime=runtime,
        block_table=resident_table,
        view=SimpleNamespace(cache=resident, block_size=resident_block),
    )
    plan = build_hisparse_prefill_staging_plan(
        source_table,
        true_lengths,
        host_block,
        staging_block_capacity=16,
    )
    metadata = SimpleNamespace(
        num_decodes=1,
        prefill=SimpleNamespace(host_staging_plan=plan),
    )
    index_group = HiSparseMLAIndexGroup.__new__(HiSparseMLAIndexGroup)
    index_group.cache = lambda layer_index: cache
    index_group.prefill_stream = torch.cuda.Stream()
    index_group.prefill_ready_events = [torch.cuda.Event()]
    original_gather = index_group.gather_fp8_prefill
    workspace = torch.full((540, 576), -907.0, dtype=torch.bfloat16, device="cuda")
    compute = torch.cuda.Stream()
    chunks = [(0, 2), (2, 4)]
    starts = [
        torch.tensor([0, lengths[start]], dtype=torch.int32, device="cuda")
        for start, _ in chunks
    ]

    def launch_chunks(outputs, guards):
        with hisparse_fp8_prefill_scope(index_group):
            for offsets, (start, end) in zip(starts, chunks, strict=True):
                count = sum(lengths[start:end])
                workspace.fill_(-907.0)
                ready = index_group.gather_fp8_prefill(
                    0,
                    source,
                    workspace[:count],
                    source_table[start:end],
                    offsets,
                    end - start,
                    metadata,
                    start,
                )
                compute.wait_event(ready)
                outputs.append(workspace[:count].clone())
                guards.append(workspace[count:].clone())

    def check_outputs(outputs, guards):
        for out, (start, end) in zip(outputs, chunks, strict=True):
            wanted = torch.cat(
                [
                    expected[table].flatten(0, 1)[:length]
                    for table, length in zip(
                        tables[start:end], lengths[start:end], strict=True
                    )
                ]
            )
            torch.testing.assert_close(out.cpu(), wanted, rtol=0, atol=0)
        assert all((guard.cpu() == -907.0).all().item() for guard in guards)

    outputs, guards = [], []
    compute.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(compute):
        resident.fill_(0)
        for host_page, gpu_pages in resident_pages.items():
            for half, gpu_page in enumerate(gpu_pages):
                if gpu_page:
                    resident[gpu_page].copy_(
                        packed[
                            host_page,
                            half * resident_block : (half + 1) * resident_block,
                        ]
                    )
        launch_chunks(outputs, guards)
    torch.cuda.current_stream().wait_stream(compute)
    check_outputs(outputs, guards)
    assert runtime.copies == 1
    if replay:
        outputs, guards = [], []
        plan.gpu_source_key = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=compute):
            launch_chunks(outputs, guards)
        graph.replay()
        check_outputs(outputs, guards)
        # Capture retains the plan, while replay must read both current sources.
        source[4, :, 528:].view(torch.bfloat16).fill_(-7)
        resident[3, :, 528:].view(torch.bfloat16).fill_(-13)
        expected[4, :, 512:].fill_(-7)
        expected[2, :resident_block, 512:].fill_(-13)
        graph.replay()
        check_outputs(outputs, guards)
        assert runtime.copies == 2
    assert index_group.gather_fp8_prefill == original_gather


def test_fp8_prefill_scope_restores_instance_override_on_failure():
    from hcu_vllm_das.models.hy_v4_flash.hisparse_fp8_prefill import (
        hisparse_fp8_prefill_scope,
    )

    original = object()
    index_group = SimpleNamespace(gather_fp8_prefill=original)
    with (
        pytest.raises(RuntimeError, match="failed forward"),
        hisparse_fp8_prefill_scope(index_group),
    ):
        assert callable(index_group.gather_fp8_prefill)
        raise RuntimeError("failed forward")
    assert index_group.gather_fp8_prefill is original
