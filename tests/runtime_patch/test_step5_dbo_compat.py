# SPDX-License-Identifier: Apache-2.0
"""DBO must attach to the plugin backend without a vLLM source module."""

from types import SimpleNamespace
import dataclasses

import numpy as np
import pytest
import torch


def test_dbo_wraps_plugin_dsa_schedule_once(monkeypatch):
    import sys
    from vllm_hcu.model_executor.layers import step4_dsa as dsa
    from vllm_hcu.v1 import hcu_dbo
    from vllm.v1.worker import ubatching

    calls = []
    result = object()

    def baseline(self, md):
        calls.append(md)
        return result

    monkeypatch.setattr(dsa.Step4DSAState, "schedule", baseline)
    monkeypatch.delitem(
        sys.modules, "vllm.model_executor.layers.step4_dsa", raising=False
    )
    monkeypatch.setattr(ubatching, "dbo_enabled", lambda: False)
    hcu_dbo._patch_dsa()
    wrapper = dsa.Step4DSAState.schedule
    assert getattr(wrapper, "_hcu_dbo", False)
    assert wrapper(SimpleNamespace(), "metadata") is result
    assert calls == ["metadata"]
    hcu_dbo._patch_dsa()
    assert dsa.Step4DSAState.schedule is wrapper


@pytest.mark.parametrize("mirrors", [False, True])
def test_dbo_slices_clean_upstream_batch_and_dynamic_host_mirrors(monkeypatch, mirrors):
    from vllm.v1.worker.gpu import ubatch_utils as uu
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    from vllm_hcu.v1 import hcu_dbo

    # Preserve the genuine upstream constructor: the old source patch added
    # host-mirror fields that this clean-base dataclass deliberately lacks.
    original = uu._slice_input_batch
    monkeypatch.setattr(uu, "_slice_input_batch", original)
    hcu_dbo._patch_slicing()
    batch = InputBatch.make_dummy(2, 12, InputBuffers(2, 12, torch.device("cpu")))
    assert "block_tables_np" not in {f.name for f in dataclasses.fields(batch)}
    if mirrors:
        batch.seq_lens_cpu = torch.tensor([16, 26], dtype=torch.int32)
        batch.num_computed_tokens_cpu = torch.tensor([10, 20], dtype=torch.int32)
        batch.block_tables_np = (np.array([[100, 101], [200, 201]]),)
    sliced = uu._slice_input_batch(
        batch,
        UBatchSlice(slice(0, 2), slice(4, 9)),
        torch.empty(3, dtype=torch.int32),
        torch.empty(2, dtype=torch.int32),
    )
    assert isinstance(sliced, InputBatch)
    np.testing.assert_array_equal(sliced.query_start_loc_np, [0, 2, 5])
    assert sliced.max_query_len == 3
    if mirrors:
        torch.testing.assert_close(
            sliced.seq_lens_cpu, torch.tensor([16, 23], dtype=torch.int32)
        )
        torch.testing.assert_close(
            sliced.num_computed_tokens_cpu, torch.tensor([14, 20], dtype=torch.int32)
        )
        np.testing.assert_array_equal(
            sliced.block_tables_np[0], batch.block_tables_np[0]
        )
        torch.testing.assert_close(
            batch.seq_lens_cpu, torch.tensor([16, 26], dtype=torch.int32)
        )
    else:
        assert sliced.seq_lens_cpu is None
        assert sliced.num_computed_tokens_cpu is None
        assert sliced.block_tables_np is None


def test_mtp_tail_preserves_dynamic_mirrors_on_clean_input_batch(monkeypatch):
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm_hcu.patch.worker.framework_opt import patch_mtp_prefill_tail as tail

    monkeypatch.setattr(tail, "_TAIL", 4)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    batch = InputBatch.make_dummy(2, 12, InputBuffers(2, 12, torch.device("cpu")))
    batch.is_prefilling_np[:] = True
    batch.has_prefill = True
    batch.input_ids.copy_(torch.arange(12))
    batch.positions.copy_(torch.arange(12))
    batch.seq_lens_cpu = torch.tensor([16, 26], dtype=torch.int32)
    batch.num_computed_tokens_cpu = torch.tensor([10, 20], dtype=torch.int32)
    batch.block_tables_np = (np.array([[100, 101], [200, 201]]),)
    hidden = torch.arange(24).reshape(12, 2)
    owner = SimpleNamespace(dp_size=1, device=torch.device("cpu"))
    trimmed, trimmed_hidden, sync = tail._trim(owner, batch, hidden, None)
    np.testing.assert_array_equal(trimmed.query_start_loc_np, [0, 4, 8])
    torch.testing.assert_close(
        trimmed.input_ids, torch.tensor([2, 3, 4, 5, 8, 9, 10, 11], dtype=torch.int32)
    )
    torch.testing.assert_close(trimmed_hidden, hidden[[2, 3, 4, 5, 8, 9, 10, 11]])
    torch.testing.assert_close(trimmed.seq_lens_cpu, batch.seq_lens_cpu)
    torch.testing.assert_close(
        trimmed.num_computed_tokens_cpu, torch.tensor([12, 22], dtype=torch.int32)
    )
    assert trimmed.block_tables_np is batch.block_tables_np
    assert sync is None and batch.num_tokens == 12


@pytest.mark.parametrize("peer_rows,expected_ubatches,expected_votes", [
    ([4] * 512 + [2048], 1, [0, 1]),
    ([4] * 256 + [3072], 2, [1, 1]),
], ids=["unsafe-peer-veto", "safe-mixed-dbo"])
def test_step5_real_cpu_dispatch_agrees_before_slicing(
    monkeypatch, peer_rows, expected_ubatches, expected_votes
):
    """Exercise installed dispatch/sync bodies and the full Step5 entry point.

    Only the DP communication boundary is simulated. Stop immediately after
    dispatch, before input preparation or model/device work can begin.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.gpu.model_runner import BatchReqState
    from vllm_hcu.model_executor.layers import step4_dsa as dsa
    from vllm_hcu.v1 import hcu_dbo, step5_execution

    assert step5_execution.dispatch_cg_and_sync_dp is dp_utils.dispatch_cg_and_sync_dp
    original_threshold = dp_utils.check_ubatch_thresholds
    monkeypatch.setattr(dp_utils, "check_ubatch_thresholds", original_threshold)
    monkeypatch.setattr(hcu_dbo, "DBO_MIN_TOKENS", 4096)
    hcu_dbo._patch_dp_threshold()

    cpu_group = object()
    monkeypatch.setattr(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=cpu_group))
    ops = dp_utils.dist.ReduceOp
    barriers = {ops.SUM: threading.Barrier(2), ops.MIN: threading.Barrier(2)}
    contributions = {ops.SUM: [], ops.MIN: []}
    lock = threading.Lock()

    def all_reduce(tensor, *, group, op=ops.SUM):
        assert group is cpu_group
        assert tensor.device.type == "cpu" and tensor.dtype == torch.int32
        assert tuple(tensor.shape) == ((6, 2) if op == ops.SUM else (1,))
        with lock:
            contributions[op].append(tensor.clone())
        barriers[op].wait(timeout=10)
        values = torch.stack(contributions[op])
        result = values.sum(dim=0) if op == ops.SUM else values.amin(dim=0)
        tensor.copy_(result)

    monkeypatch.setattr(dp_utils.dist, "all_reduce", all_reduce)

    def forbid_cuda_init():
        raise AssertionError("CPU dispatch regression must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_cuda_init)

    class DispatchComplete(Exception):
        def __init__(self, descriptor):
            self.descriptor = descriptor

    def stop_after_dispatch(runner, descriptor):
        assert hcu_dbo._DBO_BATCH.get() is None
        raise DispatchComplete(descriptor)

    monkeypatch.setattr(step5_execution, "record_context_graph_mode", stop_after_dispatch)

    def run_rank(rank, rows):
        qlens = np.array(rows, dtype=np.int32)
        req_ids = [f"rank{rank}-req{i}" for i in range(len(rows))]
        batch = BatchReqState(
            req_ids=req_ids, num_scheduled_tokens=qlens, num_tokens=int(qlens.sum()),
            idx_mapping_np=np.arange(len(rows), dtype=np.int32),
            prefill_len_np=qlens.copy(),
            num_computed_prefill_tokens_np=np.zeros_like(qlens),
            is_prefilling_np=qlens > 4, has_prefill=True,
        )
        builder = dsa.Step4DSAMetadataBuilder.__new__(dsa.Step4DSAMetadataBuilder)
        builder.reorder_batch_threshold = 4
        noop = lambda *args: None
        runner = SimpleNamespace(
            update_pp_decode_requests=noop, finish_requests=noop, free_states=noop,
            add_requests=noop, update_requests=noop,
            block_tables=SimpleNamespace(apply_staged_writes=noop),
            gather_batch_req_state=lambda *args: (batch, None),
            pcp_manager=None, adaptive_verification=None, lora_config=None,
            is_encoder_decoder=False, cudagraph_manager=None,
            parallel_config=SimpleNamespace(
                _hcu_step5_dbo=True, use_ubatching=True, num_ubatches=2
            ),
            dp_size=2, dp_rank=rank, decode_query_len=4, ubatch_runner=object(),
            attn_groups=[[SimpleNamespace(metadata_builders=[builder, builder])]],
        )
        scheduler = SimpleNamespace(
            num_scheduled_tokens=dict(zip(req_ids, rows)),
            total_num_scheduled_tokens=batch.num_tokens,
        )
        with pytest.raises(DispatchComplete) as stopped:
            step5_execution.execute_model(runner, scheduler, is_profile=True)
        assert hcu_dbo._DBO_BATCH.get() is None
        return stopped.value.descriptor.num_ubatches

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_rank, 0, [4096]), pool.submit(run_rank, 1, peer_rows)]
        decisions = [future.result(timeout=15) for future in futures]
    assert decisions == [expected_ubatches, expected_ubatches]
    assert len(contributions[ops.SUM]) == len(contributions[ops.MIN]) == 2
    assert sorted(int(v.item()) for v in contributions[ops.MIN]) == expected_votes
