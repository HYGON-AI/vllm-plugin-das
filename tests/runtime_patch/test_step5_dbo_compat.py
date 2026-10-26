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
