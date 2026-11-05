from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.attention.ops import deepseek_v41_swa_metadata as metadata


@pytest.fixture
def scratch(monkeypatch):
    original_empty = torch.empty
    events = []

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    class Event:
        def __init__(self):
            self.synchronizations = 0
            self.recordings = 0
            events.append(self)

        def synchronize(self):
            self.synchronizations += 1

        def record(self):
            self.recordings += 1

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(metadata, "_PTR_SCRATCH", {})
    return events


def make_job(num_slots):
    return SimpleNamespace(
        replay_start=torch.zeros(3, dtype=torch.int32),
        slot_mapping=torch.zeros(num_slots, dtype=torch.int64),
        block_table=torch.zeros(3, 8, dtype=torch.int32)[:, ::2],
        token_to_req=torch.zeros(3, dtype=torch.int32),
        is_valid=torch.ones(num_slots, dtype=torch.bool),
        indices=torch.zeros(3, 16, dtype=torch.int32)[:, ::2],
        lens=torch.zeros(3, dtype=torch.int32),
        ragged=torch.zeros(24, dtype=torch.int32),
        indptr=torch.zeros(4, dtype=torch.int32),
        num_slots=num_slots,
    )


def assert_pointer_table(views, jobs):
    for index, job in enumerate(jobs):
        for row, field in (
            ("replay_start_ptrs", "replay_start"),
            ("slot_mapping_ptrs", "slot_mapping"),
            ("block_table_ptrs", "block_table"),
            ("token_to_req_ptrs", "token_to_req"),
            ("is_valid_ptrs", "is_valid"),
            ("indices_ptrs", "indices"),
            ("lens_ptrs", "lens"),
            ("ragged_ptrs", "ragged"),
            ("indptr_ptrs", "indptr"),
        ):
            assert views[row][index].item() == getattr(job, field).data_ptr()
        assert views["block_table_strides"][index].item() == job.block_table.stride(0)
        assert views["indices_strides"][index].item() == job.indices.stride(0)
        assert views["lens_sizes"][index].item() == job.lens.numel()
        assert views["num_slots_arr"][index].item() == job.num_slots


def test_pointer_views_reused_while_addresses_and_slots_refresh(scratch):
    jobs = [make_job(3), make_job(7)]
    views = metadata._fill_ptr_scratch(jobs)
    assert_pointer_table(views, jobs)
    pointers = {name: value.data_ptr() for name, value in views.items()}
    assert metadata._fill_ptr_scratch(jobs) is views
    assert len(scratch) == 1
    assert scratch[0].recordings == 1
    assert scratch[0].synchronizations == 0
    previous_jobs = []
    for num_slots in (8, 9):
        previous_jobs.append(jobs[0])
        jobs[0] = make_job(num_slots)
        assert metadata._fill_ptr_scratch(jobs) is views
        assert_pointer_table(views, jobs)
        assert {name: value.data_ptr() for name, value in views.items()} == pointers
    assert len(scratch) == 2
    assert scratch[0].synchronizations == 1
    assert scratch[1].synchronizations == 0
    assert metadata._fill_ptr_scratch(jobs[:1]) is not views
    assert_pointer_table(views, jobs)
