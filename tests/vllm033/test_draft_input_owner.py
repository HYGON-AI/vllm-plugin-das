# SPDX-License-Identifier: Apache-2.0
"""Real Core draft-copy contracts; CPU buffers do not qualify GPU Graphs/MTP."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import speculator as official


def test_draft_input_copy_keeps_the_official_owner():
    method = official.DraftModelSpeculator._copy_request_inputs
    assert method.__module__ == official.__name__
    assert Path(inspect.getsourcefile(method)).resolve() == Path(official.__file__).resolve()
    assert not hasattr(method, '_vllm_hcu_draft_input_async_copy_wrapper')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('dummy_run', [None, False, True], ids=['default', 'normal', 'dummy'])
def test_draft_copy_preserves_owned_buffers_and_invalid_rows(dummy_run):
    # Only inject the helper's three data buffers. The production helper itself
    # runs unchanged; no model constructor or speculator execution is implied.
    buffers = SimpleNamespace(
        temperature=torch.full((6,), -99.0),
        seeds=torch.full((6,), -99, dtype=torch.int64),
        idx_mapping=torch.full((6,), 99, dtype=torch.int64),
    )
    owned = (buffers.temperature, buffers.seeds, buffers.idx_mapping)
    addresses = tuple(t.data_ptr() for t in owned)
    for epoch, indices in enumerate(([5, 1, 3, 0, 4], [2, 0], [], [4, 2, 0, 5, 1, 3], [1])):
        temperatures = torch.tensor([0, .25, .5, 1, 2, 4]) + epoch * .25
        seeds = torch.tensor([2**62 + i * 17 + epoch for i in range(6)], dtype=torch.int64)
        mapping = torch.tensor(indices, dtype=torch.int64)
        expected_temperatures, expected_seeds = temperatures.clone(), seeds.clone()
        kwargs = {} if dummy_run is None else {'dummy_run': dummy_run}
        official.DraftModelSpeculator._copy_request_inputs(
            buffers, len(indices), mapping, temperatures, seeds, **kwargs)
        current = (buffers.temperature, buffers.seeds, buffers.idx_mapping)
        assert all(a is b for a, b in zip(current, owned))
        assert tuple(t.data_ptr() for t in current) == addresses
        expected_indices = [-1] * 6 if dummy_run is True else list(indices) + [-1] * (6 - len(indices))
        assert buffers.idx_mapping.tolist() == expected_indices
        # Caller-owned inputs can be reused immediately without changing the
        # destination values. This is a CPU ownership check, not an async proof.
        temperatures.fill_(-7)
        seeds.fill_(-7)
        mapping.fill_(-7)
        torch.testing.assert_close(buffers.temperature, expected_temperatures, rtol=0, atol=0)
        torch.testing.assert_close(buffers.seeds, expected_seeds, rtol=0, atol=0)
        assert buffers.idx_mapping.tolist() == expected_indices
    assert not torch.cuda.is_initialized()
