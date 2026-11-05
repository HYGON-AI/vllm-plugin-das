import pytest
import torch

from vllm_hcu.v1.attention.ops import deepseek_v41_swa_metadata as metadata


pytestmark = pytest.mark.hcu


def make_jobs(noncausal):
    query_start = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([14, 19], dtype=torch.int32, device="cuda")
    jobs = []
    for index in range(3):
        jobs.append(metadata.SwaMetaJob(
            query_start_loc=query_start,
            seq_lens=seq_lens,
            slot_mapping=torch.tensor([0, 1, 2, 3, -1, -1, -1], dtype=torch.int64, device="cuda"),
            replay_start=torch.tensor([index, index + 1], dtype=torch.int32, device="cuda"),
            block_table=torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8) + index,
            token_to_req=torch.zeros(5, dtype=torch.int32, device="cuda"),
            is_valid=torch.zeros(7, dtype=torch.bool, device="cuda"),
            indices=torch.zeros(5, 16, dtype=torch.int32, device="cuda"),
            lens=torch.zeros(8, dtype=torch.int32, device="cuda"),
            ragged=torch.zeros(80, dtype=torch.int32, device="cuda"),
            indptr=torch.zeros(6, dtype=torch.int32, device="cuda"),
            num_tokens=5, num_reqs=2, num_slots=7, window_size=8,
            block_size=4, width=16, noncausal=noncausal, one_launch=True, capacity=8,
        ))
    return jobs


def assert_reference(job):
    starts = job.query_start_loc.cpu().tolist()
    lengths = job.seq_lens.cpu().tolist()
    table = job.block_table.cpu()
    replay = job.replay_start.cpu().tolist()
    slots = job.slot_mapping.cpu()
    expected_mapping = [0, 0, 0, 1, 1]
    expected_dense = torch.full((5, 16), -1, dtype=torch.int32)
    expected_lengths = []
    expected_ragged = []
    expected_indptr = [0]
    for row, request in enumerate(expected_mapping):
        prefix = lengths[request] - (starts[request + 1] - starts[request])
        end = lengths[request] if job.noncausal else prefix + row - starts[request] + 1
        begin = max(prefix - job.window_size, 0) if job.noncausal else max(end - job.window_size, replay[request], 0)
        row_slots = [
            int(table[request, position // job.block_size]) * job.block_size + position % job.block_size
            for position in range(begin, end)
        ] if slots[row] >= 0 else []
        expected_lengths.append(len(row_slots))
        expected_dense[row, :len(row_slots)] = torch.tensor(row_slots, dtype=torch.int32)
        expected_ragged.extend(row_slots)
        expected_indptr.append(len(expected_ragged))
    assert job.token_to_req.cpu().tolist() == expected_mapping
    assert torch.equal(job.is_valid.cpu(), slots >= 0)
    assert torch.equal(job.indices.cpu(), expected_dense)
    assert job.lens.cpu().tolist() == expected_lengths + [0, 0, 0]
    assert job.indptr.cpu().tolist() == expected_indptr
    assert job.ragged[:len(expected_ragged)].cpu().tolist() == expected_ragged


@pytest.mark.parametrize("noncausal", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_batch_pointer_refresh_matches_reference(noncausal, graph):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    jobs = make_jobs(noncausal)
    metadata._launch_batch(jobs)
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            metadata._launch_batch(jobs)
    for step in range(4):
        jobs[0].seq_lens.add_(1)
        for index, job in enumerate(jobs):
            job.replay_start.fill_(step + index)
            job.slot_mapping[4] = 4 if step % 2 else -1
            if not graph:
                job.block_table = job.block_table.clone() + 1
        if graph:
            captured.replay()
        else:
            metadata._launch_batch(jobs)
        torch.cuda.synchronize()
        for job in jobs:
            assert_reference(job)
