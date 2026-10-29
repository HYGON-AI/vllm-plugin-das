# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_hcu.v1.config_consistency import (
    engine_config_signature,
    validate_distributed_engine_config,
)


def _run_config_check(rank, init_method, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        config = make_config(max_num_seqs=128 if rank else 1)
        try:
            validate_distributed_engine_config(config, dist)
        except RuntimeError as exc:
            results.put((rank, str(exc)))
        else:
            results.put((rank, None))
    finally:
        dist.destroy_process_group()


def make_config(**scheduler):
    scheduler_values = {
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
        "max_num_scheduled_tokens": 512,
        **scheduler,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="kimi", revision=None, dtype="torch.bfloat16",
            quantization="slimquant_w4a8", max_model_len=1024,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, pipeline_parallel_size=2,
            data_parallel_size=1, world_size=16,
            distributed_executor_backend="mp",
        ),
        scheduler_config=SimpleNamespace(**scheduler_values),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_e4m3", block_size=64,
            enable_prefix_caching=True,
        ),
        use_v2_model_runner=True,
    )


class FakeDist:
    def __init__(self, signatures):
        self.signatures = signatures
        self.gathers = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return len(self.signatures)

    def all_gather_object(self, values, local):
        self.gathers += 1
        values[:] = self.signatures


def test_signature_covers_shape_affecting_scheduler_and_parallel_values():
    signature = engine_config_signature(make_config())
    assert signature["tensor_parallel_size"] == 8
    assert signature["pipeline_parallel_size"] == 2
    assert signature["max_num_seqs"] == 1
    assert signature["max_num_batched_tokens"] == 512


def test_matching_distributed_config_passes():
    config = make_config()
    dist = FakeDist([engine_config_signature(config)] * 2)
    validate_distributed_engine_config(config, dist)
    assert dist.gathers == 1


def test_scheduler_drift_fails_before_model_loading():
    config = make_config(max_num_seqs=128)
    baseline = make_config()
    signatures = [engine_config_signature(baseline), engine_config_signature(config)]
    for local in (baseline, config):
        with pytest.raises(RuntimeError, match="rank 1: max_num_seqs"):
            validate_distributed_engine_config(local, FakeDist(signatures))


def test_scheduler_drift_stops_all_real_ranks(tmp_path):
    import multiprocessing

    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo process group is unavailable")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    init_method = (tmp_path / "rendezvous").as_uri()
    processes = [ctx.Process(target=_run_config_check, args=(rank, init_method, results))
                 for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        reported = dict(results.get(timeout=5) for _ in processes)
        assert all("rank 1: max_num_seqs" in reported[rank] for rank in range(2))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()


def test_uninitialized_process_group_is_a_noop():
    class NotInitialized(FakeDist):
        def is_initialized(self):
            return False

    validate_distributed_engine_config(make_config(), NotInitialized({}))
