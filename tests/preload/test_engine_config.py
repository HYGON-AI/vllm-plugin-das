# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_hcu.v1.config_consistency import (
    engine_config_signature,
    validate_distributed_engine_config,
)


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
    def __init__(self, expected, rank=1):
        self.expected = expected
        self.rank = rank
        self.broadcasts = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def broadcast_object_list(self, values, src):
        self.broadcasts += 1
        values[0] = self.expected


def test_signature_covers_shape_affecting_scheduler_and_parallel_values():
    signature = engine_config_signature(make_config())
    assert signature["tensor_parallel_size"] == 8
    assert signature["pipeline_parallel_size"] == 2
    assert signature["max_num_seqs"] == 1
    assert signature["max_num_batched_tokens"] == 512


def test_matching_distributed_config_passes():
    config = make_config()
    dist = FakeDist(engine_config_signature(config))
    validate_distributed_engine_config(config, dist)
    assert dist.broadcasts == 1


def test_scheduler_drift_fails_before_model_loading():
    config = make_config(max_num_seqs=128)
    dist = FakeDist(engine_config_signature(make_config()))
    with pytest.raises(RuntimeError, match="max_num_seqs"):
        validate_distributed_engine_config(config, dist)


def test_uninitialized_process_group_is_a_noop():
    class NotInitialized(FakeDist):
        def is_initialized(self):
            return False

    validate_distributed_engine_config(make_config(), NotInitialized({}))
