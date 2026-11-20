# SPDX-License-Identifier: Apache-2.0
"""Real v0.33 configuration owners; no capture, weights or device execution."""

from hcu_vllm_das.adapter_api import adapter_module as _adapter_module

import inspect

import pytest
import torch

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.config.device import DeviceConfig


@pytest.mark.parametrize(
    "query_len,sizes,maximum,expected,expected_maximum",
    [
        (1, [1, 2, 4, 8, 16], 16, [1, 2, 4, 8, 16], 16),
        (3, [1, 2, 4, 8, 16], 16, [3, 6, 9], 9),
        (4, [1, 2, 4, 8, 16], 16, [4, 8, 16], 16),
        (3, [], 16, [], 16),
    ],
)
def test_official_speculative_size_rounding(query_len, sizes, maximum, expected, expected_maximum):
    config = CompilationConfig(
        cudagraph_capture_sizes=sizes,
        max_cudagraph_capture_size=maximum,
    )
    config.adjust_cudagraph_sizes_for_spec_decode(query_len)
    assert config.cudagraph_capture_sizes == expected
    assert config.max_cudagraph_capture_size == expected_maximum
    assert not torch.cuda.is_initialized()


def test_official_size_rounding_rejects_an_impossible_capture_limit():
    config = CompilationConfig(cudagraph_capture_sizes=[1, 2], max_cudagraph_capture_size=2)
    with pytest.raises(ValueError, match="No valid cudagraph sizes"):
        config.adjust_cudagraph_sizes_for_spec_decode(3)
    assert config.cudagraph_capture_sizes == [1, 2]
    assert config.max_cudagraph_capture_size == 2


@pytest.mark.parametrize(
    "backend,dp_size,expected",
    [
        ("deepep_high_throughput", 2, CUDAGraphMode.NONE),
        ("deepep_high_throughput", 1, CUDAGraphMode.FULL_DECODE_ONLY),
        ("deepep_low_latency", 2, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_official_deepep_graph_policy(backend, dp_size, expected):
    config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        use_inductor_graph_partition=False,
    )
    config.set_splitting_ops_for_v1(backend, dp_size)
    assert config.cudagraph_mode is expected
    assert not torch.cuda.is_initialized()


def _config(mode, graph_mode, inductor_partition):
    compilation = CompilationConfig(
        mode=mode,
        cudagraph_mode=graph_mode,
        use_inductor_graph_partition=inductor_partition,
    )
    config = VllmConfig(device_config=DeviceConfig("cpu"), compilation_config=compilation)
    compilation.set_splitting_ops_for_v1("naive", 1)
    return config


def test_adapter_split_registration_keeps_the_official_configuration_methods():
    names = ("adjust_cudagraph_sizes_for_spec_decode", "set_splitting_ops_for_v1")
    methods = {name: vars(CompilationConfig)[name] for name in names}
    assert tuple(inspect.signature(methods[names[0]]).parameters) == (
        "self", "uniform_decode_query_len"
    )
    owner = _adapter_module('v41_fp8_indexer')

    stock = tuple(CompilationConfig._attention_ops)
    config = _config(CompilationMode.VLLM_COMPILE, CUDAGraphMode.PIECEWISE, False)
    compilation = config.compilation_config
    original = tuple(compilation.splitting_ops)
    assert owner.OP_NAME not in original
    assert owner.register_splitting_op(config)
    assert compilation.splitting_ops == [*original, owner.OP_NAME]
    assert owner.register_splitting_op(config)
    assert compilation.splitting_ops.count(owner.OP_NAME) == 1
    assert tuple(CompilationConfig._attention_ops) == stock
    for name, method in methods.items():
        assert vars(CompilationConfig)[name] is method
        assert method.__code__.co_filename == inspect.getsourcefile(CompilationConfig)
    other = _config(CompilationMode.VLLM_COMPILE, CUDAGraphMode.PIECEWISE, False)
    assert owner.OP_NAME not in other.compilation_config.splitting_ops
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "mode,graph_mode,inductor_partition",
    [
        (CompilationMode.NONE, CUDAGraphMode.NONE, False),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.NONE, False),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.PIECEWISE, True),
    ],
)
def test_adapter_does_not_append_a_dynamo_split_outside_piecewise(mode, graph_mode, inductor_partition):
    owner = _adapter_module('v41_fp8_indexer')

    config = _config(mode, graph_mode, inductor_partition)
    before = list(config.compilation_config.splitting_ops)
    assert owner.register_splitting_op(config) is False
    assert config.compilation_config.splitting_ops == before
    assert owner.OP_NAME not in before
    assert not torch.cuda.is_initialized()
