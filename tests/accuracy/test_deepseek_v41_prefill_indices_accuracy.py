# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Device contracts for the fused DSV4.1 prefill combiner (no model required)."""

import statistics
import time

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_deepseek_v41_topk_output_size as patch

pytestmark = pytest.mark.hcu


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP accelerator")
    monkeypatch.setenv(patch._BACKEND_FLAG, "triton")
    return torch.device("cuda", torch.cuda.current_device())


def _args(query_lens, past, base, input_shape, window, topk, ratio, dtype):
    starts = [base]
    for length in query_lens:
        starts.append(starts[-1] + length)
    n = sum(query_lens)
    seq = [length + old for length, old in zip(query_lens, past)]
    gather = [min(s, length + 3) for s, length in zip(seq, query_lens)]
    gen = torch.Generator().manual_seed(41)
    indices = torch.randint(-2, 19, (n, *input_shape), generator=gen, dtype=dtype)
    # Include the large positive values which wrap to -1 on int32 conversion.
    if dtype == torch.int64 and indices.numel():
        indices.reshape(-1)[0] = 2**32 - 1
    return (
        indices, torch.tensor(starts, dtype=dtype), torch.tensor(seq, dtype=dtype),
        torch.tensor(gather, dtype=dtype), window, ratio, topk, 1024,
        0 if topk == 0 else 16,
    )


def _to_device(args, device):
    tensors = []
    for tensor in args[:4]:
        value = tensor.to(device)
        # Exercise both noncontiguous top-k and request metadata, including
        # chunk views. The wrapper must not interpret their strides as rows.
        tensors.append(torch.stack((value, value), dim=-1)[..., 0])
    return (*tensors, *args[4:])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 4, 128])
@pytest.mark.parametrize(
    "query_lens,past,base,input_shape,window,topk",
    [
        ([1], [0], 0, (8,), 128, 512),
        ([3, 1], [0, 129], 17, (2, 4), 8, 6),
        ([0, 3, 0, 2, 0], [0, 5, 0, 128, 0], 23, (8,), 129, 512),
        ([1, 7, 2], [0, 0, 130], 511, (512,), 128, 512),
        ([2, 1, 4], [130, 1, 0], 0, (8,), 0, 3),
        ([129], [0], 31, (8,), 1, 0),
        ([4097], [0], 31, (512,), 128, 512),
        ([3, 1], [0, 0], 17, (0,), 8, 6),
        ([3], [0], 17, (8,), 0, 0),
        ([4, 1], [-4, 0], 23, (8,), 128, 6),
    ],
)
def test_fused_indices_match_torch_and_cpu_reference(
    device, dtype, ratio, query_lens, past, base, input_shape, window, topk,
):
    args = _args(query_lens, past, base, input_shape, window, topk, ratio, dtype)
    # A zero-width input is meaningful (SWA only) even with topk > 0. The
    # existing Torch reshape(n, -1) supports this for nonempty n.
    reference = patch._combine_topk_swa_indices_torch(*args)
    gpu_args = _to_device(args, device)
    actual = patch._combine_topk_swa_indices(*gpu_args)
    torch_gpu = patch._combine_topk_swa_indices_torch(*gpu_args)
    torch.cuda.synchronize(device)
    for got, expected, fallback in zip(actual, reference, torch_gpu):
        assert got.dtype == torch.int32
        assert torch.equal(got.cpu(), expected)
        assert torch.equal(got, fallback)


@pytest.mark.parametrize("query_lens", [[], [0], [0, 0]])
def test_fused_empty_chunks(device, query_lens):
    args = _args(query_lens, [0] * len(query_lens), 17, (8,), 128, 512, 4, torch.int32)
    indices, lengths = patch._combine_topk_swa_indices(*_to_device(args, device))
    assert indices.shape == (0, 640)
    assert lengths.shape == (0,)


def test_fused_swa_only_accepts_model_zero_compress_ratio(device):
    args = _args([3, 1], [0, 129], 17, (8,), 128, 0, 0, torch.int32)
    reference = patch._combine_topk_swa_indices_torch(*args)
    actual = patch._combine_topk_swa_indices(*_to_device(args, device))
    for result, expected in zip(actual, reference):
        assert torch.equal(result.cpu(), expected)


def test_fused_launch_is_graph_capturable_and_writes_all_padding(device):
    args = _args([1, 7, 2], [0, 0, 130], 23, (8,), 129, 512, 4, torch.int32)
    gpu_args = _to_device(args, device)
    # Compile and warm up on the same side stream used for capture.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            patch._combine_topk_swa_indices(*gpu_args)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = patch._combine_topk_swa_indices(*gpu_args)
    graph.replay()
    expected = patch._combine_topk_swa_indices_torch(*args)
    for result, reference in zip(captured, expected):
        assert torch.equal(result.cpu(), reference)
    # The kernel uses empty outputs, so verify a second replay overwrites every
    # cell (including -1 padding), rather than relying on allocator contents.
    for result in captured:
        result.fill_(12345)
    graph.replay()
    for result, reference in zip(captured, expected):
        assert torch.equal(result.cpu(), reference)


def test_backend_switch_preserves_torch_fallback(device, monkeypatch):
    args = _args([3, 1], [0, 129], 17, (8,), 128, 512, 4, torch.int32)
    gpu_args = _to_device(args, device)
    fused = patch._combine_topk_swa_indices(*gpu_args)
    monkeypatch.setenv(patch._BACKEND_FLAG, "torch")
    fallback = patch._combine_topk_swa_indices(*gpu_args)
    monkeypatch.delenv(patch._BACKEND_FLAG)
    automatic = patch._combine_topk_swa_indices(*gpu_args)
    for got, expected, auto in zip(fused, fallback, automatic):
        assert torch.equal(got, expected)
        assert torch.equal(got, auto)


def test_fused_profile_has_no_dynamic_size_readback(device):
    args = _args([129], [0], 23, (512,), 128, 512, 4, torch.int32)
    gpu_args = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in args)
    for _ in range(3):
        patch._combine_topk_swa_indices(*gpu_args)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        patch._combine_topk_swa_indices(*gpu_args)
        torch.cuda.synchronize(device)
    names = {event.key for event in prof.key_averages()}
    assert any("_combine_topk_swa_indices_kernel" in name for name in names)
    assert not names.intersection({
        "aten::nonzero", "aten::_local_scalar_dense", "aten::repeat_interleave",
        "hipMemcpyWithStream", "cudaMemcpy",
    })


def _median_wall_us(function, args, repeats=10, trials=5):
    """Median wall-clock microseconds per call, synchronizing every call.

    This includes Python dispatch, allocation, launch and the synchronization
    itself, so it measures the caller-visible cost rather than pure kernel time.
    """
    for _ in range(repeats):
        function(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            function(*args)
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    return statistics.median(samples)


@pytest.mark.parametrize(
    ("query_lens", "input_width"),
    [([1], 512), ([128], 512), ([4096], 512), ([1, 7, 0, 4088], 512)],
)
def test_fused_combiner_is_faster_than_torch_fallback(device, query_lens, input_width):
    """The fused launch must beat the Torch fallback at real prefill sizes.

    This guards the reason for the kernel: the Torch path pays a device-to-host
    size readback per call, which is a fixed cost that dominates at small token
    counts and still matters at 4096 tokens.
    """
    args = _args(query_lens, [128] * len(query_lens), 17, (input_width,), 128, 512, 4, torch.int32)
    gpu_args = _to_device(args, device)
    reference = patch._combine_topk_swa_indices_torch(*gpu_args)
    fused = patch._combine_topk_swa_indices(*gpu_args)
    for got, expected in zip(fused, reference):
        assert torch.equal(got, expected)

    fallback_us = _median_wall_us(patch._combine_topk_swa_indices_torch, gpu_args)
    fused_us = _median_wall_us(patch._combine_topk_swa_indices, gpu_args)
    print(f"tokens={sum(query_lens)} fallback={fallback_us:.1f}us fused={fused_us:.1f}us")
    assert fused_us < fallback_us
