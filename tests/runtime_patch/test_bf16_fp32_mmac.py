# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("MMAC GEMM tests need a working GPU", allow_module_level=True)

pytestmark = pytest.mark.hcu

from vllm_hcu.utils import bf16_fp32_gemm, bf16_fp32_mmac  # noqa: E402


@pytest.mark.parametrize("m", [1, 7, 8, 16, 33, 48, 64])
@pytest.mark.parametrize("n", sorted(bf16_fp32_mmac.SUPPORTED_N))
@pytest.mark.parametrize("k", [512, 4096])
def test_matches_fp32_reference(m, n, k):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    out = bf16_fp32_mmac.linear_bf16_fp32_mmac(x, w)
    ref = x.float() @ w.float().T
    assert out.dtype == torch.float32 and out.shape == (m, n)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize(
    ("m", "n", "k", "expected"),
    [
        (48, 1024, 4096, True),
        (64, 2048, 4096, True),
        (65, 1024, 4096, False),
        (48, 1536, 4096, False),
        (48, 1024, 4000, False),
    ],
)
def test_supported_contract(m, n, k, expected):
    x = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    assert bf16_fp32_mmac.supported(x, w) is expected


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (524288, True),
        (524416, False),
        (786432, False),
    ],
)
def test_supported_limits_buffer_resource_by_bytes(monkeypatch, k, expected):
    class FakeTensor:
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        is_cuda = True

        def __init__(self, shape):
            self.shape = shape

        def dim(self):
            return 2

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 16

    monkeypatch.setattr(bf16_fp32_mmac, "_device_has_mmac", lambda _: True)
    x = FakeTensor((8, k))
    weight = FakeTensor((2048, k))
    assert bf16_fp32_mmac.supported(x, weight) is expected


@pytest.mark.parametrize("case", ["cpu", "different_device", "unsupported_arch"])
def test_direct_entry_rejects_unsupported_tensors(monkeypatch, case):
    monkeypatch.setattr(
        bf16_fp32_mmac,
        "_jit_module",
        lambda: pytest.fail("unsupported tensors reached the JIT kernel"),
    )
    if case == "cpu":
        x = torch.empty(8, 512, dtype=torch.bfloat16)
        weight = torch.empty(256, 512, dtype=torch.bfloat16)
    elif case == "different_device":
        monkeypatch.setattr(bf16_fp32_mmac, "_device_has_mmac", lambda _: True)
        x = torch.empty(8, 512, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(256, 512, dtype=torch.bfloat16)
    else:
        monkeypatch.setattr(bf16_fp32_mmac, "_device_has_mmac", lambda _: False)
        x = torch.empty(8, 512, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(256, 512, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="does not support"):
        bf16_fp32_mmac.linear_bf16_fp32_mmac(x, weight)


def test_cpp_entry_guards_input_device_before_launch_setup():
    source = bf16_fp32_mmac._SOURCE
    guard = source.index("OptionalCUDAGuard device_guard(device_of(x))")
    output_allocation = source.index("auto out = splitk")
    stream_lookup = source.index("getCurrentCUDAStream()")
    assert guard < output_allocation < stream_lookup


def test_dispatcher_routes_small_m(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_BF16_FP32_MMAC", "1")
    calls = []
    real = bf16_fp32_mmac.linear_bf16_fp32_mmac

    def spy(x, w):
        calls.append(x.shape[0])
        return real(x, w)

    monkeypatch.setattr(bf16_fp32_mmac, "linear_bf16_fp32_mmac", spy)
    w = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
    for m in (48, 128):
        x = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            bf16_fp32_gemm.linear_bf16_fp32(x, w),
            x.float() @ w.float().T,
            rtol=1e-4,
            atol=1e-3,
        )
    assert calls == [48]

    monkeypatch.setenv("VLLM_HCU_BF16_FP32_MMAC", "0")
    bf16_fp32_gemm.linear_bf16_fp32(
        torch.randn(48, 4096, dtype=torch.bfloat16, device="cuda"), w
    )
    assert calls == [48]


@pytest.mark.parametrize(
    ("custom_ops", "mmac", "expected"),
    [
        ("0", "0", False),
        ("0", "1", False),
        ("1", "0", False),
        ("1", "1", True),
    ],
)
def test_mmac_enabled_honors_custom_ops_switch(
    monkeypatch, custom_ops, mmac, expected
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", custom_ops)
    monkeypatch.setenv("VLLM_HCU_BF16_FP32_MMAC", mmac)
    assert bf16_fp32_gemm._mmac_enabled() is expected


def test_cuda_graph_replay():
    x = torch.randn(48, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    bf16_fp32_mmac.linear_bf16_fp32_mmac(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bf16_fp32_mmac.linear_bf16_fp32_mmac(x, w)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.float() @ w.float().T, rtol=1e-4, atol=1e-3)
