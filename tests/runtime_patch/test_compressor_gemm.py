# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tests.fixtures.blaslt_faults import (
    BLASLT_FAILURE_IDS,
    BLASLT_FAILURE_SITES,
    BLASLT_FAILURE_STATUSES,
    failing_blaslt_loader,
)


if not torch.cuda.is_available():
    pytest.skip("compressor GEMM tests need a working GPU", allow_module_level=True)

pytestmark = pytest.mark.hcu


from vllm_hcu.patch.worker.core_fix import (  # noqa: E402
    patch_deepseek_v4_attention as patch,
)
from vllm_hcu.utils import bf16_fp32_gemm  # noqa: E402


class CompressorLinear(torch.nn.Module):
    def __init__(self, weight, input_size):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.input_size_per_partition = input_size


class CompressorModel(torch.nn.Module):
    def __init__(self, weight, input_size):
        super().__init__()
        self.fused_wkv_wgate = CompressorLinear(weight, input_size)


@pytest.fixture
def blas(monkeypatch):
    calls = []

    def linear(x, w):
        assert w.is_contiguous()
        calls.append(w)
        return x.float() @ w.float().T

    monkeypatch.setattr(bf16_fp32_gemm, "linear_bf16_fp32", linear)
    return calls


def _model(weight, input_size=5):
    return CompressorModel(weight.clone(), input_size).cuda()


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("backend", ["torch", "blaslt"])
def test_outputs_and_dispatch(monkeypatch, blas, layout, backend):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", backend)
    x = torch.randn(7, 5, dtype=torch.bfloat16, device="cuda")
    nn = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    w = nn if layout == "nn" else nn.T.contiguous()
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == (
        layout == "nn" and backend == "blaslt"
    )
    result = patch._compressor_mm(x, model.fused_wkv_wgate)
    torch.testing.assert_close(result, x.float() @ nn.float(), rtol=1e-2, atol=1e-2)
    assert result.dtype == torch.float32
    assert len(blas) == (layout == "nn" and backend == "blaslt")
    if blas:
        assert blas[0].shape == (3, 5)


def test_master_switch_disables_blaslt(monkeypatch, blas):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    x = torch.randn(7, 5, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == 0
    result = patch._compressor_mm(x, model.fused_wkv_wgate)

    torch.testing.assert_close(result, torch.mm(x, w, out_dtype=torch.float32))
    assert not blas


@pytest.mark.parametrize("case", ["fp32", "square"])
def test_ineligible_weights_keep_torch(monkeypatch, blas, case):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    k = 5
    w = torch.randn(k, k if case == "square" else 3, dtype=dtype, device="cuda")
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == 0
    x = torch.randn(2, k, dtype=dtype, device="cuda")
    torch.testing.assert_close(
        patch._compressor_mm(x, model.fused_wkv_wgate),
        torch.mm(x, w, out_dtype=torch.float32),
    )
    assert not blas


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "invalid")
    model = _model(torch.randn(5, 3, dtype=torch.bfloat16, device="cuda"))

    with pytest.raises(ValueError, match="must be 'blaslt' or 'torch'"):
        patch.bind_compressor_nt_weights(model)


def test_registered_gemm_has_fake_impl():
    x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
    nt = torch.randn(3, 5, dtype=torch.bfloat16, device="cuda").contiguous()

    torch.library.opcheck(
        torch.ops.vllm.hcu_linear_bf16_fp32.default,
        (x, nt),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_compile_uses_materialized_buffer(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    model = _model(torch.randn(5, 3, dtype=torch.bfloat16, device="cuda"))
    assert patch.bind_compressor_nt_weights(model) == 1

    compiled = torch.compile(
        lambda x: patch._compressor_mm(x, model.fused_wkv_wgate),
        fullgraph=True,
        backend="eager",
    )
    x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        compiled(x), x.float() @ model.fused_wkv_wgate.weight.float()
    )


def test_refresh_reuses_buffer_after_hot_reload(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    weight = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    model = _model(weight)
    assert patch.bind_compressor_nt_weights(model) == 1
    linear = model.fused_wkv_wgate
    nt = linear._vllm_hcu_compressor_nt_weight
    pointer = nt.data_ptr()
    version = linear.weight._version

    new_weight = torch.randn_like(linear.weight)
    linear.weight.data.copy_(new_weight)
    assert linear.weight._version == version
    patch.refresh_compressor_nt_weights(model)
    assert nt.data_ptr() == pointer
    torch.testing.assert_close(nt, new_weight.T)

    replacement = torch.nn.Parameter(torch.randn_like(new_weight), requires_grad=False)
    linear.weight = replacement
    patch.refresh_compressor_nt_weights(model)
    assert nt.data_ptr() == pointer
    torch.testing.assert_close(nt, replacement.T)


def test_prefetch_offloaded_weights_use_live_slots(monkeypatch):
    from vllm.model_executor.offloader import (
        PrefetchOffloader,
        get_offloader,
        set_offloader,
    )

    class Layer(CompressorModel):
        def forward(self, x):
            return patch._compressor_mm(x, self.fused_wkv_wgate)

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    previous_offloader = get_offloader()
    offloader = PrefetchOffloader(
        group_size=1,
        num_in_group=1,
        prefetch_step=1,
        offload_params={"fused_wkv_wgate"},
    )
    set_offloader(offloader)
    try:
        model = torch.nn.ModuleList(
            [
                Layer(
                    torch.full(
                        (5, 3), value, dtype=torch.bfloat16, device="cuda"
                    ),
                    5,
                )
                for value in (1, 2, 3)
            ]
        )
        assert offloader.wrap_modules(iter(model)) == list(model)
        offloader.post_init()

        assert patch.bind_compressor_nt_weights(model) == 0
        assert all(
            not hasattr(layer.fused_wkv_wgate, patch._COMPRESSOR_NT_BUFFER)
            for layer in model
        )

        x = torch.ones((1, 5), dtype=torch.bfloat16, device="cuda")
        results = [layer(x) for layer in model]
        torch.cuda.synchronize()
        expected = [
            torch.full((1, 3), value * 5, dtype=torch.float32, device="cuda")
            for value in (1, 2, 3)
        ]
        for result, expected_result in zip(results, expected):
            torch.testing.assert_close(result, expected_result)
    finally:
        torch.cuda.synchronize()
        set_offloader(previous_offloader)


def test_uva_offloaded_weight_is_not_materialized(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    model = _model(torch.randn(5, 3, dtype=torch.bfloat16, device="cuda"))
    model.fused_wkv_wgate.weight._vllm_is_uva_offloaded = True

    assert patch.bind_compressor_nt_weights(model) == 0
    assert not hasattr(model.fused_wkv_wgate, patch._COMPRESSOR_NT_BUFFER)


def test_blaslt_failure_falls_back_to_torch_mm(monkeypatch):
    monkeypatch.setattr(bf16_fp32_gemm, "_HIPBLASLT_DISABLED", False)

    def fail(*args):
        raise RuntimeError("hipBLASLt matmul failed")

    monkeypatch.setattr(
        bf16_fp32_gemm,
        "_jit_hipblaslt_bf16_fp32",
        lambda: SimpleNamespace(linear_bf16_fp32=fail),
    )
    x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(3, 5, dtype=torch.bfloat16, device="cuda")

    result = bf16_fp32_gemm.linear_bf16_fp32(x, weight)

    torch.testing.assert_close(result, torch.mm(x, weight.T, out_dtype=torch.float32))
    assert bf16_fp32_gemm._HIPBLASLT_DISABLED


@pytest.mark.skipif(
    torch.version.hip is None, reason="hipBLASLt extension requires ROCm"
)
def test_blaslt_invalid_k_raises_from_extension():
    extension = bf16_fp32_gemm._jit_hipblaslt_bf16_fp32()
    x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(3, 4, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="incompatible K dimensions"):
        extension.linear_bf16_fp32(x, weight)


@pytest.mark.skipif(torch.version.hip is None, reason="requires hipBLASLt")
@pytest.mark.parametrize(
    "api,occurrence,expected_calls", BLASLT_FAILURE_SITES, ids=BLASLT_FAILURE_IDS,
)
@pytest.mark.parametrize("status", BLASLT_FAILURE_STATUSES)
def test_blaslt_status_failure_raises_and_falls_back(
    monkeypatch, api, occurrence, expected_calls, status,
):
    import torch.utils.cpp_extension as cpp_extension

    # Compile production error handling with only the selected vendor call
    # replaced by a non-throwing failure, including partially created layouts.
    monkeypatch.setattr(
        cpp_extension, "load_inline",
        failing_blaslt_loader(
            cpp_extension.load_inline, api, status, occurrence, expected_calls
        ),
    )
    monkeypatch.setenv("VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT", "1")
    monkeypatch.setattr(bf16_fp32_gemm, "_HIPBLASLT_DISABLED", False)
    jit = bf16_fp32_gemm._jit_hipblaslt_bf16_fp32
    jit.cache_clear()
    try:
        extension = bf16_fp32_gemm._jit_hipblaslt_bf16_fp32()
        x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(3, 5, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="failed with hipBLASLt status"):
            extension.linear_bf16_fp32(x, weight)
        result = bf16_fp32_gemm.linear_bf16_fp32(x, weight)
        torch.testing.assert_close(result, torch.mm(x, weight.T, out_dtype=torch.float32))
        assert bf16_fp32_gemm._HIPBLASLT_DISABLED
        # A failed backend must remain bypassed on subsequent calls.
        monkeypatch.setattr(
            bf16_fp32_gemm, "_jit_hipblaslt_bf16_fp32",
            lambda: pytest.fail("disabled backend was retried"),
        )
        torch.testing.assert_close(
            bf16_fp32_gemm.linear_bf16_fp32(x, weight), result
        )
    finally:
        # Keep injected extensions out of later tests, even after failures.
        jit.cache_clear()


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("with_draft", [False, True])
def test_compressor_nt_pool_sleep_wake(monkeypatch, level, with_draft):
    from vllm.device_allocator.cumem import CuMemAllocator, cumem_available
    from vllm.v1.worker.gpu_worker import Worker
    from tests.runtime_patch.test_compressor_weights_pool_contract import load_method

    if not cumem_available:
        pytest.skip("CuMem sleep allocator is unavailable")
    allocator = CuMemAllocator.get_instance()
    if allocator.get_current_usage() != 0:
        pytest.skip("sleep regression requires an unused CuMem allocator")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    monkeypatch.setattr(
        bf16_fp32_gemm, "linear_bf16_fp32",
        lambda x, w: torch.mm(x, w.T, out_dtype=torch.float32),
    )

    class RunnerBase:
        def load_model(self, load_dummy_weights=False):
            self.model = _model(
                torch.full((2048, 1024), 1, dtype=torch.bfloat16, device="cuda"),
                input_size=2048,
            )
            self.draft = (
                _model(
                    torch.full((2048, 1024), 2, dtype=torch.bfloat16, device="cuda"),
                    input_size=2048,
                ) if with_draft else None
            )

        def get_draft_model(self):
            return self.draft

    runner_cls = load_method(
        "vllm_hcu/v1/hcu_model_runner_v2.py", "HcuGPUModelRunnerV2", RunnerBase, {},
    )
    runner = runner_cls()
    config = SimpleNamespace(model_config=SimpleNamespace(
        enable_nccl_comm_suspend=False,
    ))
    worker = SimpleNamespace(
        model_runner=runner, vllm_config=config,
        get_draft_model=runner.get_draft_model,
        _sleep_saved_buffers={}, _sleep_saved_draft_buffers={},
        synchronize_device=torch.cuda.synchronize,
        sleep_mode_backend=SimpleNamespace(
            suspend=lambda level: allocator.sleep(
                offload_tags=("weights",) if level == 1 else ()
            ),
            resume=allocator.wake_up,
        ),
    )
    try:
        # The same initial context used by upstream Worker.load_model.
        with allocator.use_memory_pool(tag="weights"):
            runner.load_model()
        models = [model for model in (runner.model, runner.draft) if model is not None]
        caches = [model.fused_wkv_wgate._vllm_hcu_compressor_nt_weight for model in models]
        saved = [cache.cpu().clone() for cache in caches]
        pointers = [cache.data_ptr() for cache in caches]
        for pointer in pointers:
            assert any(
                data.tag == "weights" and ptr <= pointer < ptr + data.handle[1]
                for ptr, data in allocator.pointer_to_data.items()
            )
        torch.cuda.synchronize()
        before = torch.cuda.mem_get_info()[0]
        Worker.sleep(worker, level=level)
        after = torch.cuda.mem_get_info()[0]
        assert after - before >= sum(cache.numel() * cache.element_size() for cache in caches)
        Worker.wake_up(worker, tags=["weights"])
        for model, cache, expected, pointer in zip(models, caches, saved, pointers):
            assert cache.data_ptr() == pointer
            torch.testing.assert_close(cache.cpu(), expected)
            # Level 2 deliberately discards parameters; reload them before use.
            model.fused_wkv_wgate.weight.data.copy_(expected.T)
            x = torch.ones(1, 2048, dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(
                patch._compressor_mm(x, model.fused_wkv_wgate),
                torch.mm(x, expected.T.to("cuda"), out_dtype=torch.float32),
            )
    finally:
        # Never leave global allocator mappings asleep if an assertion fails.
        allocator.wake_up()
