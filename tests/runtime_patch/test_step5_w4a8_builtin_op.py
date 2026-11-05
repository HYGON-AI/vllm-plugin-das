"""Meta/Fake dispatch coverage for the optional normal hcu_ops build; no GPU kernels."""

import importlib
import pytest
import torch


@pytest.fixture(scope="module")
def native_op():
    try:
        importlib.import_module("vllm_hcu.hcu_ops")
        op = torch.ops.hcu_ops.step5_w4a8_gemm.default
    except (ImportError, AttributeError):
        pytest.skip("Build the plugin with VLLM_HCU_BUILD_STEP5_W4A8=1")
    return op


def tensors(m=16, k=4096, n=3072):
    return (
        torch.empty(m, k, dtype=torch.int8, device="meta"),
        torch.empty(2, n, k // 2, dtype=torch.int8, device="meta"),
        torch.empty(m, n, dtype=torch.bfloat16, device="meta"),
        torch.empty(m, 1, dtype=torch.float32, device="meta"),
        torch.empty(2, n, 1, dtype=torch.float32, device="meta"),
        torch.empty(m, dtype=torch.int32, device="meta"),
    )


@pytest.mark.parametrize("m,k,n", [(0, 4096, 3072), (16, 4096, 3072), (64, 1536, 4096)])
def test_meta_preserves_caller_owned_output(native_op, m, k, n):
    args = tensors(m, k, n)
    pointer = args[2]
    assert native_op(*args) is None
    assert args[2] is pointer and args[2].shape == (m, n)


def test_operator_schema_tracks_mutation(native_op):
    schema = str(native_op._schema)
    assert "Tensor(a!) output" in schema and "-> ()" in schema
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "hcu_ops::step5_w4a8_gemm", "CUDA"
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "hcu_ops::step5_w4a8_gemm", "Meta"
    )


@pytest.mark.parametrize("kind", ["scale_dtype", "ids_dtype"])
def test_invalid_metadata_is_rejected(native_op, kind):
    args = list(tensors())
    if kind == "scale_dtype":
        args[3] = args[3].to(torch.float16)
    elif kind == "ids_dtype":
        args[5] = args[5].to(torch.int64)
    with pytest.raises(RuntimeError):
        native_op(*args)


def test_fake_cuda_dispatch_does_not_run_a_gpu_kernel(native_op):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    meta = tensors()
    with FakeTensorMode() as mode:
        values = tuple(FakeTensor(mode, x, torch.device("cuda:0")) for x in meta)
        assert native_op(*values) is None
        assert values[2].shape == (16, 3072)


def test_torch_compile_captures_mutating_operator(native_op):
    def forward(x, w, out, xs, ws, ids):
        native_op(x, w, out, xs, ws, ids)
        return out

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    assert compiled(*tensors()).shape == (16, 3072)


def test_meta_accepts_unbacked_symbolic_batch(native_op):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        m = mode.shape_env.create_unbacked_symint()
        args = (
            torch.empty((m, 4096), dtype=torch.int8, device="meta"),
            torch.empty((2, 3072, 2048), dtype=torch.int8, device="meta"),
            torch.empty((m, 3072), dtype=torch.bfloat16, device="meta"),
            torch.empty((m, 1), dtype=torch.float32, device="meta"),
            torch.empty((2, 3072, 1), dtype=torch.float32, device="meta"),
            torch.empty((m,), dtype=torch.int32, device="meta"),
        )
        assert native_op(*args) is None
