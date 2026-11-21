# SPDX-License-Identifier: Apache-2.0
"""Actual TP1 HIP projections, default Inductor and changed-input Graph replay.

No group, GEMM, loader, compiler backend or transport implementation is replaced.
TP1 intentionally does not exercise multi-rank collectives or model activation.
"""
from datetime import timedelta
import os

import pytest


pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        os.environ.get("VLLM033_HCU_TEST") != "1",
        reason="requires the installed v0.33 HCU device tier",
    ),
]


@pytest.fixture(scope="module")
def native_tp1(tmp_path_factory):
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    assert torch.version.hip is not None
    assert current_platform.__class__.__module__ == "hcu_adapter.bridges.vllm"
    assert not torch.distributed.is_initialized(), "do not replace an existing group"
    torch.cuda.set_device(0)
    rendezvous = tmp_path_factory.mktemp("runtime-sp-tp1") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0,
                distributed_init_method=rendezvous.as_uri(),
                backend="nccl", timeout=timedelta(seconds=120),
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
            group = get_tp_group()
            assert group.world_size == 1 and group.rank_in_group == 0
            yield group
        finally:
            torch._dynamo.reset()
            destroy_model_parallel()
            destroy_distributed_environment()
    assert not torch.distributed.is_initialized()


def _projection(kind, dtype):
    import torch
    from hcu_vllm_das.parallel import runtime_sp_attention as sp

    with torch.device("cuda"), sp.runtime_sp_model_construction():
        common = dict(bias=False, params_dtype=dtype)
        if kind == "qkv":
            return sp.RuntimeSPQKVParallelLinear(
                256, 64, 4, 2, prefix="model.layers.0.self_attn.qkv_proj", **common)
        if kind == "row":
            return sp.RuntimeSPRowParallelLinear(
                256, 256, prefix="model.layers.0.self_attn.o_proj", **common)
        shared = "shared_experts." if kind.startswith("shared") else ""
        prefix = "model.layers.0.mlp." + shared
        if kind.endswith("down"):
            return sp.RuntimeSPRowParallelLinear(
                512, 256, prefix=prefix + "down_proj", **common)
        return sp.RuntimeSPMergedColumnParallelLinear(
            256, [512, 512], prefix=prefix + "gate_up_proj", **common)


def _values(shape, offset=0):
    import torch

    # Dyadic products and FP32-exact sums make the final dtype cast an exact
    # oracle, independent of reduction order. Padding NaNs must be cleared.
    count = 1
    for extent in shape:
        count *= extent
    return ((torch.arange(count).reshape(shape) + offset) % 11 - 5).float() / 16


@pytest.mark.parametrize("kind", (
    "qkv", "row", "dense_up", "dense_down", "shared_up", "shared_down",
))
@pytest.mark.parametrize("dtype_name", ("bfloat16", "float16"))
@pytest.mark.parametrize("masked", (False, True))
def test_official_projection_default_inductor_and_hipgraph(
    native_tp1, record_property, kind, dtype_name, masked,
):
    import torch
    import torch.nn.functional as F
    from hcu_vllm_das.parallel.runtime_sp_batch import RuntimeSPBatch, _BATCH_KEY
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    dtype = getattr(torch, dtype_name)
    layer = _projection(kind, dtype)
    assert type(layer.quant_method) is UnquantizedLinearMethod
    assert layer.quant_method._gemm_impl is dispatch_unquantized_gemm()
    assert layer._runtime_sp_group is native_tp1
    assert layer.weight.is_cuda and layer.weight.dtype == dtype
    weight = _values(tuple(layer.weight.shape), offset=3).to(dtype)
    # Exercise Core's real loader and post-load processing, without a local
    # replacement weight/quant method. There is no native/vendor source patch.
    layer.weight.weight_loader(layer.weight, weight.to("cuda"))
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight.cpu(), weight, rtol=0, atol=0)
    layer.requires_grad_(False)
    record_property("architecture", torch.cuda.get_device_properties(0).gcnArchName)
    record_property("gemm", layer.quant_method._gemm_impl.__qualname__)
    record_property("group_size", native_tp1.world_size)
    record_property("multirank_qualified", False)

    rows, width = 17, layer.weight.shape[1]
    value = torch.empty((rows, width), device="cuda", dtype=dtype)
    padding = torch.zeros(rows, device="cuda", dtype=torch.bool)
    batch = RuntimeSPBatch.create(1, 0, rows, padding, value.device)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={},
        is_padding=batch.local_padding,
        additional_kwargs={_BATCH_KEY: batch} if masked else {},
    )
    compiled = torch.compile(layer, fullgraph=True)

    def change_inputs(epoch):
        host = _values((rows, width), offset=epoch * 7).to(dtype)
        mask = torch.zeros(rows, dtype=torch.bool)
        if masked:
            mask[epoch % 3::3] = True
        batch.global_padding.copy_(mask)
        host.masked_fill_(mask[:, None], float("nan"))
        value.copy_(host)
        # The oracle never calls the candidate eager path or GPU GEMM.
        host.masked_fill_(mask[:, None], 0)
        return F.linear(host.float(), weight.float()).to(dtype)

    def check(actual, expected):
        output, bias = actual
        assert output.is_cuda and bias is None
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    try:
        with torch.inference_mode(), override_forward_context(context):
            for epoch in (0, 1):
                expected = change_inputs(epoch)
                check(compiled(value), expected)
            # Warm the actual compiled CUDA path on a side stream before
            # capture. Compilation is complete; no eager/alternate backend.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compiled(value)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            addresses = (value.data_ptr(), batch.global_padding.data_ptr(),
                         batch.local_padding.data_ptr(), layer.weight.data_ptr())
            with torch.cuda.graph(graph):
                output = compiled(value)
            output_address = output[0].data_ptr()
            for epoch in (2, 3, 4):
                expected = change_inputs(epoch)
                graph.replay()
                check(output, expected)
                assert output[0].data_ptr() == output_address
                assert addresses == (
                    value.data_ptr(), batch.global_padding.data_ptr(),
                    batch.local_padding.data_ptr(), layer.weight.data_ptr(),
                )
    finally:
        torch._dynamo.reset()
