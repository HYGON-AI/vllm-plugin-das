# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-side tests for the Qwen4Exp PLE INT8 CPU-offload lookup path.

These tests run on CPU only (no accelerator required): they verify the
dequantization numerics, N-D id handling, and env-var-driven method
selection.  The device staging in ``_pinned_int8_lookup`` is a no-op when the
ids already live on CPU, so the same code path is exercised.
"""

from types import SimpleNamespace

import torch

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_ple_int8 as patch


def _make_table(rows: int, dim: int):
    torch.manual_seed(0)
    weight = torch.randint(-128, 128, (rows, dim), dtype=torch.int8)
    weight_scale = torch.rand(rows, 1, dtype=torch.bfloat16) + 0.1
    return weight, weight_scale


def _reference(weight, weight_scale, ids, dtype):
    embeddings = torch.nn.functional.embedding(ids, weight)
    scales = torch.nn.functional.embedding(ids, weight_scale)
    return embeddings.to(dtype) * scales.to(dtype)


def test_pinned_lookup_matches_gpu_resident_dequant_1d():
    weight, weight_scale = _make_table(32, 16)
    ids = torch.tensor([0, 5, 31, 5, 0], dtype=torch.long)

    got = patch._pinned_int8_lookup(weight, weight_scale, ids, torch.bfloat16)
    want = _reference(weight, weight_scale, ids, torch.bfloat16)

    assert got.shape == (5, 16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)


def test_pinned_lookup_preserves_ngram_2d_shape():
    # ids arrive as [num_tokens, ngram_heads]; row axis is flattened for the
    # gather and restored on the output as [num_tokens, ngram_heads, dim].
    weight, weight_scale = _make_table(64, 8)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    got = patch._pinned_int8_lookup(weight, weight_scale, ids, torch.bfloat16)
    want = _reference(weight, weight_scale, ids, torch.bfloat16)

    assert got.shape == (2, 3, 8)
    assert torch.equal(got, want)


def test_offload_method_embedding_matches_resident_method():
    weight, weight_scale = _make_table(48, 12)
    ids = torch.tensor([2, 2, 47, 0], dtype=torch.long)

    resident = patch.HcuQwen4ExpPLEInt8EmbeddingMethod()
    offload = patch.HcuQwen4ExpPLEInt8OffloadEmbeddingMethod()
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        params_dtype=torch.bfloat16,
    )

    resident_out = resident.embedding(layer, ids)
    offload_out = offload.embedding(layer, ids)

    assert torch.equal(resident_out, offload_out)


def test_offload_method_is_subclass_for_loader_isinstance():
    # The ngram loader gates INT8 shard routing on isinstance against the
    # resident method; the offload method must remain a subclass so offloaded
    # runs still route scale shards correctly.
    assert issubclass(
        patch.HcuQwen4ExpPLEInt8OffloadEmbeddingMethod,
        patch.HcuQwen4ExpPLEInt8EmbeddingMethod,
    )


def test_offload_toggle_reads_env(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_PLE_CPU_OFFLOAD", raising=False)
    assert patch._should_offload_ple_to_cpu() is False
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")
    assert patch._should_offload_ple_to_cpu() is True
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "0")
    assert patch._should_offload_ple_to_cpu() is False


def test_create_weights_allocates_on_cpu():
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        if ps._TP is None:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29591",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)

        method = patch.HcuQwen4ExpPLEInt8OffloadEmbeddingMethod()
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[10],
            input_size=16,
            output_size=10,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )

    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight.shape) == (10, 16)
    assert layer.weight_scale.device.type == "cpu"
    assert layer.weight_scale.dtype == torch.bfloat16
    assert tuple(layer.weight_scale.shape) == (10, 1)


# --- UVA zero-copy variant ---------------------------------------------------

import pytest

_UVA_AVAILABLE = patch._is_uva_available()
_needs_uva = pytest.mark.skipif(
    not _UVA_AVAILABLE, reason="UVA op get_cuda_view_from_cpu_tensor unavailable"
)
_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires accelerator"
)


def test_uva_method_is_subclass_for_loader_isinstance():
    # Same isinstance gate as the offload variant: the ngram loader routes INT8
    # scale shards by isinstance against the resident method.
    assert issubclass(
        patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod,
        patch.HcuQwen4ExpPLEInt8EmbeddingMethod,
    )


@_needs_uva
@_needs_cuda
def test_uva_method_embedding_matches_resident():
    # UVA lookup must be numerically identical to the GPU-resident dequant.
    weight, weight_scale = _make_table(48, 12)
    w_pin = weight.pin_memory()
    s_pin = weight_scale.pin_memory()
    ids = torch.tensor([2, 2, 47, 0], device="cuda", dtype=torch.long)

    uva = patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=w_pin),
        weight_scale=SimpleNamespace(data=s_pin),
        params_dtype=torch.bfloat16,
        _hcu_uva_views=None,
    )
    got = uva.embedding(layer, ids)
    want = _reference(weight, weight_scale, ids.cpu(), torch.bfloat16)

    assert got.shape == (4, 12)
    assert torch.equal(got.cpu(), want)


@_needs_uva
@_needs_cuda
def test_uva_view_is_cached_across_calls():
    # The device view is created once and reused; a second lookup must not
    # rebuild it (rebuilding every step would defeat the zero-copy path).
    weight, weight_scale = _make_table(32, 8)
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=weight.pin_memory()),
        weight_scale=SimpleNamespace(data=weight_scale.pin_memory()),
        params_dtype=torch.bfloat16,
        _hcu_uva_views=None,
    )
    uva = patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()

    ids = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    uva.embedding(layer, ids)
    first = layer._hcu_uva_views
    uva.embedding(layer, ids)
    second = layer._hcu_uva_views

    assert first is not None
    assert first is second
