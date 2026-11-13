# SPDX-License-Identifier: Apache-2.0
"""W8A16 dequant-cache keying, switch gating and lm_head memory headroom.

Lives under tests/runtime_patch (not tests/ root, where the earlier
test_review_round*.py files sit) because CONTRACT_TESTS in
tools/run_patch_tests.py only selects tests/patch, tests/runtime_patch,
tests/accuracy and tests/gemma4_test -- a file at the tests/ root is never
run by the contract job.

Covers four latent correctness issues:

1. `_DEQUANT_CACHE` keyed on data_ptr + shape/dtype only, so a weight
   rewritten in place returned the previous model's dequantized tensor.
2. The AITER W8A16 MoE fast path ignored the `VLLM_HCU_USE_CUSTOM_OPS`
   master switch.
3. `is_hcu_w8a16_int8_channel` ignored `output_quant`, so a checkpoint with
   an output quantizer was routed to a scheme that cannot honour it.
4. The int8 lm_head copy was allocated lazily with no memory headroom check.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import w8a16_gemm

# The cache key, the LRU bound, the master switch and the headroom check are
# all device-independent, so these run on CPU tensors and stay in the no-GPU
# contract job.  The kernels themselves are covered in tests/accuracy.


def _reset_cache(monkeypatch=None):
    w8a16_gemm.clear_dequant_cache()
    if monkeypatch is not None:
        # The cache refuses to populate during graph capture; asking needs a
        # live device, which this job does not have.
        monkeypatch.setattr(
            w8a16_gemm.torch.cuda, "is_current_stream_capturing",
            lambda: False, raising=False)


def test_dequant_cache_reuses_a_hit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: calls.append(1) or torch.zeros(
            w.shape[0], w.shape[1], dtype=d, device=w.device))
    _reset_cache(monkeypatch)

    w = torch.zeros(8, 16, dtype=torch.int8)
    s = torch.ones(8, dtype=torch.float16)
    a = w8a16_gemm._cached_dequant(w, s, torch.float16, 0)
    b = w8a16_gemm._cached_dequant(w, s, torch.float16, 0)

    assert a is b
    assert len(calls) == 1


def test_dequant_cache_misses_after_in_place_weight_rewrite(monkeypatch):
    """A weight rewritten under the same storage must not hit the old entry.

    This is what a reload/adapter swap does; before the fix the cache key was
    data_ptr + shape + dtype, which is unchanged by an in-place copy, so the
    old dequantized tensor came back and the model produced stale logits.
    """
    calls = []
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: calls.append(1) or torch.zeros(
            w.shape[0], w.shape[1], dtype=d, device=w.device))
    _reset_cache(monkeypatch)

    w = torch.zeros(8, 16, dtype=torch.int8)
    s = torch.ones(8, dtype=torch.float16)
    first = w8a16_gemm._cached_dequant(w, s, torch.float16, 0)

    with torch.no_grad():
        w.copy_(torch.ones_like(w))
    second = w8a16_gemm._cached_dequant(w, s, torch.float16, 0)

    assert second is not first
    assert len(calls) == 2


def test_dequant_cache_misses_after_scale_replacement(monkeypatch):
    calls = []
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: calls.append(1) or torch.zeros(
            w.shape[0], w.shape[1], dtype=d, device=w.device))
    _reset_cache(monkeypatch)

    w = torch.zeros(8, 16, dtype=torch.int8)
    s1 = torch.ones(8, dtype=torch.float16)
    first = w8a16_gemm._cached_dequant(w, s1, torch.float16, 0)
    s2 = torch.full((8,), 2.0, dtype=torch.float16)
    second = w8a16_gemm._cached_dequant(w, s2, torch.float16, 0)

    assert second is not first
    assert len(calls) == 2


def test_dequant_cache_bound_is_above_any_real_working_set():
    """The bound is a runaway guard, not a working-set cap.

    A forward pass touches weights cyclically in layer order; if the bound
    fell below the live set, LRU would miss on every call and silently turn
    the cache into a per-call re-dequantize.
    """
    assert w8a16_gemm._MAX_CACHED_WEIGHTS >= 4096


def test_dequant_cache_warns_once_when_it_evicts(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: torch.zeros(w.shape[0], w.shape[1], dtype=d))
    monkeypatch.setattr(
        w8a16_gemm.logger, "warning",
        lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(w8a16_gemm, "_WARNED_EVICTION", [False])
    _reset_cache(monkeypatch)
    monkeypatch.setattr(w8a16_gemm, "_MAX_CACHED_WEIGHTS", 1)

    keep = []
    for _ in range(3):
        w = torch.zeros(8, 16, dtype=torch.int8)
        s = torch.ones(8, dtype=torch.float16)
        w8a16_gemm._cached_dequant(w, s, torch.float16, 0)
        keep.append((w, s))

    assert len(warnings) == 1


def test_dequant_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: torch.zeros(
            w.shape[0], w.shape[1], dtype=d, device=w.device))
    _reset_cache(monkeypatch)
    monkeypatch.setattr(w8a16_gemm, "_MAX_CACHED_WEIGHTS", 2)

    keep = []
    for _ in range(4):
        w = torch.zeros(8, 16, dtype=torch.int8)
        s = torch.ones(8, dtype=torch.float16)
        w8a16_gemm._cached_dequant(w, s, torch.float16, 0)
        keep.append((w, s))  # keep allocations distinct

    assert len(w8a16_gemm._DEQUANT_CACHE) == 2
    assert len(w8a16_gemm._DEQUANT_CACHE_SRC) == 2


def test_dequant_cache_respects_the_feature_switch(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    _reset_cache(monkeypatch)
    monkeypatch.setattr(
        henvs, "optional_custom_op_enabled", lambda _flag=True: False)
    w = torch.zeros(8, 16, dtype=torch.int8)
    s = torch.ones(8, dtype=torch.float16)
    assert w8a16_gemm._cached_dequant(w, s, torch.float16, 0) is None
    assert not w8a16_gemm._DEQUANT_CACHE


def test_aiter_moe_fast_path_requires_the_master_switch(monkeypatch):
    from vllm_hcu.model_executor.layers.quantization import (
        modelopt_w8a16_hcu as moe)

    monkeypatch.setattr(moe, "_HAS_AITER", True)
    monkeypatch.setattr(moe.envs, "VLLM_ROCM_USE_AITER_MOE", True)

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert not moe._aiter_moe_enabled()

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert moe._aiter_moe_enabled()

    monkeypatch.setattr(moe.envs, "VLLM_ROCM_USE_AITER_MOE", False)
    assert not moe._aiter_moe_enabled()


def test_aiter_moe_fast_path_is_off_without_aiter(monkeypatch):
    from vllm_hcu.model_executor.layers.quantization import (
        modelopt_w8a16_hcu as moe)

    monkeypatch.setattr(moe, "_HAS_AITER", False)
    monkeypatch.setattr(moe.envs, "VLLM_ROCM_USE_AITER_MOE", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert not moe._aiter_moe_enabled()


@pytest.mark.parametrize("output_quant", [None, object()])
def test_scheme_match_requires_absent_output_quant(output_quant):
    ct = pytest.importorskip("compressed_tensors")
    from compressed_tensors.config import CompressionFormat
    from compressed_tensors.quantization import QuantizationStrategy

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w8a16_hcu import (
        is_hcu_w8a16_int8_channel,
    )

    weight_quant = SimpleNamespace(
        num_bits=8,
        dynamic=False,
        strategy=QuantizationStrategy.CHANNEL.value,
        symmetric=True,
    )
    matched = is_hcu_w8a16_int8_channel(
        weight_quant, None, CompressionFormat.pack_quantized.value,
        output_quant)
    assert matched is (output_quant is None)


def test_scheme_match_accepts_the_legacy_three_argument_form():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.config import CompressionFormat
    from compressed_tensors.quantization import QuantizationStrategy

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w8a16_hcu import (
        is_hcu_w8a16_int8_channel,
    )

    weight_quant = SimpleNamespace(
        num_bits=8,
        dynamic=False,
        strategy=QuantizationStrategy.CHANNEL.value,
        symmetric=True,
    )
    assert is_hcu_w8a16_int8_channel(
        weight_quant, None, CompressionFormat.pack_quantized.value)


def test_int8_lm_head_skips_when_memory_is_tight(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_int8_lm_head as lmhead

    monkeypatch.setattr(
        lmhead.torch.cuda, "mem_get_info", lambda *a, **k: (1 << 20, 1 << 30),
        raising=False)
    assert not lmhead._has_headroom(512 << 20)


def test_int8_lm_head_builds_with_headroom(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_int8_lm_head as lmhead

    monkeypatch.setattr(
        lmhead.torch.cuda, "mem_get_info", lambda *a, **k: (8 << 30, 16 << 30),
        raising=False)
    assert lmhead._has_headroom(256 << 20)


def test_int8_lm_head_treats_an_unknown_free_memory_as_enough(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_int8_lm_head as lmhead

    def _raise(*args, **kwargs):
        raise RuntimeError("no device")

    monkeypatch.setattr(lmhead.torch.cuda, "mem_get_info", _raise,
                        raising=False)
    assert lmhead._has_headroom(1 << 30)


@pytest.mark.parametrize("entry", ["reload_weights", "finish_weight_update"])
def test_weight_updates_clear_the_dequant_cache(monkeypatch, entry):
    """Both weight-update entry points must drop cached W16 copies.

    A weight update lands new weights on fresh storage, so the old entries can
    never be hit again -- without this they stay resident (int8 + W16, ~2.6 GB
    on a 35B model) for the life of the process.  `finish_weight_update` is a
    second entry point alongside `reload_weights`; the sibling refresh hooks
    are wired into both, and so is this.
    """
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_hcu.v1.worker import HcuGPUWorker

    _reset_cache(monkeypatch)
    monkeypatch.setattr(
        w8a16_gemm, "_w8a16_dequant",
        lambda w, s, d, layout: torch.zeros(
            w.shape[0], w.shape[1], dtype=d))
    w = torch.zeros(8, 16, dtype=torch.int8)
    s = torch.ones(8, dtype=torch.float16)
    w8a16_gemm._cached_dequant(w, s, torch.float16, 0)
    assert w8a16_gemm._DEQUANT_CACHE

    # Neutralise the sibling refresh hooks; only the cache clear is under test.
    monkeypatch.setattr(Worker, entry, lambda self, *a, **k: None)
    for module, name in (
        ("vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_attention",
         "refresh_compressor_nt_weights"),
        ("vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_rocm_wo_a_layout",
         "refresh_wo_a_bf16_caches"),
        ("vllm_hcu.model_executor.layers.fused_moe.glm53_router_gemv",
         "refresh_glm53_router_gates"),
        ("vllm_hcu.model_executor.layers.kimi_k3_mixed_precision",
         "refresh_kimi_kda_fused_qkv"),
        ("vllm_hcu.model_executor.layers.quantization."
         "compressed_tensors_w4a8_runtime", "refresh_hcu_w4a8_scale_caches"),
        ("vllm_hcu.model_executor.layers.quantization."
         "grouped_w4a16_moe_runtime", "refresh_grouped_w4a16_scale_caches"),
        ("vllm_hcu.model_executor.layers.quantization."
         "grouped_w4a16_moe_runtime", "snapshot_grouped_w4a16_scale_owners"),
    ):
        monkeypatch.setattr(
            importlib.import_module(module), name, lambda *a, **k: None)

    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(model=torch.nn.Module())
    worker._weight_update_is_draft = False
    worker._grouped_w4a16_scale_owners = None
    getattr(worker, entry)()

    assert not w8a16_gemm._DEQUANT_CACHE
    assert not w8a16_gemm._DEQUANT_CACHE_SRC
