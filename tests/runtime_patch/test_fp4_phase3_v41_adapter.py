# SPDX-License-Identifier: Apache-2.0
"""Strict CPU/mock checks for the real V4.1 worker adapter."""

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from vllm_hcu.model_executor.layers.fp4_indexer import (
    Fp4QTuple,
    fp4_chunk_slots,
    fp4_cache_view,
)
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_indexer_k_layout as patch


def _v41_config(ratios=(1, 0, 1)):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                compress_ratios=list(ratios),
                kv_source_layer_ids=[0, 2],
                index_source_layer_ids=[0, 2],
            )
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(method=None),
    )


def _target_module(monkeypatch, *, fp4=False):
    target = ModuleType(patch.TARGET_MODULE)
    ops = ModuleType("vllm.models.deepseek_v41.common.ops")
    store = ModuleType("vllm.models.deepseek_v41.common.ops.indexer_k_store")
    calls = []

    def original_store(*args, **kwargs):
        calls.append(("store-off", args, kwargs))

    def original_q(*args, **kwargs):
        calls.append(("q", args, kwargs))
        values = torch.zeros(2, 3, 64, dtype=torch.uint8)
        scales = torch.zeros(2, 3, 4, dtype=torch.int32)
        return (values, scales), torch.ones(2, 3)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(("store-on", args, kwargs))
            return launch

    store._indexer_k_norm_rope_quant_store_kernel = FakeKernel()
    class FakeIndexer:
        def forward(self, qr, latent, indexer_weights, positions, rotary_emb, qr_scale=None):
            return "original-forward"

    target.DeepseekV4Indexer = FakeIndexer
    target.indexer_k_norm_rope_store = original_store
    target.fused_indexer_q_rope_quant = original_q
    target.get_current_vllm_config = lambda: _v41_config()
    ops.fused_indexer_q_rope_quant = original_q
    ops.indexer_k_norm_rope_store = original_store
    monkeypatch.setitem(sys.modules, patch.TARGET_MODULE, target)
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v41.common.ops", ops)
    monkeypatch.setitem(
        sys.modules, "vllm.models.deepseek_v41.common.ops.indexer_k_store", store
    )
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FP4_INDEXER", "1" if fp4 else "0")
    return target, ops, store, calls


def test_real_v41_callback_off_delegates_store_and_q(monkeypatch):
    target, ops, _, calls = _target_module(monkeypatch, fp4=False)
    assert patch.apply_to_module(target) is True
    k = torch.zeros(2, 128, dtype=torch.bfloat16)
    pos = torch.arange(2, dtype=torch.int64)
    cache = torch.zeros(1, 64, 132, dtype=torch.uint8)
    slots = torch.arange(2, dtype=torch.int64)
    target.indexer_k_norm_rope_store(k, pos, torch.zeros(4, 64), torch.ones(128),
                                     1e-6, cache, slots, 1, False)
    assert calls[0][0] == "store-off"
    assert patch.apply_to_module(target) is False
    assert ops.fused_indexer_q_rope_quant(torch.zeros(2, 3, 128), pos,
                                          torch.zeros(4, 64), torch.ones(2, 3),
                                          1.0, 1.0, use_fp4=False)[0][0].shape[-1] == 64


def test_real_v41_callback_on_uses_fp4_store_and_tuple_q(monkeypatch):
    target, ops, _, calls = _target_module(monkeypatch, fp4=True)
    assert patch.apply_to_module(target) is True
    k = torch.zeros(2, 128, dtype=torch.bfloat16)
    pos = torch.arange(2, dtype=torch.int64)
    cache = torch.zeros(1, 64, 68, dtype=torch.uint8)
    slots = torch.arange(2, dtype=torch.int64)
    target.indexer_k_norm_rope_store(k, pos, torch.zeros(4, 64), torch.ones(128),
                                     1e-6, cache, slots, 1, True)
    assert calls[-1][0] == "store-on"
    assert calls[-1][2]["TOKEN_STRIDE"] == 64
    assert calls[-1][2]["SCALE_DIM"] == 4
    packed, weights = ops.fused_indexer_q_rope_quant(
        pos, k.reshape(2, 1, 128), torch.zeros(4, 64), torch.ones(2, 1),
        1.0, 1.0, use_fp4=True
    )
    assert isinstance(packed, Fp4QTuple)
    assert packed.packed.shape[-1] == 64
    assert packed.scales.dtype == torch.int32
    assert packed.bf16.dtype == torch.bfloat16
    assert fp4_cache_view(cache).shape == (1, 64, 1, 68)


def test_fp4_domain_admits_ratio2_with_byte_staging_contract(monkeypatch):
    target, _, _, _ = _target_module(monkeypatch, fp4=True)
    target.get_current_vllm_config = lambda: _v41_config((1, 2, 1))
    assert patch.apply_to_module(target) is True
    assert patch._fp4_domain_requested(target) is True


def test_fp4_domain_keeps_unsupported_ratios_closed(monkeypatch):
    target, _, _, _ = _target_module(monkeypatch, fp4=True)
    assert not patch._fp4_domain_requested(target, _v41_config((1, 4, 1)))


def test_allocation_metadata_selector_and_backend_geometry_are_consistent(monkeypatch):
    target, _, _, _ = _target_module(monkeypatch, fp4=True)
    backend = ModuleType("vllm.v1.attention.backends.mla.indexer")
    backend.dsa_indexer_uses_fp4 = lambda cfg: False
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    target.dsa_indexer_uses_fp4 = lambda cfg: False

    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [128]

    target.DeepseekV41IndexerBackend = Backend
    cfg = _v41_config((0, 2, 1))
    cfg.model_config.hf_config.model_type = "deepseek_v41"
    cfg.parallel_config.data_parallel_size = 8
    cfg.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=5,
                                               enable_adaptive_verification=False)
    assert patch.apply_to_module(target)
    assert target.dsa_indexer_uses_fp4(cfg)
    assert backend.dsa_indexer_uses_fp4(cfg)
    assert Backend.get_supported_kernel_block_sizes() == [64]
    cfg.parallel_config.decode_context_parallel_size = 8
    with pytest.raises(RuntimeError, match="outside supported domain"):
        target.dsa_indexer_uses_fp4(cfg)
    monkeypatch.setenv("VLLM_HCU_USE_FP4_INDEXER", "0")
    assert not target.dsa_indexer_uses_fp4(cfg)
    assert not backend.dsa_indexer_uses_fp4(cfg)
    assert Backend.get_supported_kernel_block_sizes() == [128]


def test_multi_request_chunk_maps_local_positions_to_physical_pages():
    block_table = torch.tensor([[5, 6], [9, 10]], dtype=torch.int32)
    token_to_seq = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    # Packed K rows are grouped by request; cumulative row starts are 0 and 2.
    cu_seq_lens = torch.tensor([0, 2], dtype=torch.int32)
    slots = fp4_chunk_slots(block_table, token_to_seq, cu_seq_lens, 4)
    torch.testing.assert_close(
        slots, torch.tensor([5 * 64, 5 * 64 + 1, 9 * 64, 9 * 64 + 1])
    )
