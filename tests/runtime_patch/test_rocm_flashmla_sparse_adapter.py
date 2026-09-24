"""CPU contract checks for the opt-in ROCm FlashMLA sparse adapter."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_flashmla_sparse as patch
from vllm_hcu.platforms import envs as henvs


def _build_module(ratio, batch, calls):
    class BaseBuilder:
        def __init__(self):
            calls.append("base_swa_init")

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            calls.append("dense_swa")
            return SimpleNamespace(dense_swa=True)

    class Builder(BaseBuilder):
        _layer_types = {1: ("swaonly",), 4: ("c4a",), 128: ("c128a",)}[ratio]

        def __init__(self):
            super().__init__()
            calls.append("ragged_swa_init")
            self.decode_swa_ragged_indices_buffer = torch.zeros(4, dtype=torch.int32)
            self.decode_swa_ragged_indptr_buffer = torch.zeros(4, dtype=torch.int32)

        def build_tile_scheduler(self, num_decode_tokens):
            return {"swaonly": None, "c4a": None, "c128a": None}

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            calls.append("ragged_swa")
            return SimpleNamespace(dense_swa=True)

    class BaseMLABuilder:
        def __init__(self):
            calls.append("base_mla_init")

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            calls.append("dense_mla")
            return SimpleNamespace(dense_mla=True)

    class MLABuilder(BaseMLABuilder):
        def __init__(self):
            super().__init__()
            self.c128a_decode_topk_ragged_indices_buffer = torch.zeros(4, dtype=torch.int32)
            self.c128a_decode_topk_ragged_indptr_buffer = torch.zeros(4, dtype=torch.int32)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            calls.append("ragged_mla")
            return SimpleNamespace(dense_mla=True)

    class Attention:
        compress_ratio = ratio
        topk_indices_buffer = torch.zeros(batch, 64, dtype=torch.int32)
        swa_cache_layer = SimpleNamespace(kv_cache=torch.zeros(2, 64, 584, dtype=torch.uint8))
        attn_sink = torch.zeros(4, dtype=torch.float32)
        scale = 0.125
        def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
            calls.append("aiter")

    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV4ROCMAiterMLAAttention = Attention
    module.DeepseekV4ROCMAiterSparseSWAMetadataBuilder = Builder
    module.DeepseekV4ROCMAiterMLASparseMetadataBuilder = MLABuilder
    module.DeepseekSparseSWAMetadataBuilder = BaseBuilder
    module.DeepseekV4FlashMLAMetadataBuilder = BaseMLABuilder
    module.DeepseekV4ROCMAiterSparseSWAMetadata = SimpleNamespace
    module.DeepseekV4ROCMAiterMLASparseMetadata = SimpleNamespace
    module.rocm_sparse_attn_prefill = (
        lambda q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
        rope_head_dim, attn_sink, output, ragged_indices=None,
        ragged_indptr=None: calls.append("aiter_prefill_kernel")
    )
    return module, Attention, Builder, MLABuilder


def _swa_metadata(batch, tiles):
    return SimpleNamespace(
        num_decodes=batch,
        num_decode_tokens=batch,
        decode_swa_indices=torch.zeros(batch, 1, 64, dtype=torch.int32),
        decode_swa_lens=torch.ones(batch, dtype=torch.int32),
        tile_sched_swaonly=tiles["swaonly"],
        tile_sched_c4a=tiles["c4a"],
        tile_sched_c128a=tiles["c128a"],
        is_valid_token=torch.ones(batch, dtype=torch.bool),
        token_to_req_indices=torch.arange(batch, dtype=torch.int32),
    )


def test_disabled_patch_does_not_touch_rocm_module(monkeypatch):
    calls = []
    module, Attention, Builder, MLABuilder = _build_module(1, 1, calls)
    originals = (
        Attention._forward_decode,
        module.rocm_sparse_attn_prefill,
        Builder.build,
        Builder.__init__,
        MLABuilder.build,
        MLABuilder.__init__,
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)

    assert not patch.apply_to_module(module)
    assert originals == (
        Attention._forward_decode,
        module.rocm_sparse_attn_prefill,
        Builder.build,
        Builder.__init__,
        MLABuilder.build,
        MLABuilder.__init__,
    )


@pytest.mark.parametrize("enabled_path", ["decode", "prefill"])
def test_feature_flags_patch_only_the_selected_path(monkeypatch, enabled_path):
    calls = []
    module, Attention, Builder, MLABuilder = _build_module(1, 1, calls)
    original_decode = Attention._forward_decode
    original_prefill = module.rocm_sparse_attn_prefill
    original_builders = (Builder.build, Builder.__init__, MLABuilder.build, MLABuilder.__init__)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE",
        enabled_path == "decode",
    )
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL",
        enabled_path == "prefill",
    )

    assert patch.apply_to_module(module)
    if enabled_path == "decode":
        assert Attention._forward_decode is not original_decode
        assert module.rocm_sparse_attn_prefill is original_prefill
        assert original_builders != (
            Builder.build, Builder.__init__, MLABuilder.build, MLABuilder.__init__
        )
    else:
        assert Attention._forward_decode is original_decode
        assert module.rocm_sparse_attn_prefill is not original_prefill
        assert original_builders == (
            Builder.build, Builder.__init__, MLABuilder.build, MLABuilder.__init__
        )


@pytest.mark.parametrize("ratio", [1, 4, 128])
@pytest.mark.parametrize("batch", [1, 8])
def test_decode_contract(monkeypatch, ratio, batch):
    calls = []
    mapper = ModuleType("vllm.models.deepseek_v4.common.ops")

    def map_topk(indices, *args):
        calls.append("map")
        return torch.zeros_like(indices, dtype=torch.int32), torch.ones(
            batch, dtype=torch.int32
        )

    mapper.compute_global_topk_indices_and_lens = map_topk
    monkeypatch.setitem(sys.modules, mapper.__name__, mapper)

    flash = ModuleType("vllm_hcu.v1.attention.ops.flashmla")
    flash.is_flashmla_sparse_supported = lambda: (True, None)
    flash.get_mla_metadata = lambda: (object(), None)

    def kernel(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return torch.ones(q.shape[:-1] + (512,), dtype=q.dtype), None

    flash.flash_mla_with_kvcache = kernel
    def prefill_kernel(**kwargs):
        calls.append({"prefill": kwargs})
        return torch.ones_like(kwargs["q"]), None, None

    flash.flash_mla_sparse_fwd = prefill_kernel
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", True)
    monkeypatch.setattr(patch, "_require_flashmla_ready", lambda: calls.append("guard"))
    monkeypatch.setattr(
        patch, "_require_flashmla_prefill_ready", lambda: calls.append("prefill_guard")
    )

    module, Attention, Builder, MLABuilder = _build_module(ratio, batch, calls)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)

    # Enabled: builds route to the base builders, and the builder inits
    # release the AITER-only ragged buffers.
    swa_builder = Builder()
    assert swa_builder.decode_swa_ragged_indices_buffer is None
    assert swa_builder.decode_swa_ragged_indptr_buffer is None
    mla_builder = MLABuilder()
    assert mla_builder.c128a_decode_topk_ragged_indices_buffer is None
    assert mla_builder.c128a_decode_topk_ragged_indptr_buffer is None
    assert swa_builder.build(0, None).dense_swa
    assert mla_builder.build(0, None).dense_mla
    assert "dense_swa" in calls and "dense_mla" in calls
    assert "ragged_swa" not in calls and "ragged_mla" not in calls

    tiles = swa_builder.build_tile_scheduler(batch)
    assert tiles[Builder._layer_types[0]] is not None
    assert "guard" in calls
    metadata = _swa_metadata(batch, tiles)
    compressed = SimpleNamespace(
        block_table=torch.zeros(batch, 2, dtype=torch.int32),
        block_size=256 * ratio,
        c128a_global_decode_topk_indices=torch.zeros(batch, 1, 64, dtype=torch.int32),
        c128a_decode_topk_lens=torch.ones(batch, dtype=torch.int32),
    )
    q = torch.zeros(batch, 4, 512, dtype=torch.bfloat16)
    output = torch.zeros(batch, 4, 512, dtype=torch.bfloat16)
    Attention()._forward_decode(
        q, None if ratio == 1 else torch.zeros(2, 64, 584, dtype=torch.uint8),
        metadata, None if ratio == 1 else compressed, ratio == 1, output,
    )
    assert torch.all(output == 1)
    kwargs = next(c for c in calls if isinstance(c, dict))
    assert kwargs["head_dim_v"] == 512
    assert kwargs["q"].shape == (batch, 1, 4, 512)
    assert kwargs["topk_length"] is metadata.decode_swa_lens
    assert (kwargs["extra_k_cache"] is None) == (ratio == 1)
    assert ("map" in calls) == (ratio == 4)

    # Prefill replaces only the module-level attention kernel called by the
    # native gather/combiner loop.
    prefill_tokens = 2
    prefill_q = torch.zeros(prefill_tokens, 4, 512, dtype=torch.bfloat16)
    prefill_out = torch.zeros_like(prefill_q)
    module.rocm_sparse_attn_prefill(
        prefill_q,
        torch.zeros(16, 1, 512, dtype=torch.bfloat16),
        torch.zeros(prefill_tokens, 8, dtype=torch.int32),
        torch.full((prefill_tokens,), 8, dtype=torch.int32),
        Attention.scale,
        512,
        448,
        64,
        Attention.attn_sink,
        prefill_out,
    )
    assert torch.all(prefill_out == 1)
    prefill_call = next(c["prefill"] for c in calls if isinstance(c, dict) and "prefill" in c)
    assert prefill_call["indices"].shape == (prefill_tokens, 1, 8)
    assert prefill_call["d_v"] == 512
    assert "prefill_guard" in calls

    # Unsupported local head count (TP2 shape) fails loudly instead of
    # reaching a missing kernel instantiation.
    q32 = torch.zeros(batch, 32, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head counts"):
        Attention()._forward_decode(q32, None, metadata, None, True, q32.clone())

    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)
    disabled_builder = Builder()
    assert disabled_builder.decode_swa_ragged_indices_buffer is not None
    disabled_builder.build(0, None)
    MLABuilder().build(0, None)
    assert "ragged_swa" in calls and "ragged_mla" in calls
    Attention()._forward_decode(q, None, metadata, None, True, output)
    assert calls[-1] == "aiter"
    module.rocm_sparse_attn_prefill(
        prefill_q,
        torch.zeros(16, 1, 512, dtype=torch.bfloat16),
        torch.zeros(prefill_tokens, 8, dtype=torch.int32),
        torch.full((prefill_tokens,), 8, dtype=torch.int32),
        Attention.scale,
        512,
        448,
        64,
        Attention.attn_sink,
        prefill_out,
    )
    assert calls[-1] == "aiter_prefill_kernel"


def test_guard_rejects_unavailable_flashmla(monkeypatch):
    calls = []
    flash = ModuleType("vllm_hcu.v1.attention.ops.flashmla")
    flash.is_flashmla_sparse_supported = lambda: (False, "flash_mla is not available.")
    flash.get_mla_metadata = lambda: (object(), None)
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)

    module, _, Builder, _ = _build_module(1, 1, calls)
    assert patch.apply_to_module(module)
    patch._require_flashmla_ready.cache_clear()
    with pytest.raises(RuntimeError, match="FlashMLA decode unavailable"):
        Builder().build_tile_scheduler(1)
    patch._require_flashmla_ready.cache_clear()
