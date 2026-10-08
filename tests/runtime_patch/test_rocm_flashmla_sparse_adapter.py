# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract checks for the ROCm FlashMLA sparse adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_flashmla_sparse as patch
from vllm_hcu.platforms import envs as henvs


@pytest.fixture(autouse=True)
def custom_ops_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")


def _build_module(ratio, batch, calls, local_heads=4, cache_dtype="fp8_ds_mla"):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_attention_heads=local_heads)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )

    class BaseBuilder:
        def __init__(self):
            calls.append("base_swa_init")
            self.vllm_config = vllm_config

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False, replay_start=None):
            calls.append("dense_swa")
            calls.append(("replay_start", replay_start))
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

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False, replay_start=None):
            calls.append("ragged_swa")
            calls.append(("replay_start", replay_start))
            return SimpleNamespace(dense_swa=True)

    class BaseMLABuilder:
        def __init__(self):
            calls.append("base_mla_init")
            self.vllm_config = vllm_config

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
    module.DeepseekV4SparseMLAMetadataBuilder = BaseMLABuilder
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
    prefill_q = torch.zeros(prefill_tokens, 64, 512, dtype=torch.bfloat16)
    prefill_out = torch.zeros_like(prefill_q)
    prefill_sink = torch.zeros(64, dtype=torch.float32)
    module.rocm_sparse_attn_prefill(
        prefill_q,
        torch.zeros(16, 1, 512, dtype=torch.bfloat16),
        torch.zeros(prefill_tokens, 8, dtype=torch.int32),
        torch.full((prefill_tokens,), 8, dtype=torch.int32),
        Attention.scale,
        512,
        448,
        64,
        prefill_sink,
        prefill_out,
    )
    assert torch.all(prefill_out == 1)
    prefill_call = next(c["prefill"] for c in calls if isinstance(c, dict) and "prefill" in c)
    assert prefill_call["indices"].shape == (prefill_tokens, 1, 8)
    assert prefill_call["d_v"] == 512
    assert "prefill_guard" in calls

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
        prefill_sink,
        prefill_out,
    )
    assert calls[-1] == "aiter_prefill_kernel"


@pytest.mark.parametrize(
    ("heads", "uses_flashmla"),
    [(16, True), (32, False), (64, True), (128, True)],
)
def test_decode_falls_back_for_unsupported_local_head_counts(
    monkeypatch, heads, uses_flashmla
):
    calls = []
    mapper = ModuleType("vllm.models.deepseek_v4.common.ops")
    mapper.compute_global_topk_indices_and_lens = lambda *args: (None, None)
    monkeypatch.setitem(sys.modules, mapper.__name__, mapper)
    flash = ModuleType("vllm_hcu.v1.attention.ops.flashmla")
    flash.is_flashmla_sparse_supported = lambda: (True, None)
    flash.get_mla_metadata = lambda: (object(), None)

    def kernel(**kwargs):
        calls.append("flashmla_decode")
        return torch.ones_like(kwargs["q"]), None

    flash.flash_mla_with_kvcache = kernel
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)
    monkeypatch.setattr(patch, "_require_flashmla_ready", lambda: calls.append("guard"))

    module, Attention, Builder, MLABuilder = _build_module(
        1, 1, calls, local_heads=heads
    )
    assert patch.apply_to_module(module)
    swa_builder = Builder()
    mla_builder = MLABuilder()
    if uses_flashmla:
        assert swa_builder.decode_swa_ragged_indices_buffer is None
        assert mla_builder.c128a_decode_topk_ragged_indices_buffer is None
    else:
        assert swa_builder.decode_swa_ragged_indices_buffer is not None
        assert mla_builder.c128a_decode_topk_ragged_indices_buffer is not None

    swa_builder.build(0, None)
    mla_builder.build(0, None)
    tiles = swa_builder.build_tile_scheduler(1)
    metadata = _swa_metadata(1, tiles)
    q = torch.zeros(1, heads, 512, dtype=torch.bfloat16)
    Attention()._forward_decode(q, None, metadata, None, True, q.clone())

    assert ("flashmla_decode" in calls) is uses_flashmla
    assert ("guard" in calls) is uses_flashmla
    assert ("aiter" in calls) is not uses_flashmla
    assert ("dense_swa" in calls) is uses_flashmla
    assert ("ragged_swa" in calls) is not uses_flashmla


@pytest.mark.parametrize(
    ("heads", "uses_flashmla"),
    [(16, False), (32, False), (64, True), (128, True)],
)
def test_prefill_falls_back_for_unsupported_local_head_counts(
    monkeypatch, heads, uses_flashmla
):
    calls = []
    flash = ModuleType("vllm_hcu.v1.attention.ops.flashmla")
    flash.is_flashmla_sparse_supported = lambda: (True, None)

    def prefill_kernel(**kwargs):
        calls.append("flashmla_prefill")
        return torch.ones_like(kwargs["q"]), None, None

    flash.flash_mla_sparse_fwd = prefill_kernel
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", True)
    monkeypatch.setattr(
        patch, "_require_flashmla_prefill_ready", lambda: calls.append("prefill_guard")
    )

    module, _, _, _ = _build_module(1, 1, calls)
    assert patch.apply_to_module(module)
    tokens = 2
    q = torch.zeros(tokens, heads, 512, dtype=torch.bfloat16)
    output = torch.zeros_like(q)
    module.rocm_sparse_attn_prefill(
        q,
        torch.zeros(16, 1, 512, dtype=torch.bfloat16),
        torch.zeros(tokens, 8, dtype=torch.int32),
        torch.full((tokens,), 8, dtype=torch.int32),
        0.125,
        512,
        448,
        64,
        torch.zeros(heads, dtype=torch.float32),
        output,
    )

    assert ("flashmla_prefill" in calls) is uses_flashmla
    assert ("prefill_guard" in calls) is uses_flashmla
    assert ("aiter_prefill_kernel" in calls) is not uses_flashmla
    if uses_flashmla:
        assert torch.all(output == 1)


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


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "float16"])
def test_plain_cache_keeps_native_builders_and_decode(monkeypatch, cache_dtype):
    calls = []
    module, Attention, Builder, MLABuilder = _build_module(
        1, 1, calls, cache_dtype=cache_dtype
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)
    monkeypatch.setattr(patch, "_require_flashmla_ready", lambda: pytest.fail("FP8 guard reached"))
    assert patch.apply_to_module(module)
    swa_builder, mla_builder = Builder(), MLABuilder()
    assert swa_builder.decode_swa_ragged_indices_buffer is not None
    assert mla_builder.c128a_decode_topk_ragged_indices_buffer is not None
    swa_builder.build(0, None)
    mla_builder.build(0, None)
    tiles = swa_builder.build_tile_scheduler(1)
    assert all(value is None for value in tiles.values())
    Attention.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    )
    q = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
    Attention()._forward_decode(q, None, _swa_metadata(1, tiles), None, True, q.clone())
    assert calls[-1] == "aiter"
    assert "ragged_swa" in calls and "ragged_mla" in calls


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "bfloat16"])
def test_swa_replay_start_is_forwarded(monkeypatch, cache_dtype):
    calls = []
    module, _, Builder, _ = _build_module(1, 1, calls, cache_dtype=cache_dtype)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)
    patch.apply_to_module(module)
    replay = torch.tensor([3], dtype=torch.int32)
    Builder().build(0, None, replay_start=replay)
    assert next(item[1] for item in calls if isinstance(item, tuple) and item[0] == "replay_start") is replay


@pytest.mark.parametrize("width,dtype", [(576, torch.bfloat16), (512, torch.float16)])
def test_prefill_preserves_other_query_layouts(monkeypatch, width, dtype):
    calls = []
    module, _, _, _ = _build_module(1, 1, calls)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", True)
    monkeypatch.setattr(patch, "_require_flashmla_prefill_ready", lambda: pytest.fail("FP8 guard reached"))
    patch.apply_to_module(module)
    q = torch.zeros(1, 64, width, dtype=dtype)
    module.rocm_sparse_attn_prefill(
        q, torch.zeros(8, 1, width, dtype=dtype),
        torch.zeros(1, 8, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
        0.125, width, width - 64, 64, torch.zeros(64), q.clone(),
    )
    assert calls == ["aiter_prefill_kernel"]


def test_v0281_installed_source_api_contract(monkeypatch):
    """Use real target method signatures without importing GPU model modules."""
    import ast
    from tests.fixtures.vllm_source import resolve_target_vllm_root

    root = resolve_target_vllm_root()
    rocm_tree = ast.parse((root / "vllm/models/deepseek_v4/amd/rocm.py").read_text())
    ops_tree = ast.parse((root / "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py").read_text())
    calls = []
    module, Attention, Builder, MLABuilder = _build_module(1, 1, calls)

    def signature_stub(node):
        import copy
        node = copy.deepcopy(node)
        node.decorator_list = []
        node.body = [ast.Pass()]
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(tree), "<target-signature>", "exec"), namespace)
        return namespace[node.name]

    for name, cls, method in (
        ("DeepseekV4ROCMAiterMLAAttention", Attention, "_forward_decode"),
        ("DeepseekV4ROCMAiterSparseSWAMetadataBuilder", Builder, "build"),
        ("DeepseekV4ROCMAiterMLASparseMetadataBuilder", MLABuilder, "build"),
    ):
        target_cls = next(node for node in rocm_tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        target_fn = next(node for node in target_cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
        setattr(cls, method, signature_stub(target_fn))
    prefill = next(node for node in ops_tree.body if isinstance(node, ast.FunctionDef) and node.name == "rocm_sparse_attn_prefill")
    module.rocm_sparse_attn_prefill = signature_stub(prefill)
    assert any(isinstance(node, ast.ImportFrom) and any(alias.name == "DeepseekV4SparseMLAMetadataBuilder" for alias in node.names) for node in rocm_tree.body)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", True)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)


def test_adaptive_splits_reaches_native_decode(monkeypatch):
    calls = []
    module, Attention, _, _ = _build_module(1, 1, calls, local_heads=32)

    def native(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output, adaptive_splits):
        calls.append(("adaptive_splits", adaptive_splits))

    Attention._forward_decode = native
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", False)
    patch.apply_to_module(module)
    q = torch.zeros(1, 32, 512, dtype=torch.bfloat16)
    Attention()._forward_decode(q, None, None, None, True, q.clone(), adaptive_splits=True)
    assert calls == [("adaptive_splits", True)]


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("already_patched", [False, True])
def test_master_disabled_retains_rocm_attention_and_metadata(
    monkeypatch, heads, already_patched
):
    calls = []
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", True)
    module, Attention, Builder, MLABuilder = _build_module(
        1, 2, calls, local_heads=heads
    )
    originals = (
        Attention._forward_decode, module.rocm_sparse_attn_prefill,
        Builder.build, Builder.__init__, Builder.build_tile_scheduler,
        MLABuilder.build, MLABuilder.__init__,
    )
    if already_patched:
        assert patch.apply_to_module(module)

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    # No provider should be imported or validated with the master disabled.
    for name in (
        "flash_mla", "flash_mla.cuda", "flash_mla.flash_mla_interface",
        "boltops", "boltops.mla", "vllm_hcu.v1.attention.ops.flashmla",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    assert not patch.apply_to_module(module)
    assert not patch.apply()
    if not already_patched:
        assert originals == (
            Attention._forward_decode, module.rocm_sparse_attn_prefill,
            Builder.build, Builder.__init__, Builder.build_tile_scheduler,
            MLABuilder.build, MLABuilder.__init__,
        )

    builder, mla_builder = Builder(), MLABuilder()
    assert builder.decode_swa_ragged_indices_buffer is not None
    assert mla_builder.c128a_decode_topk_ragged_indices_buffer is not None
    builder.build(0, None)
    mla_builder.build(0, None)
    assert builder.build_tile_scheduler(2) == {
        "swaonly": None, "c4a": None, "c128a": None,
    }
    q = torch.zeros(2, heads, 512, dtype=torch.bfloat16)
    Attention()._forward_decode(q, None, None, None, True, q.clone())
    module.rocm_sparse_attn_prefill(
        q, torch.zeros(3, 1, 512, dtype=torch.bfloat16),
        torch.zeros(2, 4, dtype=torch.int32),
        torch.ones(2, dtype=torch.int32), 0.125, 512, 448, 64,
        torch.zeros(heads), q.clone(),
    )
    assert "ragged_swa" in calls and "ragged_mla" in calls
    assert "dense_swa" not in calls and "dense_mla" not in calls
    assert "aiter" in calls and "aiter_prefill_kernel" in calls
