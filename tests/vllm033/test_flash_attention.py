from __future__ import annotations

import importlib
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import pytest
import torch

PLUGIN_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))


def load_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_vllm_seams: bool = True,
    kv_cache_layout: str = "LBHNC",
):
    class AttentionImpl:
        dcp_world_size = pcp_world_size = 1

        @classmethod
        def __class_getitem__(cls, item):
            return cls

    class FlashAttentionBackend:
        @classmethod
        def get_preferred_block_size(cls, default_block_size):
            return default_block_size

        @classmethod
        def supports_kv_cache_dtype(cls, kv_cache_dtype):
            return kv_cache_dtype != "fp8_e5m2"

        @classmethod
        def supports_combination(
            cls,
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        ):
            if kv_cache_dtype == "fp8_e5m2":
                return "upstream rejects fp8_e5m2"
            return "delegated-combination-result"

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    attention = ModuleType("vllm.v1.attention")
    attention.__path__ = []
    backends = ModuleType("vllm.v1.attention.backends")
    backends.__path__ = []
    official = ModuleType("vllm.v1.attention.backends.flash_attn")
    official.FlashAttentionMetadata = SimpleNamespace
    public_api = ModuleType("vllm.v1.attention.backend")
    public_api.AttentionImpl = AttentionImpl
    official.FlashAttentionBackend = FlashAttentionBackend
    layouts = ModuleType("vllm.v1.kv_cache_layout")
    layouts.KVCacheLayout = SimpleNamespace(LBHNC="LBHNC", LBNHC="LBNHC")
    config = ModuleType("vllm.config")
    cache_config = SimpleNamespace(get_resolved_kv_cache_layout=lambda: kv_cache_layout)
    config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
        cache_config=cache_config
    )
    for name, module in (
        ("vllm", vllm),
        ("vllm.config", config),
        ("vllm.v1", v1),
        ("vllm.v1.attention", attention),
        ("vllm.v1.attention.backend", public_api),
        ("vllm.v1.attention.backends", backends),
        ("vllm.v1.attention.backends.flash_attn", official),
        ("vllm.v1.kv_cache_layout", layouts),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules,
        "hcu_vllm_das.attention.flash_attn",
        raising=False,
    )
    return importlib.import_module("hcu_vllm_das.attention.flash_attn")


def test_import_requires_only_public_attention_api(monkeypatch):
    flash = load_flash_attention(monkeypatch, with_vllm_seams=False)
    assert flash.HcuFlashAttentionImpl.__bases__[0].__name__ == "AttentionImpl"


def test_constructs_against_target_public_abstract_base(monkeypatch):
    """Execute the target ABC, including its real __new__, without native ops.

    Only group/config dependencies are CPU doubles; constructor/abstract and
    fusion contracts come from the selected source rather than a fake class.
    Full installed vLLM imports remain a separate native-ABI gate.
    """
    root = Path(os.environ["VLLM_TARGET_ROOT"])
    source = root / "vllm/v1/attention/backend.py"
    tree = ast.parse(source.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"AttentionImplBase", "AttentionImpl"}
    ]
    namespace = dict(
        ABC=ABC,
        abstractmethod=abstractmethod,
        Generic=Generic,
        ClassVar=ClassVar,
        T=TypeVar("T"),
        AttentionType=SimpleNamespace(DECODER="decoder"),
    )
    # Compile annotations without evaluating framework-only type names.
    import __future__

    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    flash = load_flash_attention(monkeypatch)
    public = sys.modules["vllm.v1.attention.backend"]
    public.AttentionImpl = namespace["AttentionImpl"]
    groups = ModuleType("vllm.distributed.parallel_state")
    groups.get_dcp_world_size_and_rank = lambda enabled: (1, 0)
    groups.get_pcp_group = lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", groups)
    sys.modules["vllm.config"].get_current_vllm_config_or_none = lambda: None
    monkeypatch.delitem(sys.modules, "hcu_vllm_das.attention.flash_attn")
    flash = importlib.import_module("hcu_vllm_das.attention.flash_attn")
    impl = make_impl(flash)
    assert impl.dcp_world_size == impl.pcp_world_size == 1
    assert impl.fused_output_quant_supported(None) is False
    assert impl.fused_qk_norm_rope_kvcache_supported() is False
    assert impl.fused_rope_kvcache_supported() is False


def metadata(tokens=2):
    return SimpleNamespace(
        num_actual_tokens=tokens,
        causal=True,
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        max_query_len=tokens,
        seq_lens=torch.tensor([7], dtype=torch.int32),
        max_seq_len=7,
        block_table=torch.tensor([[1]], dtype=torch.int32),
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        dcp_context_kv_lens=None,
    )


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("dtype", ("bfloat16", "fp8_e5m2"))
def test_forward_preserves_output_padding_and_passes_paged_metadata(
    monkeypatch, layout, dtype
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=layout)
    impl = make_impl(flash, kv_cache_dtype=dtype)
    query = torch.randn(5, 4, 64)
    output = torch.full_like(query, -99.0)
    layer = SimpleNamespace(_k_scale=torch.tensor(0.5), _v_scale=torch.tensor(0.25))
    cache_dtype = torch.uint8 if dtype == "fp8_e5m2" else torch.bfloat16
    cache = torch.empty_strided(
        (2, 4, 64, 128),
        (32768, 8192, 128, 1) if layout == "LBHNC" else (32768, 128, 512, 1),
        dtype=cache_dtype,
    )
    calls = []

    def provider(**arguments):
        calls.append(arguments)
        arguments["out"].fill_(7)

    monkeypatch.setattr(impl, "_run_flash_attention", provider)
    batch = metadata()
    result = impl.forward(layer, query, query, query, cache, batch, output)

    assert result is output
    torch.testing.assert_close(output[:2], torch.full_like(output[:2], 7.0))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], -99.0))
    assert len(calls) == 1
    arguments = calls[0]
    assert arguments["q"].shape == (2, 4, 64)
    assert arguments["out"].data_ptr() == output.data_ptr()
    assert arguments["block_table"] is batch.block_table
    assert arguments["cu_seqlens_q"] is batch.query_start_loc
    assert arguments["seqused_k"] is batch.seq_lens
    assert arguments["max_seqlen_k"] == 7
    assert arguments["q_descale"] is None
    if dtype == "fp8_e5m2":
        torch.testing.assert_close(arguments["k_descale"], torch.full((1, 4), 0.5))
        torch.testing.assert_close(arguments["v_descale"], torch.full((1, 4), 0.25))
    else:
        assert arguments["k_descale"] is arguments["v_descale"] is None


def test_forward_profiles_without_provider_and_leaves_empty_batch_untouched(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch)
    impl = make_impl(flash)
    monkeypatch.setattr(
        impl,
        "_run_flash_attention",
        lambda **kw: pytest.fail("unexpected provider call"),
    )
    output = torch.ones(3, 4, 64)
    assert impl.forward(None, None, None, None, None, None, output) is output
    assert torch.count_nonzero(output) == 0
    output.fill_(3)
    assert impl.forward(None, None, None, None, None, metadata(0), output) is output
    assert torch.all(output == 3)


@pytest.mark.parametrize(
    "field,value",
    (
        ("causal", False),
        ("causal", torch.tensor([True])),
        ("mm_prefix_query_range_tensor", torch.empty(2, 2)),
        ("rswa_prefix_lens", torch.tensor([1])),
        ("dcp_context_kv_lens", torch.tensor([1])),
    ),
)
def test_forward_rejects_unsupported_metadata_before_provider(
    monkeypatch, field, value
):
    flash = load_flash_attention(monkeypatch)
    impl = make_impl(flash)
    batch = metadata()
    setattr(batch, field, value)
    with pytest.raises(NotImplementedError, match="metadata"):
        impl.forward(None, None, None, None, None, batch, None)


@pytest.mark.parametrize("argument", ("output_scale", "output_block_scale"))
def test_forward_rejects_fused_output_quantization(monkeypatch, argument):
    impl = make_impl(load_flash_attention(monkeypatch))
    with pytest.raises(NotImplementedError, match="output quantization"):
        impl.forward(
            None, None, None, None, None, None, None, **{argument: torch.tensor(1.0)}
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"attn_type": "encoder"},
        {"sliding_window": 128},
        {"sinks": torch.empty(4)},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"head_size": 96},
        {"num_kv_heads": 3},
    ),
)
def test_constructor_rejects_unsupported_configuration(monkeypatch, kwargs):
    flash = load_flash_attention(monkeypatch)
    with pytest.raises((ValueError, NotImplementedError)):
        make_impl(flash, **kwargs)


def make_impl(flash, **kwargs):
    arguments = dict(num_heads=4, head_size=64, scale=0.125)
    arguments.update(kwargs)
    return flash.HcuFlashAttentionImpl(**arguments)


@pytest.mark.parametrize(
    (
        "layout",
        "packed_inner_strides",
        "read_shape",
        "read_inner_strides",
    ),
    (
        (
            "LBHNC",
            lambda h, n, d: (n * 2 * d, 2 * d, 1),
            lambda b, h, n, d: (b, h, n, d),
            lambda h, n, d: (n * d, d, 1),
        ),
        (
            "LBNHC",
            lambda h, n, d: (2 * d, h * 2 * d, 1),
            lambda b, h, n, d: (b, n, h, d),
            lambda h, n, d: (h * d, d, 1),
        ),
    ),
)
def test_dense_views_separate_kv_and_preserve_block_stride(
    monkeypatch,
    layout,
    packed_inner_strides,
    read_shape,
    read_inner_strides,
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=layout)
    num_blocks, num_heads, block_size, head_size = 2, 3, 4, 5
    half_page = num_heads * block_size * head_size
    block_stride = 2 * half_page + 17
    storage_offset = 13
    raw = torch.zeros(storage_offset + block_stride * num_blocks)
    packed = torch.as_strided(
        raw,
        size=(num_blocks, num_heads, block_size, 2 * head_size),
        stride=(
            block_stride,
            *packed_inner_strides(num_heads, block_size, head_size),
        ),
        storage_offset=storage_offset,
    )

    impl = make_impl(flash)
    impl.head_size = head_size
    key, value = impl._get_kv_cache_views(packed)
    write_key, write_value = impl._get_kv_cache_write_views(packed)
    key.fill_(11)
    value.fill_(22)

    assert (
        key.shape
        == value.shape
        == read_shape(num_blocks, num_heads, block_size, head_size)
    )
    assert (
        key.stride()
        == value.stride()
        == (
            block_stride,
            *read_inner_strides(num_heads, block_size, head_size),
        )
    )
    assert key.storage_offset() == storage_offset
    assert value.storage_offset() == storage_offset + half_page
    assert (
        write_key.shape
        == write_value.shape
        == (
            num_blocks,
            block_size,
            num_heads,
            head_size,
        )
    )
    expected_write_key = key.transpose(1, 2) if layout == "LBHNC" else key
    expected_write_value = value.transpose(1, 2) if layout == "LBHNC" else value
    assert write_key.stride() == expected_write_key.stride()
    assert write_value.stride() == expected_write_value.stride()
    torch.testing.assert_close(
        raw[storage_offset : storage_offset + half_page],
        torch.full((half_page,), 11.0),
    )
    torch.testing.assert_close(
        raw[storage_offset + half_page : storage_offset + 2 * half_page],
        torch.full((half_page,), 22.0),
    )
    torch.testing.assert_close(
        raw[storage_offset + block_stride : storage_offset + block_stride + half_page],
        torch.full((half_page,), 11.0),
    )
    torch.testing.assert_close(
        raw[
            storage_offset + block_stride + half_page : storage_offset
            + block_stride
            + 2 * half_page
        ],
        torch.full((half_page,), 22.0),
    )
    torch.testing.assert_close(
        raw[storage_offset + 2 * half_page : storage_offset + block_stride],
        torch.zeros(17),
    )


@pytest.mark.parametrize("manager_page", (64, 128, 576, 1088, 2176))
def test_upstream_accepts_manager_pages_and_selects_64_token_kernel(
    monkeypatch, manager_page
):
    """Run target admission and block selection, preserving their distinct roles."""
    import __future__

    root = Path(os.environ["VLLM_TARGET_ROOT"])
    backend_source = root / "vllm/v1/attention/backend.py"
    backend_tree = ast.parse(backend_source.read_text())
    base = next(
        node
        for node in backend_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionBackend"
    )
    method = next(
        node
        for node in base.body
        if isinstance(node, ast.FunctionDef) and node.name == "supports_block_size"
    )
    # All exercised declarations are fixed integer kernel sizes, not MultipleOf.
    namespace = {"MultipleOf": type("MultipleOf", (), {})}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(backend_source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    flash = load_flash_attention(monkeypatch)
    parent = flash.HcuFlashAttentionBackend.__bases__[0]
    monkeypatch.setattr(
        parent, "supports_block_size", namespace["supports_block_size"], raising=False
    )
    assert "supports_block_size" not in flash.HcuFlashAttentionBackend.__dict__
    assert flash.HcuFlashAttentionBackend.supports_block_size(manager_page)
    assert not flash.HcuFlashAttentionBackend.supports_block_size(manager_page + 1)

    worker_source = root / "vllm/v1/worker/utils.py"
    definitions = [
        node
        for node in ast.parse(worker_source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_block_size_is_supported", "select_common_block_size"}
    ]
    assert len(definitions) == 2
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(worker_source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    assert (
        namespace["select_common_block_size"](
            manager_page, [flash.HcuFlashAttentionBackend]
        )
        == 64
    )


def test_backend_selects_fa2_dual_layout_and_64_token_pages(monkeypatch):
    flash = load_flash_attention(monkeypatch)

    impl = make_impl(flash)

    assert impl.vllm_flash_attn_version == 2
    assert flash.HcuFlashAttentionBackend.get_impl_cls() is type(impl)
    assert flash.HcuFlashAttentionBackend.supported_kv_cache_layouts() == (
        "LBHNC",
        "LBNHC",
    )
    assert flash.HcuFlashAttentionBackend.get_supported_kernel_block_sizes() == [64]
    assert flash.HcuFlashAttentionBackend.get_preferred_block_size(16) == 64
    assert flash.HcuFlashAttentionBackend.get_preferred_block_size(128) == 64
    assert flash.HcuFlashAttentionBackend.supported_kv_cache_dtypes == [
        "auto",
        "float16",
        "bfloat16",
        "fp8_e5m2",
    ]
    assert flash.HcuFlashAttentionBackend.supports_dcp() is False
    assert flash.HcuFlashAttentionBackend.supports_mm_prefix() is False
    assert flash.HcuFlashAttentionBackend.supports_sink() is False
    assert flash.HcuFlashAttentionBackend.supports_rswa() is False
    assert flash.HcuFlashAttentionBackend.supports_attn_type("decoder") is True
    assert flash.HcuFlashAttentionBackend.supports_attn_type("encoder") is False
    assert flash.HcuFlashAttentionBackend.supports_sliding_window() is False
    assert flash.HcuFlashAttentionBackend.supports_non_causal() is False
    assert flash.HcuFlashAttentionBackend.supports_batch_invariance() is False


def test_backend_accepts_only_hcu_e5m2_extension_and_delegates_other_dtypes(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch)
    backend = flash.HcuFlashAttentionBackend
    common = dict(
        head_size=128,
        dtype=torch.bfloat16,
        block_size=64,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=object(),
    )

    assert backend.supports_kv_cache_dtype("fp8_e5m2") is True
    assert (
        backend.supports_combination(
            kv_cache_dtype="fp8_e5m2",
            **common,
        )
        is None
    )
    assert backend.supports_kv_cache_dtype("bfloat16") is True
    assert (
        backend.supports_combination(
            kv_cache_dtype="bfloat16",
            **common,
        )
        is None
    )


@pytest.mark.parametrize("constructor_style", ("keyword", "positional"))
def test_e5m2_constructor_preserves_public_dtype_contract(
    monkeypatch,
    constructor_style,
):
    flash = load_flash_attention(monkeypatch)

    if constructor_style == "keyword":
        impl = make_impl(flash, kv_cache_dtype="fp8_e5m2")
    else:
        impl = flash.HcuFlashAttentionImpl(
            4,
            64,
            0.125,
            2,
            None,
            None,
            "fp8_e5m2",
        )

    assert impl.kv_cache_dtype == "fp8_e5m2"
    assert impl.supports_quant_query_input is False


def test_non_e5m2_constructor_uses_unquantized_query_input(monkeypatch):
    flash = load_flash_attention(monkeypatch)

    impl = make_impl(flash, kv_cache_dtype="bfloat16")

    assert impl.kv_cache_dtype == "bfloat16"
    assert impl.supports_quant_query_input is False


def test_e5m2_cache_update_uses_plugin_writer_and_resolved_write_views(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LBHNC")
    calls = []
    monkeypatch.setattr(
        flash,
        "_reshape_and_cache_e5m2",
        lambda *args: calls.append(args),
    )
    impl = make_impl(flash)
    impl.kv_cache_dtype = "fp8_e5m2"
    impl.head_size = 4
    key = torch.randn(2, 3, 4)
    value = torch.randn_like(key)
    packed = torch.empty(1, 3, 5, 8, dtype=torch.uint8)
    slots = torch.tensor([0, 1], dtype=torch.long)
    layer = SimpleNamespace(
        _k_scale=torch.tensor(0.5),
        _v_scale=torch.tensor(0.25),
    )

    impl.do_kv_cache_update(layer, key, value, packed, slots)

    assert len(calls) == 1
    written_key, written_value = impl._get_kv_cache_write_views(packed)
    assert calls[0][0] is key
    assert calls[0][1] is value
    assert calls[0][2].shape == written_key.shape
    assert calls[0][2].stride() == written_key.stride()
    assert calls[0][2].data_ptr() == written_key.data_ptr()
    assert calls[0][3].shape == written_value.shape
    assert calls[0][3].stride() == written_value.stride()
    assert calls[0][3].data_ptr() == written_value.data_ptr()
    assert calls[0][4] is slots
    assert calls[0][5] is layer._k_scale
    assert calls[0][6] is layer._v_scale
    assert not hasattr(impl, "upstream_cache_update")


@pytest.mark.parametrize("layout", ("LBHNC", "LBNHC"))
@pytest.mark.parametrize("kv_cache_dtype", ("auto", "bfloat16", "float16"))
def test_non_e5m2_cache_update_uses_native_writer_with_dense_views(
    monkeypatch, layout, kv_cache_dtype
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=layout)
    calls = []
    native = ModuleType("vllm._custom_ops")
    native.reshape_and_cache_flash = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", native)
    impl = make_impl(flash, kv_cache_dtype=kv_cache_dtype)
    impl.head_size = 4
    strides = (120, 40, 8, 1) if layout == "LBHNC" else (120, 8, 24, 1)
    packed = torch.empty_strided((1, 3, 5, 8), strides)
    key = torch.randn(2, 3, 4)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 1], dtype=torch.long)
    layer = SimpleNamespace(_k_scale=torch.tensor(1.0), _v_scale=torch.tensor(1.0))

    impl.do_kv_cache_update(layer, key, value, packed, slots)

    assert len(calls) == 1
    k, v = impl._get_kv_cache_write_views(packed)
    assert calls[0][:2] == (key, value)
    assert calls[0][2].data_ptr() == k.data_ptr()
    assert calls[0][3].data_ptr() == v.data_ptr()
    assert calls[0][2].stride() == k.stride()
    assert calls[0][3].stride() == v.stride()
    assert calls[0][4] is slots
    # v0.33 native dtype dispatch owns the public unquantized aliases.
    assert calls[0][5] == kv_cache_dtype


@pytest.mark.parametrize(
    "incoming_dtype",
    (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2),
)
def test_e5m2_vendor_boundary_reinterprets_kv_bytes_only(
    monkeypatch,
    incoming_dtype,
):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs) or "output",
    )
    impl = make_impl(flash)
    impl.kv_cache_dtype = "fp8_e5m2"
    query = torch.empty(1, 4, 64, dtype=torch.bfloat16)
    raw_key = torch.arange(64, dtype=torch.uint8)
    raw_value = torch.arange(64, dtype=torch.uint8).flip(0)
    key = raw_key if incoming_dtype == torch.uint8 else raw_key.view(incoming_dtype)
    value = (
        raw_value if incoming_dtype == torch.uint8 else raw_value.view(incoming_dtype)
    )
    k_descale = torch.tensor(0.5)
    v_descale = torch.tensor(0.25)

    result = impl._run_flash_attention(
        q=query,
        k=key,
        v=value,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    assert result == "output"
    assert calls[0]["q"] is query
    assert calls[0]["k"].dtype == torch.float8_e5m2
    assert calls[0]["v"].dtype == torch.float8_e5m2
    assert calls[0]["k"].view(torch.uint8).data_ptr() == raw_key.data_ptr()
    assert calls[0]["v"].view(torch.uint8).data_ptr() == raw_value.data_ptr()
    assert calls[0]["k_descale"] is k_descale
    assert calls[0]["v_descale"] is v_descale


def test_e5m2_vendor_boundary_rejects_non_byte_kv_storage(monkeypatch):
    flash = load_flash_attention(monkeypatch)
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: pytest.fail("vendor must not run"),
    )
    impl = make_impl(flash)
    impl.kv_cache_dtype = "fp8_e5m2"

    with pytest.raises(ValueError, match="one-byte K/V storage"):
        impl._run_flash_attention(
            q=torch.empty(1, 4, 64, dtype=torch.bfloat16),
            k=torch.empty(64, dtype=torch.bfloat16),
            v=torch.empty(64, dtype=torch.bfloat16),
        )


def test_non_e5m2_vendor_boundary_preserves_kv_dtype(monkeypatch):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = make_impl(flash)
    impl.kv_cache_dtype = "auto"
    key = torch.empty(64, dtype=torch.float8_e4m3fn)
    value = torch.empty_like(key)

    impl._run_flash_attention(q="q", k=key, v=value)

    assert calls[0]["k"] is key
    assert calls[0]["v"] is value


@pytest.mark.parametrize("head_size", (64, 128, 192, 256))
def test_backend_accepts_vendor_supported_head_sizes(monkeypatch, head_size):
    flash = load_flash_attention(monkeypatch)

    assert flash.HcuFlashAttentionBackend.supports_head_size(head_size) is True


@pytest.mark.parametrize("head_size", (-64, 0, 8, 32, 80, 96, 264))
def test_backend_rejects_vendor_unsupported_head_sizes(monkeypatch, head_size):
    flash = load_flash_attention(monkeypatch)

    assert flash.HcuFlashAttentionBackend.supports_head_size(head_size) is False


@pytest.mark.parametrize(
    ("kv_cache_layout", "kernel_layout"),
    (("LBHNC", "bhsd"), ("LBNHC", "bshd")),
)
def test_varlen_boundary_forwards_fa2_keywords_without_fa4_options(
    monkeypatch, kv_cache_layout, kernel_layout
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=kv_cache_layout)
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        return "vendor-output"

    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: vendor)
    impl = make_impl(flash)

    result = impl._run_flash_attention(
        q="q",
        k="k",
        v="v",
        out="out",
        dynamic_causal=None,
        mask_mod=None,
        aux_tensors=None,
        num_splits=3,
    )

    assert result == "vendor-output"
    assert calls == [
        {
            "q": "q",
            "k": "k",
            "v": "v",
            "out": "out",
            "num_splits": 3,
            "layout": kernel_layout,
            "fa_version": 2,
        }
    ]


def test_layout_selection_uses_resolved_config_when_geometry_is_ambiguous(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LBNHC")
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = make_impl(flash)
    impl.head_size = 8
    packed = torch.empty_strided(
        (1, 4, 4, 16),
        (256, 16, 64, 1),
    )

    key, _ = impl._get_kv_cache_views(packed)
    impl._run_flash_attention(q="q", k=key, v=key)

    assert key.shape == (1, 4, 4, 8)
    assert key.stride() == (256, 32, 8, 1)
    assert calls[0]["layout"] == "bshd"


def test_resolved_layout_rejects_other_layout_strides(monkeypatch):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LBHNC")
    impl = make_impl(flash)
    impl.head_size = 8
    nhd_packed = torch.empty_strided(
        (1, 4, 4, 16),
        (256, 16, 64, 1),
    )

    with pytest.raises(ValueError, match="strides do not match resolved layout"):
        impl._get_kv_cache_views(nhd_packed)


def test_unknown_resolved_layout_fails_closed(monkeypatch):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LHBNC")
    impl = make_impl(flash)
    impl.head_size = 8

    with pytest.raises(ValueError, match="unsupported resolved KV cache layout"):
        impl._get_kv_cache_views(torch.empty(1, 1, 1, 16))


def test_varlen_boundary_avoids_vendor_short_prefill_mtp_misclassification(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch)
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        return "vendor-output"

    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: vendor)
    impl = make_impl(flash)

    impl._run_flash_attention(
        q=torch.empty(5, 14, 64),
        k=torch.empty(1, 2, 64, 64),
        v=torch.empty(1, 2, 64, 64),
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen_q=5,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )

    assert calls[0]["max_seqlen_q"] == 6


@pytest.mark.parametrize('mode',('eligible','decode','fp8','capacity'))
def test_opt_in_prefill_composition_keeps_decode_fp8_and_capacity_fallbacks(monkeypatch,mode):
    flash=load_flash_attention(monkeypatch)
    from hcu_vllm_das.attention import prefill_gather
    impl=make_impl(flash)
    impl._prefill_gather_enabled=True
    calls=[]
    monkeypatch.setattr(flash,'_vendor_varlen_func',lambda:lambda **kw:calls.append(('vendor',kw)))
    monkeypatch.setattr(prefill_gather,'run',lambda vendor,kw,layout:calls.append(('gather',kw)))
    nq=1 if mode=='decode' else 4
    if mode=='fp8':
        impl.kv_cache_dtype='fp8_e5m2'
        scales=ModuleType('hcu_vllm_das.attention.e5m2_scales')
        scales.run_e5m2_attention=lambda vendor,kw:calls.append(('fp8',kw))
        monkeypatch.setitem(sys.modules,scales.__name__,scales)
    cache_dtype=torch.uint8 if mode=='fp8' else torch.bfloat16
    impl._run_flash_attention(q=torch.empty(nq,4,64,dtype=torch.bfloat16),
        out=torch.empty(nq,4,64,dtype=torch.bfloat16),
        k=torch.empty(1,2,64,64,dtype=cache_dtype),v=torch.empty(1,2,64,64,dtype=cache_dtype),
        cu_seqlens_q=torch.tensor([0,nq],dtype=torch.int32),
        seqused_k=torch.tensor([64],dtype=torch.int32),max_seqlen_q=nq,
        max_seqlen_k=262145 if mode=='capacity' else 64,
        block_table=torch.zeros((1,1),dtype=torch.int32))
    assert len(calls)==1
    assert calls[0][0]==('gather' if mode=='eligible' else 'fp8' if mode=='fp8' else 'vendor')
    assert calls[0][1]['max_seqlen_q']==(5 if mode in ('capacity','fp8') else nq)


@pytest.mark.parametrize("query_len", (1, 3, 16))
def test_e5m2_paged_descale_adapter_handles_decode_and_prefill(monkeypatch, query_len):
    flash = load_flash_attention(monkeypatch)
    calls = []
    adapter = ModuleType("hcu_vllm_das.attention.e5m2_scales")
    adapter.run_e5m2_attention = lambda vendor, kwargs: calls.append(
        ("descale", kwargs)
    )
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(("native", kwargs)),
    )
    impl = make_impl(flash, kv_cache_dtype="fp8_e5m2")
    impl._run_flash_attention(
        q=torch.empty(query_len, 4, 64),
        k=torch.empty(1, 2, 64, 64, dtype=torch.uint8),
        v=torch.empty(1, 2, 64, 64, dtype=torch.uint8),
        cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32),
        max_seqlen_q=query_len,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    assert len(calls) == 1
    assert calls[0][0] == "descale"
    assert calls[0][1]["k"].dtype == torch.float8_e5m2


@pytest.mark.parametrize(
    ("max_seqlen_q", "query_tokens", "batch_size", "has_block_table"),
    (
        (1, 1, 1, True),
        (16, 16, 1, True),
        (5, 9, 2, True),
        (5, 5, 1, False),
    ),
)
def test_varlen_boundary_preserves_non_rectangular_or_non_paged_shapes(
    monkeypatch,
    max_seqlen_q,
    query_tokens,
    batch_size,
    has_block_table,
):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = make_impl(flash)

    impl._run_flash_attention(
        q=torch.empty(query_tokens, 4, 64),
        k=torch.empty(1, 2, 64, 64),
        v=torch.empty(1, 2, 64, 64),
        cu_seqlens_q=torch.arange(
            batch_size + 1,
            dtype=torch.int32,
        ),
        max_seqlen_q=max_seqlen_q,
        block_table=(
            torch.zeros((batch_size, 1), dtype=torch.int32) if has_block_table else None
        ),
    )

    assert calls[0]["max_seqlen_q"] == max_seqlen_q


@pytest.mark.parametrize("name", ("dynamic_causal", "mask_mod", "aux_tensors"))
def test_varlen_boundary_rejects_active_fa4_options(monkeypatch, name):
    flash = load_flash_attention(monkeypatch)
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: pytest.fail("vendor must not run"),
    )
    impl = make_impl(flash)

    with pytest.raises(NotImplementedError, match=name):
        impl._run_flash_attention(q="q", k="k", v="v", **{name: object()})
