# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime contracts for PP partitioning and maintained HCU attention paths.

Legacy custom FlashAttention code remains present but is outside this release's
supported runtime contract; functional attention coverage is scoped accordingly.
"""

from __future__ import annotations

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.framework_opt import patch_distributed_utils
from vllm_hcu.patch.worker.op_opt import patch_triton_unified_attention
from vllm_hcu.platforms import envs as hcu_envs


def _load_hcu_flash_attention_module(monkeypatch: pytest.MonkeyPatch):
    hcu_ops = ModuleType("vllm_hcu.hcu_ops")
    hcu_ops.__spec__ = ModuleSpec(hcu_ops.__name__, loader=None)
    monkeypatch.setitem(sys.modules, hcu_ops.__name__, hcu_ops)

    flash_attn_extension = ModuleType("flash_attn")
    flash_attn_extension.__spec__ = ModuleSpec(
        flash_attn_extension.__name__,
        loader=None,
    )
    def layout_entrypoint(
        q=None,
        k=None,
        v=None,
        *,
        layout="bshd",
        **kwargs,
    ):
        del q, k, v, layout, kwargs

    for symbol in (
        "flash_attn_varlen_func",
        "hg_flash_attn_varlen_func",
        "varlen_fwd_unified",
    ):
        setattr(flash_attn_extension, symbol, layout_entrypoint)
    flash_attn_extension.vllm_flash_attn_varlen_func = (
        lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, flash_attn_extension.__name__, flash_attn_extension)

    return importlib.import_module("vllm_hcu.v1.attention.backends.flash_attn")


def _load_hcu_fa_utils_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kv_cache_layout: str,
    missing_layout_on: str | None = None,
):
    """Load the real HCU FA boundary against observable kernel doubles."""

    from vllm.v1.attention.backends import utils as attention_utils

    monkeypatch.setattr(
        attention_utils,
        "get_kv_cache_layout",
        lambda: kv_cache_layout,
    )

    hcu_ops = ModuleType("vllm_hcu.hcu_ops")
    hcu_ops.__spec__ = ModuleSpec(hcu_ops.__name__, loader=None)
    monkeypatch.setitem(sys.modules, hcu_ops.__name__, hcu_ops)

    calls: dict[str, list[dict[str, object]]] = {}
    flash_attn_extension = ModuleType("flash_attn")
    flash_attn_extension.__spec__ = ModuleSpec(
        flash_attn_extension.__name__,
        loader=None,
    )

    def make_entrypoint(name: str):
        calls[name] = []
        if name == missing_layout_on:

            def entrypoint(q=None, k=None, v=None, **kwargs):
                calls[name].append(dict(kwargs))

        else:

            def entrypoint(
                q=None,
                k=None,
                v=None,
                *,
                layout="unrouted",
                **kwargs,
            ):
                calls[name].append({"layout": layout, **kwargs})

        return entrypoint

    for symbol in (
        "flash_attn_varlen_func",
        "vllm_flash_attn_varlen_func",
        "hg_flash_attn_varlen_func",
        "varlen_fwd_unified",
    ):
        setattr(flash_attn_extension, symbol, make_entrypoint(symbol))
    monkeypatch.setitem(sys.modules, flash_attn_extension.__name__, flash_attn_extension)
    monkeypatch.delitem(
        sys.modules,
        "vllm_hcu.v1.attention.backends.fa_utils",
        raising=False,
    )

    module = importlib.import_module("vllm_hcu.v1.attention.backends.fa_utils")
    return module, calls


@pytest.mark.parametrize(
    ("kv_cache_layout", "expected_fa_layout"),
    [("HND", "bhsd"), ("NHD", "bshd")],
)
@pytest.mark.parametrize(
    "entrypoint_name",
    [
        "flash_attn_varlen_func",
        "hg_flash_attn_varlen_func",
        "varlen_fwd_unified",
    ],
)
def test_hcu_fa_entrypoints_map_kv_cache_layout_to_kernel_layout(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_layout: str,
    expected_fa_layout: str,
    entrypoint_name: str,
) -> None:
    fa_utils, calls = _load_hcu_fa_utils_module(
        monkeypatch,
        kv_cache_layout=kv_cache_layout,
    )

    getattr(fa_utils, entrypoint_name)(q="q", k="k", v="v")

    assert calls[entrypoint_name] == [{"layout": expected_fa_layout}]


def test_hcu_fa_boundary_rejects_interface_without_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="hg_flash_attn_varlen_func.*layout"):
        _load_hcu_fa_utils_module(
            monkeypatch,
            kv_cache_layout="HND",
            missing_layout_on="hg_flash_attn_varlen_func",
        )


def test_pp_size_one_ignores_invalid_manual_partition(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, int, int]] = []

    def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int):
        calls.append((num_hidden_layers, pp_rank, pp_size))
        raise ValueError("invalid VLLM_PP_LAYER_PARTITION")

    module = ModuleType(patch_distributed_utils.TARGET_MODULE)
    module.get_pp_indices = get_pp_indices

    assert patch_distributed_utils.apply_to_module(module) is True
    assert patch_distributed_utils.apply_to_module(module) is False
    assert module.get_pp_indices(61, 0, 1) == (0, 61)
    assert calls == []
    with pytest.raises(ValueError, match="invalid VLLM_PP_LAYER_PARTITION"):
        module.get_pp_indices(61, 0, 2)
    assert calls == [(61, 0, 2)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "varlen"),
        ("VLLM_HCU_USE_FLASH_ATTN", "classic"),
        ("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "cutlass"),
        ("VLLM_HCU_USE_FLASH_ATTN_VARLEN", "varlen"),
    ],
)
def test_flash_attention_mode_legacy_priority_and_default(
    monkeypatch: pytest.MonkeyPatch, name: str | None, expected: str
):
    for variable in (
        "VLLM_HCU_USE_FLASH_ATTN",
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED",
        "VLLM_HCU_USE_FLASH_ATTN_VARLEN",
        "VLLM_HCU_USE_CUSTOM_FLASH_ATTN",
    ):
        monkeypatch.delenv(variable, raising=False)
    if name is not None:
        monkeypatch.setenv(name, "1")
    assert hcu_envs.resolve_hcu_flash_attn_mode(None) == expected


def test_explicit_flash_attention_mode_wins_over_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_USE_FLASH_ATTN", "1")
    assert hcu_envs.resolve_hcu_flash_attn_mode("unified") == "cutlass"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"hcu_flash_attn_mode": 1}, TypeError),
        ({"hcu_flash_attn_mode": "future"}, ValueError),
    ],
)
def test_platform_flash_attention_mode_rejects_invalid_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    error: type[Exception],
):
    import torch
    import vllm.config as vllm_config_module

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Properties", (), {"gcnArchName": "gfx936"})(),
    )
    import vllm_hcu.platforms.hcu as hcu_module

    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: {"additional_config": {"hcu": payload}},
    )
    with pytest.raises(error):
        hcu_module.get_hcu_flash_attn_mode()


@pytest.mark.parametrize(
    ("mode", "expected_shape", "expected_block_dim"),
    [
        ("cutlass", (3, 2, 64, 4, 128), 0),
        ("classic", (3, 2, 64, 4, 128), 0),
        ("varlen", (3, 2, 64, 4, 128), 0),
    ],
)
def test_flash_attention_kv_cache_contract_follows_resolved_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_shape: object,
    expected_block_dim: int,
):
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: mode)
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    backend = flash_attn.HcuFlashAttentionBackend

    assert backend.get_kv_cache_shape(3, 64, 4, 128) == expected_shape
    assert backend.get_kv_cache_block_dim(64, 4, 128) == expected_block_dim
    assert backend.indexes_kv_by_block_stride() is True
    assert backend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
    assert backend.get_kv_cache_stride_order(True) == (1, 0, 2, 3, 4, 5)


def test_cutlass_block_first_hnd_stride_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "cutlass")
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "HND")
    backend = flash_attn.HcuFlashAttentionBackend

    assert backend.get_kv_cache_shape(3, 64, 4, 128) == (3, 2, 64, 4, 128)
    assert backend.get_kv_cache_block_dim(64, 4, 128) == 0
    assert backend.get_kv_cache_stride_order() == (0, 1, 3, 2, 4)
    assert backend.get_kv_cache_stride_order(True) == (1, 4, 0, 2, 3, 5)
    assert backend.indexes_kv_by_block_stride() is True


@pytest.mark.parametrize(
    ("layout", "kv_cache_dtype", "expected_writer"),
    [
        ("NHD", "auto", "aiter"),
        ("HND", "auto", "triton"),
        ("NHD", "fp8_e4m3", "hcu"),
        ("HND", "fp8_e4m3", "hcu"),
        ("NHD", "fp8_e5m2", "hcu"),
        ("HND", "fp8_e5m2", "hcu"),
    ],
)
def test_flash_cache_writer_dispatches_by_physical_layout(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    kv_cache_dtype: str,
    expected_writer: str,
) -> None:
    fa_utils = importlib.import_module(
        "vllm_hcu.v1.attention.backends.fa_utils"
    )
    calls: list[tuple[str, tuple[object, ...]]] = []
    aiter_cache_module = ModuleType("aiter.ops.cache")
    aiter_cache_module.reshape_and_cache_flash = (
        lambda *args: calls.append(("aiter", args))
    )
    monkeypatch.setitem(sys.modules, "aiter.ops.cache", aiter_cache_module)

    triton_module_name = (
        "vllm.v1.attention.ops.triton_reshape_and_cache_flash"
    )
    triton_module = ModuleType(triton_module_name)
    triton_module.triton_reshape_and_cache_flash = (
        lambda *args: calls.append(("triton", args))
    )
    monkeypatch.setitem(sys.modules, triton_module_name, triton_module)
    monkeypatch.setattr(
        torch.ops.hcu_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(("hcu", args)),
        raising=False,
    )

    key = torch.zeros(2, 1, 8)
    value = torch.ones_like(key)
    if layout == "HND":
        key_cache = torch.empty(3, 2, 4, 8).permute(0, 2, 1, 3)
        value_cache = torch.empty(3, 2, 4, 8).permute(0, 2, 1, 3)
    else:
        key_cache = torch.empty(3, 4, 2, 8)
        value_cache = torch.empty_like(key_cache)
    slots = torch.tensor([0, 5])
    scale = torch.tensor(1.0)
    fa_utils.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        kv_cache_dtype,
        scale,
        scale,
    )

    assert len(calls) == 1
    assert calls[0][0] == expected_writer
    assert calls[0][1][:5] == (key, value, key_cache, value_cache, slots)


def test_hcu_flash_attention_mm_prefix_is_explicitly_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    backend = flash_attn.HcuFlashAttentionBackend

    assert backend.supports_mm_prefix() is False
    reason = backend.supports_combination(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        device_capability=flash_attn.DeviceCapability(9, 0),
    )
    assert reason is not None and "mm_prefix" in reason


def test_hcu_flash_attention_rswa_is_explicitly_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    config = SimpleNamespace(model_config=SimpleNamespace(rswa_window=128))

    with pytest.raises(NotImplementedError, match="R-SWA"):
        flash_attn.FlashAttentionMetadataBuilder(
            None,
            [],
            config,
            torch.device("cpu"),
        )


def _bare_flash_attention_builder(flash_attn, *, sliding_window: int = 32):
    builder = object.__new__(flash_attn.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.aot_sliding_window = (sliding_window - 1, 0)
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    builder.kv_cache_dtype = torch.float16
    builder.num_heads_q = 4
    builder.num_heads_kv = 2
    builder.headdim = 64
    builder.block_size = 64
    builder.dcp_world_size = 1
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.device = torch.device("cpu")
    builder.kv_cache_spec = SimpleNamespace(sliding_window=sliding_window)
    return builder


def test_gqa_pcp_plan_reaches_flash_attention_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the per-step plan makes FlashAttention read local-row metadata."""

    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    builder = _bare_flash_attention_builder(flash_attn)
    plan = object()

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_common_attention_metadata(causal=True),
        pcp_plan=plan,
    )

    assert flash_attn.FlashAttentionImpl.supports_pcp is True
    assert metadata.pcp_plan is plan


def _common_attention_metadata(*, causal: bool | torch.Tensor):
    return SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=4,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        causal=causal,
    )


@pytest.mark.parametrize(
    ("causal", "expected_window"),
    [(True, (31, 0)), (False, (31, 31))],
)
def test_hcu_flash_attention_aligns_aot_and_forward_window_semantics(
    monkeypatch: pytest.MonkeyPatch,
    causal: bool,
    expected_window: tuple[int, int],
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    builder = _bare_flash_attention_builder(flash_attn)
    scheduler_calls: list[dict[str, object]] = []

    def get_scheduler_metadata(**kwargs):
        scheduler_calls.append(kwargs)
        return torch.zeros(1, dtype=torch.int32)

    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", get_scheduler_metadata)

    metadata = builder.build(
        0,
        _common_attention_metadata(causal=causal),
    )

    assert scheduler_calls[0]["window_size"] == expected_window
    assert metadata.sliding_window == expected_window


def test_hcu_flash_attention_rejects_per_request_causal_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    builder = _bare_flash_attention_builder(flash_attn)

    with pytest.raises(NotImplementedError, match="Per-request causal masks"):
        builder.build(
            0,
            _common_attention_metadata(
                causal=torch.tensor([True], dtype=torch.bool),
            ),
        )


@pytest.mark.parametrize(
    ("metadata_window", "fallback_window", "causal", "expected"),
    [
        (None, (7, 0), True, [7, 0]),
        (None, (7, 0), False, [7, 7]),
        ((31, 31), (7, 0), True, [31, 31]),
        (None, (-1, -1), False, [-1, -1]),
        (None, None, False, None),
    ],
)
def test_hcu_flash_attention_resolves_native_window_for_each_causal_mode(
    monkeypatch: pytest.MonkeyPatch,
    metadata_window: tuple[int, int] | None,
    fallback_window: tuple[int, int] | None,
    causal: bool,
    expected: list[int] | None,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)

    assert (
        flash_attn._get_native_sliding_window(
            metadata_window,
            fallback_window,
            causal,
        )
        == expected
    )


@pytest.mark.parametrize("attn_type_name", ["ENCODER", "ENCODER_ONLY"])
def test_hcu_flash_attention_encoder_window_is_symmetric(
    monkeypatch: pytest.MonkeyPatch,
    attn_type_name: str,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **kwargs: 3)
    monkeypatch.setattr(flash_attn, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        flash_attn,
        "flash_attn_supports_quant_query_input",
        lambda: False,
    )
    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "classic")

    impl = flash_attn.FlashAttentionImpl(
        num_heads=1,
        head_size=64,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=8,
        kv_cache_dtype="auto",
        attn_type=getattr(flash_attn.AttentionType, attn_type_name),
    )

    assert impl.sliding_window == (7, 7)


def test_hcu_flash_attention_backend_rejects_pcp_with_dcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attention backend must fail closed if config validation is bypassed."""

    from vllm.distributed import parallel_state

    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    monkeypatch.setattr(
        parallel_state,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda **kwargs: 3)
    monkeypatch.setattr(flash_attn, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        flash_attn,
        "flash_attn_supports_quant_query_input",
        lambda: False,
    )
    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "classic")

    with pytest.raises(ValueError, match="does not support decode context"):
        flash_attn.FlashAttentionImpl(
            num_heads=1,
            head_size=64,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )


@pytest.mark.parametrize(
    ("metadata_window", "expected_window"),
    [(None, [7, 0]), ((31, 31), [31, 31])],
)
def test_hcu_flash_attention_forward_uses_metadata_window_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
    metadata_window: tuple[int, int] | None,
    expected_window: list[int],
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    impl.attn_type = flash_attn.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    impl.num_kv_heads = 1
    impl.supports_quant_query_input = False
    impl.dcp_world_size = 1
    impl.scale = 1.0
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0.0
    impl.sinks = None
    impl.sliding_window = (7, 0)

    calls: list[dict[str, object]] = []

    def native_forward(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "classic")
    monkeypatch.setattr(
        flash_attn,
        "canonicalize_singleton_dim_strides",
        lambda tensor: tensor,
    )
    monkeypatch.setattr(flash_attn, "hg_flash_attn_varlen_func", native_forward)

    metadata = SimpleNamespace(
        num_actual_tokens=1,
        use_cascade=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        scheduler_metadata=None,
        causal=True,
        sliding_window=metadata_window,
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    query = torch.zeros(1, 1, 2)

    impl.forward(
        layer,
        query,
        query,
        query,
        torch.zeros(1, 2, 1, 1, 2),
        metadata,
        torch.empty_like(query),
    )

    assert calls[0]["window_size"] == expected_window


@pytest.mark.parametrize(
    ("kv_cache_layout", "expected_fa_layout"),
    [("HND", "bhsd"), ("NHD", "bshd")],
)
def test_hcu_flash_attention_varlen_mode_routes_layout_to_native_interface(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_layout: str,
    expected_fa_layout: str,
) -> None:
    _, calls = _load_hcu_fa_utils_module(
        monkeypatch,
        kv_cache_layout=kv_cache_layout,
    )
    monkeypatch.delitem(
        sys.modules,
        "vllm_hcu.v1.attention.backends.flash_attn",
        raising=False,
    )
    flash_attn = importlib.import_module(
        "vllm_hcu.v1.attention.backends.flash_attn"
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    impl.attn_type = flash_attn.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    impl.num_kv_heads = 1
    impl.supports_quant_query_input = False
    impl.dcp_world_size = 1
    impl.scale = 1.0
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0.0
    impl.sinks = None
    impl.sliding_window = (-1, -1)

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "canonicalize_singleton_dim_strides",
        lambda tensor: tensor,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        use_cascade=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        scheduler_metadata=None,
        causal=True,
        sliding_window=None,
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    query = torch.zeros(1, 1, 2)

    impl.forward(
        layer,
        query,
        query,
        query,
        torch.zeros(1, 2, 1, 1, 2),
        metadata,
        torch.empty_like(query),
    )

    assert len(calls["flash_attn_varlen_func"]) == 1
    assert calls["flash_attn_varlen_func"][0]["layout"] == expected_fa_layout
    assert calls["hg_flash_attn_varlen_func"] == []
    assert calls["varlen_fwd_unified"] == []


@pytest.mark.parametrize(
    "kernel_result",
    [("out", "lse"), ("out", "lse", "attention-probabilities")],
)
def test_hcu_native_varlen_lse_adapter_accepts_two_or_three_results(
    monkeypatch: pytest.MonkeyPatch,
    kernel_result: tuple[str, ...],
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    monkeypatch.setattr(
        flash_attn,
        "_select_flash_attn_varlen_func",
        lambda: lambda **kwargs: kernel_result,
    )

    assert flash_attn._call_select_flash_attn_with_lse(q="query") == (
        "out",
        "lse",
    )


def test_hcu_native_varlen_lse_adapter_requests_nonpaged_lse_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    calls: list[dict[str, object]] = []

    def native(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("return_attn_probs"):
            return "out-only"
        return "out", "lse", "attention-probabilities"

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "_select_flash_attn_varlen_func",
        lambda: native,
    )

    assert flash_attn._call_select_flash_attn_with_lse(
        q="query",
        block_table=None,
        return_softmax_lse=True,
    ) == ("out", "lse")
    assert calls == [
        {
            "q": "query",
            "block_table": None,
            "return_softmax_lse": True,
            "return_attn_probs": True,
        }
    ]


def test_hcu_native_varlen_lse_adapter_bypasses_paged_decode_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    calls: list[dict[str, object]] = []

    def native(**kwargs):
        calls.append(kwargs)
        if kwargs["window_size"] == [-1, -1]:
            return "out-only"
        return "out", "lse"

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "_select_flash_attn_varlen_func",
        lambda: native,
    )

    assert flash_attn._call_select_flash_attn_with_lse(
        q="query",
        block_table="paged-cache",
        max_seqlen_k=128,
        window_size=[-1, -1],
        return_softmax_lse=True,
    ) == ("out", "lse")
    assert calls[0]["window_size"] == [128, -1]


def test_hcu_varlen_cascade_consumes_two_value_lse_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    native_calls: list[dict[str, object]] = []
    merge_calls: list[tuple[object, ...]] = []

    def native(**kwargs):
        native_calls.append(kwargs)
        query = kwargs["q"]
        return torch.zeros_like(query), torch.zeros(
            query.shape[-2], query.shape[0]
        )

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "_select_flash_attn_varlen_func",
        lambda: native,
    )
    monkeypatch.setattr(
        flash_attn,
        "merge_attn_states",
        lambda *args: merge_calls.append(args),
    )
    query = torch.zeros(2, 1, 64)
    output = torch.empty_like(query)
    cache = torch.zeros(4, 64, 1, 64)

    flash_attn.cascade_attention(
        output,
        query,
        cache,
        cache,
        cu_query_lens=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        cu_prefix_query_lens=torch.tensor([0, 2], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([128], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([64, 64], dtype=torch.int32),
        max_kv_len=192,
        softmax_scale=1.0,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        block_table=torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int32),
        common_prefix_len=128,
        max_num_splits=0,
        fa_version=3,
    )

    assert len(native_calls) == 2
    assert [call["window_size"] for call in native_calls] == [
        [128, -1],
        [64, -1],
    ]
    assert len(merge_calls) == 1


def test_hcu_varlen_dcp_requests_lse_from_paged_and_nonpaged_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    native_calls: list[dict[str, object]] = []
    merge_calls: list[tuple[object, ...]] = []

    def native(**kwargs):
        native_calls.append(kwargs)
        if kwargs.get("block_table") is None:
            assert kwargs["return_attn_probs"] is True
        else:
            assert kwargs["window_size"] == [8, -1]
        query = kwargs["q"]
        return torch.zeros_like(query), torch.zeros(
            query.shape[-2], query.shape[0]
        )

    class _DcpGroup:
        @staticmethod
        def all_gather(tensor, dim):
            assert dim == 1
            return tensor

    class _Workspace:
        @staticmethod
        def get_simultaneous(spec):
            shape, dtype = spec
            return (torch.empty(shape, dtype=dtype),)

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "_select_flash_attn_varlen_func",
        lambda: native,
    )
    monkeypatch.setattr(flash_attn, "get_dcp_group", lambda: _DcpGroup())
    monkeypatch.setattr(
        flash_attn,
        "current_workspace_manager",
        lambda: _Workspace(),
    )
    monkeypatch.setattr(
        flash_attn,
        "merge_attn_states",
        lambda *args: merge_calls.append(args),
    )
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    impl.num_heads = 1
    impl.dcp_world_size = 1
    impl.head_size = 64
    impl._dcp_dtype = torch.float32
    impl.scale = 1.0
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0.0
    impl.dcp_combine = lambda out, lse, group, return_lse: (out, lse)
    query = torch.zeros(1, 1, 64)
    cache = torch.zeros(2, 64, 1, 64)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        sliding_window=None,
        causal=True,
        dcp_context_kv_lens=torch.tensor([8], dtype=torch.int32),
        max_dcp_context_kv_len=8,
        scheduler_metadata=None,
    )

    impl._forward_with_dcp(
        query,
        query,
        query,
        cache,
        cache,
        torch.empty_like(query),
        metadata,
    )

    assert len(native_calls) == 2
    assert native_calls[0]["block_table"] is metadata.block_table
    assert native_calls[1].get("block_table") is None
    assert len(merge_calls) == 1


def test_block_first_kv_update_passes_axis_one_views_to_aiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.attn_type = flash_attn.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "cutlass")
    monkeypatch.setattr(
        flash_attn,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
    )

    key = torch.zeros(2, 1, 8)
    value = torch.ones_like(key)
    cache = torch.empty(3, 2, 4, 1, 8)
    slots = torch.tensor([0, 5])
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )

    impl.do_kv_cache_update(layer, key, value, cache, slots)

    assert len(calls) == 1
    writer_key, writer_value, key_cache, value_cache, writer_slots, *_ = calls[0]
    assert writer_key is key
    assert writer_value is value
    assert writer_slots is slots
    expected_key_cache, expected_value_cache = cache.unbind(1)
    assert key_cache.stride() == expected_key_cache.stride()
    assert value_cache.stride() == expected_value_cache.stride()
    assert key_cache.storage_offset() == expected_key_cache.storage_offset()
    assert value_cache.storage_offset() == expected_value_cache.storage_offset()


def test_pcp_kv_update_does_not_retain_gathered_tokens_without_dcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DCP=1 writes gathered K/V immediately and must not retain it per layer."""

    flash_attn = _load_hcu_flash_attention_module(monkeypatch)
    impl = object.__new__(flash_attn.FlashAttentionImpl)
    impl.attn_type = flash_attn.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    impl.use_pcp = True
    impl.pcp_world_size = 2
    impl.dcp_world_size = 1
    impl._pcp_kv = {}

    class _PCPGroup:
        @staticmethod
        def all_gather(tensor, dim):
            assert dim == 0
            assert tensor.is_contiguous()
            return torch.cat((tensor, tensor + 10), dim=0)

    writes: list[tuple[object, ...]] = []
    monkeypatch.setattr(flash_attn, "get_pcp_group", lambda: _PCPGroup())
    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: "varlen")
    monkeypatch.setattr(
        flash_attn,
        "reshape_and_cache_flash",
        lambda *args: writes.append(args),
    )
    qkv = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1)
    key = qkv[:, 0]
    value = (qkv + 100)[:, 1]
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    cache = torch.empty(4, 2, 1, 1, 1)
    slots = torch.tensor([0, 1, 2, 3])
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )

    impl.do_kv_cache_update(layer, key, value, cache, slots)

    gathered_key, gathered_value = writes[0][0], writes[0][1]
    assert gathered_key.flatten().tolist() == [0.0, 2.0, 10.0, 12.0]
    assert gathered_value.flatten().tolist() == [101.0, 103.0, 111.0, 113.0]
    assert layer.layer_name not in impl._pcp_kv


def test_platform_block_copy_and_swap_use_axis_zero():
    from vllm_hcu.platforms.hcu import HCUPlatform

    src = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    inserted = torch.full_like(src, -1)
    host = torch.full_like(src, -1)
    src_blocks = torch.tensor([3, 1])
    dst_blocks = torch.tensor([0, 2])

    HCUPlatform.insert_blocks_to_device(src, inserted, src_blocks, dst_blocks)
    HCUPlatform.swap_out_blocks_to_host(src, host, src_blocks, dst_blocks)

    torch.testing.assert_close(inserted[dst_blocks], src[src_blocks])
    torch.testing.assert_close(host[dst_blocks], src[src_blocks])
    assert torch.count_nonzero(inserted[[1, 3]] + 1) == 0
    assert torch.count_nonzero(host[[1, 3]] + 1) == 0


class _FakeKernel:
    def __init__(self) -> None:
        self.launches: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def __getitem__(self, grid: object):
        def launch(*args: object, **kwargs: object):
            self.launches.append((grid, args, kwargs))
            return "launched"

        return launch


def _unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    rswa_prefix_lens=None,
    rswa_window=None,
    use_alibi_sqrt=False,
    kv_quant_mode=None,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback=-1,
    use_td=False,
    mm_prefix_clamp_sliding_window=False,
):
    del (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        seq_threshold_3D,
        num_par_softmax_segments,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        mm_prefix_range,
        rswa_prefix_lens,
        rswa_window,
        use_alibi_sqrt,
        kv_quant_mode,
        k_scale_cache,
        v_scale_cache,
        chunk_lookback,
        use_td,
        mm_prefix_clamp_sliding_window,
    )


def test_unified_attention_proxy_forces_single_stage_and_preserves_other_kwargs():
    kernel = _FakeKernel()
    module = ModuleType(patch_triton_unified_attention.TARGET_MODULE)
    module.unified_attention = _unified_attention
    module.kernel_unified_attention = kernel

    assert patch_triton_unified_attention.apply_to_module(module) is True
    assert patch_triton_unified_attention.apply_to_module(module) is False
    launcher = module.kernel_unified_attention[(3, 7)]
    assert launcher("q", num_stages=2, num_warps=8) == "launched"
    assert kernel.launches == [
        ((3, 7), ("q",), {"num_stages": 1, "num_warps": 8})
    ]
