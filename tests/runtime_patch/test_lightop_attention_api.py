# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


def _load_fused_qkv_impl():
    """Load the lazy LightOp boundary without importing the full vLLM adapter."""
    source = (
        REPO / "vllm_hcu/model_executor/layers/attention_runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "fused_qkv_split_rmsnorm_rope_kv_store_impl"
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "split_kv_cache": lambda *_args, **_kwargs: None,
        "logger": SimpleNamespace(warning_once=lambda *_args, **_kwargs: None),
    }
    exec(compile(module, "attention_runtime_contract", "exec"), namespace)
    return namespace[function.name]


def test_fused_qkv_runtime_uses_categorized_lightop_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A categorized-only LightOp package remains usable at the lazy boundary."""
    calls: list[tuple[object, ...]] = []

    def split_qkv(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            torch.ones((1, 1, 2)),
            torch.ones((1, 1, 2)),
            torch.ones((1, 1, 2)),
        )

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    attention = ModuleType("lightop.attention")
    attention.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant = split_qkv
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                slot_mapping={},
                no_compile_layers={"layer": SimpleNamespace(kv_cache=None)},
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.platforms.hcu",
        SimpleNamespace(get_hcu_flash_attn_mode=lambda: "other"),
    )

    result = _load_fused_qkv_impl()(
        torch.ones((1, 6)),
        torch.tensor([0]),
        "layer",
        "auto",
        torch.ones((1, 2)),
        torch.ones(2),
        torch.ones(2),
        1e-5,
        2,
        2,
        2,
        2,
        16,
    )

    assert len(calls) == 1
    assert all(tensor.shape == (1, 1, 2) for tensor in result)


def test_fused_qkv_runtime_rejects_top_level_lightop_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy boundary must not retry the removed top-level export."""
    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    lightop.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant = (
        lambda *_args, **_kwargs: pytest.fail("legacy top-level kernel called")
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.delitem(sys.modules, "lightop.attention", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                slot_mapping={},
                no_compile_layers={"layer": SimpleNamespace(kv_cache=None)},
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.platforms.hcu",
        SimpleNamespace(get_hcu_flash_attn_mode=lambda: "other"),
    )

    with pytest.raises(ImportError):
        _load_fused_qkv_impl()(
            torch.ones((1, 6)),
            torch.tensor([0]),
            "layer",
            "auto",
            torch.ones((1, 2)),
            torch.ones(2),
            torch.ones(2),
            1e-5,
            2,
            2,
            2,
            2,
            16,
        )


def _runtime():
    return importlib.import_module("vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse")


@pytest.mark.parametrize(
    ("aiter_enabled", "force_aiter_triton"),
    ((False, False), (True, False), (False, True)),
)
@pytest.mark.parametrize("is_gfx938", (False, True))
def test_sparse_mla_master_off_prefill_uses_boltops(
    monkeypatch: pytest.MonkeyPatch,
    aiter_enabled: bool,
    force_aiter_triton: bool,
    is_gfx938: bool,
) -> None:
    """The master-off prefill route never reaches AITER or LightOp."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: aiter_enabled)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: is_gfx938)
    monkeypatch.setattr(
        runtime,
        "mqa_logits_module",
        lambda: pytest.fail("master-off prefill queried AITER MQA"),
    )
    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: pytest.fail("master-off prefill queried LightOp MQA"),
    )

    calls: list[tuple[object, ...]] = []
    output = object()

    def boltops_mqa(
        q,
        k,
        scales,
        normalized_weights,
        starts,
        ends,
        *,
        D_out=None,
        backend=None,
    ):
        calls.append(
            (
                q,
                k,
                scales,
                normalized_weights,
                starts,
                ends,
                D_out,
                backend,
            )
        )
        return output

    monkeypatch.setattr(
        runtime,
        "_load_boltops_mqa_logits",
        lambda: boltops_mqa,
        raising=False,
    )
    q = torch.ones((4, 1, 2), dtype=torch.bfloat16)
    k = torch.ones((3, 2), dtype=torch.bfloat16)
    scales = torch.ones(3)
    weights = torch.arange(8, dtype=torch.float16).reshape(2, 4).transpose(0, 1)
    starts = torch.zeros(4, dtype=torch.int32)
    ends = torch.full((4,), 3, dtype=torch.int32)

    result = runtime.rocm_fp8_mqa_logits(
        q,
        (k, scales),
        weights,
        starts,
        ends,
        force_aiter_triton=force_aiter_triton,
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][1] is k
    assert calls[0][2] is None
    assert calls[0][3].dtype is torch.float32
    assert calls[0][3].is_contiguous()
    assert torch.equal(calls[0][3], weights.float().contiguous())
    assert calls[0][4] is starts
    assert calls[0][5] is ends
    assert calls[0][6:] == (None, None)


@pytest.mark.parametrize("is_gfx938", (False, True))
def test_sparse_mla_master_off_prefill_uses_torch_when_boltops_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    is_gfx938: bool,
) -> None:
    """A missing optional BoltOPs package retains the portable reference."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: is_gfx938)
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_load_boltops_mqa_logits",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: pytest.fail("master-off fallback queried LightOp MQA"),
    )
    calls: list[tuple[object, ...]] = []
    output = object()

    def torch_reference(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(runtime, "fp8_mqa_logits_torch", torch_reference)
    q = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    kv = (torch.ones((1, 2), dtype=torch.bfloat16), torch.ones(1))
    weights = torch.ones((1, 1))
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.ones(1, dtype=torch.int32)

    result = runtime.rocm_fp8_mqa_logits(q, kv, weights, starts, ends)

    assert result is output
    expected_kv = (kv[0], None)
    assert calls == [(q, expected_kv, weights, starts, ends)]


@pytest.mark.parametrize("boltops_available", (False, True))
def test_sparse_mla_master_off_preserves_fp8_scale_off_gfx938(
    monkeypatch: pytest.MonkeyPatch,
    boltops_available: bool,
) -> None:
    """GLM5Next kpool stays FP8 even when the device is not gfx938."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)

    def boltops_reference(q, k, scale, weights, starts, ends, **_kwargs):
        return runtime.fp8_mqa_logits_torch(
            q,
            (k, scale),
            weights,
            starts,
            ends,
        )

    monkeypatch.setattr(
        runtime,
        "_load_boltops_mqa_logits",
        lambda: boltops_reference if boltops_available else None,
    )
    q = torch.ones((1, 32, 128), dtype=torch.float8_e4m3fn)
    k = torch.stack(
        (
            torch.ones(128, dtype=torch.float8_e4m3fn),
            torch.full((128,), 2.0, dtype=torch.float8_e4m3fn),
        )
    )
    scales = torch.tensor([4.0, 1.0], dtype=torch.float32)
    weights = torch.zeros((1, 32), dtype=torch.float32)
    weights[0, 0] = 1.0

    logits = runtime.rocm_fp8_mqa_logits(
        q,
        (k, scales),
        weights,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        force_aiter_triton=True,
    )

    assert torch.equal(logits, torch.tensor([[512.0, 256.0]]))
    assert logits.argmax(dim=-1).item() == 0


def test_sparse_mla_boltops_prefill_rejects_public_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present but incompatible BoltOPs install must fail visibly."""
    runtime = _runtime()
    assert hasattr(runtime, "_load_boltops_mqa_logits"), (
        "the sparse-indexer runtime must own a lazy BoltOPs MQA resolver"
    )
    resolver = runtime._load_boltops_mqa_logits
    resolver.cache_clear()
    incompatible = SimpleNamespace(triton_mqa_logits=lambda q: q)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: incompatible,
    )
    try:
        with pytest.raises(RuntimeError, match="triton_mqa_logits signature"):
            resolver()
    finally:
        resolver.cache_clear()


@pytest.mark.parametrize(
    ("aiter_enabled", "force_aiter_triton"),
    ((False, False), (True, False), (False, True)),
)
def test_sparse_mla_master_off_decode_uses_boltops_paged_mqa(
    monkeypatch: pytest.MonkeyPatch,
    aiter_enabled: bool,
    force_aiter_triton: bool,
) -> None:
    """Master-off decode must use packed row-major BoltOps, never AITER/LightOp."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: aiter_enabled)
    monkeypatch.setattr(
        runtime,
        "paged_mqa_logits_module",
        lambda: pytest.fail("master-off decode queried AITER paged MQA"),
    )
    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: pytest.fail("master-off decode queried LightOp paged MQA"),
    )

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def boltops_paged(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3]

    monkeypatch.setattr(
        runtime,
        "_load_boltops_paged_mqa_logits",
        lambda: boltops_paged,
        raising=False,
    )
    q = torch.empty((2, 1, 8, 128), dtype=torch.float8_e4m3fn)
    cache = torch.empty((3, 64, 1, 132), dtype=torch.uint8)
    weights = torch.arange(16, dtype=torch.float16).reshape(8, 2).transpose(0, 1)
    context_lens = torch.tensor([63, 65], dtype=torch.int32)
    block_tables = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0),
        128,
        force_aiter_triton=force_aiter_triton,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is q
    assert args[1] is cache
    assert args[2].dtype is torch.float32
    assert args[2].is_contiguous()
    assert torch.equal(args[2], weights.float().contiguous())
    assert args[3] is result
    assert result.shape == (2, 128)
    assert result.dtype is torch.float32
    assert args[4] is context_lens
    assert args[5] is block_tables
    assert args[6] == 128
    assert kwargs == {
        "kv_block_size": 64,
        "preshuffle": False,
        "fuse_output_init": True,
    }


@pytest.mark.parametrize("block_size", (1, 64))
def test_sparse_mla_boltops_decode_preserves_exact_mtp_pool_ends(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
) -> None:
    """Exact KPool ends must remain distinct for every MTP query row."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def boltops_paged(*args, **kwargs):
        calls.append((args, kwargs))
        return args[3]

    monkeypatch.setattr(
        runtime,
        "_load_boltops_paged_mqa_logits",
        lambda: boltops_paged,
    )
    q = (
        torch.arange(6, dtype=torch.float32)
        .reshape(2, 3, 1, 1)
        .expand(2, 3, 8, 128)
        .contiguous()
        .to(torch.float8_e4m3fn)
    )
    cache = torch.empty((10, block_size, 1, 132), dtype=torch.uint8)
    weights = torch.ones((6, 8), dtype=torch.float32)
    context_lens = torch.tensor(
        [[300, 300, 301], [1, 1, 2]],
        dtype=torch.int32,
    )
    block_tables = torch.tensor(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        dtype=torch.int32,
    )

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0),
        320,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].shape == (6, 1, 8, 128)
    assert torch.equal(args[0].float().reshape_as(q), q.float())
    assert args[1] is cache
    assert args[2] is weights
    assert args[3] is result
    assert result.shape == (6, 320)
    assert torch.equal(args[4], context_lens.reshape(-1))
    assert torch.equal(args[5], block_tables.repeat_interleave(3, dim=0))
    assert args[6] == 320
    assert kwargs == {
        "kv_block_size": block_size,
        "preshuffle": False,
        "fuse_output_init": block_size == 64,
    }


def test_sparse_mla_master_off_bf16_decode_skips_boltops_paged_mqa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BoltOps packed paged-MQA must not receive the gfx936 BF16 K-cache."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        runtime,
        "_load_boltops_paged_mqa_logits",
        lambda: pytest.fail("BF16 K-cache was routed to packed BoltOps paged-MQA"),
    )
    calls: list[tuple[object, ...]] = []
    output = object()

    def paged_mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: SimpleNamespace(paged_mqa_logits=paged_mqa_logits),
    )
    q = torch.empty((2, 1, 8, 128), dtype=torch.bfloat16)
    cache = torch.empty((3, 64, 1, 128), dtype=torch.bfloat16)
    weights = torch.ones((2, 8), dtype=torch.float32)
    context_lens = torch.tensor([63, 65], dtype=torch.int32)
    block_tables = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0),
        128,
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][1] is cache


@pytest.mark.parametrize("block_size", (16, 32))
def test_sparse_mla_master_off_unsupported_boltops_page_uses_existing_backend(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
) -> None:
    """Unsupported BoltOps pages must retain the gfx938 normal-layout backend."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        runtime,
        "_load_boltops_paged_mqa_logits",
        lambda: pytest.fail("unsupported page size queried BoltOps paged-MQA"),
    )
    calls: list[tuple[object, ...]] = []
    output = object()

    def paged_mqa_logits(*args, **kwargs):
        calls.append((*args, kwargs))
        return output

    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: SimpleNamespace(paged_mqa_logits=paged_mqa_logits),
    )
    q = torch.empty((1, 1, 8, 128), dtype=torch.float8_e4m3fn)
    cache = torch.empty((3, block_size, 1, 132), dtype=torch.uint8)

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        torch.ones((1, 8), dtype=torch.float32),
        torch.tensor([block_size], dtype=torch.int32),
        torch.tensor([[0, -1]], dtype=torch.int32),
        torch.empty(0),
        block_size * 2,
        kcache_layout="normal",
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][1] is cache


def test_sparse_mla_boltops_decode_rejects_public_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present but incompatible public paged-MQA API must fail visibly."""
    runtime = _runtime()
    assert hasattr(runtime, "_load_boltops_paged_mqa_logits"), (
        "the sparse-indexer runtime must own a lazy BoltOps paged-MQA resolver"
    )
    resolver = runtime._load_boltops_paged_mqa_logits
    resolver.cache_clear()
    incompatible = SimpleNamespace(triton_fp8_paged_mqa_logits=lambda q: q)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: incompatible)
    try:
        with pytest.raises(RuntimeError, match="paged_mqa_logits signature"):
            resolver()
    finally:
        resolver.cache_clear()


def test_sparse_mla_boltops_decode_resolver_allows_missing_optional_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent BoltOps install is a backend miss, not a startup failure."""
    runtime = _runtime()
    resolver = runtime._load_boltops_paged_mqa_logits
    resolver.cache_clear()

    def missing_module(name: str):
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )

    monkeypatch.setattr(runtime.importlib, "import_module", missing_module)
    try:
        assert resolver() is None
    finally:
        resolver.cache_clear()


def test_sparse_mla_master_off_missing_boltops_decode_uses_existing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Master-off decode falls back without re-entering the AITER route."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "_load_boltops_paged_mqa_logits", lambda: None)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        runtime,
        "paged_mqa_logits_module",
        lambda: pytest.fail("master-off BoltOps fallback queried AITER"),
    )
    calls: list[tuple[object, ...]] = []
    output = object()

    def paged_mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: SimpleNamespace(paged_mqa_logits=paged_mqa_logits),
    )
    q = (
        torch.arange(6, dtype=torch.float32)
        .reshape(2, 3, 1, 1)
        .expand(2, 3, 8, 128)
        .contiguous()
        .to(torch.float8_e4m3fn)
    )
    cache = torch.empty((10, 64, 1, 132), dtype=torch.uint8)
    context_lens = torch.tensor(
        [[300, 300, 301], [1, 1, 2]],
        dtype=torch.int32,
    )
    block_tables = torch.tensor(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        dtype=torch.int32,
    )

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        torch.ones((6, 8), dtype=torch.float32),
        context_lens,
        block_tables,
        torch.empty(0),
        320,
        force_aiter_triton=True,
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0][0].shape == (6, 1, 8, 128)
    assert torch.equal(calls[0][0].float().reshape_as(q), q.float())
    assert calls[0][1] is cache
    assert torch.equal(calls[0][3], context_lens.reshape(-1))
    assert torch.equal(calls[0][4], block_tables.repeat_interleave(3, dim=0))


@pytest.mark.parametrize(
    ("block_size", "kcache_layout", "expected_backend"),
    ((1, "preshuffle", "torch"), (64, "normal", "lightop")),
)
def test_sparse_mla_master_off_ignores_forced_aiter_on_gfx938_fallback(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    kcache_layout: str,
    expected_backend: str,
) -> None:
    """The GLM5Next force flag cannot bypass the disabled master switch."""
    runtime = _runtime()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "_load_boltops_paged_mqa_logits", lambda: None)
    monkeypatch.setattr(
        runtime,
        "paged_mqa_logits_module",
        lambda: pytest.fail("master-off gfx938 fallback queried AITER"),
    )
    outputs = {"torch": object(), "lightop": object()}
    calls: list[str] = []

    def torch_reference(*_args):
        calls.append("torch")
        return outputs["torch"]

    def lightop_reference(*_args, **_kwargs):
        calls.append("lightop")
        return outputs["lightop"]

    monkeypatch.setattr(runtime, "fp8_paged_mqa_logits_torch", torch_reference)
    monkeypatch.setattr(
        runtime,
        "_get_lightop_attention",
        lambda: SimpleNamespace(paged_mqa_logits=lightop_reference),
    )
    q = torch.zeros((1, 1, 8, 128), dtype=torch.float8_e4m3fn)
    cache = torch.empty((2, block_size, 1, 132), dtype=torch.uint8)

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        torch.ones((1, 8), dtype=torch.float32),
        torch.tensor([block_size], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        torch.empty(0),
        block_size,
        force_aiter_triton=True,
        kcache_layout=kcache_layout,
    )

    assert result is outputs[expected_backend]
    assert calls == [expected_backend]


@pytest.mark.parametrize("abi_change", ("kind", "default", "uninspectable"))
def test_sparse_mla_boltops_decode_rejects_subtle_public_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
    abi_change: str,
) -> None:
    """Every part of the audited paged-MQA call contract must fail closed."""
    runtime = _runtime()
    resolver = runtime._load_boltops_paged_mqa_logits
    resolver.cache_clear()
    from boltops.mqa_logits import triton_fp8_paged_mqa_logits

    if abi_change == "uninspectable":

        class Kernel:
            @property
            def __signature__(self):
                raise ValueError("opaque callable")

            def __call__(self, *args, **kwargs):
                return args, kwargs

        kernel = Kernel()
    else:
        expected = inspect.signature(triton_fp8_paged_mqa_logits)
        parameters = list(expected.parameters.values())
        index = next(
            index
            for index, parameter in enumerate(parameters)
            if parameter.name == ("q_fp8" if abi_change == "kind" else "bucket_tokens")
        )
        parameters[index] = parameters[index].replace(
            kind=(
                inspect.Parameter.POSITIONAL_ONLY
                if abi_change == "kind"
                else parameters[index].kind
            ),
            default=(4096 if abi_change == "default" else parameters[index].default),
        )

        def kernel(*args, **kwargs):
            return args, kwargs

        kernel.__signature__ = expected.replace(parameters=parameters)

    incompatible = SimpleNamespace(triton_fp8_paged_mqa_logits=kernel)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: incompatible)
    try:
        with pytest.raises(RuntimeError, match="paged_mqa_logits signature"):
            resolver()
    finally:
        resolver.cache_clear()


@pytest.mark.parametrize("is_gfx938", [False, True])
def test_sparse_mla_uses_categorized_mqa_abi_with_fp32_contiguous_weights(
    monkeypatch: pytest.MonkeyPatch,
    is_gfx938: bool,
) -> None:
    runtime = _runtime()
    calls: list[tuple[object, ...]] = []
    output = object()

    def mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: is_gfx938)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(mqa_logits=mqa_logits),
        raising=False,
    )
    monkeypatch.delattr(runtime, "lightop", raising=False)

    weights = torch.arange(8, dtype=torch.float16).reshape(2, 4).transpose(0, 1)
    scale = torch.ones(3)
    result = runtime.rocm_fp8_mqa_logits(
        torch.ones((4, 1, 2)),
        (torch.ones((3, 2)), scale),
        weights,
        torch.zeros(4, dtype=torch.int32),
        torch.full((4,), 3, dtype=torch.int32),
    )

    assert result is output
    assert len(calls) == 1
    assert len(calls[0]) == 6
    supplied_weights = calls[0][2]
    assert supplied_weights.dtype is torch.float32
    assert supplied_weights.is_contiguous()
    assert torch.equal(supplied_weights, weights.float().contiguous())
    assert calls[0][5] is (scale if is_gfx938 else None)


def test_sparse_mla_does_not_retry_legacy_namespace(monkeypatch):
    runtime = _runtime()
    legacy = SimpleNamespace(mqa_logits=lambda *_: pytest.fail("legacy called"))
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(runtime, "lightop_attention", SimpleNamespace())
    monkeypatch.setattr(runtime, "lightop", legacy, raising=False)
    with pytest.raises(AttributeError):
        runtime.rocm_fp8_mqa_logits(
            torch.ones((1, 1, 2)),
            (torch.ones((1, 2)), torch.ones(1)),
            torch.ones((1, 1)),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )


def test_chunked_sparse_mla_uses_new_abi_and_categorized_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    q_fp8 = torch.ones((4, 1, 2), dtype=torch.float16)
    k_fp8 = torch.ones((3, 2))
    weights = torch.arange(8, dtype=torch.float16).reshape(2, 4).transpose(0, 1)
    cu_seqlen_ks = torch.zeros(4, dtype=torch.int32)
    cu_seqlen_ke = torch.full((4,), 3, dtype=torch.int32)
    logits_buffer = torch.zeros(128 * 128)
    topk_indices_buffer = torch.empty((4, 1), dtype=torch.int32)
    marker = 37.5
    topk_marker = 17

    def mqa_logits(
        q: torch.Tensor,
        k: torch.Tensor,
        kernel_weights: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        kv_scale: torch.Tensor | None,
        clean_logit: bool,
        D_out: torch.Tensor,
    ) -> None:
        assert q.data_ptr() == q_fp8.data_ptr()
        assert torch.equal(q, q_fp8)
        assert k is k_fp8
        assert torch.equal(kernel_weights, weights.float().contiguous())
        assert kernel_weights.dtype is torch.float32
        assert kernel_weights.is_contiguous()
        assert ks.data_ptr() == cu_seqlen_ks.data_ptr()
        assert torch.equal(ks, cu_seqlen_ks)
        assert ke.data_ptr() == cu_seqlen_ke.data_ptr()
        assert torch.equal(ke, cu_seqlen_ke)
        assert kv_scale is None
        assert clean_logit is True
        assert D_out.shape == (128, 128)
        assert D_out.dtype is torch.float32
        assert D_out.data_ptr() == logits_buffer.data_ptr()
        D_out.fill_(marker)

    def top_k_prefill(
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        topk_indices: torch.Tensor,
        num_rows: int,
        row_stride: int,
        column_stride: int,
        topk_tokens: int,
    ) -> None:
        assert logits.shape == (4, 3)
        assert torch.all(logits == marker)
        assert torch.equal(row_starts, cu_seqlen_ks)
        assert torch.equal(row_ends, cu_seqlen_ke)
        assert topk_indices.data_ptr() == topk_indices_buffer.data_ptr()
        assert topk_indices.shape == topk_indices_buffer.shape
        assert num_rows == 4
        assert row_stride == 128
        assert column_stride == 1
        assert topk_tokens == 1
        topk_indices.fill_(topk_marker)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            mqa_logits=mqa_logits,
            top_k_per_row_prefill=top_k_prefill,
        ),
        raising=False,
    )
    monkeypatch.delattr(runtime, "op", raising=False)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    monkeypatch.setattr(runtime, "get_logits_buffer", lambda _device: logits_buffer)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_LIGHTOP_TOPK", True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)

    runtime.mqa_logits_inner_chunked(
        SimpleNamespace(
            token_start=0,
            token_end=4,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
        ),
        q_fp8,
        k_fp8,
        weights,
        torch.ones(3),
        topk_indices_buffer,
        1,
    )

    assert torch.equal(
        topk_indices_buffer,
        torch.full_like(topk_indices_buffer, topk_marker),
    )


def test_sparse_mla_topk_helpers_use_categorized_attention_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    prefill_calls: list[tuple[object, ...]] = []
    decode_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            top_k_per_row_prefill=lambda *args: prefill_calls.append(args),
            top_k_per_row_decode=lambda *args: decode_calls.append(args),
        ),
        raising=False,
    )
    monkeypatch.delattr(runtime, "op", raising=False)
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    topk = torch.empty((2, 2), dtype=torch.int32)

    runtime._lightop_topk_indices_prefill(
        logits,
        torch.tensor([0, 1]),
        torch.tensor([6, 5]),
        topk,
        2,
    )
    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([6, 5]),
        1,
        topk,
        2,
    )

    assert len(prefill_calls) == 1
    assert len(decode_calls) == 1
