# SPDX-License-Identifier: Apache-2.0
"""Portable contracts for the model-local, eager-only decode adapter."""

import ast
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_deepseek_v41_rocm_flashmla_decode as patch
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError
from vllm_hcu.platforms import envs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from tests.fixtures.vllm_source import resolve_target_vllm_root


def packed(page=64):
    stride = ((page * 584 + 575) // 576) * 576
    return torch.zeros(4, stride, dtype=torch.uint8).as_strided(
        (4, page, 584), (stride, 584, 1),
    )


def inputs(heads=64, page=None, swa_page=64):
    q = torch.zeros(3, heads, 512, dtype=torch.bfloat16)
    args = dict(
        q=q, swa_cache=packed(swa_page), kv_cache=None if page is None else packed(page),
        swa_indices=torch.arange(128, dtype=torch.int32).repeat(3, 1),
        swa_lens=torch.tensor([0, 1, 128], dtype=torch.int32), scheduler=object(),
        scale=512**-0.5, attn_sink=torch.zeros(heads + 2), output=torch.full_like(q, 7),
        head_dim=512, nope_head_dim=448, rope_head_dim=64,
    )
    if page is not None:
        args.update(
            topk_indices=torch.arange(128, dtype=torch.int32).repeat(3, 1),
            block_table=torch.zeros(3, 4, dtype=torch.int32),
            token_to_req_indices=torch.arange(3, dtype=torch.int32),
            is_valid_token=torch.ones(3, dtype=torch.bool),
        )
    return args


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "1")
    monkeypatch.setattr(runtime, "_supported_device", lambda q: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("child,master,expected", [
    (None, None, True), (None, "1", True), (None, "0", False),
    ("0", None, False), ("0", "1", False), ("1", "0", False),
    ("true", "false", False), ("1", "1", True), ("True", None, True),
])
def test_gates(monkeypatch, child, master, expected):
    for name, value in (("VLLM_HCU_DSV41_FLASHMLA_DECODE", child), ("VLLM_HCU_USE_CUSTOM_OPS", master)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert envs.dsv41_flashmla_decode_enabled() is expected
    assert envs.hcu_vllm_environment_variables["VLLM_HCU_DSV41_FLASHMLA_DECODE"]() is expected


def forbidden(*args, **kwargs):
    pytest.fail("unsupported/disabled decode must not resolve or launch native kernel")


def test_disabled_is_lazy(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "0")
    monkeypatch.setattr(runtime, "_supported_device", forbidden)
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    args = inputs()
    assert not runtime.try_flashmla_decode(**args)
    assert torch.all(args["output"] == 7)


@pytest.mark.parametrize("change", [
    dict(head_dim=576), dict(nope_head_dim=512), dict(rope_head_dim=0),
    dict(max_decode_query_len=5), dict(kv_cache_dtype="nvfp4_ds_mla"), dict(scheduler=None),
    dict(q=torch.zeros(3, 16, 512, dtype=torch.bfloat16)), dict(q=torch.zeros(3, 64, 512)),
    dict(swa_cache=torch.zeros(4, 64, 584, dtype=torch.uint8)),
    dict(swa_cache=packed(128)), dict(swa_cache=None),
    dict(swa_indices=None), dict(swa_indices=torch.zeros(3, 127, dtype=torch.int32)),
    dict(swa_indices=torch.zeros(3, 128, dtype=torch.int64)),
    dict(swa_indices=torch.zeros(3, 2, 128, dtype=torch.int32)),
    dict(swa_lens=None), dict(swa_lens=torch.zeros(2, dtype=torch.int32)),
    dict(attn_sink=torch.zeros(63)), dict(attn_sink=torch.zeros(64, dtype=torch.bfloat16)),
    dict(output=torch.zeros(3, 64, 512)),
])
def test_unsupported_does_not_write(enabled, monkeypatch, change):
    args = inputs()
    args.update(change)
    before = args["output"].clone()
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    assert not runtime.try_flashmla_decode(**args)
    torch.testing.assert_close(args["output"], before)


@pytest.mark.parametrize("field", ["topk_indices", "block_table", "token_to_req_indices", "is_valid_token"])
def test_missing_history_metadata(enabled, monkeypatch, field):
    args = inputs(page=32)
    args[field] = None
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    assert not runtime.try_flashmla_decode(**args)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [None, 32, 64])
@pytest.mark.parametrize("swa_page", [32, 64])
def test_dispatch_compacts_holes_preserves_buffers_and_uses_fresh_plan(enabled, monkeypatch, heads, page, swa_page):
    args = inputs(heads, page, swa_page)
    original = args["swa_indices"].clone()
    calls = []
    plan = object()

    def map_indices(topk, requests, table, block_size, valid):
        assert block_size == page
        mapped = topk.clone()
        mapped[:, 1::2] = -1  # Valid entries exist beyond the valid-count prefix.
        mapped[0] = -1
        return mapped, torch.tensor([0, 64, 64], dtype=torch.int32)

    def kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"].shape == (3, 1, heads, 512)
        assert kwargs["k_cache"].shape == (4, swa_page, 1, 584)
        assert kwargs["k_cache"].data_ptr() == args["swa_cache"].data_ptr()
        assert kwargs["k_cache"].stride(0) == args["swa_cache"].stride(0)
        assert kwargs["tile_scheduler_metadata"] is plan
        assert kwargs["attn_sink"].shape == (heads,)
        assert torch.all(kwargs["indices"][0] == -1)
        assert torch.all(kwargs["indices"][1, 0, 1:] == -1)
        if page is not None:
            torch.testing.assert_close(kwargs["extra_indices_in_kvcache"][1, 0, :64], torch.arange(0, 128, 2, dtype=torch.int32))
            assert torch.all(kwargs["extra_indices_in_kvcache"][0] == -1)
            assert torch.all(kwargs["extra_indices_in_kvcache"][:, :, 64:] == -1)
        kwargs["indices"].fill_(-1)
        return torch.full_like(kwargs["q"], 2), None

    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (kernel, map_indices, lambda: (plan, None)))
    storage = torch.full((6, heads, 512), 7, dtype=torch.bfloat16)
    args["output"] = storage[::2]
    history_before = None if page is None else args["topk_indices"].clone()
    assert runtime.try_flashmla_decode(**args)
    assert len(calls) == 1
    assert torch.all(storage[::2] == 2) and torch.all(storage[1::2] == 7)
    torch.testing.assert_close(args["swa_indices"], original)
    if page is not None:
        torch.testing.assert_close(args["topk_indices"], history_before)


def test_capture_and_unavailable_library_fall_back(enabled, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    assert not runtime.try_flashmla_decode(**inputs())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: None)
    assert not runtime.try_flashmla_decode(**inputs())


def test_native_failure_propagates(enabled, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("HIP launch failed")
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (fail, forbidden, lambda: (object(), None)))
    with pytest.raises(RuntimeError, match="HIP launch failed"):
        runtime.try_flashmla_decode(**inputs())


def model():
    calls = []
    class Attention:
        def __init__(self, *args, **kwargs):
            pass

        def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
            calls.append((q, kv_cache, swa_metadata, attn_metadata, swa_only, output))
            return "original"
    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV41ROCMAiterMLAAttention = Attention
    module.rocm_sparse_attn_prefill = object()
    module.rocm_sparse_attn_decode = object()
    instance = Attention()
    instance.compress_ratio = 0
    instance.swa_cache_layer = SimpleNamespace(kv_cache=packed())
    instance.scale = 512**-0.5
    instance.attn_sink = torch.zeros(64)
    instance.head_dim, instance.nope_head_dim, instance.rope_head_dim = 512, 448, 64
    instance.kv_cache_dtype = "fp8_ds_mla"
    instance.topk_indices_buffer = torch.zeros(3, 128, dtype=torch.int32)
    swa = SimpleNamespace(
        num_decode_tokens=3, num_decodes=3, max_decode_query_len=1,
        tile_sched_swaonly=object(), tile_sched_c1a=object(), tile_sched_c2a=object(),
        decode_swa_indices=torch.zeros(3, 128, dtype=torch.int32),
        decode_swa_lens=torch.ones(3, dtype=torch.int32),
        token_to_req_indices=torch.arange(3, dtype=torch.int32),
        is_valid_token=torch.ones(3, dtype=torch.bool),
    )
    return module, instance, swa, calls


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_model_local_patch_and_exact_fallback(enabled, monkeypatch, ratio):
    module, instance, swa, calls = model()
    cls = type(instance)
    original = cls._forward_decode
    prefill, shared_decode = module.rocm_sparse_attn_prefill, module.rocm_sparse_attn_decode
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_context_supported", lambda self: True)
    monkeypatch.setattr(runtime, "try_flashmla_decode", lambda **kw: False)
    args = inputs()
    cache = None if ratio == 0 else packed(64 // ratio)
    metadata = None if ratio == 0 else SimpleNamespace(block_size=64, block_table=torch.zeros(3, 4, dtype=torch.int32))
    instance.compress_ratio = ratio
    assert instance._forward_decode(args["q"], cache, swa, metadata, ratio == 0, args["output"]) == "original"
    seen = []
    def selected(**kwargs):
        seen.append(kwargs)
        return True
    monkeypatch.setattr(runtime, "try_flashmla_decode", selected)
    assert instance._forward_decode(args["q"], cache, swa, metadata, ratio == 0, args["output"]) is None
    assert len(calls) == 1
    expected = {0: swa.tile_sched_swaonly, 1: swa.tile_sched_c1a, 2: swa.tile_sched_c2a}[ratio]
    assert seen[0]["scheduler"] is expected and seen[0]["kv_cache"] is cache
    assert module.rocm_sparse_attn_prefill is prefill and module.rocm_sparse_attn_decode is shared_decode
    monkeypatch.setattr(patch, "_context_supported", lambda self: False)
    assert instance._forward_decode(args["q"], cache, swa, metadata, ratio == 0, args["output"]) == "original"
    assert len(seen) == 1
    cls._forward_decode = original
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)


def test_signature_drift_fails_before_mutation():
    module, instance, _, _ = model()
    type(instance)._forward_decode = lambda self, q: None
    with pytest.raises(PatchCompatibilityError, match="signature"):
        patch.apply_to_module(module)
    assert not hasattr(type(instance), patch._MARKER)
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch.apply_to_module(ModuleType("vllm.models.deepseek_v4.amd.rocm"))


def test_disabled_model_wrapper_does_not_probe_config(monkeypatch):
    module, instance, swa, calls = model()
    patch.apply_to_module(module)
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "0")
    monkeypatch.setattr(patch, "_context_supported", forbidden)
    monkeypatch.setattr(runtime, "try_flashmla_decode", forbidden)
    args = inputs()
    assert instance._forward_decode(args["q"], None, swa, None, True, args["output"]) == "original"
    assert len(calls) == 1


def test_missing_history_does_not_silently_become_swa_only(enabled, monkeypatch):
    module, instance, swa, calls = model()
    instance.compress_ratio = 1
    patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_context_supported", lambda self: True)
    monkeypatch.setattr(runtime, "try_flashmla_decode", forbidden)
    args = inputs()
    metadata = SimpleNamespace(block_size=64, block_table=torch.zeros(3, 4, dtype=torch.int32))
    assert instance._forward_decode(args["q"], None, swa, metadata, False, args["output"]) == "original"
    assert len(calls) == 1


@pytest.mark.parametrize("dcp,pcp,speculation,expected", [
    (1, 1, None, True), (2, 1, None, False), (1, 2, None, False),
    (1, 1, object(), False),
])
def test_context_parallel_and_speculation_gate(monkeypatch, dcp, pcp, speculation, expected):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp,
        ),
        speculative_config=speculation,
    )
    assert patch._config_supported(config) is expected


@pytest.mark.parametrize("keyword_config", [False, True])
def test_forward_needs_no_global_config(enabled, monkeypatch, keyword_config):
    module, template, swa, calls = model()
    cls = type(template)
    assert patch.apply_to_module(module)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1,
        ), speculative_config=None,
    )
    instance = cls(vllm_config=config) if keyword_config else cls(config)
    instance.__dict__.update(template.__dict__)
    assert patch._context_supported(instance)
    # Reproduce MRV2 warmup's missing global config rather than mocking the gate.
    global_config = ModuleType("vllm.config")
    def no_global_context():
        raise AssertionError("Current vLLM config is not set")
    global_config.get_current_vllm_config = no_global_context
    monkeypatch.setitem(sys.modules, "vllm.config", global_config)
    selected = []
    monkeypatch.setattr(runtime, "try_flashmla_decode", lambda **kwargs: selected.append(kwargs) or True)
    args = inputs()
    assert instance._forward_decode(args["q"], None, swa, None, True, args["output"]) is None
    assert len(selected) == 1 and not calls
    assert not patch._context_supported(template), "pre-patch instances must conservatively fall back"


def test_real_source_signature_and_registration():
    root = resolve_target_vllm_root()
    tree = ast.parse((root / "vllm/models/deepseek_v41/amd/rocm.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41ROCMAiterMLAAttention")
    decode = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_decode")
    assert [arg.arg for arg in decode.args.args] == [
        "self", "q", "kv_cache", "swa_metadata", "attn_metadata", "swa_only", "output",
    ]
    assert not decode.args.defaults and not decode.args.kwonlyargs
    # SWA pages are a model-owned setting, not the service's --block-size.
    # Keep this real-source contract beside the dispatch tests so a synthetic
    # 64-token SWA fixture cannot hide another serving-only layout mismatch.
    attention_tree = ast.parse((root / "vllm/models/deepseek_v41/attention.py").read_text())
    swa_calls = [
        node for node in ast.walk(attention_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "DeepseekV4SWACache"
    ]
    assert len(swa_calls) == 1
    swa_page = next(kw.value for kw in swa_calls[0].keywords if kw.arg == "block_size")
    assert isinstance(swa_page, ast.Constant) and swa_page.value == 32
    from vllm_hcu.patch.worker import worker_callback_names
    assert (patch.PATCH_ID, patch.TARGET_MODULE) in worker_callback_names()
