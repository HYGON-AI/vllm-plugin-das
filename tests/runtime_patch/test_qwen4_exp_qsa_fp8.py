# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_patch():
    return importlib.import_module(
        "vllm_hcu.patch.worker.core_fix.patch_qwen4_exp_qsa_fp8"
    )


def _owner_module(qsa_fp8_patch) -> ModuleType:
    module = ModuleType(qsa_fp8_patch.TARGET_MODULE)

    class Backend:
        supported_kv_cache_dtypes = ["auto", "bfloat16"]

        @classmethod
        def supports_kv_cache_dtype(cls, kv_cache_dtype):
            if kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                return False
            return kv_cache_dtype in cls.supported_kv_cache_dtypes

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
            del (
                cls,
                head_size,
                dtype,
                block_size,
                use_mla,
                use_sparse,
                use_mm_prefix,
                device_capability,
            )
            if has_sink:
                return "sink constraint remains active"
            if kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                return "generic FlashAttention rejects FP8 on HCU"
            return None

    class Impl:
        def __init__(self, *args, **kwargs):
            del kwargs
            self.kv_cache_dtype = args[6]
            self.dcp_world_size = 1
            self.alibi_slopes = None
            self.sinks = None
            self.sliding_window = (-1, -1)
            self.head_size = 4
            if self.kv_cache_dtype not in ("auto", "bfloat16"):
                raise NotImplementedError(
                    "Qwen4Exp QSA requires a BF16 main KV cache"
                )
            self.supports_quant_query_input = False

        def do_kv_cache_update(
            self,
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        ):
            del layer, key, value, kv_cache, slot_mapping
            self.upstream_cache_update_called = True

        def forward_qsa(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            token_to_req,
            output_scale=None,
            output_block_scale=None,
        ):
            del (
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                token_to_req,
                output_scale,
                output_block_scale,
            )
            output.fill_(7)
            return output

    class Attention:
        def __init__(
            self,
            *,
            vllm_config,
            config,
            layer_id,
            quant_config=None,
            reduce_results=True,
            prefix="",
        ):
            del config, layer_id, quant_config, reduce_results, prefix
            cache_dtype = vllm_config.cache_config.cache_dtype
            self.original_cache_dtype = cache_dtype
            if cache_dtype not in ("auto", "bfloat16"):
                raise NotImplementedError(
                    "Qwen4Exp QSA requires a BF16 main KV cache"
                )
            self.kv_cache_dtype = cache_dtype
            self.kv_cache_torch_dtype = torch.bfloat16
            self.impl = Impl(1, 4, 0.5, 1, None, None, cache_dtype)

    module.Qwen4ExpQSAFlashAttentionBackend = Backend
    module.Qwen4ExpQSAFlashAttentionImpl = Impl
    module.Qwen4ExpQSAAttention = Attention
    module.canonicalize_singleton_dim_strides = lambda tensor: tensor
    return module


def _config(cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def test_qsa_owner_accepts_fp8_cache_and_restores_shared_config(
    monkeypatch: pytest.MonkeyPatch,
):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)

    def reader(*args, **kwargs):
        return None

    def cache_writer(*args, **kwargs):
        return None

    monkeypatch.setattr(qsa_fp8_patch, "get_qsa_fp8_reader", lambda **kwargs: reader)
    monkeypatch.setattr(
        qsa_fp8_patch,
        "_load_hcu_cache_writer",
        lambda: cache_writer,
        raising=False,
    )
    qsa_fp8_patch.apply_to_module(module)
    config = _config("fp8_e4m3")

    layer = module.Qwen4ExpQSAAttention(
        vllm_config=config,
        config=object(),
        layer_id=0,
    )

    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert layer.original_cache_dtype == "auto"
    assert layer.kv_cache_dtype == "fp8_e4m3"
    assert layer.kv_cache_torch_dtype is torch.uint8
    assert layer.impl.kv_cache_dtype == "fp8_e4m3"
    assert layer.impl._vllm_hcu_qsa_fp8_reader is reader
    assert layer.impl._vllm_hcu_qsa_fp8_cache_writer is cache_writer
    supported = module.Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
    assert "fp8_e4m3" in supported
    assert "fp8_e5m2" in supported


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_qsa_backend_advertises_fp8_through_capability_contract(cache_dtype: str):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)

    qsa_fp8_patch.apply_to_module(module)
    backend = module.Qwen4ExpQSAFlashAttentionBackend

    assert tuple(inspect.signature(backend.supports_kv_cache_dtype).parameters) == (
        "kv_cache_dtype",
    )
    assert tuple(inspect.signature(backend.supports_combination).parameters) == (
        "head_size",
        "dtype",
        "kv_cache_dtype",
        "block_size",
        "use_mla",
        "has_sink",
        "use_sparse",
        "use_mm_prefix",
        "device_capability",
    )
    assert backend.supports_kv_cache_dtype(cache_dtype)
    assert (
        backend.supports_combination(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=cache_dtype,
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=object(),
        )
        is None
    )
    assert (
        backend.supports_combination(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=cache_dtype,
            block_size=16,
            use_mla=False,
            has_sink=True,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=object(),
        )
        == "sink constraint remains active"
    )


def test_real_qsa_backend_accepts_fp8_capability_contract():
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "hcu"
    code = """
import torch

from vllm_hcu.patch.worker import prepare_worker_patches

prepare_worker_patches()
import vllm.models.qwen4_exp.amd.model
from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAFlashAttentionBackend
from vllm.platforms.interface import DeviceCapability

backend = Qwen4ExpQSAFlashAttentionBackend
for cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
    assert backend.supports_kv_cache_dtype(cache_dtype)
    assert backend.supports_combination(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=cache_dtype,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=DeviceCapability(9, 0),
    ) is None
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("cache_dtype", "expected_dtype"),
    [
        ("fp8_e4m3", torch.float8_e4m3fn),
        ("fp8_e5m2", torch.float8_e5m2),
    ],
)
def test_qsa_fp8_forward_uses_native_view_and_device_scales(
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: str,
    expected_dtype: torch.dtype,
):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)
    seen = {}

    def reader(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
        k_scale=None,
        v_scale=None,
    ):
        seen.update(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            logical_indices=logical_indices,
            block_table=block_table,
            token_to_req=token_to_req,
            out=out,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out.fill_(3)
        return out

    monkeypatch.setattr(
        qsa_fp8_patch,
        "get_qsa_fp8_reader",
        lambda **kwargs: reader,
    )
    qsa_fp8_patch.apply_to_module(module)
    impl = module.Qwen4ExpQSAFlashAttentionImpl(
        1, 4, 0.5, 1, None, None, cache_dtype
    )
    layer = SimpleNamespace(
        topk_indices_buffer=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        _k_scale=torch.tensor([0.75], dtype=torch.float32),
        _v_scale=torch.tensor([1.25], dtype=torch.float32),
    )
    query = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    kv_cache = torch.zeros((1, 1, 2, 8), dtype=torch.uint8)
    output = torch.empty_like(query)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
    )
    token_to_req = torch.tensor([0, 0], dtype=torch.int32)

    result = impl.forward_qsa(
        layer,
        query,
        query,
        query,
        kv_cache,
        metadata,
        output,
        token_to_req,
    )

    assert result is output
    assert seen["key_cache"].dtype is expected_dtype
    assert seen["value_cache"].dtype is expected_dtype
    assert seen["logical_indices"].shape == (1, 2)
    assert seen["token_to_req"].shape == (1,)
    assert seen["out"].data_ptr() == output[:1].data_ptr()
    assert seen["k_scale"] is layer._k_scale
    assert seen["v_scale"] is layer._v_scale
    assert torch.all(output[0] == 3)


def test_qsa_fp8_cache_update_uses_hcu_writer(monkeypatch: pytest.MonkeyPatch):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)
    writes = []
    monkeypatch.setattr(
        qsa_fp8_patch,
        "_load_hcu_cache_writer",
        lambda: lambda *args: writes.append(args),
        raising=False,
    )
    monkeypatch.setattr(
        qsa_fp8_patch,
        "get_qsa_fp8_reader",
        lambda **kwargs: lambda *args, **kwargs: None,
    )
    qsa_fp8_patch.apply_to_module(module)
    impl = module.Qwen4ExpQSAFlashAttentionImpl(
        1, 4, 0.5, 1, None, None, "fp8_e5m2"
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor([0.75], dtype=torch.float32),
        _v_scale=torch.tensor([1.25], dtype=torch.float32),
    )
    key = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    value = key + 1
    kv_cache = torch.zeros((2, 1, 4, 8), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 5], dtype=torch.int64)

    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    assert not hasattr(impl, "upstream_cache_update_called")
    assert len(writes) == 1
    writer_args = writes[0]
    assert writer_args[:2] == (key, value)
    assert writer_args[2].shape == (2, 4, 1, 4)
    assert writer_args[3].shape == (2, 4, 1, 4)
    assert writer_args[4:] == (
        slot_mapping,
        "fp8_e5m2",
        layer._k_scale,
        layer._v_scale,
    )


def test_qsa_bf16_cache_update_keeps_upstream_path(monkeypatch: pytest.MonkeyPatch):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)
    monkeypatch.setattr(
        qsa_fp8_patch,
        "_load_hcu_cache_writer",
        lambda: pytest.fail("BF16 must not load the HCU cache writer"),
        raising=False,
    )
    qsa_fp8_patch.apply_to_module(module)
    impl = module.Qwen4ExpQSAFlashAttentionImpl(
        1, 4, 0.5, 1, None, None, "bfloat16"
    )

    impl.do_kv_cache_update(object(), object(), object(), object(), object())

    assert impl.upstream_cache_update_called


def test_qsa_fp8_cache_update_respects_custom_ops_master_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)
    monkeypatch.setattr(
        qsa_fp8_patch,
        "_load_hcu_cache_writer",
        lambda: pytest.fail("master gate must disable the HCU writer"),
        raising=False,
    )
    monkeypatch.setattr(
        qsa_fp8_patch,
        "get_qsa_fp8_reader",
        lambda **kwargs: lambda *args, **kwargs: None,
    )
    qsa_fp8_patch.apply_to_module(module)
    impl = module.Qwen4ExpQSAFlashAttentionImpl(
        1, 4, 0.5, 1, None, None, "fp8_e5m2"
    )

    impl.do_kv_cache_update(object(), object(), object(), object(), object())

    assert impl.upstream_cache_update_called


def test_qsa_bf16_forward_keeps_upstream_path(monkeypatch: pytest.MonkeyPatch):
    qsa_fp8_patch = _load_patch()
    module = _owner_module(qsa_fp8_patch)
    monkeypatch.setattr(
        qsa_fp8_patch,
        "get_qsa_fp8_reader",
        lambda **kwargs: pytest.fail("BF16 must not select an FP8 reader"),
    )
    qsa_fp8_patch.apply_to_module(module)
    impl = module.Qwen4ExpQSAFlashAttentionImpl(
        1, 4, 0.5, 1, None, None, "bfloat16"
    )
    output = torch.empty((1, 1, 4), dtype=torch.bfloat16)

    result = impl.forward_qsa(
        object(),
        output,
        output,
        output,
        torch.empty(0),
        object(),
        output,
        torch.empty(0, dtype=torch.int32),
    )

    assert result is output
    assert torch.all(output == 7)
