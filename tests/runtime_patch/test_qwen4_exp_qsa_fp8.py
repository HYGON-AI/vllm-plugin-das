# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
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

    monkeypatch.setattr(qsa_fp8_patch, "get_qsa_fp8_reader", lambda **kwargs: reader)
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
    supported = module.Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
    assert "fp8_e4m3" in supported
    assert "fp8_e5m2" in supported


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
