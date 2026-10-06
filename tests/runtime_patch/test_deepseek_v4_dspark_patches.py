# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v4_attention,
    patch_deepseek_v4_dspark_target,
    patch_deepseek_v4_load_weights,
    patch_deepseek_v4_bf16_compressor,
    patch_deepseek_v4_rocm_compressor_fusion,
    patch_deepseek_v4_rocm_bf16_cache,
    patch_deepseek_v4_rocm_dspark_metadata,
    patch_deepseek_v4_rocm_wo_a_layout,
    patch_mhc_backend,
)
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_bf16_compressor_writes_finite_576_dim_row_at_last_rope_position() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU GPU to execute the Triton kernel")

    from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_compressor import (
        compress_norm_rope_store_bf16,
    )

    device = torch.device("cuda")
    head_dim = 576
    state_width = head_dim
    state_cache = torch.ones((1, 1, 2 * head_dim), dtype=torch.bfloat16, device=device)
    kv_cache = torch.empty((1, 1, head_dim), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.cat(
        (
            torch.ones((1, 32), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 32), dtype=torch.bfloat16, device=device),
        ),
        dim=1,
    )
    zero = torch.zeros((1,), dtype=torch.int32, device=device)

    compress_norm_rope_store_bf16(
        state_cache=state_cache,
        num_actual=1,
        token_to_req_indices=zero,
        positions=zero,
        slot_mapping=zero,
        block_table=zero.view(1, 1),
        block_size=1,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=zero),
        rms_norm_weight=torch.ones((head_dim,), dtype=torch.bfloat16, device=device),
        rms_norm_eps=1e-6,
        head_dim=head_dim,
        rope_head_dim=64,
        compress_ratio=1,
        overlap=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(kv_cache, torch.ones_like(kv_cache))


def test_bf16_compressor_never_calls_fp8_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    def fp8_store(
        state_cache,
        num_actual,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        state_width,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata,
        pdl_kwargs,
        head_dim,
        rope_head_dim,
        compress_ratio,
        overlap,
        use_fp4_cache,
        rms_norm_weight,
        rms_norm_eps,
        quant_block,
        token_stride,
        scale_dim,
    ) -> None:
        del (
            state_cache,
            num_actual,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            state_width,
            cos_sin_cache,
            k_cache_metadata,
            pdl_kwargs,
            head_dim,
            rope_head_dim,
            compress_ratio,
            overlap,
            use_fp4_cache,
            rms_norm_weight,
            rms_norm_eps,
            quant_block,
            token_stride,
            scale_dim,
        )
        calls.append(("fp8", kv_cache.dtype))

    def fp8_two_stage_store(
        state_cache,
        num_actual,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        state_width,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata,
        pdl_kwargs,
        head_dim,
        rope_head_dim,
        compress_ratio,
        overlap,
        use_fp4_cache,
        rms_norm_weight,
        rms_norm_eps,
        quant_block,
        token_stride,
        scale_dim,
        num_decode_tokens,
        compress_scratch,
    ) -> None:
        del num_decode_tokens, compress_scratch
        fp8_store(
            state_cache,
            num_actual,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            state_width,
            cos_sin_cache,
            kv_cache,
            k_cache_metadata,
            pdl_kwargs,
            head_dim,
            rope_head_dim,
            compress_ratio,
            overlap,
            use_fp4_cache,
            rms_norm_weight,
            rms_norm_eps,
            quant_block,
            token_stride,
            scale_dim,
        )

    module = _module(
        patch_deepseek_v4_bf16_compressor.TARGET_MODULE,
        compress_norm_rope_store_triton=fp8_store,
        compress_norm_rope_store_two_stage_triton=fp8_two_stage_store,
    )
    monkeypatch.setattr(
        patch_deepseek_v4_bf16_compressor,
        "compress_norm_rope_store_bf16",
        lambda **kwargs: calls.append(("bf16", kwargs["kv_cache"].dtype)),
    )
    patch_deepseek_v4_bf16_compressor.apply_to_module(module)

    kwargs = dict(
        state_cache=object(),
        num_actual=1,
        token_to_req_indices=object(),
        positions=object(),
        slot_mapping=object(),
        block_table=object(),
        block_size=4,
        state_width=1024,
        cos_sin_cache=object(),
        k_cache_metadata=object(),
        pdl_kwargs={},
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=object(),
        rms_norm_eps=1e-6,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
    )
    module.compress_norm_rope_store_triton(
        kv_cache=torch.empty((1, 1, 512), dtype=torch.bfloat16),
        **kwargs,
    )
    module.compress_norm_rope_store_triton(
        kv_cache=torch.empty((1,), dtype=torch.uint8),
        **kwargs,
    )

    assert calls == [
        ("bf16", torch.bfloat16),
        ("fp8", torch.uint8),
    ]


def test_rocm_bf16_cache_uses_plain_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def packed_fp8_gather(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=False,
    ):
        calls.append(("fp8", (use_fnuz,)))

    def bf16_gather(*args):
        calls.append(("bf16", args))

    def sparse_decode(
        q,
        kv_cache,
        swa_k_cache,
        swa_only,
        topk_indices,
        topk_lens,
        swa_indices,
        swa_lens,
        swa_ragged_indices,
        swa_ragged_indptr,
        topk_ragged_indices,
        topk_ragged_indptr,
        attn_sink,
        scale,
        head_dim,
        nope_head_dim,
        rope_head_dim,
        output,
        extra_cache_nan_free=False,
        adaptive_splits=False,
    ):
        return None

    module = _module(
        patch_deepseek_v4_rocm_bf16_cache.TARGET_MODULE,
        dequantize_and_gather_k_cache=packed_fp8_gather,
        rocm_sparse_attn_decode=sparse_decode,
    )
    monkeypatch.setattr(
        patch_deepseek_v4_rocm_bf16_cache,
        "gather_bf16_k_cache",
        bf16_gather,
    )
    patch_deepseek_v4_rocm_bf16_cache.apply_to_module(module)

    common = (object(), object(), object(), 256, 7)
    module.dequantize_and_gather_k_cache(
        object(),
        torch.empty((1, 1, 512), dtype=torch.bfloat16),
        *common,
        use_fnuz=True,
    )
    module.dequantize_and_gather_k_cache(
        object(),
        torch.empty((1,), dtype=torch.uint8),
        *common,
        use_fnuz=True,
    )

    assert calls[0][0] == "bf16"
    assert calls[0][1][-2:] == (256, 7)
    assert calls[1] == ("fp8", (True,))


def test_rocm_bf16_sparse_decode_uses_plain_cache_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def packed_fp8_gather(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=False,
    ):
        return None

    def fp8_sparse_decode(
        q,
        kv_cache,
        swa_k_cache,
        swa_only,
        topk_indices,
        topk_lens,
        swa_indices,
        swa_lens,
        swa_ragged_indices,
        swa_ragged_indptr,
        topk_ragged_indices,
        topk_ragged_indptr,
        attn_sink,
        scale,
        head_dim,
        nope_head_dim,
        rope_head_dim,
        output,
        extra_cache_nan_free=False,
        adaptive_splits=False,
    ):
        calls.append(("fp8", swa_k_cache.dtype))

    def bf16_sparse_decode(*args, **kwargs):
        calls.append(("bf16", kwargs["swa_k_cache"].dtype))

    module = _module(
        patch_deepseek_v4_rocm_bf16_cache.TARGET_MODULE,
        dequantize_and_gather_k_cache=packed_fp8_gather,
        rocm_sparse_attn_decode=fp8_sparse_decode,
    )
    monkeypatch.setattr(
        patch_deepseek_v4_rocm_bf16_cache,
        "bf16_sparse_attn_decode",
        bf16_sparse_decode,
        raising=False,
    )
    patch_deepseek_v4_rocm_bf16_cache.apply_to_module(module)

    common = dict(
        q=object(),
        kv_cache=None,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=object(),
        swa_lens=object(),
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=None,
        scale=1.0,
        head_dim=576,
        nope_head_dim=512,
        rope_head_dim=64,
        output=object(),
    )
    module.rocm_sparse_attn_decode(
        swa_k_cache=torch.empty((1, 1, 576), dtype=torch.bfloat16),
        **common,
    )
    module.rocm_sparse_attn_decode(
        swa_k_cache=torch.empty((1,), dtype=torch.uint8),
        **common,
    )

    assert calls == [
        ("bf16", torch.bfloat16),
        ("fp8", torch.uint8),
    ]


@pytest.mark.parametrize(
    "weight",
    (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
    ),
)
def test_compressor_mm_accepts_hcu_nn_and_upstream_nt_layouts(
    weight: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        patch_deepseek_v4_attention.torch,
        "mm",
        lambda lhs, rhs, *, out_dtype: lhs @ rhs.to(out_dtype),
    )
    hidden = torch.tensor([[2.0, 3.0]])

    result = patch_deepseek_v4_attention._compressor_mm(hidden, weight)

    torch.testing.assert_close(result, torch.tensor([[8.0, 18.0, 28.0]]))


def test_attention_parallel_projection_preserves_rocm_fused_weight_override() -> None:
    """The HCU wrapper must not reinterpret a preshuffled fused Q/KV weight."""

    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
        ):
            del vllm_config, prefix, topk_indices_buffer, aux_stream_list
            self.aux_stream_list = None
            self.ln_events = [object(), object(), object(), object()]
            self.compressor = None
            self.indexer = None

        def _fused_wqa_wkv_gemm(self, hidden_states):
            return hidden_states + 7

        def fused_wqa_wkv(self, hidden_states):
            del hidden_states
            raise AssertionError(
                "preshuffled ROCm fused Q/KV weight used the generic linear path"
            )

        def _run_parallel_input_projections(self, hidden_states):
            return hidden_states, None, None, None

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            del positions, llama_4_scaling
            return hidden_states

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            del kv, positions, attn_metadata
            return q

    def execute_in_parallel(
        main_fn,
        aux_fns,
        start_event,
        done_events,
        aux_streams,
        enable,
    ):
        del start_event, done_events, aux_streams, enable
        return main_fn(), tuple(fn() if fn is not None else None for fn in aux_fns)

    module = _module(
        patch_deepseek_v4_attention.TARGET_MODULE,
        DeepseekV4Attention=DeepseekV4Attention,
        execute_in_parallel=execute_in_parallel,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1),
    )
    patch_deepseek_v4_attention.apply_to_module(module)
    attention = DeepseekV4Attention(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(expert_dtype="fp8")
            ),
            quant_config=None,
        ),
        "layer.attn",
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    result = attention._run_parallel_input_projections(hidden_states)

    torch.testing.assert_close(result[0], hidden_states + 7)
    assert result[1:] == (None, None, None)


def test_attention_fp8_ds_mla_insert_uses_non_pcp_lightop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.dtype, torch.dtype, bool, object, int]] = []

    def lightop_insert(
        q,
        kv,
        kv_norm_weight,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
    ) -> None:
        del kv, cos_sin_cache, eps
        calls.append(
            (
                positions.dtype,
                slot_mapping.dtype,
                slot_mapping.is_contiguous(),
                kv_norm_weight,
                block_size,
            )
        )
        q.add_(4)
        cache.fill_(9)

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    lightop_attention = ModuleType("lightop.attention")
    lightop_attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32 = (
        lightop_insert
    )
    lightop.attention = lightop_attention  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "lightop",
        lightop,
    )
    monkeypatch.setitem(sys.modules, "lightop.attention", lightop_attention)

    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
        ):
            del vllm_config, prefix, topk_indices_buffer, aux_stream_list

        def _run_parallel_input_projections(self, hidden_states):
            return hidden_states

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            del positions, llama_4_scaling
            return hidden_states

        def _fused_qnorm_rope_kv_insert(
            self,
            q,
            kv,
            positions,
            attn_metadata,
        ):
            del q, kv, positions, attn_metadata
            return "official"

    module = _module(
        patch_deepseek_v4_attention.TARGET_MODULE,
        DeepseekV4Attention=DeepseekV4Attention,
        execute_in_parallel=lambda *args, **kwargs: None,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1),
    )
    patch_deepseek_v4_attention.apply_to_module(module)

    attention = DeepseekV4Attention(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(expert_dtype="fp8")
            ),
            quant_config=None,
        ),
        "layer.attn",
    )
    attention.swa_cache_layer = SimpleNamespace(
        prefix="layer.swa",
        kv_cache=torch.zeros((2, 16), dtype=torch.uint8),
    )
    attention.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(4))
    attention.eps = 1e-6
    kv_norm_weight = object()
    attention.kv_norm = SimpleNamespace(
        weight=SimpleNamespace(data=kv_norm_weight)
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([3, 99], dtype=torch.int64)[::2],
        block_size=16,
    )
    q = torch.zeros((1, 2, 4))

    result = attention._fused_qnorm_rope_kv_insert(
        q,
        torch.ones((1, 4)),
        torch.tensor([7], dtype=torch.int32),
        {"layer.swa": metadata},
    )

    assert result is q
    assert calls == [(torch.int64, torch.int32, True, kv_norm_weight, 16)]
    assert q.tolist() == [[[4.0] * 4, [4.0] * 4]]
    assert torch.count_nonzero(attention.swa_cache_layer.kv_cache == 9) == 32


@pytest.mark.parametrize(
    ("quant_name", "quant_format", "expert_dtype"),
    (
        ("compressed-tensors", "int-quantized", "int8"),
        # DeepSeek-V4-Flash-0731 mixed W4A8 checkpoints store wo_a as a
        # standalone BF16 tensor without a weight scale.  The SlimQuant
        # facade must therefore leave it unquantized while constructing the
        # attention layer even though the routed experts use INT4.
        ("slimquant_w4a8", None, "int4"),
    ),
)
def test_attention_int8_wo_a_is_excluded_only_during_construction(
    quant_name: str,
    quant_format: str | None,
    expert_dtype: str,
) -> None:
    seen_ignore: list[list[str]] = []
    seen_fp8_ds_mla_layout: list[bool] = []

    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
        ):
            del prefix, topk_indices_buffer, aux_stream_list
            seen_ignore.append(list(vllm_config.quant_config.ignore))
            seen_fp8_ds_mla_layout.append(self.use_fp8_ds_mla_layout)

        def _run_parallel_input_projections(self, hidden_states):
            return hidden_states

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            del positions, llama_4_scaling
            return hidden_states

        def _fused_qnorm_rope_kv_insert(
            self,
            q,
            kv,
            positions,
            attn_metadata,
        ):
            del kv, positions, attn_metadata
            return q

    module = _module(
        patch_deepseek_v4_attention.TARGET_MODULE,
        DeepseekV4Attention=DeepseekV4Attention,
        execute_in_parallel=lambda *args, **kwargs: None,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1),
    )
    quant_config = SimpleNamespace(
        ignore=[],
        quant_format=quant_format,
        get_name=lambda: quant_name,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(expert_dtype=expert_dtype)
        ),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
        quant_config=quant_config,
    )

    patch_deepseek_v4_attention.apply_to_module(module)
    DeepseekV4Attention(vllm_config, "model.layers.3.attn")

    assert seen_ignore == [["model.layers.3.attn.wo_a"]]
    assert seen_fp8_ds_mla_layout == [False]
    assert quant_config.ignore == []


@pytest.mark.parametrize(
    ("is_gfx938", "expected_dtype", "expected_head_dim"),
    (
        (False, torch.bfloat16, 128),
        (True, torch.uint8, 132),
    ),
)
def test_deepseek_v4_indexer_cache_matches_hcu_reader_contract(
    is_gfx938: bool,
    expected_dtype: torch.dtype,
    expected_head_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gfx936 uses the BF16 reader; gfx938 retains its quantized cache."""

    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
        ):
            del vllm_config, prefix, topk_indices_buffer, aux_stream_list

        def _run_parallel_input_projections(self, hidden_states):
            return hidden_states

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            del positions, llama_4_scaling
            return hidden_states

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            del kv, positions, attn_metadata
            return q

    class DeepseekV4Indexer:
        def __init__(
            self,
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            compress_ratio=1,
            prefix="",
            aux_stream=None,
        ):
            del (
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                compress_ratio,
                prefix,
                aux_stream,
            )
            self.head_dim = 128
            self.k_cache = SimpleNamespace(dtype=torch.uint8, head_dim=132)

    def fused_indexer_q_rope_quant(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=False,
    ):
        del (
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            use_fp4,
        )
        return "quantized"

    module = _module(
        patch_deepseek_v4_attention.TARGET_MODULE,
        DeepseekV4Attention=DeepseekV4Attention,
        DeepseekV4Indexer=DeepseekV4Indexer,
        fused_indexer_q_rope_quant=fused_indexer_q_rope_quant,
        execute_in_parallel=lambda *args, **kwargs: None,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1),
    )
    monkeypatch.setattr(
        patch_deepseek_v4_attention,
        "on_gfx938",
        lambda: is_gfx938,
        raising=False,
    )

    patch_deepseek_v4_attention.apply_to_module(module)
    indexer = DeepseekV4Indexer(None, None, 1, 1, None, None, None)

    assert indexer.k_cache.dtype is expected_dtype
    assert indexer.k_cache.head_dim == expected_head_dim
    q_result = module.fused_indexer_q_rope_quant(
        torch.tensor([0]),
        torch.ones((1, 1, 4), dtype=torch.bfloat16),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        torch.ones((1, 1)),
        0.5,
        0.25,
    )
    if is_gfx938:
        assert q_result == "quantized"
    else:
        query, scaled_weights = q_result
        assert query.dtype is torch.bfloat16
        torch.testing.assert_close(scaled_weights, torch.tensor([[0.125]]))


def test_deepseek_v4_bf16_indexer_query_applies_gptj_rope_and_scales_weights():
    q = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    # rotary_dim=4: first half is cos, second half is sin. With cos=0 and
    # sin=1, interleaved GPT-J pairs [a,b] become [-b,a].
    cos_sin_cache = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    weights = torch.tensor([[8.0]])

    rotated_q, scaled_weights = (
        patch_deepseek_v4_attention._bf16_indexer_q_rope(
            torch.tensor([0]),
            q,
            cos_sin_cache,
            weights,
            softmax_scale=0.5,
            head_scale=0.25,
        )
    )

    torch.testing.assert_close(
        rotated_q,
        torch.tensor(
            [[[10.0, 20.0, 30.0, 40.0, -2.0, 1.0, -4.0, 3.0]]],
            dtype=torch.bfloat16,
        ),
    )
    torch.testing.assert_close(scaled_weights, torch.tensor([[1.0]]))


@pytest.mark.parametrize(
    ("quant_name", "quant_format", "expert_dtype"),
    (
        ("compressed-tensors", "float-quantized", "fp8"),
        ("compressed-tensors", "int-quantized", "fp8"),
        ("compressed-tensors", "float-quantized", "int8"),
        ("other-quantizer", "int-quantized", "int8"),
    ),
)
def test_attention_wo_a_exclusion_rejects_other_quantization_schemes(
    quant_name: str,
    quant_format: str,
    expert_dtype: str,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(expert_dtype=expert_dtype)
        ),
        quant_config=SimpleNamespace(
            quant_format=quant_format,
            get_name=lambda: quant_name,
        ),
    )

    assert not patch_deepseek_v4_attention._requires_unquantized_int8_wo_a(
        config
    )


def test_scale_alias_matches_only_non_inv_scale_parameters() -> None:
    scale_alias = patch_deepseek_v4_load_weights._scale_alias

    assert scale_alias("layers.0.attn.weight_scale") == (
        "layers.0.attn.weight_scale_inv"
    )
    assert scale_alias("layers.0.ffn.experts.w13_weight_scale") == (
        "layers.0.ffn.experts.w13_weight_scale_inv"
    )
    assert scale_alias("layers.0.attn.weight_scale_inv") is None
    assert scale_alias("layers.0.attn.weight") is None


def test_load_weights_exposes_channel_scale_alias_only_to_official_loader() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    observed_names: list[set[str]] = []

    class DeepseekV4Model(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            yield "layers.0.attn.fused_wqa_wkv.weight_scale", parameter

        def load_weights(self, weights):
            del weights
            params = dict(self.named_parameters())
            observed_names.append(set(params))
            assert (
                params["layers.0.attn.fused_wqa_wkv.weight_scale_inv"]
                is parameter
            )
            return {"layers.0.attn.fused_wqa_wkv.weight_scale"}

    module = _module(
        patch_deepseek_v4_load_weights.TARGET_MODULE,
        DeepseekV4Model=DeepseekV4Model,
    )
    patch_deepseek_v4_load_weights.apply_to_module(module)
    model = DeepseekV4Model()
    loaded = model.load_weights([("unused", torch.tensor(1.0))])

    assert loaded == {"layers.0.attn.fused_wqa_wkv.weight_scale"}
    assert observed_names == [
        {
            "layers.0.attn.fused_wqa_wkv.weight_scale",
            "layers.0.attn.fused_wqa_wkv.weight_scale_inv",
        }
    ]
    assert dict(model.named_parameters()) == {
        "layers.0.attn.fused_wqa_wkv.weight_scale": parameter
    }


def test_dspark_target_keeps_original_forward_without_aux_layers() -> None:
    calls: list[tuple[object, ...]] = []

    class DeepseekV4DecoderLayer:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
            fuse_heterogeneous_shared_expert=False,
        ) -> None:
            del (
                vllm_config,
                prefix,
                topk_indices_buffer,
                aux_stream_list,
                fuse_heterogeneous_shared_expert,
            )

    class DeepseekV4Model:
        aux_hidden_state_layers: tuple[int, ...] = ()

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=None,
        ):
            calls.append((input_ids, positions, intermediate_tensors, inputs_embeds))
            return "official"

    class DeepseekV4ForCausalLM:
        def __init__(self) -> None:
            self.model = DeepseekV4Model()

    module = _module(
        patch_deepseek_v4_dspark_target.TARGET_MODULE,
        DeepseekV4DecoderLayer=DeepseekV4DecoderLayer,
        DeepseekV4Model=DeepseekV4Model,
        DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
    )
    patch_deepseek_v4_dspark_target.apply_to_module(module)

    model = DeepseekV4Model()
    result = model.forward("ids", "positions", None)

    assert result == "official"
    assert calls == [("ids", "positions", None, None)]
    causal = DeepseekV4ForCausalLM()
    causal.set_aux_hidden_state_layers((4, 8, 12))
    assert causal.model.aux_hidden_state_layers == (4, 8, 12)
    assert causal.supports_eagle3 is True


def test_dspark_target_binds_boltops_mhc_and_preserves_official_norm() -> None:
    bind = getattr(
        patch_deepseek_v4_dspark_target,
        "_bind_deepseek_v4_boltops_mhc",
        None,
    )
    assert callable(bind), "DeepSeek-V4 must expose a model-scoped mHC binder"

    one = torch.tensor(1.0)
    two = torch.tensor(2.0)
    three = torch.tensor(3.0)
    four = torch.tensor(4.0)
    calls: list[tuple[str, tuple[object, ...]]] = []

    class Backend:
        @staticmethod
        def mhc_pre(*args):
            calls.append(("pre", args))
            return one, two, three

        @staticmethod
        def mhc_post(*args):
            calls.append(("post", args))
            return four

        @staticmethod
        def mhc_fused_post_pre(*args):
            calls.append(("fused", args))
            return one, two, three, four

    class Op:
        def __init__(self) -> None:
            self._forward_method = lambda *args, **kwargs: None

    layer = SimpleNamespace(
        mhc_pre=Op(),
        mhc_post=Op(),
        mhc_fused_post_pre=Op(),
    )
    mhc = SimpleNamespace(
        _apply_mhc_norm=lambda value, weight, eps: value + weight + eps
    )
    bind(layer, mhc, Backend)

    pre = layer.mhc_pre._forward_method(
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )
    post = layer.mhc_post._forward_method(one, one, one, one)
    fused = layer.mhc_fused_post_pre._forward_method(
        one,
        one,
        one,
        one,
        one,
        one,
        one,
        1e-5,
        0.0,
        0.0,
        1.0,
        1,
        norm_weight=two,
        norm_eps=0.5,
    )

    assert pre[:2] == (one, two)
    assert torch.equal(pre[2], torch.tensor(5.5))
    assert post is four
    assert fused[:3] == (one, two, three)
    assert torch.equal(fused[3], torch.tensor(6.5))
    assert [name for name, _ in calls] == ["pre", "post", "fused"]
    assert len(calls[0][1]) == 10
    assert len(calls[2][1]) == 14


def test_dspark_target_binds_every_constructed_decoder_mhc_instance() -> None:
    original_forward_method = lambda *args, **kwargs: None

    class Op:
        def __init__(self) -> None:
            self._forward_method = original_forward_method

    class DeepseekV4DecoderLayer:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
            fuse_heterogeneous_shared_expert=False,
        ) -> None:
            del (
                vllm_config,
                prefix,
                topk_indices_buffer,
                aux_stream_list,
                fuse_heterogeneous_shared_expert,
            )
            self.mhc_pre = Op()
            self.mhc_post = Op()
            self.mhc_fused_post_pre = Op()

    class DeepseekV4Model:
        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=None,
        ):
            del input_ids, positions, intermediate_tensors, inputs_embeds
            return None

    class DeepseekV4ForCausalLM:
        pass

    module = _module(
        patch_deepseek_v4_dspark_target.TARGET_MODULE,
        DeepseekV4DecoderLayer=DeepseekV4DecoderLayer,
        DeepseekV4Model=DeepseekV4Model,
        DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
    )
    patch_deepseek_v4_dspark_target.apply_to_module(module)

    layer = DeepseekV4DecoderLayer(object(), "model.layers.0")

    assert layer.mhc_pre._forward_method is not original_forward_method
    assert layer.mhc_post._forward_method is not original_forward_method
    assert layer.mhc_fused_post_pre._forward_method is not original_forward_method


def test_dspark_ragged_copy_grows_to_real_nnz_without_overflow() -> None:
    class DeepseekV4ROCMAiterSparseSWAMetadataBuilder:
        def __init__(self) -> None:
            self.is_dspark = True
            self.noncausal_index_width = 6
            self.window_size = 2
            self._max_tokens = 2
            self.device = "cpu"
            self.decode_swa_ragged_indices_buffer = torch.empty(4, dtype=torch.int32)

    def baseline(
        ragged_indices,
        ragged_indptr,
        ragged_indices_buffer,
        ragged_indptr_buffer,
        num_rows,
        max_entries_per_row,
    ):
        raise AssertionError(
            "baseline capacity is too small: "
            f"{ragged_indices.numel()} > {num_rows * max_entries_per_row}"
        )

    module = _module(
        patch_deepseek_v4_rocm_dspark_metadata.TARGET_MODULE,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder=(
            DeepseekV4ROCMAiterSparseSWAMetadataBuilder
        ),
        _copy_ragged_to_graph_buffers=baseline,
    )
    patch_deepseek_v4_rocm_dspark_metadata.apply_to_module(module)
    builder = DeepseekV4ROCMAiterSparseSWAMetadataBuilder()
    assert builder.decode_swa_ragged_indices_buffer.numel() == 12

    ragged, indptr = module._copy_ragged_to_graph_buffers(
        torch.arange(6, dtype=torch.int32),
        torch.tensor([0, 3, 6], dtype=torch.int32),
        builder.decode_swa_ragged_indices_buffer,
        torch.empty(3, dtype=torch.int32),
        2,
        2,
    )

    assert ragged[:6].tolist() == list(range(6))
    assert indptr.tolist() == [0, 3, 6]


def _compressor_fusion_attention(
    main_weight: torch.Tensor,
    indexer_weight: torch.Tensor,
    calls: list[str],
):
    class DeepseekV4ROCMAiterMLAAttention:
        def __init__(self) -> None:
            self._fused_compressor_weight = None
            self.compressor = SimpleNamespace(
                fused_wkv_wgate=SimpleNamespace(weight=main_weight)
            )
            self.indexer = SimpleNamespace(
                compressor=SimpleNamespace(
                    fused_wkv_wgate=SimpleNamespace(weight=indexer_weight)
                )
            )

        def prepare_compressor_gemm_fusion(self) -> bool:
            calls.append("upstream")
            if main_weight.shape[1] != indexer_weight.shape[1]:
                raise ValueError("DeepSeek V4 compressor weights must share K")
            return True

    return DeepseekV4ROCMAiterMLAAttention


def test_rocm_compressor_fusion_skips_hcu_nn_layout() -> None:
    calls: list[str] = []
    attention_cls = _compressor_fusion_attention(
        torch.empty(4096, 2048),
        torch.empty(4096, 512),
        calls,
    )
    module = _module(
        patch_deepseek_v4_rocm_compressor_fusion.TARGET_MODULE,
        DeepseekV4ROCMAiterMLAAttention=attention_cls,
    )

    patch_deepseek_v4_rocm_compressor_fusion.apply_to_module(module)

    assert attention_cls().prepare_compressor_gemm_fusion() is False
    assert calls == []


def test_rocm_compressor_fusion_keeps_upstream_nt_layout() -> None:
    calls: list[str] = []
    attention_cls = _compressor_fusion_attention(
        torch.empty(2048, 4096),
        torch.empty(512, 4096),
        calls,
    )
    module = _module(
        patch_deepseek_v4_rocm_compressor_fusion.TARGET_MODULE,
        DeepseekV4ROCMAiterMLAAttention=attention_cls,
    )

    patch_deepseek_v4_rocm_compressor_fusion.apply_to_module(module)

    assert attention_cls().prepare_compressor_gemm_fusion() is True
    assert calls == ["upstream"]


def test_rocm_wo_a_cache_accepts_hcu_nn_layout() -> None:
    calls: list[torch.Tensor] = []

    def get_cached_wo_a_bf16(
        wo_a,
        n_local_groups,
        o_lora_rank,
        hidden_dim,
    ):
        del n_local_groups, o_lora_rank, hidden_dim
        calls.append(wo_a.weight)
        return wo_a.weight

    module = _module(
        patch_deepseek_v4_rocm_wo_a_layout.TARGET_MODULE,
        _get_cached_wo_a_bf16=get_cached_wo_a_bf16,
    )
    patch_deepseek_v4_rocm_wo_a_layout.apply_to_module(module)

    # Checkpoint layout is [groups * rank, hidden].  HCU's NN linear loader
    # stores the unquantized parameter transposed as [hidden, groups * rank].
    logical_weight = torch.arange(24).view(6, 4)
    wo_a = SimpleNamespace(weight=logical_weight.T)
    result = module._get_cached_wo_a_bf16(wo_a, 2, 3, 4)

    torch.testing.assert_close(result, logical_weight)
    torch.testing.assert_close(calls[0], logical_weight)


def test_rocm_wo_a_cache_keeps_upstream_layout() -> None:
    def get_cached_wo_a_bf16(
        wo_a,
        n_local_groups,
        o_lora_rank,
        hidden_dim,
    ):
        del n_local_groups, o_lora_rank, hidden_dim
        return wo_a.weight

    module = _module(
        patch_deepseek_v4_rocm_wo_a_layout.TARGET_MODULE,
        _get_cached_wo_a_bf16=get_cached_wo_a_bf16,
    )
    patch_deepseek_v4_rocm_wo_a_layout.apply_to_module(module)

    logical_weight = torch.arange(24).view(6, 4)
    wo_a = SimpleNamespace(weight=logical_weight)

    torch.testing.assert_close(
        module._get_cached_wo_a_bf16(wo_a, 2, 3, 4),
        logical_weight,
    )


def test_mhc_backend_switch_masks_aiter_only_when_hcu_option_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(patch_mhc_backend.TARGET_MODULE, HAS_AITER_MHC=True)
    monkeypatch.setattr(
        "vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC",
        False,
    )

    assert patch_mhc_backend.apply_to_module(module) is True
    assert module.HAS_AITER_MHC is False
    assert patch_mhc_backend.apply_to_module(module) is False


def test_load_weights_rejects_stale_patch_marker() -> None:
    class DeepseekV4Model:
        _vllm_hcu_deepseek_v4_load_weights_applied = True

        def load_weights(self, weights):
            return weights

    module = _module(
        patch_deepseek_v4_load_weights.TARGET_MODULE,
        DeepseekV4Model=DeepseekV4Model,
    )

    with pytest.raises(PatchCompatibilityError, match="marker.*stale"):
        patch_deepseek_v4_load_weights.apply_to_module(module)


def test_attention_patch_rejects_incompatible_forward_signature() -> None:
    class DeepseekV4Attention:
        def _run_parallel_input_projections(self):
            return None

    module = _module(
        patch_deepseek_v4_attention.TARGET_MODULE,
        DeepseekV4Attention=DeepseekV4Attention,
        execute_in_parallel=lambda *args, **kwargs: None,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1),
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_deepseek_v4_attention.apply_to_module(module)


def test_dspark_target_patch_rejects_incompatible_forward_signature() -> None:
    class DeepseekV4DecoderLayer:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
            fuse_heterogeneous_shared_expert=False,
        ) -> None:
            del (
                vllm_config,
                prefix,
                topk_indices_buffer,
                aux_stream_list,
                fuse_heterogeneous_shared_expert,
            )

    class DeepseekV4Model:
        def forward(self, input_ids):
            return input_ids

    class DeepseekV4ForCausalLM:
        pass

    module = _module(
        patch_deepseek_v4_dspark_target.TARGET_MODULE,
        DeepseekV4DecoderLayer=DeepseekV4DecoderLayer,
        DeepseekV4Model=DeepseekV4Model,
        DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_deepseek_v4_dspark_target.apply_to_module(module)


def test_ragged_patch_rejects_incompatible_copy_signature() -> None:
    class DeepseekV4ROCMAiterSparseSWAMetadataBuilder:
        def __init__(self, *args, **kwargs):
            pass

    module = _module(
        patch_deepseek_v4_rocm_dspark_metadata.TARGET_MODULE,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder=(
            DeepseekV4ROCMAiterSparseSWAMetadataBuilder
        ),
        _copy_ragged_to_graph_buffers=lambda ragged: ragged,
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_deepseek_v4_rocm_dspark_metadata.apply_to_module(module)


def test_mhc_patch_rejects_missing_capability_flag() -> None:
    module = _module(patch_mhc_backend.TARGET_MODULE)

    with pytest.raises(PatchCompatibilityError, match="HAS_AITER_MHC"):
        patch_mhc_backend.apply_to_module(module)
