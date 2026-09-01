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

        def attn_gemm_parallel_execute(self, hidden_states):
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


def test_attention_int8_wo_a_is_excluded_only_during_construction() -> None:
    seen_ignore: list[list[str]] = []

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

        def attn_gemm_parallel_execute(self, hidden_states):
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
        quant_format="int-quantized",
        get_name=lambda: "compressed-tensors",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(expert_dtype="int8")
        ),
        quant_config=quant_config,
    )

    patch_deepseek_v4_attention.apply_to_module(module)
    DeepseekV4Attention(vllm_config, "model.layers.3.attn")

    assert seen_ignore == [["model.layers.3.attn.wo_a"]]
    assert quant_config.ignore == []


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
        def attn_gemm_parallel_execute(self):
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
    class DeepseekV4Model:
        def forward(self, input_ids):
            return input_ids

    class DeepseekV4ForCausalLM:
        pass

    module = _module(
        patch_deepseek_v4_dspark_target.TARGET_MODULE,
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
