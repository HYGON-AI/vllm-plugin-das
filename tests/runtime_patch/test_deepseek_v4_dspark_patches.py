# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.core_fix import patch_config_utils
from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v4_attention,
    patch_deepseek_v4_dspark_target,
    patch_deepseek_v4_load_weights,
    patch_deepseek_v4_rocm_dspark_metadata,
    patch_deepseek_v4_rocm_wo_a_layout,
    patch_deepseek_v41_dspark_load_weights,
    patch_dspark_draft_a2a,
    patch_dspark_query_capacity,
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
            model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
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
    attention.kv_norm = SimpleNamespace(weight=SimpleNamespace(data=kv_norm_weight))
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
        quant_format="int-quantized",
        get_name=lambda: "compressed-tensors",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="int8")),
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

    assert not patch_deepseek_v4_attention._requires_unquantized_int8_wo_a(config)


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
            assert params["layers.0.attn.fused_wqa_wkv.weight_scale_inv"] is parameter
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


def test_dsv41_dspark_load_weights_exposes_both_channel_scale_aliases() -> None:
    direct_parameter = torch.nn.Parameter(torch.ones(1))
    inverse_parameter = torch.nn.Parameter(torch.ones(1))

    class DSparkDeepseekV4ForCausalLM(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            yield "model.layers.0.attn.weight_scale", direct_parameter
            yield "model.layers.1.attn.weight_scale_inv", inverse_parameter

        def load_weights(self, weights):
            del weights
            params = dict(self.named_parameters())
            assert params["model.layers.0.attn.weight_scale_inv"] is direct_parameter
            assert params["model.layers.1.attn.weight_scale"] is inverse_parameter
            return set(params)

    module = _module(
        patch_deepseek_v41_dspark_load_weights.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    patch_deepseek_v41_dspark_load_weights.apply_to_module(module)
    model = DSparkDeepseekV4ForCausalLM()

    assert model.load_weights([("unused", torch.tensor(1.0))]) == {
        "model.layers.0.attn.weight_scale",
        "model.layers.0.attn.weight_scale_inv",
        "model.layers.1.attn.weight_scale",
        "model.layers.1.attn.weight_scale_inv",
    }
    assert dict(model.named_parameters()) == {
        "model.layers.0.attn.weight_scale": direct_parameter,
        "model.layers.1.attn.weight_scale_inv": inverse_parameter,
    }


def test_config_replace_ignores_hcu_runtime_attributes() -> None:
    @dataclass
    class ParallelConfig:
        tensor_parallel_size: int
        cached_world_size: int = field(init=False, default=1)

    def replace(dataclass_instance, /, **kwargs):
        del dataclass_instance, kwargs
        raise AssertionError("HCU wrapper should replace the upstream implementation")

    def is_init_field(cls, name):
        return next(item for item in fields(cls) if item.name == name).init

    module = _module(
        patch_config_utils.TARGET_MODULE,
        replace=replace,
        is_init_field=is_init_field,
    )
    patch_config_utils.apply_to_module(module)
    config = ParallelConfig(tensor_parallel_size=4)
    config._vllm_hcu_deepep_auto = True

    copied = module.replace(config, tensor_parallel_size=8)

    assert copied == ParallelConfig(tensor_parallel_size=8)
    assert not hasattr(copied, "_vllm_hcu_deepep_auto")


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
    module = _mhc_module(
        HAS_AITER_MHC=True,
        HAS_TILELANG_MHC=True,
    )
    monkeypatch.setattr(
        "vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC",
        False,
    )

    assert patch_mhc_backend.apply_to_module(module) is True
    assert module.HAS_AITER_MHC is False
    assert module.HAS_TILELANG_MHC is False
    assert patch_mhc_backend.apply_to_module(module) is False


def test_mhc_backend_keeps_importable_tilelang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _mhc_module(
        HAS_AITER_MHC=False,
        HAS_TILELANG_MHC=True,
    )
    monkeypatch.setattr(
        patch_mhc_backend,
        "_tilelang_runtime_available",
        lambda: True,
    )

    assert patch_mhc_backend.apply_to_module(module) is True
    assert module.HAS_TILELANG_MHC is True


def test_hcu_aiter_mhc_exposes_main_delayed_batch_heuristic() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.aiter_ops import rocm_aiter_ops

    assert rocm_aiter_ops.mhc_fused_post_pre_delayed_prefers_unfused(0) is False
    assert rocm_aiter_ops.mhc_fused_post_pre_delayed_prefers_unfused(1_000_000) is True


def test_mhc_backend_rejects_partial_aiter_delayed_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _mhc_module(
        HAS_AITER_MHC=True,
        HAS_TILELANG_MHC=False,
    )
    monkeypatch.setattr(
        "vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC",
        True,
    )
    monkeypatch.setattr(
        patch_mhc_backend,
        "_aiter_mhc_runtime_complete",
        lambda _module: False,
    )

    assert patch_mhc_backend.apply_to_module(module) is True
    assert module.HAS_AITER_MHC is False


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


def test_dspark_draft_parallel_config_keeps_ep_and_uses_ag_rs() -> None:
    """The DSpark drafter keeps EP8 but avoids the target DeepEP LL backend.

    Upstream keeps the target's ``enable_expert_parallel`` on the drafter, so
    with ``--all2all-backend deepep_low_latency`` the MTP drafter also selects
    DeepEP LL and its buffer allocation OOMs next to the target weights.  The
    HCU patch selects AG-RS for the drafter only.  This matches SGLang's
    speculative ``a2a=none`` behavior for DP+EP: gather DP-local token shards,
    run the local EP expert shard, then reduce-scatter the partial outputs.
    It must not be replaced by ``NoDPEP``, which would leave each rank with
    only its pre-gather token shard.
    """

    from dataclasses import dataclass as _dataclass
    from dataclasses import fields as _fields

    @_dataclass
    class ParallelConfig:
        pipeline_parallel_size: int = 1
        tensor_parallel_size: int = 1
        enable_expert_parallel: bool = True
        all2all_backend: str = "deepep_low_latency"
        enable_eplb: bool = False
        enable_elastic_ep: bool = False

    def replace(dataclass_instance, /, **kwargs):
        init_fields = {item.name for item in _fields(dataclass_instance) if item.init}
        updated = {name: getattr(dataclass_instance, name) for name in init_fields}
        updated.update(kwargs)
        return type(dataclass_instance)(**updated)

    def original_parallel_config(parallel_config, tensor_parallel_size):
        return replace(
            parallel_config,
            tensor_parallel_size=tensor_parallel_size,
            enable_eplb=False,
            enable_elastic_ep=False,
        )

    def load_dspark_model(target_model, vllm_config):
        return target_model, vllm_config

    target = ParallelConfig(tensor_parallel_size=1, enable_expert_parallel=True)
    module = _module(
        patch_dspark_draft_a2a.TARGET_MODULE,
        _get_dspark_parallel_config=original_parallel_config,
        load_dspark_model=load_dspark_model,
    )
    patch_dspark_draft_a2a.apply_to_module(module)

    result = module._get_dspark_parallel_config(target, 1)

    assert result.enable_expert_parallel is True
    assert result.all2all_backend == "allgather_reducescatter"
    assert result.tensor_parallel_size == 1
    assert result.enable_eplb is False
    # idempotent: a second application must be a no-op, not a double wrap
    assert patch_dspark_draft_a2a.apply_to_module(module) is False
    again = module._get_dspark_parallel_config(target, 1)
    assert again.enable_expert_parallel is True
    assert again.all2all_backend == "allgather_reducescatter"


def test_dspark_draft_parallel_config_keeps_dense_drafter_untouched() -> None:
    """A drafter that already runs without EP is returned unchanged."""

    from dataclasses import dataclass as _dataclass
    from dataclasses import fields as _fields

    @_dataclass
    class ParallelConfig:
        tensor_parallel_size: int = 1
        enable_expert_parallel: bool = False

    def replace(dataclass_instance, /, **kwargs):
        init_fields = {item.name for item in _fields(dataclass_instance) if item.init}
        updated = {name: getattr(dataclass_instance, name) for name in init_fields}
        updated.update(kwargs)
        return type(dataclass_instance)(**updated)

    def original_parallel_config(parallel_config, tensor_parallel_size):
        return replace(parallel_config, tensor_parallel_size=tensor_parallel_size)

    def load_dspark_model(target_model, vllm_config):
        return target_model, vllm_config

    target = ParallelConfig(tensor_parallel_size=1, enable_expert_parallel=False)
    module = _module(
        patch_dspark_draft_a2a.TARGET_MODULE,
        _get_dspark_parallel_config=original_parallel_config,
        load_dspark_model=load_dspark_model,
    )
    patch_dspark_draft_a2a.apply_to_module(module)

    result = module._get_dspark_parallel_config(target, 1)
    assert result.enable_expert_parallel is False
    assert result.tensor_parallel_size == 1


def test_dspark_draft_parallel_config_rejects_signature_drift() -> None:
    def wrong_signature(parallel_config, tensor_parallel_size, extra):
        del parallel_config, tensor_parallel_size, extra

    module = _module(
        patch_dspark_draft_a2a.TARGET_MODULE,
        _get_dspark_parallel_config=wrong_signature,
        load_dspark_model=lambda target_model, vllm_config: (target_model, vllm_config),
    )
    with pytest.raises(PatchCompatibilityError):
        patch_dspark_draft_a2a.apply_to_module(module)


@pytest.mark.parametrize("backend", [None, "triton", "deep_gemm"])
def test_dspark_draft_load_keeps_backend_owners_consistent(backend) -> None:
    @dataclass
    class KernelConfig:
        moe_backend: str = "deep_gemm"

    @dataclass
    class SpeculativeConfig:
        moe_backend: str | None = None

    @dataclass
    class VllmConfig:
        kernel_config: KernelConfig = field(default_factory=KernelConfig)
        speculative_config: SpeculativeConfig = field(default_factory=SpeculativeConfig)
        additional_config: dict = field(
            default_factory=lambda: {
                "hcu": {"moe_backend": "deep_gemm"},
                "unrelated": 7,
            }
        )

    def original_parallel_config(parallel_config, tensor_parallel_size):
        return parallel_config

    observed: list[str] = []

    def load_dspark_model(target_model, vllm_config):
        from vllm_hcu.patch.config import get_hcu_config

        observed.append(vllm_config.kernel_config.moe_backend)
        assert get_hcu_config(vllm_config).moe_backend == (
            "deep_gemm" if backend == "deep_gemm" else "auto"
        )
        assert vllm_config.additional_config["unrelated"] == 7
        return target_model

    module = _module(
        patch_dspark_draft_a2a.TARGET_MODULE,
        _get_dspark_parallel_config=original_parallel_config,
        load_dspark_model=load_dspark_model,
    )
    patch_dspark_draft_a2a.apply_to_module(module)
    config = VllmConfig()
    config.speculative_config.moe_backend = backend

    model = torch.nn.Module()
    assert module.load_dspark_model(model, config) is model
    assert observed == [backend or "triton"]
    assert config.kernel_config.moe_backend == "deep_gemm"
    assert config.additional_config["hcu"] == {"moe_backend": "deep_gemm"}


def test_dspark_ag_rs_prepare_finalize_bypasses_target_manager(monkeypatch) -> None:
    """Gather and reduce through the draft manager, including FP8 scales."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

    calls = []
    manager = SimpleNamespace(
        dispatch=lambda *args, **kwargs: calls.append("dispatch") or args[:3],
        combine=lambda tensor, **kwargs: calls.append("combine") or tensor + 3,
    )

    def target_group():
        raise AssertionError("draft must not call target DeepEP manager")

    monkeypatch.setattr(naive_dp_ep, "get_ep_group", target_group)
    monkeypatch.setattr(
        naive_dp_ep,
        "_quantize_and_setup_dispatch",
        lambda a1, config, defer: (a1, None, None),
    )
    monkeypatch.setattr(
        patch_dspark_draft_a2a, "_make_draft_ag_rs_manager", lambda _: manager
    )
    pf = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=8)
    target_manager = SimpleNamespace()
    patch_dspark_draft_a2a._bind_draft_manager(pf, target_manager)
    hidden = torch.ones(2, 4)
    weights = torch.ones(2, 1)
    ids = torch.zeros(2, 1, dtype=torch.int64)

    prepared = pf.prepare(hidden, weights, ids, 128, None, False, None)
    assert prepared[0] is hidden
    output = torch.empty_like(hidden)
    reducer = SimpleNamespace(apply=lambda **kwargs: hidden * 2)
    pf.finalize(output, hidden, weights, ids, False, reducer)
    torch.testing.assert_close(output, hidden * 5)
    assert calls == ["dispatch", "combine"]
    assert naive_dp_ep.get_ep_group is target_group


def test_dspark_draft_does_not_bind_no_dp_ep() -> None:
    """Protect the SGLang DP-gather semantics from a local-only regression."""

    class MoEPrepareAndFinalizeNoDPEPModular:
        def prepare(self):
            pass

        def finalize(self):
            pass

    target_manager = SimpleNamespace()
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()

    result = patch_dspark_draft_a2a.maybe_bind_draft_ag_rs(
        prepare_finalize, target_manager
    )

    assert result is prepare_finalize
    assert not hasattr(prepare_finalize, "_vllm_hcu_ag_rs_manager")


@pytest.mark.parametrize("num_reqs", [32, 64, 256, 1024])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_dspark_query_capacity_covers_markov_anchors(num_reqs, sample_from_anchor):
    """Expanded queries must fit even when target profiling uses one token/req."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    steps = 5
    width = steps + (not sample_from_anchor)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=num_reqs, max_num_batched_tokens=256
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=steps,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(sample_from_anchor=sample_from_anchor)
            ),
        ),
    )
    draft_config = patch_dspark_query_capacity._draft_capacity_config(config)
    capacity = draft_config.scheduler_config.max_num_batched_tokens
    assert capacity >= num_reqs * width
    assert config.scheduler_config.max_num_batched_tokens == 256

    speculator = object.__new__(DSparkSpeculator)
    speculator._draft_topk = None
    speculator.num_speculative_steps = steps
    speculator.sample_indices = torch.arange(num_reqs * steps)
    speculator.sample_idx_mapping = torch.zeros(num_reqs * steps)
    speculator.sample_pos = torch.zeros(num_reqs * steps)
    speculator._anchor_idx = torch.arange(num_reqs) * width
    ids = torch.arange(capacity, dtype=torch.int32) % 7
    speculator.input_buffers = SimpleNamespace(input_ids=ids)
    speculator.use_confidence_head = False
    speculator.draft_tokens = torch.zeros(num_reqs, steps, dtype=torch.int64)
    seen = []

    def markov_embed(prev):
        seen.append(prev.clone())
        return prev[:, None].float()

    speculator.model = SimpleNamespace(
        compute_draft_logits=lambda hidden: hidden,
        markov_embed=markov_embed,
        markov_bias=lambda embed: embed,
    )
    speculator._sample_logits = lambda logits, *args: logits[:, 0].long() + 1
    speculator._sample_sequential(num_reqs, torch.zeros(num_reqs * steps, 1))

    anchors = ids[speculator._anchor_idx].long()
    torch.testing.assert_close(seen[0], anchors.to(torch.int32))
    for step in range(steps):
        torch.testing.assert_close(speculator.draft_tokens[:, step], anchors + step + 1)
    if (num_reqs - 1) * width >= 256:
        with pytest.raises(IndexError):
            ids[:256][speculator._anchor_idx]


def test_dspark_query_capacity_expands_shared_slots_before_capture():
    seen = []

    class DSparkSpeculator:
        def __init__(self, vllm_config, device):
            self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        def set_attn(
            self,
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        ):
            seen.append(block_tables.slot_mappings)

    module = _module(
        patch_dspark_query_capacity.TARGET_MODULE, DSparkSpeculator=DSparkSpeculator
    )
    assert patch_dspark_query_capacity.apply_to_module(module)
    assert not patch_dspark_query_capacity.apply_to_module(module)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=256),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        ),
    )
    drafter = DSparkSpeculator(config, "cpu")
    slots = torch.arange(512).view(2, 256)
    tables = SimpleNamespace(slot_mappings=slots, max_num_batched_tokens=256)
    drafter.set_attn(None, None, tables, None, None)
    assert tables.slot_mappings.shape == (2, 320)
    torch.testing.assert_close(tables.slot_mappings[:, :256], slots)
    assert torch.all(tables.slot_mappings[:, 256:] == -1)
    assert seen[0] is tables.slot_mappings
    drafter.set_attn(None, None, tables, None, None)
    assert seen[1] is seen[0]
    assert config.scheduler_config.max_num_batched_tokens == 256


@pytest.fixture(autouse=True)
def _isolate_aiter_post_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        patch_mhc_backend, "_aiter_mhc_post_runtime_available", lambda: False
    )


def _mhc_module(**attributes: object) -> ModuleType:
    class MHCPostOp:
        def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

        def forward_native(self, x, residual, post_layer_mix, comb_res_mix):
            return (
                torch.einsum("...ij,...ih->...jh", comb_res_mix, residual.float())
                + post_layer_mix * x.unsqueeze(-2).float()
            ).to(residual.dtype)

    return _module(patch_mhc_backend.TARGET_MODULE, MHCPostOp=MHCPostOp, **attributes)


def _enable_aiter_post_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        patch_mhc_backend, "_aiter_mhc_post_runtime_available", lambda: True
    )
    monkeypatch.setattr(
        patch_mhc_backend, "_aiter_mhc_post_supported", lambda *args: True
    )
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC", True)
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC_POST", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")


def test_mhc_post_uses_aiter_when_delayed_pre_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing delayed Pre interface must not disable a working Post op."""
    _enable_aiter_post_probe(monkeypatch)
    module = _mhc_module(HAS_AITER_MHC=False, HAS_TILELANG_MHC=False)
    expected = torch.tensor([7.0])
    calls = []

    def aiter_post(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(patch_mhc_backend, "_call_aiter_mhc_post", aiter_post)
    patch_mhc_backend.apply_to_module(module)
    tensors = tuple(torch.tensor([float(i)]) for i in range(4))

    actual = module.MHCPostOp().forward_hip(*tensors)

    assert actual is expected
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], tensors))
    assert module.HAS_AITER_MHC is False


@pytest.mark.parametrize("disabled", ("master", "aiter", "post"))
def test_mhc_post_switch_restores_original_dispatch(
    disabled: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_aiter_post_probe(monkeypatch)
    if disabled == "master":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    elif disabled == "aiter":
        monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC", False)
    else:
        monkeypatch.setattr(
            "vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC_POST", False
        )
    monkeypatch.setattr(
        patch_mhc_backend,
        "_call_aiter_mhc_post",
        lambda *args: pytest.fail("disabled Post reached AITER"),
    )
    module = _mhc_module(HAS_AITER_MHC=False, HAS_TILELANG_MHC=False)
    op = module.MHCPostOp()
    x = torch.ones(1, 8, dtype=torch.bfloat16)
    residual = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    post = torch.ones(1, 4, 1)
    comb = torch.eye(4).unsqueeze(0)
    expected = op.forward_hip(x, residual, post, comb)
    patch_mhc_backend.apply_to_module(module)

    torch.testing.assert_close(op.forward_hip(x, residual, post, comb), expected)


@pytest.mark.parametrize("explicit", (None, "legacy", "auto"))
def test_mhc_post_selects_v3_before_warmup_and_preserves_user_override(
    explicit: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_aiter_post_probe(monkeypatch)
    if explicit is None:
        monkeypatch.delenv("AITER_MHC_POST_KERNEL", raising=False)
    else:
        monkeypatch.setenv("AITER_MHC_POST_KERNEL", explicit)
    module = _mhc_module(HAS_AITER_MHC=False, HAS_TILELANG_MHC=False)
    patch_mhc_backend.apply_to_module(module)

    import os

    assert os.environ["AITER_MHC_POST_KERNEL"] == (explicit or "v3")
    assert patch_mhc_backend.apply_to_module(module) is False


@pytest.mark.parametrize("hidden_size", (256, 512, 768, 1280, 2560))
def test_mhc_post_rejects_v3_shapes_with_invalid_b32_b64_output(
    hidden_size: int,
) -> None:
    """These installed HIP variants failed direct numerical probes on gfx938."""
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    mode = FakeTensorMode()

    def fake(shape, dtype):
        return FakeTensor(
            mode,
            torch.empty(shape, dtype=dtype, device="meta"),
            torch.device("cuda:0"),
        )

    tensors = (
        fake((2, hidden_size), torch.bfloat16),
        fake((2, 4, hidden_size), torch.bfloat16),
        fake((2, 4, 1), torch.float32),
        fake((2, 4, 4), torch.float32),
    )
    assert patch_mhc_backend._aiter_mhc_post_supported(*tensors) is False


def test_mhc_post_unavailable_runtime_keeps_original_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AITER_MHC_POST_KERNEL", raising=False)
    module = _mhc_module(HAS_AITER_MHC=False, HAS_TILELANG_MHC=False)
    original = module.MHCPostOp.forward_hip
    patch_mhc_backend.apply_to_module(module)
    assert module._vllm_hcu_original_post_forward_hip is original
    import os

    assert "AITER_MHC_POST_KERNEL" not in os.environ


def test_mhc_post_rejects_incompatible_target_before_mutating_flags() -> None:
    module = _mhc_module(HAS_AITER_MHC=True, HAS_TILELANG_MHC=True)
    module.MHCPostOp.forward_hip = lambda self, x: x
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_mhc_backend.apply_to_module(module)
    assert module.HAS_AITER_MHC is True
    assert module.HAS_TILELANG_MHC is True


def test_dspark_trailing_mhc_post_uses_same_aiter_route(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_mhc_post_dspark

    _enable_aiter_post_probe(monkeypatch)
    expected = torch.tensor([5.0])
    monkeypatch.setattr(
        patch_mhc_backend, "_call_aiter_mhc_post", lambda *args: expected
    )

    def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
        return residual

    module = _module(patch_mhc_post_dspark.TARGET_MODULE, mhc_post_torch=mhc_post_torch)
    assert patch_mhc_post_dspark.apply_to_module(module) is True
    assert module.mhc_post_torch(None, None, None, None) is expected
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    residual = torch.tensor([2.0])
    assert module.mhc_post_torch(None, residual, None, None) is residual
    assert patch_mhc_post_dspark.apply_to_module(module) is False


def test_dspark_mhc_post_rejects_changed_signature():
    from vllm_hcu.patch.worker.op_opt import patch_mhc_post_dspark

    module = _module(patch_mhc_post_dspark.TARGET_MODULE, mhc_post_torch=lambda x: x)
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_mhc_post_dspark.apply_to_module(module)


@pytest.fixture
def staged_mhc(monkeypatch):
    """Model the staged AITER contract to check delayed mixing on CPU."""
    from vllm_hcu.model_executor.layers import mhc_runtime as runtime
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch  # noqa: F401

    def project(gemm, sqrsum, source, fn, tile_k, use_tf32):
        assert not use_tf32
        flat = source.flatten(1).float()
        gemm[0].copy_(flat @ fn.t())
        sqrsum[0].copy_(flat.square().sum(-1))

    def coefficients(gemm, sqrsum, scale, base, hidden, eps, pre_eps,
                     sinkhorn_eps, post_mult, repeat):
        mixes = gemm.sum(0) * torch.rsqrt(sqrsum.sum(0) / (4 * hidden) + eps)[:, None]
        pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + pre_eps
        post = torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8]) * post_mult
        comb = torch.softmax((mixes[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4), -1)
        comb = comb + sinkhorn_eps
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
        for _ in range(repeat - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
        return pre, post, comb

    def big_fuse(post, comb, layer_input, gemm, sqrsum, scale, base, residual,
                 eps, pre_eps, sinkhorn_eps, post_mult, repeat):
        pre, post_value, comb_value = coefficients(
            gemm, sqrsum, scale, base, residual.shape[-1], eps, pre_eps,
            sinkhorn_eps, post_mult, repeat,
        )
        post.copy_(post_value.unsqueeze(-1))
        comb.copy_(comb_value)
        layer_input.copy_((pre.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype))

    monkeypatch.setattr(runtime, "runtime", lambda: runtime.MhcRuntime(
        project=project, big_fuse=big_fuse,
    ))

    def pre_mix(gemm, sqrsum, scale, base, hc, width, eps, pre_eps):
        mixes = gemm.sum(0) * torch.rsqrt(sqrsum.sum(0) / width + eps)[:, None]
        return torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + pre_eps

    monkeypatch.setattr(torch.ops.vllm, "mhc_pre_mix_triton", pre_mix, raising=False)
    monkeypatch.setattr(torch.ops.vllm, "hc_collapse_triton",
                        lambda residual, pre: (residual.float() * pre.unsqueeze(-1)).sum(1).to(residual.dtype),
                        raising=False)
    return runtime


@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("identity_entry", [False, True])
def test_delayed_pre_preserves_previous_mix_and_embedding_projection(
    staged_mhc, embedding, identity_entry,
):
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    torch.manual_seed(18)
    residual = torch.randn(3, 4, 32, dtype=torch.bfloat16)
    x = torch.randn(3, 32, dtype=torch.bfloat16) if embedding else None
    fn = torch.randn(24, 32 if embedding else 128) * 0.05
    scale = torch.ones(3)
    base = torch.randn(24) * 0.1
    previous = None if identity_entry else torch.rand(3, 4)
    args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    actual = staged_mhc.delayed_pre(*args, pre_mix=previous, x=x)
    expected = mhc_pre_delayed_torch(*args, pre_mix=previous, x=x)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
    if identity_entry:
        assert actual[2].untyped_storage().data_ptr() != residual.untyped_storage().data_ptr()


def test_tilelang_pre_receives_compact_projection_and_unfused_norm(staged_mhc, monkeypatch):
    runtime = staged_mhc
    native = runtime.runtime()
    seen = []

    def tile_pre(gemm, sqrsum, scale, base, flat, post, comb, layer_input,
                 hidden, eps, pre_eps, sinkhorn_eps, post_mult, repeat,
                 n_splits, mhc_mult):
        assert gemm.is_contiguous() and gemm.shape[-1] == 24
        assert flat.shape == (3, 4, 32)
        assert post.shape == (3, 4) and comb.shape == (3, 16)
        assert n_splits == 1 and mhc_mult == 4
        seen.append(True)
        native.big_fuse(post.unsqueeze(-1), comb.reshape(3, 4, 4),
                        layer_input, gemm, sqrsum, scale, base, flat, eps, pre_eps,
                        sinkhorn_eps, post_mult, repeat)

    monkeypatch.setattr(runtime, "runtime", lambda: runtime.MhcRuntime(
        project=native.project, big_fuse=native.big_fuse, tilelang_pre=tile_pre,
    ))
    residual = torch.randn(3, 4, 32, dtype=torch.bfloat16)
    args = (residual, torch.randn(24, 128), torch.ones(3), torch.zeros(24),
            1e-6, 1e-6, 1e-6, 2.0, 20)
    output = runtime.pre(*args)
    assert seen == [True]
    # The operator accepts no decoder norm weight; the layer must apply it.
    assert output[2].shape == (3, 32)


def test_tilelang_post_squeezes_mix_for_sglang_contract(monkeypatch):
    from vllm_hcu.model_executor.layers import mhc_runtime as runtime

    residual = torch.ones(2, 4, 32, dtype=torch.bfloat16)
    x = torch.ones(2, 32, dtype=torch.bfloat16)
    post = torch.ones(2, 4, 1)
    comb = torch.eye(4).repeat(2, 1, 1)

    def tile_post(x, residual, post, comb):
        assert post.shape == (2, 4)
        return (residual + post.unsqueeze(-1) * x.unsqueeze(1)).to(residual.dtype)

    monkeypatch.setattr(runtime, "runtime", lambda: runtime.MhcRuntime(tilelang_post=tile_post))
    torch.testing.assert_close(runtime.post(x, residual, post, comb), residual * 2)


def test_missing_tilelang_enum_does_not_disable_native_mhc(monkeypatch):
    from vllm_hcu.model_executor.layers import mhc_runtime as runtime

    native = SimpleNamespace(mhc_pre=lambda: None, mhc_post=lambda: None,
                             mhc_pre_gemm_sqrsum=lambda: None,
                             mhc_pre_big_fuse=lambda: None)
    def load(name):
        if name == "aiter.ops.tilelang":
            raise ImportError("cannot import name 'Enum'")
        return native

    monkeypatch.setattr(runtime.importlib, "import_module", load)
    runtime.get_runtime.cache_clear()
    try:
        ops = runtime.get_runtime(True)
        assert ops.pre is native.mhc_pre and ops.post is native.mhc_post
        assert ops.delayed_available and ops.tilelang_pre is None
    finally:
        runtime.get_runtime.cache_clear()


def test_real_fused_mhc_reorders_outputs_and_missing_interface_decomposes(monkeypatch):
    from vllm_hcu.model_executor.layers import mhc_runtime as runtime

    tensors = [torch.tensor([float(i)]) for i in range(4)]
    monkeypatch.setattr(runtime, "runtime", lambda: runtime.MhcRuntime(
        fused=lambda *args: tuple(tensors),
    ))
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC_FUSED", True)
    args = (torch.ones(1),) * 7 + (1e-6, 1e-6, 1e-6, 2.0, 20)
    result = runtime.fused_post_pre(*args)
    assert result == (tensors[3], tensors[0], tensors[1], tensors[2])
    monkeypatch.setattr(runtime, "runtime", lambda: runtime.MhcRuntime())
    monkeypatch.setattr(runtime, "post", lambda *args: tensors[3])
    monkeypatch.setattr(runtime, "pre", lambda *args: tuple(tensors[:3]))
    assert runtime.fused_post_pre(*args) == result


def test_empty_delayed_pre_does_not_launch_aiter(monkeypatch):
    from vllm_hcu.model_executor.layers import mhc_runtime as runtime

    monkeypatch.setattr(runtime, "runtime", lambda: pytest.fail("empty input launched AITER"))
    output = runtime.delayed_pre(torch.empty(0, 4, 1024, dtype=torch.bfloat16),
                                 torch.empty(24, 4096), torch.ones(3), torch.zeros(24),
                                 1e-6, 1e-6, 1e-6, 2.0, 20)
    assert [t.shape for t in output] == [(0, 4, 1), (0, 4, 4), (0, 1024), (0, 4)]


def test_post_off_stays_native_when_delayed_pre_becomes_available(monkeypatch):
    """Enabling Pre must not silently reactivate a disabled Post operator."""
    _enable_aiter_post_probe(monkeypatch)
    monkeypatch.setattr("vllm_hcu.platforms.envs.VLLM_HCU_USE_AITER_MHC_POST", False)
    monkeypatch.setattr(patch_mhc_backend, "_aiter_mhc_runtime_complete", lambda m: True)
    module = _mhc_module(HAS_AITER_MHC=True, HAS_TILELANG_MHC=False)

    def original_aiter(self, x, residual, post_layer_mix, comb_res_mix):
        pytest.fail("disabled Post used the original AITER route")

    module.MHCPostOp.forward_hip = original_aiter
    patch_mhc_backend.apply_to_module(module)
    residual = torch.ones(2, 4, 16)
    post = torch.ones(2, 4, 1)
    comb = torch.eye(4).expand(2, -1, -1)
    output = module.MHCPostOp().forward_hip(torch.ones(2, 16), residual, post, comb)
    torch.testing.assert_close(output, residual * 2)


def test_mhc_pre_changed_signature_is_rejected_before_any_patch(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_mhc_pre_backend

    class MHCPreOp:
        def forward_hip(self, residual):
            return residual

        def forward_native(self, residual):
            return residual

    original = MHCPreOp.forward_hip
    module = _module(patch_mhc_pre_backend.TARGET_MODULE, MHCPreOp=MHCPreOp)
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_mhc_pre_backend.apply_to_module(module)
    assert MHCPreOp.forward_hip is original
    assert not hasattr(module, "_vllm_hcu_mhc_pre_backend_applied")
