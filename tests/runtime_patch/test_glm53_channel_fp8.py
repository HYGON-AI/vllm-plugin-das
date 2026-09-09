# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_glm5next_channel_fp8,
    patch_rocm_mla_sparse_metadata,
)
from vllm_hcu.patch.platform.core_fix import patch_vllm_config
from vllm_hcu.patch.worker.op_opt import patch_glm5next_kda_conv_weight


def _fake_glm_model_module() -> ModuleType:
    module = ModuleType(patch_glm5next_channel_fp8.TARGET_MODULE)

    class Glm5NextForConditionalGeneration:
        def __init__(self, *, vllm_config, prefix=""):
            del vllm_config, prefix

    class Glm5NextDecoderLayer:
        def __init__(
            self,
            vllm_config,
            config,
            layer_idx,
            prefix="",
            topk_indices_buffer=None,
            is_mtp_layer=False,
            **kwargs,
        ):
            del (
                vllm_config,
                config,
                layer_idx,
                prefix,
                topk_indices_buffer,
                is_mtp_layer,
                kwargs,
            )

    module.Glm5NextForConditionalGeneration = Glm5NextForConditionalGeneration
    module.Glm5NextDecoderLayer = Glm5NextDecoderLayer
    return module


def _fake_kda_module() -> tuple[ModuleType, type]:
    module = ModuleType(patch_glm5next_kda_conv_weight.TARGET_MODULE)

    class Glm5NextLinearAttention:
        def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
            del qkv_proj_states, g1, beta, core_attn_out
            if self._merged_conv_weight is None:
                def _w(conv):
                    weight = conv.weight
                    return weight.view(weight.size(0), weight.size(2))

                self._merged_conv_weight = torch.cat(
                    [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                    dim=0,
                ).contiguous()
            return self._merged_conv_weight

    module.Glm5NextLinearAttention = Glm5NextLinearAttention
    return module, Glm5NextLinearAttention


def _fake_kda_layer(layer_cls: type, *, nn_layout: bool):
    layer = layer_cls()
    layer.local_projection_size = 5
    layer.conv_size = 4
    layer._merged_conv_weight = None
    logical_weights = []
    for offset, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
        logical = torch.arange(20, dtype=torch.float32).reshape(5, 1, 4) + 100 * offset
        stored = logical.permute(2, 1, 0).contiguous() if nn_layout else logical
        setattr(layer, name, SimpleNamespace(weight=stored))
        logical_weights.append(logical.reshape(5, 4))
    return layer, torch.cat(logical_weights, dim=0).contiguous()


def test_glm5next_kda_restores_each_nn_conv_weight_before_merge(monkeypatch) -> None:
    module, layer_cls = _fake_kda_module()
    monkeypatch.setattr(patch_glm5next_kda_conv_weight, "use_nn_layout", lambda: True)

    assert patch_glm5next_kda_conv_weight.apply_to_module(module) is True
    assert patch_glm5next_kda_conv_weight.apply_to_module(module) is False
    layer, expected = _fake_kda_layer(layer_cls, nn_layout=True)

    merged = layer._forward(None, None, None, None)
    torch.testing.assert_close(merged, expected)
    first_data_ptr = merged.data_ptr()
    assert layer._forward(None, None, None, None).data_ptr() == first_data_ptr


def test_glm5next_kda_leaves_official_layout_untouched(monkeypatch) -> None:
    module, layer_cls = _fake_kda_module()
    monkeypatch.setattr(patch_glm5next_kda_conv_weight, "use_nn_layout", lambda: False)
    patch_glm5next_kda_conv_weight.apply_to_module(module)
    layer, expected = _fake_kda_layer(layer_cls, nn_layout=False)

    torch.testing.assert_close(layer._forward(None, None, None, None), expected)


def test_glm5next_kda_rejects_unknown_nn_conv_shape(monkeypatch) -> None:
    module, layer_cls = _fake_kda_module()
    monkeypatch.setattr(patch_glm5next_kda_conv_weight, "use_nn_layout", lambda: True)
    patch_glm5next_kda_conv_weight.apply_to_module(module)
    layer, _ = _fake_kda_layer(layer_cls, nn_layout=True)
    layer.q_conv1d.weight = torch.empty(7, 1, 3)

    with pytest.raises(RuntimeError, match="incompatible NN layout"):
        layer._forward(None, None, None, None)


def test_hcu_flashmla_sparse_accepts_glm5next_absorbed_head_size() -> None:
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseBackend,
    )

    assert HcuFlashMLASparseBackend.get_supported_head_sizes() == [512, 576]


def test_hcu_glm5next_kpool_cache_keeps_compression_without_deepgemm_page() -> None:
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    attention = ModuleType(patch_glm5next_channel_fp8.ATTENTION_MODULE)

    class DeepseekV32IndexerCache:
        def get_kv_cache_spec(self, vllm_config):
            del vllm_config
            return MLAAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
            )

    class Glm5NextIndexerCache(DeepseekV32IndexerCache):
        def get_kv_cache_spec(self, vllm_config):
            del vllm_config
            raise AssertionError("DeepGEMM requires a 32-state page")

    attention.Glm5NextIndexerCache = Glm5NextIndexerCache
    patch_glm5next_channel_fp8._patch_glm5next_indexer_cache(attention)

    cache = Glm5NextIndexerCache()
    cache._index_kpool = 4
    spec = cache.get_kv_cache_spec(None)
    assert spec.block_size == 64
    assert spec.tokens_per_state == 4
    assert spec.storage_block_size is None
    assert spec.num_states == 16


def test_glm5next_graph_auto_enables_breakable_cudagraph(monkeypatch) -> None:
    from vllm.config.compilation import CUDAGraphMode

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Glm5NextForConditionalGeneration"],
            enforce_eager=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )

    patch_vllm_config._normalize_glm5next_breakable_cudagraph(config)

    assert __import__("os").environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"


def test_rocm_sparse_builder_disables_missing_aiter_metadata_api(
    monkeypatch,
) -> None:
    aiter = ModuleType("aiter")
    aiter.dtypes = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "aiter", aiter)

    module = ModuleType(patch_rocm_mla_sparse_metadata.TARGET_MODULE)

    class ROCMAiterMLASparseMetadataBuilder:
        def __init__(self):
            from aiter import get_mla_metadata_info_v1

            self._use_persistent_metadata = True
            self.metadata_info = get_mla_metadata_info_v1("unused")

    module.ROCMAiterMLASparseMetadataBuilder = ROCMAiterMLASparseMetadataBuilder
    patch_rocm_mla_sparse_metadata.apply_to_module(module)

    builder = ROCMAiterMLASparseMetadataBuilder()
    assert builder._use_persistent_metadata is False
    assert builder.metadata_info == ((0, torch.int32),) * 6
    assert not hasattr(aiter, "get_mla_metadata_info_v1")


def test_kpool_indexer_uses_official_triton_path_without_aiter() -> None:
    module = ModuleType(patch_glm5next_channel_fp8.KPOOL_MODULE)
    module.rocm_aiter_ops = SimpleNamespace(is_enabled=lambda: False)

    class SparseAttnIndexerKpool:
        def forward_hip(
            self,
            hidden_states,
            q_quant,
            k,
            weights,
            *,
            gate_score=None,
            compress_ape=None,
            index_kpool=1,
            positions=None,
        ):
            del hidden_states, q_quant, k, weights
            del gate_score, compress_ape, index_kpool, positions
            return "official-hip"

        def forward_cuda(self, *args, **kwargs):
            return "official-triton", args, kwargs

    module.SparseAttnIndexerKpool = SparseAttnIndexerKpool
    patch_glm5next_channel_fp8._patch_sparse_indexer_kpool(module)

    indexer = SparseAttnIndexerKpool()
    result, args, kwargs = indexer.forward_hip(
        "hidden",
        "query",
        "key",
        "weights",
        gate_score="gate",
        compress_ape="compress",
        index_kpool=4,
        positions="positions",
    )
    assert result == "official-triton"
    assert args == ("hidden", "query", "key", "weights")
    assert kwargs == {
        "gate_score": "gate",
        "compress_ape": "compress",
        "index_kpool": 4,
        "positions": "positions",
    }
    assert indexer.forward_hip(
        "hidden", "query", "key", "weights", index_kpool=1
    ) == "official-hip"


def test_glm5next_forced_sparse_triton_requires_packaged_modules(
    monkeypatch,
) -> None:
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse

    monkeypatch.setattr(sparse, "mqa_logits_module", lambda: None)
    with pytest.raises(RuntimeError, match="fp8_mqa_logits Triton module"):
        sparse.rocm_fp8_mqa_logits(
            torch.empty(1, 1, 1),
            (torch.empty(1, 1), torch.empty(1)),
            torch.empty(1, 1),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            force_aiter_triton=True,
        )

    monkeypatch.setattr(sparse, "paged_mqa_logits_module", lambda: None)
    with pytest.raises(RuntimeError, match="pa_mqa_logits Triton module"):
        sparse.rocm_fp8_paged_mqa_logits(
            torch.empty(1, 1, 1, 1),
            torch.empty(1, 1, 1, 5, dtype=torch.uint8),
            torch.empty(1, 1),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.empty(0),
            1,
            force_aiter_triton=True,
        )


def test_indexer_derives_gate_weight_from_hcu_nn_layout() -> None:
    attention = ModuleType(patch_glm5next_channel_fp8.ATTENTION_MODULE)

    class Indexer:
        def forward(self, hidden_states, qr, positions, rotary_emb):
            del qr, positions, rotary_emb
            if self._wp_fp32 is None:
                self._wp_fp32 = (
                    self.wk_weights_proj.weight.data[self.head_dim :, :]
                    .t()
                    .contiguous()
                    .float()
                )
            return torch.mm(hidden_states.float(), self._wp_fp32)

    attention.Indexer = Indexer
    patch_glm5next_channel_fp8._patch_indexer_nn_layout(attention)

    instance = Indexer()
    instance.head_dim = 3
    instance.n_head = 2
    instance._wp_fp32 = None
    logical_weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    instance.wk_weights_proj = SimpleNamespace(
        weight=SimpleNamespace(data=logical_weight.t().contiguous())
    )

    hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
    result = instance.forward(hidden_states, None, None, None)
    expected_weight = logical_weight[3:, :].t().float()
    torch.testing.assert_close(instance._wp_fp32, expected_weight)
    torch.testing.assert_close(result, hidden_states.float() @ expected_weight)


def test_indexer_keeps_official_output_input_layout() -> None:
    attention = ModuleType(patch_glm5next_channel_fp8.ATTENTION_MODULE)

    class Indexer:
        def forward(self, hidden_states, qr, positions, rotary_emb):
            del qr, positions, rotary_emb
            if self._wp_fp32 is None:
                self._wp_fp32 = (
                    self.wk_weights_proj.weight.data[self.head_dim :, :]
                    .t()
                    .contiguous()
                    .float()
                )
            return torch.mm(hidden_states.float(), self._wp_fp32)

    attention.Indexer = Indexer
    patch_glm5next_channel_fp8._patch_indexer_nn_layout(attention)

    instance = Indexer()
    instance.head_dim = 3
    instance.n_head = 2
    instance._wp_fp32 = None
    logical_weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    instance.wk_weights_proj = SimpleNamespace(
        weight=SimpleNamespace(data=logical_weight.contiguous())
    )

    hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
    result = instance.forward(hidden_states, None, None, None)
    expected_weight = logical_weight[3:, :].t().float()
    torch.testing.assert_close(instance._wp_fp32, expected_weight)
    torch.testing.assert_close(result, hidden_states.float() @ expected_weight)


def test_channel_fp8_projection_kept_in_bf16_is_dequantized() -> None:
    module = _fake_glm_model_module()
    module._FP8_ATTN_PROJS = {
        ".kv_a_proj_with_mqa.": (
            "kv_a",
            "fused_qkv_a_proj",
            1,
            True,
        )
    }

    def official(
        name,
        tensor,
        buf,
        params_dict,
        loaded_params,
        kv_a_pad_size,
    ):
        del name, tensor, buf, params_dict, loaded_params, kv_a_pad_size
        return False

    module._try_load_fp8_attn_proj = official
    patch_glm5next_channel_fp8.apply_to_module(module)

    loaded = []

    def weight_loader(param, tensor, shard_id):
        del param
        loaded.append((tensor, shard_id))

    target = "layers.11.self_attn.fused_qkv_a_proj.weight"
    params = {target: SimpleNamespace(weight_loader=weight_loader)}
    buffered = {}
    loaded_params = set()
    weight = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.bfloat16)

    helper = module._try_load_fp8_attn_proj
    assert helper(
        "layers.11.self_attn.kv_a_proj_with_mqa.weight",
        weight,
        buffered,
        params,
        loaded_params,
        1,
    )
    assert helper(
        "layers.11.self_attn.kv_a_proj_with_mqa.weight_scale",
        scale,
        buffered,
        params,
        loaded_params,
        1,
    )

    expected = torch.nn.functional.pad(
        (weight.float() * scale.float()).to(torch.bfloat16),
        (0, 0, 0, 1),
    )
    torch.testing.assert_close(loaded[0][0], expected)
    assert loaded[0][1] == 1
    assert loaded_params == {target}
    assert buffered == {}


def test_quantized_channel_fp8_target_stays_on_official_loader_path() -> None:
    module = _fake_glm_model_module()
    module._FP8_ATTN_PROJS = {
        ".o_proj.": ("o_proj", "o_proj", None, False)
    }

    def official(
        name,
        tensor,
        buf,
        params_dict,
        loaded_params,
        kv_a_pad_size,
    ):
        del name, tensor, buf, params_dict, loaded_params, kv_a_pad_size
        raise AssertionError("wrapper must leave this tensor to the caller")

    module._try_load_fp8_attn_proj = official
    patch_glm5next_channel_fp8.apply_to_module(module)
    params = {
        "layers.3.self_attn.o_proj.weight": object(),
        "layers.3.self_attn.o_proj.weight_scale": object(),
    }

    assert not module._try_load_fp8_attn_proj(
        "layers.3.self_attn.o_proj.weight_scale",
        torch.ones(2, 1),
        {},
        params,
        set(),
        0,
    )
def test_glm53_flashmla_nope_query_skips_empty_rope_concat():
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        _normalize_nope_query,
    )

    ql_nope = torch.empty(2, 4, 512)
    q_pe = torch.empty(2, 4, 0)
    rope_q = torch.empty(2, 4, 64)

    assert _normalize_nope_query((ql_nope, q_pe)) is ql_nope
    assert _normalize_nope_query((ql_nope, rope_q)) == (ql_nope, rope_q)
    assert _normalize_nope_query(ql_nope) is ql_nope
