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


def test_glm5next_shared_gate_uses_deepgemm_only_when_opted_in(monkeypatch) -> None:
    calls = []

    class Kernel:
        _hcu_fp8_backend = "lightop"

        def apply_scaled_mm(self, *, A, B, As, Bs, out_dtype, bias, output_shape):
            del A, B, As, Bs, bias
            return torch.full(output_shape, -1, dtype=out_dtype)

    def fake_module():
        module = ModuleType(patch_glm5next_channel_fp8.TARGET_MODULE)

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
                del vllm_config, config, layer_idx, prefix, topk_indices_buffer, kwargs
                self.is_mtp_layer = is_mtp_layer
                self.other_kernel = Kernel()
                gate = SimpleNamespace(scheme=SimpleNamespace(fp8_linear=Kernel()))
                self.mlp = SimpleNamespace(
                    shared_experts=SimpleNamespace(gate_up_proj=gate)
                )

        module.Glm5NextDecoderLayer = Glm5NextDecoderLayer
        return module

    deepgemm = ModuleType("deepgemm")

    def fp8_gemm(activation, weight, output):
        calls.append((activation, weight))
        output.fill_(7)

    deepgemm.fp8_gemm = fp8_gemm
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    monkeypatch.setenv("VLLM_HCU_GLM53_GATE_UP_DEEPGEMM", "1")

    module = fake_module()
    assert patch_glm5next_channel_fp8._patch_glm5next_shared_gate_deepgemm(module)
    layer = module.Glm5NextDecoderLayer(None, None, 3)
    gate_kernel = layer.mlp.shared_experts.gate_up_proj.scheme.fp8_linear
    A = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
    B = torch.ones((3, 4), dtype=torch.float8_e4m3fn).t()
    kwargs = {
        "A": A,
        "B": B,
        "As": torch.ones((2, 1), dtype=torch.float32),
        "Bs": torch.ones((3, 1), dtype=torch.float32),
        "out_dtype": torch.bfloat16,
        "bias": None,
        "output_shape": (2, 3),
    }
    assert torch.all(gate_kernel.apply_scaled_mm(**kwargs) == 7)
    assert len(calls) == 1
    # An equal element count is insufficient: the original scaled-mm contract
    # rejects transposed scale layouts, so they must stay on the original path.
    assert torch.all(
        gate_kernel.apply_scaled_mm(
            **{**kwargs, "As": torch.ones((1, 2), dtype=torch.float32)}
        )
        == -1
    )
    assert torch.all(
        gate_kernel.apply_scaled_mm(
            **{**kwargs, "Bs": torch.ones((1, 3), dtype=torch.float32)}
        )
        == -1
    )
    assert torch.all(
        gate_kernel.apply_scaled_mm(
            **{**kwargs, "As": torch.ones((2, 2), dtype=torch.float32)[:, :1]}
        )
        == -1
    )
    assert torch.all(
        gate_kernel.apply_scaled_mm(
            **{**kwargs, "Bs": torch.ones((3, 2), dtype=torch.float32)[:, :1]}
        )
        == -1
    )
    assert len(calls) == 1
    assert torch.all(layer.other_kernel.apply_scaled_mm(**kwargs) == -1)
    assert torch.all(
        gate_kernel.apply_scaled_mm(**{**kwargs, "B": B.contiguous()}) == -1
    )
    assert len(calls) == 1

    mtp_layer = module.Glm5NextDecoderLayer(None, None, 41, is_mtp_layer=True)
    mtp_kernel = mtp_layer.mlp.shared_experts.gate_up_proj.scheme.fp8_linear
    assert torch.all(mtp_kernel.apply_scaled_mm(**kwargs) == -1)
    assert len(calls) == 1

    monkeypatch.delenv("VLLM_HCU_GLM53_GATE_UP_DEEPGEMM")
    disabled_module = fake_module()
    patch_glm5next_channel_fp8._patch_glm5next_shared_gate_deepgemm(
        disabled_module
    )
    disabled_layer = disabled_module.Glm5NextDecoderLayer(None, None, 3)
    disabled_kernel = (
        disabled_layer.mlp.shared_experts.gate_up_proj.scheme.fp8_linear
    )
    assert torch.all(disabled_kernel.apply_scaled_mm(**kwargs) == -1)
    assert len(calls) == 1


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


@pytest.mark.parametrize(
    "architecture",
    [
        "Glm5NextForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MTP",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeMTP",
    ],
)
def test_hcu_recurrent_graph_auto_enables_breakable_cudagraph(
    monkeypatch, architecture: str
) -> None:
    from vllm.config.compilation import CUDAGraphMode

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[architecture],
            enforce_eager=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )

    patch_vllm_config._normalize_hcu_breakable_cudagraph(config)

    assert __import__("os").environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"


def test_hcu_breakable_normalization_wraps_official_enable_hook(monkeypatch) -> None:
    from vllm.config.compilation import CompilationMode

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            enforce_eager=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=None,
            mode=CompilationMode.VLLM_COMPILE,
        ),
    )

    def official_maybe_enable(self) -> bool:
        enabled = (
            __import__("os").environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
        )
        if enabled:
            self.compilation_config.mode = CompilationMode.NONE
        return enabled

    wrapped = patch_vllm_config._wrap_maybe_enable_breakable_cudagraph(
        official_maybe_enable
    )

    assert wrapped(config) is True
    assert config.compilation_config.mode == CompilationMode.NONE


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


@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
@pytest.mark.parametrize(
    "layout", ["regular", "collapsed", "transposed", "contiguous", "five_dim"]
)
def test_glm5next_upstream_paged_mqa_uses_physical_page_after_layout_normalization(
    monkeypatch, page_size: int, layout: str
) -> None:
    """The wrapper must classify the same physical page as the dispatcher."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

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
            return None

    kpool = ModuleType(patch_glm5next_channel_fp8.KPOOL_MODULE)
    kpool.SparseAttnIndexerKpool = SparseAttnIndexerKpool
    monkeypatch.setattr(hcu_sparse, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        upstream_sparse,
        "rocm_fp8_paged_mqa_logits",
        upstream_sparse.rocm_fp8_paged_mqa_logits,
    )
    calls = []
    output = torch.ones((1, 2))

    def dispatcher(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(hcu_sparse, "rocm_fp8_paged_mqa_logits", dispatcher)
    patch_glm5next_channel_fp8._patch_sparse_indexer_kpool(kpool)
    cache = torch.empty((2, page_size, 1, 132), dtype=torch.uint8)
    supplied_cache = {
        "regular": cache,
        "collapsed": cache[:, :1],
        "transposed": cache.transpose(1, 2),
        "contiguous": torch.empty((2, 1, page_size, 132), dtype=torch.uint8),
        "five_dim": cache.unsqueeze(2),
    }[layout]
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    result = upstream_sparse.rocm_fp8_paged_mqa_logits(
        torch.empty((1, 1, 1, 128), dtype=torch.float8_e4m3fn),
        supplied_cache,
        torch.ones((1, 1)),
        torch.tensor([2], dtype=torch.int32),
        table,
        torch.empty(0),
        2,
    )
    assert result is output
    assert len(calls) == 1
    assert calls[0][0][1] is supplied_cache
    assert calls[0][0][4] is table
    assert calls[0][1]["force_aiter_triton"] is (page_size == 1)


def test_glm5next_upstream_explicit_aiter_rejects_preshuffled_gfx938_pages(
    monkeypatch,
) -> None:
    """Explicit AITER requests must reach the layout-aware dispatcher."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

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
            return None

    kpool = ModuleType(patch_glm5next_channel_fp8.KPOOL_MODULE)
    kpool.SparseAttnIndexerKpool = SparseAttnIndexerKpool
    monkeypatch.setattr(hcu_sparse, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        upstream_sparse,
        "rocm_fp8_paged_mqa_logits",
        upstream_sparse.rocm_fp8_paged_mqa_logits,
    )
    patch_glm5next_channel_fp8._patch_sparse_indexer_kpool(kpool)

    with pytest.raises(
        RuntimeError,
        match="AITER paged-MQA does not support gfx938 preshuffled KPool pages",
    ):
        upstream_sparse.rocm_fp8_paged_mqa_logits(
            torch.empty((1, 1, 1, 128), dtype=torch.float8_e4m3fn),
            torch.empty((2, 16, 1, 132), dtype=torch.uint8),
            torch.ones((1, 1)),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[1, 0]], dtype=torch.int32),
            torch.empty(0),
            2,
            force_aiter_triton=True,
        )


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

    monkeypatch.setattr(sparse, "_ON_GFX942", True)
    monkeypatch.setattr(sparse, "on_gfx938", lambda: False)
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


def test_channel_int8_projection_kept_in_bf16_is_dequantized() -> None:
    module = _fake_glm_model_module()
    module._FP8_ATTN_PROJS = {
        ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False)
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

    target = "layers.0.self_attn.fused_qkv_a_proj.weight"
    params = {target: SimpleNamespace(weight_loader=weight_loader)}
    buffered = {}
    loaded_params = set()
    weight = torch.tensor([[10, -20], [30, 40]], dtype=torch.int8)
    scale = torch.tensor([[0.01], [0.02]], dtype=torch.float32)

    helper = module._try_load_fp8_attn_proj
    assert helper(
        "layers.0.self_attn.q_a_proj.weight",
        weight,
        buffered,
        params,
        loaded_params,
        0,
    )
    assert helper(
        "layers.0.self_attn.q_a_proj.weight_scale",
        scale,
        buffered,
        params,
        loaded_params,
        0,
    )

    expected = (weight.float() * scale).to(torch.bfloat16)
    torch.testing.assert_close(loaded[0][0], expected)
    assert loaded[0][1] == 0
    assert loaded_params == {target}
    assert buffered == {}


@pytest.mark.parametrize(
    ("suffix", "target_suffix", "expected_shard_id"),
    [
        (".kv_b_proj.", ".kv_b_proj.weight", None),
        (".indexer.wq_b.", ".indexer.wq_b.weight", None),
        (".indexer.wk.", ".indexer.wk_weights_proj.weight", 0),
    ],
)
def test_channel_int8_checkpoint_only_projection_is_dequantized(
    suffix: str,
    target_suffix: str,
    expected_shard_id: int | None,
) -> None:
    module = _fake_glm_model_module()
    module._FP8_ATTN_PROJS = {}

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

    def weight_loader(param, tensor, *args):
        del param
        loaded.append((tensor, args[0] if args else None))

    prefix = "layers.0.self_attn"
    target = f"{prefix}{target_suffix}"
    params = {target: SimpleNamespace(weight_loader=weight_loader)}
    buffered = {}
    loaded_params = set()
    weight = torch.tensor([[10, -20], [30, 40]], dtype=torch.int8)
    scale = torch.tensor([[0.01], [0.02]], dtype=torch.float32)

    helper = module._try_load_fp8_attn_proj
    assert helper(
        f"{prefix}{suffix}weight",
        weight,
        buffered,
        params,
        loaded_params,
        0,
    )
    assert helper(
        f"{prefix}{suffix}weight_scale",
        scale,
        buffered,
        params,
        loaded_params,
        0,
    )

    expected = (weight.float() * scale).to(torch.bfloat16)
    torch.testing.assert_close(loaded[0][0], expected)
    assert loaded[0][1] == expected_shard_id
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
