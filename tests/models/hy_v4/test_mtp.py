# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm_hcu.models.hy_v4 import model as target_module


def _mtp():
    name = "vllm_hcu.models.hy_v4.mtp"
    assert importlib.util.find_spec(name) is not None, "Native HYV4 MTP model is missing"
    return importlib.import_module(name)


def _minimal_draft(mtp, monkeypatch):
    """Real draft loader with a small tree; only distributed allocation is omitted."""
    from vllm.model_executor.models import utils
    # The pinned owner caches by id(model). Isolate short-lived test models
    # so a recycled id cannot inherit a previous fixture's PP ownership.
    monkeypatch.setattr(utils, "_model_to_pp_missing_layer_names", {})
    monkeypatch.setattr(target_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(target_module, "get_tensor_model_parallel_rank", lambda: 0)
    model = object.__new__(mtp.HYV4MTP)
    nn.Module.__init__(model)
    model.config = HYV4Config(num_hidden_layers=2, n_routed_experts=0, num_attention_heads=4)
    model.config.num_experts = 0
    model.quant_config = None
    model.checkpoint_quant_config = None
    model.num_redundant_experts = 0
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(4, 2)
    layer = nn.Module()
    model.model.layers = nn.ModuleDict({"2": layer})
    for name in ("enorm", "hnorm", "final_layernorm"):
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.ones(2))
        setattr(layer, name, norm)
    layer.eh_proj = nn.Linear(4, 2, bias=False)
    layer.shared_head = nn.Module()
    layer.shared_head.head = nn.Linear(2, 4, bias=False)
    layer.mtp_block = nn.Module()
    layer.mtp_block.mlp = nn.Module()
    layer.mtp_block.mlp.gate_up_proj = nn.Linear(2, 4, bias=False)

    def load_merged(param, value, shard_id):
        assert shard_id in (0, 1)
        with torch.no_grad():
            param[2 * shard_id:2 * shard_id + 2].copy_(value)

    layer.mtp_block.mlp.gate_up_proj.weight.weight_loader = load_merged
    model.get_expert_mapping = lambda: []
    return model


def _weights():
    prefix = "model.mtp_layers.0."
    return [("model.embed_tokens.weight", torch.full((4, 2), 2.0)),
            ("lm_head.weight", torch.full((4, 2), 3.0)),
            *[(prefix + name + ".weight", torch.full((2,), 4.0))
              for name in ("enorm", "hnorm", "final_layernorm")],
            (prefix + "eh_proj.weight", torch.full((2, 4), 5.0)),
            (prefix + "mlp.gate_proj.weight", torch.full((2, 2), 6.0)),
            (prefix + "mlp.up_proj.weight", torch.full((2, 2), 7.0)),
            ("model.layers.0.unknown_target.weight", torch.tensor(float("nan"))),
            ("model.norm.weight", torch.tensor(float("nan")))]


def _constructed_tied_draft(monkeypatch, *, source=None, algo=None, tied=True, with_mlp=False, shared_mlp=False, decoder=None):
    """Keep the native constructor, vocab/head modules and strict loader."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers import linear, vocab_parallel_embedding
    from vllm.model_executor import parameter
    from vllm.model_executor.models import utils

    mtp = _mtp()
    monkeypatch.setattr(utils, "_model_to_pp_missing_layer_names", {})
    for module in (target_module, vocab_parallel_embedding, linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    class Block(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.block_type = "feedforward"
            self.self_attn = SimpleNamespace(is_sparse=False)
            if with_mlp:
                # Keep the actual merged-linear allocation and shard loader.
                assert kwargs["quant_config"] is None
                self.mlp = nn.Module()
                dense = self.mlp
                if shared_mlp:
                    dense = self.mlp.shared_experts = nn.Module()
                dense.gate_up_proj = linear.MergedColumnParallelLinear(
                    2, [2, 2], bias=False, quant_config=kwargs["quant_config"],
                    prefix=kwargs["prefix"] + (".mlp.shared_experts" if shared_mlp else ".mlp") + ".gate_up_proj")

    # Decoder kernels are unrelated to the embedding/head allocation boundary.
    monkeypatch.setattr(mtp, "HYV4DecoderLayer", decoder or Block)
    config = HYV4Config(num_hidden_layers=2, hidden_size=2, vocab_size=4,
                       n_routed_experts=0, tie_word_embeddings=tied, mtp_quant_algo=algo,
                       pad_token_id=0, bos_token_id=1, eos_token_id=2)
    current = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        quant_config=source, cache_config=None,
        parallel_config=SimpleNamespace(enable_eplb=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )
    with set_current_vllm_config(VllmConfig()):
        return mtp.HYV4MTP(vllm_config=current)


def _tied_weights():
    return [(name, value) for name, value in _weights()
            if name != "lm_head.weight" and ".mlp." not in name]


@pytest.mark.parametrize("head_aliases", [
    (), ("lm_head.weight",), ("model.mtp_layers.0.shared_head.head.weight",),
    ("model.layers.2.shared_head.head.weight",),
    ("lm_head.weight", "model.mtp_layers.0.shared_head.head.weight",
     "model.layers.2.shared_head.head.weight"),
])
@pytest.mark.parametrize("head_first", [False, True])
def test_mtp_tied_checkpoint_loads_without_requiring_head(head_aliases, head_first, monkeypatch):
    draft = _constructed_tied_draft(monkeypatch)
    embedding = draft.model.embed_tokens.weight
    pointer = embedding.data_ptr()
    weights = _tied_weights()
    aliases = [(name, torch.full((4, 2), 9.0)) for name in head_aliases]
    loaded = draft.load_weights(iter(aliases + weights if head_first else weights + aliases))
    assert loaded == set(dict(draft.named_parameters()))
    assert "model.embed_tokens.weight" in loaded
    assert "model.layers.2.shared_head.head.weight" not in loaded
    head = draft.model.layers["2"].shared_head.head
    assert head.weight is embedding
    assert head.weight.data_ptr() == pointer
    torch.testing.assert_close(embedding[:4], torch.full((4, 2), 2.0))
    assert not hasattr(draft, "_checkpoint_accounting")


def test_mtp_tied_checkpoint_still_requires_canonical_embedding(monkeypatch):
    draft = _constructed_tied_draft(monkeypatch)
    weights = [(name, value) for name, value in _tied_weights()
               if name != "model.embed_tokens.weight"]
    weights.append(("lm_head.weight", torch.ones(4, 2)))
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint.*model.embed_tokens.weight"):
        draft.load_weights(iter(weights))
    assert not hasattr(draft, "_checkpoint_accounting")


@pytest.mark.parametrize("pp_size", [1, 2])
def test_mtp_tied_checkpoint_preserves_pinned_eagle_pp_embedding_ownership(pp_size, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    draft = _constructed_tied_draft(monkeypatch)
    draft.load_weights(iter(_tied_weights()))
    own_embed = draft.model.embed_tokens
    own_pointer = own_embed.weight.data_ptr()
    assert draft.model.layers["2"].shared_head.head.weight is own_embed.weight
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(4, 2)
    target.lm_head = nn.Linear(2, 4, bias=False)
    target.lm_head.weight = target.model.embed_tokens.weight
    monkeypatch.setattr(utils, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size))
    loaded = utils.load_eagle_model(target, SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=object(), kv_cache_dtype=None,
            attention_backend=None,
        )))
    assert loaded.model.layers["2"].shared_head.head is target.lm_head
    assert loaded.lm_head is target.lm_head
    assert loaded.model.embed_tokens is (target.model.embed_tokens if pp_size == 1 else own_embed)
    if pp_size == 2:
        assert loaded.model.embed_tokens.weight.data_ptr() == own_pointer
        assert own_pointer != target.model.embed_tokens.weight.data_ptr()
        torch.testing.assert_close(loaded.model.embed_tokens.weight[:4], torch.full((4, 2), 2.0))
    else:
        assert loaded.model.embed_tokens.weight is loaded.lm_head.weight


@pytest.mark.parametrize("layout", ["mtp_layers", "nextn_layers"])
def test_mtp_checkpoint_loads_complete_draft_and_ignores_target(layout, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    weights = _weights()
    if layout == "nextn_layers":
        weights = [(name.replace("model.mtp_layers.0.", "model.layers.2."), value) for name, value in weights]
    loaded = model.load_weights(iter(weights))
    assert loaded == set(dict(model.named_parameters()))
    layer = model.model.layers["2"]
    torch.testing.assert_close(layer.mtp_block.mlp.gate_up_proj.weight,
                               torch.tensor([[6., 6.], [6., 6.], [7., 7.], [7., 7.]]))
    assert model.model.embed_tokens.weight[0].tolist() == [2., 2.]
    assert layer.shared_head.head.weight[0].tolist() == [3., 3.]
    assert layer.eh_proj.weight[0].tolist() == [5., 5., 5., 5.]
    assert not hasattr(model, "_checkpoint_accounting")


@pytest.mark.parametrize("missing", ["mlp.up_proj", "enorm", "eh_proj", "lm_head", "model.embed_tokens"])
def test_mtp_load_requires_every_parameter_and_packed_shard(missing, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint"):
        model.load_weights((name, value) for name, value in _weights() if missing not in name)
    assert not hasattr(model, "_checkpoint_accounting")
    model.load_weights(iter(_weights()))


@pytest.mark.parametrize("name", ["model.mtp_layers.0.enorm.weight", "model.layers.2.enorm.weight"])
def test_mtp_load_rejects_duplicate_checkpoint_aliases(name, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    with pytest.raises(RuntimeError, match="Duplicate HY V4"):
        model.load_weights(iter(_weights() + [(name, torch.ones(2))]))


@pytest.mark.parametrize("name", ["model.mtp_layers.1.enorm.weight", "model.layers.3.enorm.weight",
                                    "model.mtp_layers.bad.enorm.weight", "model.mtp_layers.0.unknown.weight"])
def test_mtp_load_rejects_unknown_or_extra_draft_weights(name, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    with pytest.raises((ValueError, RuntimeError), match="HY.?V4|HY V4"):
        model.load_weights(iter(_weights() + [(name, torch.ones(2))]))


def test_mtp_pp_missing_layer_is_excluded_from_required_parameters(monkeypatch):
    from vllm.model_executor.models.utils import PPMissingLayer
    model = _minimal_draft(_mtp(), monkeypatch)
    model.model.layers["2"] = PPMissingLayer()
    loaded = model.load_weights(iter(_weights()))
    assert loaded == {"model.embed_tokens.weight"}


@pytest.mark.parametrize("raw", [False, True])
def test_mtp_fp8_router_reuses_strict_local_decoder(raw, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    model.model.layers["2"].mtp_block.mlp.gate = nn.Linear(2, 2, bias=False)
    scale = torch.tensor([[128], [129]], dtype=torch.uint8) if raw else torch.tensor([[2.], [4.]])
    weights = _weights() + [
        ("model.mtp_layers.0.mlp.router.gate.weight_scale_inv", scale),
        ("model.mtp_layers.0.mlp.router.gate.weight", torch.ones(2, 2).to(torch.float8_e4m3fn))]
    loaded = model.load_weights(iter(weights))
    torch.testing.assert_close(model.model.layers["2"].mtp_block.mlp.gate.weight,
                               torch.tensor([[2., 2.], [4., 4.]]))
    assert loaded == set(dict(model.named_parameters()))
    with pytest.raises(RuntimeError, match="Incomplete HY V4 FP8"):
        model.load_weights(iter(weights[:-1]))
    with pytest.raises(RuntimeError, match="Duplicate HY V4"):
        model.load_weights(iter(weights + [weights[-2]]))


def test_mtp_raw_block_expert_scale_normalization():
    mtp = _mtp()
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    quant.is_scale_e8m0 = True
    raw = torch.tensor([[127, 128]], dtype=torch.uint8)
    name, value = mtp._prepare_mtp_fp8_expert_scale(quant, "model.layers.2.mtp_block.mlp.experts.down_proj.scale", raw)
    assert name.endswith("down_proj_weight_scale_inv") or name.endswith("down_proj.weight_scale_inv")
    assert value.dtype == torch.float8_e8m0fnu
    assert value.data_ptr() == raw.data_ptr()
    torch.testing.assert_close(value.float(), torch.tensor([[1., 2.]]))


@pytest.mark.parametrize("scale_suffix", ["_scale", ".scale", ".weight_scale_inv"])
def test_mtp_fused_experts_load_every_weight_and_scale_shard(scale_suffix, monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    model = _minimal_draft(_mtp(), monkeypatch)
    model.config.num_experts = 2
    model.quant_config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    model.quant_config.is_scale_e8m0 = True
    mlp = model.model.layers["2"].mtp_block.mlp
    del mlp.gate_up_proj
    mlp.experts = nn.Module()
    owner = mlp.experts.routed_experts = nn.Module()
    for tag, shape in [("w13_weight", (2, 4, 2)), ("w2_weight", (2, 2, 2)),
                       ("w13_weight_scale_inv", (2, 4, 1)), ("w2_weight_scale_inv", (2, 2, 1))]:
        param = nn.Parameter(torch.zeros(shape), requires_grad=False)
        def loader(param, value, name, shard_id, expert_id, return_success=False):
            with torch.no_grad():
                if shard_id == "w2":
                    param[expert_id].copy_(value.float())
                else:
                    offset = 0 if shard_id == "w1" else 2
                    param[expert_id, offset:offset + 2].copy_(value.float())
            return True
        param.weight_loader = loader
        setattr(owner, tag, param)
    weights = [(name, value) for name, value in _weights() if ".mlp." not in name]
    prefix = "model.mtp_layers.0.mlp.experts."
    weights += [(prefix + "gate_up_proj", torch.full((2, 4, 2), 3.)),
                (prefix + "down_proj", torch.full((2, 2, 2), 4.)),
                (prefix + "gate_up_proj" + scale_suffix, torch.full((2, 4, 1), 128, dtype=torch.uint8)),
                (prefix + "down_proj" + scale_suffix, torch.full((2, 2, 1), 129, dtype=torch.uint8))]
    loaded = model.load_weights(iter(weights))
    assert loaded == set(dict(model.named_parameters()))
    torch.testing.assert_close(owner.w13_weight, torch.full((2, 4, 2), 3.))
    torch.testing.assert_close(owner.w2_weight, torch.full((2, 2, 2), 4.))
    torch.testing.assert_close(owner.w13_weight_scale_inv, torch.full((2, 4, 1), 2.))
    torch.testing.assert_close(owner.w2_weight_scale_inv, torch.full((2, 2, 1), 4.))
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint"):
        model.load_weights(iter(weights[:-1]))


def test_mtp_runtime_kv_scale_keeps_quantization_mapper_owner(monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
    model = _minimal_draft(_mtp(), monkeypatch)
    model.quant_config = Fp8Config(is_checkpoint_fp8_serialized=True)
    block = model.model.layers["2"].mtp_block
    block.self_attn = nn.Module()
    block.self_attn.attn = nn.Module()
    block.self_attn.attn.k_scale = KVCacheScaleParameter()
    model.load_weights(iter(_weights() + [
        ("model.mtp_layers.0.self_attn.k_cache.scale", torch.tensor(2.))]))
    assert block.self_attn.attn.k_scale.item() == 2.


def _predictor(mtp):
    draft = object.__new__(mtp.HYV4MTP)
    nn.Module.__init__(draft)
    draft.model = object.__new__(mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(draft.model)
    layer = nn.Module()
    layer.mtp_block = nn.Module()
    attn = layer.mtp_block.self_attn = nn.Module()
    attn.is_sparse = True
    attn.indexer = nn.Module()
    attn.indexer.indexer_op = nn.Module()
    attn.mla_attn = nn.Module()
    attn.mla_attn.impl = SimpleNamespace(topk_indices_buffer=None)
    layer.shared_head = nn.Module()
    layer.shared_head.head = nn.Linear(2, 4, bias=False)
    draft.model.layers = nn.ModuleDict({"2": layer})
    draft.model.embed_tokens = nn.Embedding(4, 2)
    draft.model.topk_indices_buffer = torch.zeros(4, 2, dtype=torch.int32)
    return draft


def test_mtp_topk_skip_control_keeps_shared_storage_stable():
    draft = _predictor(_mtp())
    before = draft.model.topk_indices_buffer.data_ptr()
    draft.model.set_skip_topk(True)
    assert draft.model.layers["2"].mtp_block.self_attn.skip_topk is True
    draft.model.set_skip_topk(False)
    assert draft.model.layers["2"].mtp_block.self_attn.skip_topk is False
    assert draft.model.topk_indices_buffer.data_ptr() == before


def test_mtp_topk_compaction_keeps_shared_storage_stable():
    draft = _predictor(_mtp())
    buffer = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32)
    draft.set_topk_indices_buffer(buffer)
    draft.model.compact_topk_indices(torch.tensor([2, 0]))
    assert buffer.tolist() == [[30, 31], [10, 11], [30, 31], [40, 41]]
    assert draft.model.topk_indices_buffer.data_ptr() == buffer.data_ptr()
    draft.model.compact_topk_indices(torch.tensor([], dtype=torch.int64))
    assert buffer.tolist() == [[30, 31], [10, 11], [30, 31], [40, 41]]


@pytest.mark.parametrize("pp_size", [1, 2])
def test_mtp_target_and_draft_share_all_topk_storage_through_pinned_loader(pp_size, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import utils
    draft = _predictor(_mtp())
    target = nn.Module()
    target.model = nn.Module()
    target.model.topk_indices_buffer = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    target.model.embed_tokens = nn.Embedding(4, 2)
    target.lm_head = nn.Linear(2, 4, bias=False)
    own_embed = draft.model.embed_tokens
    monkeypatch.setattr(utils, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size))
    loaded = utils.load_eagle_model(target, SimpleNamespace(speculative_config=SimpleNamespace(
        draft_model_config=object(), kv_cache_dtype=None, attention_backend=None,
    )))
    attn = loaded.model.layers["2"].mtp_block.self_attn
    for owner in (loaded.model, attn, attn.indexer, attn.indexer.indexer_op, attn.mla_attn.impl):
        assert owner.topk_indices_buffer.data_ptr() == target.model.topk_indices_buffer.data_ptr()
    attn.mla_attn.impl.topk_indices_buffer[0, 0] = 41
    assert target.model.topk_indices_buffer[0, 0] == 41
    assert loaded.model.layers["2"].shared_head.head is target.lm_head
    assert loaded.model.embed_tokens is (target.model.embed_tokens if pp_size == 1 else own_embed)


def test_mtp_local_argmax_uses_shared_head_and_delegates_from_wrapper():
    mtp = _mtp()
    predictor = object.__new__(mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 2

    class SharedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 4, bias=False)

        def forward(self, hidden_states):
            return hidden_states + 7

    layer = nn.Module()
    layer.shared_head = SharedHead()
    predictor.layers = nn.ModuleDict({"2": layer})
    calls = []

    class Processor:
        def get_top_tokens(self, lm_head, hidden_states):
            calls.append((lm_head, hidden_states.clone()))
            return torch.tensor([3, 1], dtype=torch.int64)

    predictor.logits_processor = Processor()
    draft = object.__new__(mtp.HYV4MTP)
    nn.Module.__init__(draft)
    draft.model = predictor
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = draft.get_top_tokens(hidden_states)

    assert torch.equal(actual, torch.tensor([3, 1], dtype=torch.int64))
    assert len(calls) == 1
    assert calls[0][0] is layer.shared_head.head
    torch.testing.assert_close(calls[0][1], hidden_states + 7)


def test_mtp_layer_fuses_embeddings_and_previous_hidden_then_final_residual(monkeypatch):
    mtp = _mtp()
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.config import VllmConfig, set_current_vllm_config
    monkeypatch.setattr(RMSNorm, "forward", RMSNorm.forward_native)
    monkeypatch.setattr(mtp, "HYV4SharedHead", lambda *args: nn.Identity())
    class Block(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["config"].enable_ihc is False
            assert len(kwargs["config"].layer_types) == 3
        def forward(self, *, positions, hidden_states, residual):
            assert residual is None
            return hidden_states * 2, hidden_states
    monkeypatch.setattr(mtp, "HYV4DecoderLayer", Block)
    config = HYV4Config(num_hidden_layers=2, hidden_size=2, layer_types=["full_attention"] * 2)
    with set_current_vllm_config(VllmConfig()):
        layer = mtp.HYV4MultiTokenPredictorLayer(config, "model.layers.2", vllm_config=object(), model_config=object())
    with torch.no_grad():
        layer.eh_proj.weight.copy_(torch.tensor([[1., 0., 0., 1.], [0., 1., 1., 0.]]))
    embed, previous = torch.tensor([[3., 4.]]), torch.tensor([[1., 2.]])
    mixed = torch.nn.functional.linear(torch.cat([
        torch.nn.functional.rms_norm(embed, (2,), eps=config.rms_norm_eps),
        torch.nn.functional.rms_norm(previous, (2,), eps=config.rms_norm_eps)], -1), layer.eh_proj.weight)
    expected = torch.nn.functional.rms_norm(mixed * 3, (2,), eps=config.rms_norm_eps)
    actual = layer(torch.tensor([1]), torch.tensor([4]), previous, embed)
    torch.testing.assert_close(actual, expected)


def test_mtp_predictor_constructs_one_layer_and_reuses_it_for_all_steps(monkeypatch):
    mtp = _mtp()
    class Layer(nn.Module):
        def __init__(self, config, prefix, **kwargs):
            super().__init__()
            assert prefix == "model.layers.2"
            self.mtp_block = nn.Module()
            self.mtp_block.block_type = "feedforward"
            self.mtp_block.self_attn = SimpleNamespace(is_sparse=False)
            self.shared_head = nn.Module()
            self.shared_head.head = nn.Identity()
            self.shared_head.forward = lambda hidden: hidden
        def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds):
            return previous_hidden_states + inputs_embeds
    monkeypatch.setattr(mtp, "HYV4MultiTokenPredictorLayer", Layer)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda *args: object())
    config = HYV4Config(num_hidden_layers=2, hidden_size=2, vocab_size=4, index_topk=2)
    current = SimpleNamespace(model_config=SimpleNamespace(hf_config=config),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        quant_config=None, cache_config=None, scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        parallel_config=SimpleNamespace(enable_eplb=False))
    draft = mtp.HYV4MTP(vllm_config=current)
    assert list(draft.model.layers) == ["2"]
    assert draft.model.num_mtp_layers == 1
    assert draft.num_moe_layers == 0
    positions = torch.tensor([0, 3])
    ids = torch.tensor([1, 2])
    hidden = torch.tensor([[2., 3.], [4., 5.]])
    embeds = torch.tensor([[6., 7.], [8., 9.]])
    pointer = draft.model.topk_indices_buffer.data_ptr()
    for step in range(3):
        actual = draft(ids, positions, hidden, inputs_embeds=embeds, spec_step_idx=step)
        torch.testing.assert_close(actual, torch.tensor([[2., 3.], [12., 14.]]))
        assert draft.model.topk_indices_buffer.data_ptr() == pointer
    torch.testing.assert_close(embeds, torch.tensor([[6., 7.], [8., 9.]]))
    for method, args in [(draft.set_eplb_state, (None, None, None)),
                         (draft.update_physical_experts_metadata, (2, 2))]:
        with pytest.raises(NotImplementedError, match="static placement plan"):
            method(*args)


def test_mtp_sampling_uses_current_sampler_greedy_contract():
    mtp = _mtp()
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.sampler import Sampler
    draft = object.__new__(mtp.HYV4MTP)
    nn.Module.__init__(draft)
    draft.sampler = Sampler()
    metadata = SamplingMetadata(temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None, frequency_penalties=torch.zeros(2),
        presence_penalties=torch.zeros(2), repetition_penalties=torch.ones(2),
        output_token_ids=[[], []], allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors())
    result = draft.sample(torch.tensor([[0., 2., 1.], [3., 1., 2.]]), metadata)
    assert result.sampled_token_ids.tolist() == [[1], [0]]
    assert result.sampled_token_ids.dtype == torch.int32
