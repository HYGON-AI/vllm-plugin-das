# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Torch/Core constructor contracts with explicit construction spies.

No attention/MoE kernel or complete Flash model is executed here. Only module
aliases in the plugin are patched; Core classes and globals stay untouched.
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config

from hcu_vllm_das.models.hy_v4_flash import core_model as port

core = importlib.import_module("vllm.models.hy_v4.nvidia.model")


class EmbeddingSpy(nn.Module):
    def __init__(self, vocab, hidden, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.empty(vocab, hidden))


class NormSpy(nn.Identity):
    def __init__(self, hidden, eps):
        super().__init__()
        self.hidden, self.eps = hidden, eps


class HeadSpy(nn.Identity):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class DecoderSpy(core.HYV4DecoderLayer):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.kwargs = kwargs
        self.block_type = "feedforward"


class MoESpy(port.HYV4MoEFused):
    def __init__(self):
        nn.Module.__init__(self)
        self.experts = nn.Identity()
        for name, value in {
            "n_logical_experts": 8,
            "n_physical_experts": 10,
            "n_local_physical_experts": 5,
            "n_routed_experts": 8,
            "n_redundant_experts": 2,
        }.items():
            setattr(self, name, value)


class MoEDecoderSpy(DecoderSpy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_type = "moe"
        self.mlp = MoESpy()


def config(*, ihc=False, sparse=True, tied=False):
    hf = NS(
        routed_scaling_factor=1.25,
        n_routed_experts=8,
        moe_intermediate_size=16,
        n_shared_experts=1,
        norm_topk_prob=True,
        hidden_size=8,
        vocab_size=17,
        num_hidden_layers=3,
        rms_norm_eps=1e-5,
        enable_ihc=ihc,
        hc_mult=4,
        hc_eps=1e-6,
        tie_word_embeddings=tied,
    )
    if sparse:
        hf.index_topk = 5
    return NS(
        model_config=NS(hf_config=hf),
        cache_config=object(),
        quant_config=None,
        parallel_config=NS(eplb_config=NS(num_redundant_experts=2)),
        scheduler_config=NS(max_num_batched_tokens=7),
    )


@pytest.fixture
def construction(monkeypatch):
    pp = NS(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(port, "get_pp_group", lambda: pp)
    monkeypatch.setattr(port, "current_platform", NS(device_type="cpu"))
    monkeypatch.setattr(port, "VocabParallelEmbedding", EmbeddingSpy)
    monkeypatch.setattr(port, "ParallelLMHead", EmbeddingSpy)
    monkeypatch.setattr(port, "RMSNorm", NormSpy)
    monkeypatch.setattr(port, "HYV4HCHeadLayer", HeadSpy)

    def layers(count, factory, prefix):
        return (
            0,
            count,
            nn.ModuleList([factory(prefix=f"{prefix}.{i}") for i in range(count)]),
        )

    monkeypatch.setattr(port, "make_layers", layers)
    # Real Core context: LogitsProcessor is not replaced by a spy.
    with set_current_vllm_config(VllmConfig()):
        yield pp


def make_model(cfg, layer=DecoderSpy, prefix="outer.model"):
    class LocalModel(port.HYV4ModelWithFactories):
        decoder_layer_cls = layer

    return LocalModel(vllm_config=cfg, prefix=prefix)


@pytest.mark.parametrize("ihc", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_decoder_dispatch_config_and_buffers(construction, ihc, sparse):
    cfg = config(ihc=ihc, sparse=sparse)
    model = make_model(cfg)
    assert isinstance(model, core.HYV4Model)
    assert model.config is cfg.model_config.hf_config
    assert model.config.num_experts == 8
    assert model.config.router_scaling_factor == 1.25
    assert model.config.expert_hidden_dim == 16
    assert model.config.num_shared_experts == 1
    assert model.config.route_norm is True
    for i, layer in enumerate(model.layers):
        assert type(layer) is DecoderSpy
        assert layer.kwargs["vllm_config"] is cfg
        assert layer.kwargs["config"] is model.config
        assert layer.kwargs["cache_config"] is cfg.cache_config
        assert layer.kwargs["quant_config"] is cfg.quant_config
        assert layer.kwargs["prefix"] == f"outer.model.layers.{i}"
        assert layer.kwargs["topk_indices_buffer"] is model.topk_indices_buffer
    if sparse:
        assert model.topk_indices_buffer.shape == (7, 5)
        assert model.topk_indices_buffer.dtype == torch.int32
        assert model.topk_indices_buffer.device.type == "cpu"
    else:
        assert model.topk_indices_buffer is None
    intermediate = model.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert intermediate["hidden_states"].shape == (2, 32 if ihc else 8)
    if not ihc:
        assert intermediate["residual"].shape == (2, 8)
    assert model.num_moe_layers == 0
    assert model.num_redundant_experts == 0  # Matches Core's no-MoE case.


@pytest.mark.parametrize("rank", ["first", "middle", "last", "only"])
@pytest.mark.parametrize("tied", [False, True])
def test_pp_embedding_norm_and_tied_head(construction, rank, tied):
    construction.is_first_rank = rank in ("first", "only")
    construction.is_last_rank = rank in ("last", "only")
    cfg = config(tied=tied, ihc=True)
    model = make_model(cfg)
    expected_embed = construction.is_first_rank or (tied and construction.is_last_rank)
    assert isinstance(model.embed_tokens, EmbeddingSpy) == expected_embed
    assert isinstance(model.norm, NormSpy) == construction.is_last_rank
    assert isinstance(model.hc_head, HeadSpy) == construction.is_last_rank

    class LM(port.HYV4ForCausalLMWithFactories):
        def model_cls(self, **kwargs):
            assert kwargs["prefix"] == "outer.model"
            assert kwargs["vllm_config"] is cfg
            return model

    lm = LM(vllm_config=cfg, prefix="outer")
    assert lm.model is model
    assert isinstance(lm.lm_head, EmbeddingSpy) == construction.is_last_rank
    assert lm.make_empty_intermediate_tensors is model.make_empty_intermediate_tensors
    if construction.is_last_rank:
        assert lm.lm_head.kwargs["prefix"] == "outer.lm_head"
        assert lm.lm_head.kwargs["quant_config"] is cfg.quant_config
        assert (
            lm.lm_head.weight is getattr(model.embed_tokens, "weight", None)
        ) == tied


def test_moe_metadata_and_expert_collection(construction):
    model = make_model(config(), MoEDecoderSpy)
    assert model.num_moe_layers == 3
    assert model.num_logical_experts == 8
    assert model.num_physical_experts == 10
    assert model.num_local_physical_experts == 5
    assert model.num_routed_experts == 8
    assert model.num_redundant_experts == 2
    assert model.moe_layers == [layer.mlp.experts for layer in model.layers]
    assert model.expert_weights == []
    assert model.num_expert_groups == 1


def test_pp_missing_layers_not_in_moe_metadata(construction, monkeypatch):
    def layers(count, factory, prefix):
        return (
            1,
            2,
            nn.ModuleList(
                [
                    port.PPMissingLayer(),
                    factory(prefix=f"{prefix}.1"),
                    port.PPMissingLayer(),
                ]
            ),
        )

    monkeypatch.setattr(port, "make_layers", layers)
    model = make_model(config(), MoEDecoderSpy)
    assert (model.start_layer, model.end_layer) == (1, 2)
    assert model.num_moe_layers == 1
    assert model.moe_layers == [model.layers[1].mlp.experts]


def test_quant_config_mapping_and_empty_prefix(construction):
    cfg = config()
    cfg.quant_config = NS(packed_modules_mapping=None)

    class Model(port.HYV4ModelWithFactories):
        decoder_layer_cls = DecoderSpy

    class LM(port.HYV4ForCausalLMWithFactories):
        model_cls = Model

    lm = LM(vllm_config=cfg)
    assert isinstance(lm.model, Model)
    assert cfg.quant_config.packed_modules_mapping is lm.packed_modules_mapping
    assert lm.model.quant_config is cfg.quant_config
    assert lm.model.layers[0].kwargs["prefix"] == "model.layers.0"
    assert lm.lm_head.kwargs["prefix"] == "lm_head"
    assert lm.num_redundant_experts == 2
    assert lm.logits_processor.org_vocab_size == 17


def test_inherited_execution_and_loading_and_unmodified_core():
    assert port.HYV4ModelWithFactories.decoder_layer_cls is core.HYV4DecoderLayer
    assert port.HYV4ForCausalLMWithFactories.model_cls is port.HYV4ModelWithFactories
    assert "decoder_layer_cls" not in core.HYV4Model.__dict__
    assert "model_cls" not in core.HYV4ForCausalLM.__dict__
    for name in (
        "forward",
        "load_weights",
        "get_expert_mapping",
        "update_physical_experts_metadata",
        "embed_input_ids",
    ):
        assert getattr(port.HYV4ModelWithFactories, name) is getattr(
            core.HYV4Model, name
        )
    for name in ("forward", "load_weights", "compute_logits", "get_expert_mapping"):
        assert getattr(port.HYV4ForCausalLMWithFactories, name) is getattr(
            core.HYV4ForCausalLM, name
        )


@pytest.mark.parametrize(
    "plugin_cls,core_cls,dispatch",
    [
        (
            port.HYV4ModelWithFactories,
            core.HYV4Model,
            (
                "lambda prefix: HYV4DecoderLayer(",
                "lambda prefix: self.decoder_layer_cls(",
            ),
        ),
        (
            port.HYV4ForCausalLMWithFactories,
            core.HYV4ForCausalLM,
            ("self.model = HYV4Model(", "self.model = self.model_cls("),
        ),
    ],
)
def test_constructor_matches_installed_core_except_explicit_dispatch(
    plugin_cls, core_cls, dispatch
):
    # Drift check only: no AST execution, rebinding, or production source loading.
    import ast
    import inspect
    import textwrap

    expected = textwrap.dedent(inspect.getsource(core_cls.__init__))
    expected = expected.replace("super().__init__()", "nn.Module.__init__(self)")
    assert dispatch[0] in expected
    expected = expected.replace(*dispatch)
    actual = textwrap.dedent(inspect.getsource(plugin_cls.__init__))
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))


def test_decoder_factory_failure_propagates_without_core_changes(construction):
    error = RuntimeError("construction witness")

    class BrokenDecoder(core.HYV4DecoderLayer):
        def __init__(self, **kwargs):
            raise error

    with pytest.raises(RuntimeError) as caught:
        make_model(config(), BrokenDecoder)
    assert caught.value is error
    assert "decoder_layer_cls" not in core.HYV4Model.__dict__
