# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""patch_qwen4_exp_disable_qsa: dense replacement of QSA full-attention layers."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_disable_qsa as disable_qsa
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

INDEXER_HEADS = 4


def _config(indexer_n_heads: int | None = INDEXER_HEADS) -> SimpleNamespace:
    return SimpleNamespace(indexer_n_heads=indexer_n_heads)


def _vllm_config(config: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config))


def _owner(*, scale_loading: bool = False) -> ModuleType:
    """A stand-in for ``vllm.models.qwen4_exp.amd.model``."""

    owner = ModuleType(disable_qsa.TARGET_MODULE)
    owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES = ["_k_scale", "_v_scale"]

    class Qwen4ExpDecoderLayer:
        def __init__(self, vllm_config, layer_type, prefix=""):
            config = vllm_config.model_config.hf_text_config
            self.seen_indexer_heads = getattr(config, "indexer_n_heads", None)
            self.layer_type = layer_type
            if getattr(config, "raise_in_init", False):
                raise RuntimeError("layer construction failed")

    class Qwen4ExpModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            self._qsa_layer_ids = frozenset()
            self.seen_qsa_layer_ids = None

        def load_weights(self, weights):
            self.seen_qsa_layer_ids = self._qsa_layer_ids
            if scale_loading:
                return _native_load_weights(self, weights)
            return set()

    def _native_load_weights(model, weights):
        # Mirrors the native flow: the QSA scale remap reads the owner set
        # lazily, then AutoWeightsLoader consumes the renamed weights.
        pytest.importorskip("vllm.models.qwen4_exp.amd.model")
        from vllm.model_executor.models.utils import AutoWeightsLoader
        from vllm.models.qwen4_exp.amd.model import _remap_qsa_cache_scale_name

        remapped = (
            (_remap_qsa_cache_scale_name(name, model._qsa_layer_ids), weight)
            for name, weight in weights
        )
        loader = AutoWeightsLoader(
            model,
            ignore_unexpected_suffixes=owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(remapped)

    owner.Qwen4ExpDecoderLayer = Qwen4ExpDecoderLayer
    owner.Qwen4ExpModel = Qwen4ExpModel
    return owner


@pytest.fixture
def switch_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(disable_qsa, "_DISABLE_QSA", True)


def _build_layer(owner, layer_type, config):
    return owner.Qwen4ExpDecoderLayer(_vllm_config(config), layer_type, "layers.0")


def test_adapter_exposes_registry_contract():
    assert disable_qsa.PATCH_ID == "worker.core_fix.qwen4_exp.disable_qsa"
    assert disable_qsa.TARGET_MODULE == "vllm.models.qwen4_exp.amd.model"
    assert callable(disable_qsa.apply_to_module)
    assert set(disable_qsa.TARGETS) == {
        f"{disable_qsa.TARGET_MODULE}.Qwen4ExpDecoderLayer.__init__",
        f"{disable_qsa.TARGET_MODULE}.Qwen4ExpModel.load_weights",
        f"{disable_qsa.TARGET_MODULE}._QWEN4_EXP_IGNORED_MISSING_SUFFIXES",
    }


def test_switch_off_leaves_qsa_behavior_untouched(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(disable_qsa, "_DISABLE_QSA", False)
    owner = _owner()

    assert disable_qsa.apply_to_module(owner) is True

    config = _config()
    layer = _build_layer(owner, "full_attention", config)
    assert layer.seen_indexer_heads == INDEXER_HEADS
    assert config.indexer_n_heads == INDEXER_HEADS
    assert owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES == ["_k_scale", "_v_scale"]

    model = owner.Qwen4ExpModel()
    model._qsa_layer_ids = frozenset({0, 3})
    model.load_weights([])
    assert model.seen_qsa_layer_ids == frozenset({0, 3})


def test_full_attention_layer_is_built_dense_and_config_restored(switch_on):
    owner = _owner()
    disable_qsa.apply_to_module(owner)

    config = _config()
    layer = _build_layer(owner, "full_attention", config)

    assert layer.seen_indexer_heads is None
    assert config.indexer_n_heads == INDEXER_HEADS


def test_config_is_restored_when_layer_construction_raises(switch_on):
    owner = _owner()
    disable_qsa.apply_to_module(owner)

    config = _config()
    config.raise_in_init = True
    with pytest.raises(RuntimeError, match="layer construction failed"):
        _build_layer(owner, "full_attention", config)

    assert config.indexer_n_heads == INDEXER_HEADS


@pytest.mark.parametrize(
    ("layer_type", "indexer_n_heads"),
    [("linear_attention", INDEXER_HEADS), ("full_attention", None)],
)
def test_layers_without_qsa_are_not_touched(switch_on, layer_type, indexer_n_heads):
    owner = _owner()
    disable_qsa.apply_to_module(owner)

    config = _config(indexer_n_heads)
    layer = _build_layer(owner, layer_type, config)

    assert layer.seen_indexer_heads == indexer_n_heads
    assert config.indexer_n_heads == indexer_n_heads


def test_indexer_weight_suffix_is_added_once_and_apply_is_idempotent(switch_on):
    owner = _owner()
    suffixes = owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES

    assert disable_qsa.apply_to_module(owner) is True
    assert disable_qsa.apply_to_module(owner) is False

    assert suffixes.count(".self_attn.indexer.") == 1


def test_stale_marker_is_rejected(switch_on):
    owner = _owner()
    setattr(owner, disable_qsa._MARKER, True)

    with pytest.raises(PatchCompatibilityError, match="stale"):
        disable_qsa.apply_to_module(owner)


def test_incompatible_layer_signature_is_rejected(switch_on):
    owner = _owner()

    def broken_init(self, vllm_config):
        del self, vllm_config

    owner.Qwen4ExpDecoderLayer.__init__ = broken_init

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        disable_qsa.apply_to_module(owner)


def test_wrong_module_is_rejected(switch_on):
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        disable_qsa.apply_to_module(ModuleType("not.the.qwen4exp.model"))


def test_qsa_layer_ids_follow_the_layers_that_were_actually_built(switch_on):
    owner = _owner()
    disable_qsa.apply_to_module(owner)

    model = owner.Qwen4ExpModel()
    dense = nn.Module()
    dense.self_attn = nn.Module()
    dense.self_attn.indexer = None
    qsa = nn.Module()
    qsa.self_attn = nn.Module()
    qsa.self_attn.indexer = nn.Module()
    linear = nn.Module()
    linear.linear_attn = nn.Module()
    model.layers = nn.ModuleList([dense, qsa, linear, dense])
    # Computed from the unmodified config before any layer existed.
    model._qsa_layer_ids = frozenset({0, 1, 3})

    model.load_weights([])

    assert model.seen_qsa_layer_ids == frozenset({1})


def _scale_model(owner, *, with_indexer: bool) -> nn.Module:
    model = owner.Qwen4ExpModel()

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn._k_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.attn._v_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.indexer = nn.Module() if with_indexer else None

    layer = nn.Module()
    layer.self_attn = Attention()
    model.layers = nn.ModuleList([layer])
    return model


def _scale_weights():
    return [
        ("layers.0.self_attn.attn._k_scale", torch.tensor(0.125)),
        ("layers.0.self_attn.attn._v_scale", torch.tensor(0.25)),
    ]


def test_dense_layer_loads_non_unit_kv_cache_scales(switch_on):
    owner = _owner(scale_loading=True)
    disable_qsa.apply_to_module(owner)
    model = _scale_model(owner, with_indexer=False)
    # The unmodified config marks layer 0 as a QSA owner.
    model._qsa_layer_ids = frozenset({0})

    model.load_weights(_scale_weights())

    attn = model.layers[0].self_attn.attn
    assert attn._k_scale.item() == pytest.approx(0.125)
    assert attn._v_scale.item() == pytest.approx(0.25)


def test_stale_qsa_layer_ids_would_drop_dense_kv_cache_scales():
    """Control: without the wrapper the remap silently loses the scales."""

    owner = _owner(scale_loading=True)
    model = _scale_model(owner, with_indexer=False)
    model._qsa_layer_ids = frozenset({0})

    model.load_weights(_scale_weights())

    attn = model.layers[0].self_attn.attn
    assert attn._k_scale.item() == pytest.approx(1.0)
    assert attn._v_scale.item() == pytest.approx(1.0)


def test_checkpoint_indexer_weights_are_skipped_for_dense_layers(switch_on):
    pytest.importorskip("vllm.model_executor.models.utils")
    from vllm.model_executor.models.utils import AutoWeightsLoader

    owner = _owner()
    disable_qsa.apply_to_module(owner)

    model = _scale_model(owner, with_indexer=False)
    weights = [
        ("layers.0.self_attn.indexer.q_proj.weight", torch.zeros(2, 2)),
        ("layers.0.self_attn.indexer.k_proj.weight", torch.zeros(2, 2)),
    ]

    loader = AutoWeightsLoader(
        model,
        ignore_unexpected_suffixes=owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
    )
    assert loader.load_weights(iter(weights)) == set()


def test_indexer_weights_without_the_suffix_are_rejected_by_the_loader():
    """Control: the ignore rule is what keeps dense loading from raising."""

    pytest.importorskip("vllm.model_executor.models.utils")
    from vllm.model_executor.models.utils import AutoWeightsLoader

    owner = _owner()
    model = _scale_model(owner, with_indexer=False)
    loader = AutoWeightsLoader(
        model,
        ignore_unexpected_suffixes=owner._QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
    )

    with pytest.raises(ValueError, match="no module or parameter"):
        loader.load_weights(
            iter([("layers.0.self_attn.indexer.q_proj.weight", torch.zeros(2, 2))])
        )
