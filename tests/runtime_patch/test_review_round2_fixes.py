# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the second PR review round (CPU-only).

P1-1: cached mrope eligibility must not leave rot_emb unbound on the
      second call (UnboundLocalError inside the fused branch).
P1-2: w2 per-channel scale must be copied whole under TP>1 (its shard is
      on the input dim; the scale is per output channel).
P1-3: expert scales must map global->local expert ids (EP-aware) and skip
      non-local experts instead of indexing the wrong slot.
P2-4: the W8A16 RoutedExperts interception must honor exclude_modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


# --- P1-1: rot_emb bound on every call --------------------------------------

class MRotaryEmbedding:  # name must match the layout probe
    is_neox_style = True
    rotary_dim = 64
    mrope_interleaved = True
    mrope_section = (11, 11, 10)

    def _match_cos_sin_cache_dtype(self, q):
        return torch.ones(1024, 64, dtype=q.dtype)


def _mrope_layer():
    return SimpleNamespace(
        head_dim=256,
        rotary_emb=MRotaryEmbedding(),
        attn_output_gate=False,
        q_size=4096, kv_size=1024, num_heads=16,
        q_norm=SimpleNamespace(
            weight=torch.ones(256), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(
            weight=torch.ones(256), variance_epsilon=1e-6),
    )


def _mrope_call(layer, monkeypatch, positions):
    import vllm_hcu.patch.worker.op_opt.patch_qwen35_mrope as pm
    import vllm_hcu.model_executor.layers.rotary_embedding.fused_rmsnorm_mrope as fm

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FUSED_RMSNORM_MROPE", "1")
    calls = []
    monkeypatch.setattr(
        fm, "fused_rmsnorm_mrope",
        lambda q, k, qw, kw, cache, pos, **kw_: (calls.append(1) or (q, k)))
    qkv = torch.randn(positions.shape[-1], 4096 + 1024 + 1024)
    out = pm._TEST_project_qkv_gate(
        layer, qkv, positions,
        original=lambda self, q, p: ("orig",))
    return out, calls


def test_mrope_repeated_and_1d_to_2d(monkeypatch):
    layer = _mrope_layer()
    pos2d = torch.arange(9).view(3, 3).t().contiguous()
    # Cold cache then warm cache: both fused calls must succeed (P1-1: the
    # second call used to raise UnboundLocalError on rot_emb).
    _, calls = _mrope_call(layer, monkeypatch, pos2d)
    _, calls2 = _mrope_call(layer, monkeypatch, pos2d)
    assert len(calls) == 1 and len(calls2) == 1
    # 1D positions -> original path; then a 2D call still fused.
    out, _ = _mrope_call(layer, monkeypatch, torch.arange(3))
    assert out == ("orig",)
    _, calls3 = _mrope_call(layer, monkeypatch, pos2d)
    assert len(calls3) == 1


# --- P1-2 + P1-3: scale loader TP/EP semantics ------------------------------

def _mapping(expert, shard):
    proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[shard]
    slot = {"w1": "w13", "w3": "w13", "w2": "w2"}[shard]
    return (f"experts.{slot}_", f"experts.{expert}.{proj}.", expert, shard)


def _layer(tp_rank=0, tp_size=1, local_experts=4, inter=512, hidden=2048,
           mapping=None, ep_map=None):
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        moe_config=SimpleNamespace(tp_rank=tp_rank, tp_size=tp_size),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(local_experts, 2, inter, dtype=torch.float16),
            requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(local_experts, hidden, dtype=torch.float16),
            requires_grad=False),
    )
    layer.get_expert_mapping = lambda: mapping or []
    if ep_map is not None:
        layer._map_global_expert_id_to_local_expert_id = ep_map.__getitem__
    return layer


def _load(layer, expert_name, weight):
    import vllm_hcu.patch.worker.op_opt.moe.patch_layer as pl
    return pl._TEST_load_per_expert_channel_scale(layer, expert_name, weight)


def test_w2_scale_full_copy_under_tp2():
    hidden = 2048
    layer = _layer(tp_rank=1, tp_size=2, mapping=[_mapping(0, "w2")])
    ckpt = torch.randn(hidden)
    assert _load(layer, "0.down_proj.weight_scale", ckpt) == \
        ["w2_weight_scale"]
    # Full copy on every rank — no tp_rank slice (P1-2).
    want = ckpt.to(torch.float16)
    assert torch.equal(layer.w2_weight_scale.data[0], want)


def test_w2_scale_checkpoint_len_mismatch_falls_back():
    layer = _layer(tp_size=2, mapping=[_mapping(0, "w2")])
    assert _load(layer, "0.down_proj.weight_scale",
                 torch.randn(4096)) is None


def test_w1_scale_tp_slice_unchanged():
    inter = 512
    layer = _layer(tp_rank=1, tp_size=2, inter=inter,
                   mapping=[_mapping(0, "w1")])
    ckpt = torch.randn(inter * 2)
    assert _load(layer, "0.gate_proj.weight_scale", ckpt) == \
        ["w13_weight_scale"]
    want = ckpt[inter:].to(torch.float16)
    assert torch.equal(layer.w13_weight_scale.data[0, 0], want)


def test_ep_maps_global_to_local_and_skips_nonlocal():
    inter = 512
    # EP=2, rank holds global experts 2,3 -> local slots 0,1.
    layer = _layer(local_experts=2, inter=inter,
                   mapping=[_mapping(2, "w1")],
                   ep_map={0: -1, 1: -1, 2: 0, 3: 1})
    ckpt = torch.randn(inter)
    assert _load(layer, "2.gate_proj.weight_scale", ckpt) == \
        ["w13_weight_scale"]
    # Global expert 2 wrote LOCAL slot 0 (P1-3), not slot 2.
    assert torch.equal(layer.w13_weight_scale.data[0, 0],
                       ckpt.to(torch.float16))
    # Non-local expert (global 0 -> -1): fall through, slots untouched.
    layer.get_expert_mapping = lambda: [_mapping(0, "w1")]
    assert _load(layer, "0.gate_proj.weight_scale",
                 torch.randn(inter)) is None
    assert torch.equal(layer.w13_weight_scale.data[0, 0],
                       ckpt.to(torch.float16))
    assert torch.equal(layer.w13_weight_scale.data[1, 0],
                       torch.ones(inter, dtype=torch.float16))


# --- P2-4: exclude_modules honored -------------------------------------------

def test_w8a16_moe_interception_respects_exclusion():
    pytest.importorskip("vllm")
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import vllm.model_executor.layers.quantization.modelopt as mo

    # Concrete subclass so __new__ works without the abstract machinery.
    cfg = mo.ModelOptFp8Config.__new__(mo.ModelOptFp8Config)
    cfg.quant_method = "W8A16"
    cfg.is_layer_excluded = lambda prefix: prefix == "model.layers.0.mlp.experts"

    class RoutedExperts:  # name must match the interception probe
        moe_config = SimpleNamespace(tp_rank=0, tp_size=1)

    # Excluded MoE layer: the W8A16 interception must not fire; the
    # upstream dispatch returns None for a non-linear excluded layer.
    assert cfg.get_quant_method(
        RoutedExperts(), "model.layers.0.mlp.experts") is None
