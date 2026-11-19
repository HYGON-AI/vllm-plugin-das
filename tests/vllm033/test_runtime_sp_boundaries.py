# SPDX-License-Identifier: Apache-2.0
"""Actual official model loops, with explicit CPU layer/collective doubles."""
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adapter_extension/src"))
from hcu_vllm_das.parallel import runtime_sp_attention as projections
from hcu_vllm_das.parallel import runtime_sp_batch as layout
from hcu_vllm_das.parallel import runtime_sp_models as models
from hcu_vllm_das.parallel import runtime_sp_moe_runner as runners
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.models import glm4_moe, hy_v3
from vllm.sequence import IntermediateTensors


def context(mask):
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
                          is_padding=mask, additional_kwargs={"preserved": object()})


@pytest.mark.parametrize("family", ["glm4", "hy3"])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("stage", ["only", "first", "middle", "last"])
@pytest.mark.parametrize("source", ["ids", "embeds"])
def test_official_outer_and_inner_loops_keep_full_embedding_and_pp_boundaries(
        monkeypatch, family, rank, stage, source):
    tokens, hidden, size = 7, 5, 4  # Includes runner padding AND SP alignment padding.
    ids = torch.arange(tokens)
    x = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden) / 16
    residual = x / 2
    mask = torch.tensor([False] * 5 + [True] * 2)
    masked_x = x.masked_fill(mask[:, None], 0)
    masked_residual = residual.masked_fill(mask[:, None], 0)
    first, last = stage in ("only", "first"), stage in ("only", "last")
    pp = SimpleNamespace(is_first_rank=first, is_last_rank=last)
    group = SimpleNamespace(world_size=size, rank_in_group=rank)
    monkeypatch.setattr(models, "get_tp_group", lambda: group)
    monkeypatch.setattr(models, "get_pp_group", lambda: pp)
    module = glm4_moe if family == "glm4" else hy_v3
    monkeypatch.setattr(module, "get_pp_group", lambda: pp)
    monkeypatch.setattr(models, "check_attention_layers", lambda: None)
    monkeypatch.setattr(models, "check_moe_factory_policy", lambda: None)

    embed_calls, layer_calls = [], []
    class Leaf(nn.Module):
        def forward(self, positions, states, residual, **kwargs):
            batch = layout.current_sp_batch()
            assert states.shape == (2, hidden)
            torch.testing.assert_close(positions, torch.tensor([0, 1, 2, 3, 4, 5, 6, 0]))
            torch.testing.assert_close(get_forward_context().is_padding, batch.local_padding)
            get_forward_context().moe_layer_index += 1
            layer_calls.append(True)
            return states + 2, 3 * states if residual is None else residual + 3 * states

    class Norm(nn.Module):
        def forward(self, states, residual=None):
            return states if residual is None else (states + residual, residual)

    inner = nn.Module()
    official_inner = module.Glm4MoeModel if family == "glm4" else module.HYV3Model
    inner.forward = MethodType(official_inner.forward, inner)
    inner.layers = nn.ModuleList([Leaf()])
    inner.start_layer, inner.end_layer, inner.norm = 0, 1, Norm()
    def embed(input_ids):
        torch.testing.assert_close(input_ids, ids)
        assert layout.current_sp_batch() is None  # No token sharding before vocabulary TP.
        embed_calls.append(input_ids.clone())
        return x.clone()
    inner.embed_input_ids = embed

    cls = models.RuntimeSPGlm4MoeForCausalLM if family == "glm4" else models.RuntimeSPHYV3ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.model = inner
    model._runtime_sp_enabled = True
    model._runtime_sp_group, model._runtime_sp_size = group, size
    model._runtime_sp_rank, model._runtime_sp_pid = rank, os.getpid()
    out_hidden = masked_x + 2
    out_residual = 3 * masked_x + (0 if first else masked_residual)
    expected = [out_hidden + out_residual] if last else [out_hidden, out_residual]
    expected = [value.masked_fill(mask[:, None], 0) for value in expected]
    gathered = []
    def gather(local, dim):
        assert dim == 0 and layout.current_sp_batch() is None
        full = torch.nn.functional.pad(expected[len(gathered)], (0, 0, 0, 1))
        torch.testing.assert_close(local, full[rank * 2:(rank + 1) * 2])
        gathered.append(True)
        return full
    group.all_gather = gather
    original = context(mask)
    incoming = None if first else IntermediateTensors({"hidden_states": x, "residual": residual})
    with override_forward_context(original):
        output = model(ids, torch.arange(tokens), incoming,
                       x if source == "embeds" else None)
        assert get_forward_context() is original and original.is_padding is mask
        assert original.moe_layer_index == 1
        assert layout._BATCH_KEY not in original.additional_kwargs
    values = [output] if last else [output["hidden_states"], output["residual"]]
    for actual, target in zip(values, expected):
        torch.testing.assert_close(actual, target)
        assert actual.shape == (tokens, hidden)
    assert len(embed_calls) == int(first and source == "ids") and len(layer_calls) == 1
    # PP input storage remains intact even when the inner loop is in-place.
    torch.testing.assert_close(x, torch.arange(tokens * hidden).reshape(tokens, hidden) / 16)
    torch.testing.assert_close(residual, x / 2)


@pytest.mark.parametrize("tokens", [0, 1, 4, 5, 8, 9])
@pytest.mark.parametrize("rank", range(4))
def test_token_alignment_positions_and_dynamic_padding_values(tokens, rank):
    mask = torch.zeros(tokens, dtype=torch.bool)
    if tokens:
        mask[-1] = True
    batch = layout.RuntimeSPBatch.create(4, rank, tokens, mask, torch.device("cpu"))
    x = torch.ones(tokens, 3)
    positions = batch.positions(torch.arange(tokens))
    assert positions.shape[0] == batch.padded_tokens and batch.padded_tokens % 4 == 0
    local = batch.shard(x)
    assert local.shape == (batch.padded_tokens // 4, 3)
    assert torch.all(local[batch.local_padding] == 0)
    # Changing the producer mask changes data, not a cached num_actual_tokens scalar.
    mask.fill_(False)
    changed = layout.RuntimeSPBatch.create(4, rank, tokens, mask, torch.device("cpu"))
    if tokens and rank == (tokens - 1) // max(1, batch.padded_tokens // 4):
        assert torch.any(batch.local_padding != changed.local_padding)


@pytest.mark.parametrize("naive", [False, True])
@pytest.mark.parametrize("dp", [1, 2])
def test_moe_mask_follows_official_pre_or_post_dispatch(monkeypatch, naive, dp):
    batch = layout.RuntimeSPBatch.create(4, 3, 7,
        torch.tensor([False] * 6 + [True]), torch.device("cpu"))
    parent = context(batch.local_padding)
    parent.additional_kwargs[layout._BATCH_KEY] = batch
    full = batch.global_padding if dp == 1 else torch.cat((batch.global_padding, torch.ones(8, dtype=torch.bool)))
    sizes = [2] * (4 * dp)
    gathered = []
    def gather(mask, dim, sizes):
        assert mask is batch.local_padding and dim == 0 and sizes == [2] * 8
        gathered.append(True)
        return full
    ep = SimpleNamespace(world_size=4 * dp, rank_in_group=3, all_gatherv=gather)
    monkeypatch.setattr(runners, "get_ep_group", lambda: ep)
    monkeypatch.setattr(runners.RuntimeSPMoERunner, "do_naive_dispatch_combine", property(lambda _: naive))
    if dp == 2:
        parent.dp_metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: sizes)
    expected = full if naive else batch.local_padding
    def apply(self, states, logits, shared, ids, overlapping):
        torch.testing.assert_close(get_forward_context().is_padding, expected)
        assert states.shape[0] == expected.shape[0]
        assert shared is sentinel and ids is sentinel and overlapping
        return states
    sentinel = object()
    monkeypatch.setattr(runners.MoERunner, "_apply_quant_method", apply)
    runner = runners.RuntimeSPMoERunner.__new__(runners.RuntimeSPMoERunner)
    nn.Module.__init__(runner)
    x = torch.ones(expected.shape[0], 3)
    with override_forward_context(parent):
        assert runner._apply_quant_method(x, x, sentinel, sentinel, True) is x
        assert get_forward_context() is parent and parent.is_padding is batch.local_padding
    assert len(gathered) == int(naive and dp == 2)


@pytest.mark.parametrize("family", ["glm4", "hy3"])
def test_official_loop_dynamo_keeps_local_tokens_and_global_positions(family, monkeypatch):
    module = glm4_moe if family == "glm4" else hy_v3
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr(module, "get_pp_group", lambda: pp)
    class Projection(nn.Module):
        def forward(self, positions, hidden_states, residual, **kwargs):
            hidden_states = layout.mask_local_padding(hidden_states)
            batch = layout.current_sp_batch()
            rows = hidden_states.shape[0]
            local_positions = positions[batch.rank * rows:(batch.rank + 1) * rows, None]
            return hidden_states + local_positions, hidden_states
    inner = nn.Module()
    inner.layers, inner.start_layer, inner.end_layer = nn.ModuleList([Projection()]), 0, 1
    cls = module.Glm4MoeModel if family == "glm4" else module.HYV3Model
    forward = torch.compile(MethodType(cls.forward, inner), backend="eager", fullgraph=True, dynamic=True)
    mask = torch.zeros(8, dtype=torch.bool)
    batch = layout.RuntimeSPBatch.create(4, 3, 8, mask, torch.device("cpu"))
    ctx = context(batch.local_padding)
    ctx.additional_kwargs[layout._BATCH_KEY] = batch
    x = torch.ones(2, 3)
    with override_forward_context(ctx):
        for epoch in (0, 1, 3):
            batch.local_padding[1] = epoch > 0
            positions = torch.arange(8) + epoch
            actual = forward(None, positions, None, x)
            expected = x.masked_fill(batch.local_padding[:, None], 0)
            torch.testing.assert_close(actual["hidden_states"], expected + positions[6:, None])
            torch.testing.assert_close(actual["residual"], expected)
    torch._dynamo.reset()


@pytest.mark.parametrize("rank", range(4))
def test_attention_unwritten_nan_padding_is_cleared_before_exchange_and_gemm(monkeypatch, rank):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.model_executor.layers import linear
    from vllm.model_executor import parameter

    group = SimpleNamespace(world_size=4, rank_in_group=rank)
    monkeypatch.setattr(projections, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: rank)
    full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) / 16
    full[5:] = torch.nan  # FlashAttention only initializes actual token rows.
    cleared = full.clone()
    cleared[5:] = 0
    batch = layout.RuntimeSPBatch.create(4, rank, 8,
        torch.tensor([False] * 5 + [True] * 3), torch.device("cpu"))
    def exchange(actual_group, send):
        assert actual_group is group and torch.isfinite(send).all()
        torch.testing.assert_close(send.reshape(8, 4), cleared[:, rank * 4:(rank + 1) * 4])
        return cleared[rank * 2:(rank + 1) * 2].reshape(2, 4, 4).permute(1, 0, 2).contiguous()
    monkeypatch.setattr(projections, "_exchange", exchange)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        with projections.runtime_sp_model_construction():
            layer = projections.RuntimeSPRowParallelLinear(16, 5, bias=False,
                prefix="model.layers.0.self_attn.o_proj")
        layer.weight.data.fill_(1)
        layer.quant_method._gemm_impl = lambda layer, x, w, b: torch.nn.functional.linear(x, w, b)
        ctx = context(batch.local_padding)
        ctx.additional_kwargs[layout._BATCH_KEY] = batch
        with override_forward_context(ctx):
            actual, bias = layer(full[:, rank * 4:(rank + 1) * 4])
        expected = cleared[rank * 2:(rank + 1) * 2].sum(-1, keepdim=True).expand(-1, 5)
        torch.testing.assert_close(actual, expected)
        assert bias is None and torch.isfinite(actual).all()


def test_boundary_restores_context_and_preserves_cursor_on_exception(monkeypatch):
    group = SimpleNamespace(world_size=4, rank_in_group=2)
    monkeypatch.setattr(models, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    class Stock(nn.Module):
        def embed_input_ids(self, ids):
            return torch.ones(ids.shape[0], 3)
        def forward(self, *args):
            ctx = get_forward_context()
            assert layout.current_sp_batch() is not None
            ctx.moe_layer_index += 2
            raise RuntimeError("official layer failure")
    class Candidate(models.RuntimeSPModelBoundary, Stock):
        pass
    model = Candidate.__new__(Candidate)
    nn.Module.__init__(model)
    model._runtime_sp_enabled = True
    model._check_runtime_sp = lambda: group
    ctx = context(torch.zeros(7, dtype=torch.bool))
    with override_forward_context(ctx), pytest.raises(RuntimeError, match="official layer failure"):
        model(torch.arange(7), torch.arange(7))
    with override_forward_context(ctx):
        assert get_forward_context() is ctx and layout.current_sp_batch() is None
    assert ctx.moe_layer_index == 2


def test_disabled_boundary_delegates_without_sp_state_or_collectives(monkeypatch):
    monkeypatch.setattr(models, "get_tp_group", lambda: pytest.fail("ordinary TP must not enter SP"))
    expected = object()
    class Stock(nn.Module):
        def forward(self, ids, positions, intermediate, embeds):
            assert (ids, positions, intermediate, embeds) == (1, 2, 3, 4)
            return expected
    class Candidate(models.RuntimeSPModelBoundary, Stock):
        pass
    model = Candidate.__new__(Candidate)
    nn.Module.__init__(model)
    model._runtime_sp_enabled = False
    assert model(1, 2, 3, 4) is expected
