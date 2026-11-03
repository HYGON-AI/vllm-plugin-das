# SPDX-License-Identifier: Apache-2.0
"""Check Kimi PP auxiliary capture against a single decoder execution."""
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.sequence import IntermediateTensors
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_pp_aux as patch


@pytest.mark.parametrize("attn_res", [False, True])
@pytest.mark.parametrize("split,total,aux_layers", [
    (6, 12, (2, 4, 6, 8, 10)),
    (47, 93, (3, 24, 48, 72, 90)),
])
def test_aux_values_and_order_across_pipeline_boundary(attn_res, split, total, aux_layers):
    from bisect import bisect_right

    pp = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)

    class Decoder(EagleModelMixin):
        def __init__(self, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise AssertionError("PP auxiliary forward was bypassed")

        def _maybe_add_hidden_state(self, states, idx, hidden, residual):
            return super()._maybe_add_hidden_state(
                states, idx, hidden, None if attn_res else residual
            )

    module = ModuleType(patch.TARGET_MODULE)
    module.torch = torch
    module.IntermediateTensors = IntermediateTensors
    module.get_pp_group = lambda: pp
    module.cdiv = lambda a, b: (a + b - 1) // b
    module._apply_attn_res = lambda hidden, *args: hidden
    module.KimiLinearModel = Decoder
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)

    def layer(positions, hidden_states, residual):
        return hidden_states + 1, (residual if attn_res else torch.zeros_like(hidden_states))

    def decoder(start, end, upstream):
        obj = object.__new__(Decoder)
        obj.config = SimpleNamespace(attn_res_block_size=12 if attn_res else None)
        obj.start_layer, obj.end_layer = start, end
        obj.layers = [layer] * total
        obj.aux_hidden_state_layers = aux_layers
        obj._aux_slot_base_cached = upstream
        obj._aux_upstream_total_cached = upstream if pp.is_last_rank else 0
        obj.output_attn_res_proj = obj.output_attn_res_norm = None
        return obj

    positions = torch.tensor([0, 1])
    initial = torch.zeros(2, 3)
    first = decoder(0, split, 0)
    intermediate = first.forward(None, positions, None, inputs_embeds=initial)
    upstream_count = bisect_right(aux_layers, split)
    assert set(intermediate.tensors) == {"hidden_states", "residual"} | {
        "aux_hidden_states_" + str(i) for i in range(upstream_count)}
    pp.is_first_rank, pp.is_last_rank = False, True
    last = decoder(split, total, upstream_count)
    hidden, aux = last.forward(None, positions, intermediate)
    torch.testing.assert_close(hidden, initial + total)
    assert len(aux) == len(aux_layers)  # Sending stage owns the boundary tap.
    for value, expected in zip(aux, aux_layers):
        torch.testing.assert_close(value, initial + expected)


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("needs_embedding", [False, True])
def test_shared_embedding_only_materialized_on_draft_stage(monkeypatch, stage, needs_embedding):
    from vllm.model_executor.models import utils

    monkeypatch.setattr(utils, "spec_decode_needs_target_embed", lambda config: needs_embedding)
    pp = SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == 2)
    existing_embedding = object()

    class Missing:
        pass

    class Decoder:
        def __init__(self, *, vllm_config, prefix=""):
            self.config = SimpleNamespace(vocab_size=16, hidden_size=4)
            self.embed_tokens = existing_embedding if stage == 0 else Missing()

        def forward(self, *args, **kwargs):
            return "original"

    module = ModuleType(patch.TARGET_MODULE)
    module.torch = torch
    module.IntermediateTensors = IntermediateTensors
    module.get_pp_group = lambda: pp
    module.cdiv = lambda a, b: (a + b - 1) // b
    module._apply_attn_res = lambda *args: None
    module.KimiLinearModel = Decoder
    module.PPMissingLayer = Missing
    allocated = []
    module.VocabParallelEmbedding = lambda *args, **kwargs: allocated.append((args, kwargs)) or "shared"
    patch.apply_to_module(module)
    model = Decoder(vllm_config=object(), prefix="model")
    assert bool(allocated) == (stage == 2 and needs_embedding)
    if stage == 0:
        assert model.embed_tokens is existing_embedding
    if allocated:
        assert allocated == [((16, 4), {"prefix": "model.embed_tokens"})]


@pytest.mark.parametrize("pp_size,aux_layers", [(1, (3,)), (2, ())])
def test_original_forward_retained_without_pp_aux(pp_size, aux_layers):
    class Decoder:
        def __init__(self, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return args, kwargs

    module = ModuleType(patch.TARGET_MODULE)
    module.torch = torch
    module.IntermediateTensors = IntermediateTensors
    module.get_pp_group = lambda: SimpleNamespace(world_size=pp_size)
    module.cdiv = lambda a, b: (a + b - 1) // b
    module._apply_attn_res = lambda *args: None
    module.KimiLinearModel = Decoder
    patch.apply_to_module(module)
    model = object.__new__(Decoder)
    model.aux_hidden_state_layers = aux_layers
    assert model.forward(1, 2, custom=True) == ((1, 2), {"custom": True})
