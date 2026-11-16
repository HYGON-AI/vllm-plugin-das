# SPDX-License-Identifier: Apache-2.0
"""Round-5 review: w2 parameter-level channel loader.

The default RoutedExperts weight_loader calls _load_per_tensor_weight_scale
which reshape(())s 1-D scales into scalars. The module-level interception
in patch_layer only covers load_weights chains; models that call
param.weight_loader directly (e.g. Llama4) bypass it entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu import (
    ModelOptW8A16MoEMethod,
)


def _fake_layer(experts=4, hidden=2048, ep_map=None):
    layer = SimpleNamespace()
    layer._map_global_expert_id_to_local_expert_id = (
        ep_map.__getitem__ if ep_map else lambda e: e)
    return layer


def _default_loader(param, loaded_weight, weight_name, shard_id, expert_id,
                    return_success=False):
    """Stub of the stock loader: fails on 1-D vectors like the real one."""
    if loaded_weight.ndim == 1 and "scale" in weight_name:
        raise RuntimeError(
            f"shape '[]' is invalid for input of size {loaded_weight.numel()}")
    return None


def test_w2_param_loader_handles_1d_channel_scale():
    """Direct param.weight_loader call with a 1-D w2 scale must succeed
    (P2: previously reshape(()) crash via the default loader)."""
    layer = _fake_layer(experts=4, hidden=2048)
    loader = ModelOptW8A16MoEMethod._make_w8a16_w2_scale_loader(
        layer, _default_loader)
    param = SimpleNamespace(
        data=torch.ones(4, 2048, dtype=torch.float16), dtype=torch.float16)
    scale = torch.randn(2048)
    result = loader(param, scale, "experts.0.down_proj.weight_scale",
                    "w2", 0, return_success=True)
    assert result is True
    assert torch.equal(param.data[0], scale.to(torch.float16))


def test_w2_param_loader_skips_non_local_expert():
    ep_map = {0: -1, 1: -1, 2: 0, 3: 1}
    layer = _fake_layer(experts=2, hidden=2048, ep_map=ep_map)
    loader = ModelOptW8A16MoEMethod._make_w8a16_w2_scale_loader(
        layer, _default_loader)
    param = SimpleNamespace(
        data=torch.ones(2, 2048, dtype=torch.float16), dtype=torch.float16)
    # Global expert 0 is non-local on this rank -> skip
    result = loader(param, torch.randn(2048),
                    "experts.0.down_proj.weight_scale", "w2", 0,
                    return_success=True)
    assert result is False


def test_w2_param_loader_falls_back_for_non_scale():
    layer = _fake_layer()
    loader = ModelOptW8A16MoEMethod._make_w8a16_w2_scale_loader(
        layer, _default_loader)
    param = SimpleNamespace(
        data=torch.zeros(4, 2048, dtype=torch.int8), dtype=torch.int8)
    # Non-scale weight: default loader handles it (no crash on 2-D)
    weight = torch.zeros(4, 2048, dtype=torch.int8)
    result = loader(param, weight, "experts.0.down_proj.weight",
                    "w2", 0, return_success=True)
    assert result is None  # default loader returned None (no-op for 2-D)


def test_w2_param_loader_falls_back_for_wrong_length():
    """A scale of the wrong length (per-tensor) falls to the default."""
    layer = _fake_layer(experts=4, hidden=2048)
    loader = ModelOptW8A16MoEMethod._make_w8a16_w2_scale_loader(
        layer, _default_loader)
    param = SimpleNamespace(
        data=torch.ones(4, 2048, dtype=torch.float16), dtype=torch.float16)
    # Wrong length: scalar scale (numel=1) should go to default loader.
    # Default loader stub returns None for non-1-D; but 1-D numel=1 would
    # raise from the stub — so verify the fall-through by checking the
    # param-level loader's numel guard routes it to default (which for
    # this test we make a recording stub).
    calls = []

    def recording_default(param, loaded_weight, weight_name, shard_id,
                          expert_id, return_success=False):
        calls.append(weight_name)
        return None

    loader = ModelOptW8A16MoEMethod._make_w8a16_w2_scale_loader(
        layer, recording_default)
    loader(param, torch.tensor([1.0]), "experts.0.down_proj.weight_scale",
           "w2", 0, return_success=True)
    assert len(calls) == 1  # routed to default
