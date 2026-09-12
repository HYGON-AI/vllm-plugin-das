import importlib
import importlib.util

import pytest
import torch


@pytest.fixture
def order():
    def invoke(*args, **kwargs):
        name = "vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch"
        assert importlib.util.find_spec(name) is not None, "locality order implementation missing"
        return importlib.import_module(name).build_locality_fair_replica_order(*args, **kwargs)
    return invoke


def test_local_gpu_then_node_and_balanced_fallback(order):
    mapping = torch.tensor([[[0, 4, -1], [1, 2, -1], [3, 5, -1]]])
    original = mapping.clone()
    outputs = [order(mapping, ep_rank=r, ep_size=8, num_nodes=2,
                     num_physical_experts=8) for r in range(8)]
    assert [o[0, 0, 0].item() for o in outputs] == [0, 0, 0, 0, 4, 4, 4, 4]
    assert outputs[1][0, 1, 0] == 1
    assert outputs[2][0, 1, 0] == 2
    assert torch.equal(original, mapping)
    for rank, result in enumerate(outputs):
        assert torch.equal(result, order(mapping, ep_rank=rank, ep_size=8,
            num_nodes=2, num_physical_experts=8))
        assert torch.equal(result.sort(-1).values, mapping.sort(-1).values)
    counts = torch.bincount(torch.tensor([o[0, 1, 0] for o in outputs]), minlength=8)
    assert counts[1] == counts[2] == 4


def test_layer_offsets_preserve_chunked_order(order):
    mapping = torch.tensor([[[0, 2, 4, 6]], [[1, 3, 5, 7]]])
    kwargs = dict(ep_rank=7, ep_size=8, num_nodes=2, num_physical_experts=8)
    assert torch.equal(order(mapping, **kwargs)[1:],
                       order(mapping[1:], layer_offset=1, **kwargs))


@pytest.mark.parametrize("kwargs", [dict(ep_rank=-1), dict(ep_size=0),
    dict(num_nodes=3), dict(num_physical_experts=3)])
def test_invalid_topology(order, kwargs):
    args = dict(ep_rank=0, ep_size=4, num_nodes=2, num_physical_experts=4)
    with pytest.raises(ValueError):
        order(torch.tensor([[[0, 1]]]), **(args | kwargs))


@pytest.mark.parametrize("mapping", [[[[0, 0]]], [[[-1, -1]]], [[[0, 4]]], [[[-2, 0]]]])
def test_invalid_replica_maps(order, mapping):
    with pytest.raises(ValueError):
        order(torch.tensor(mapping), ep_rank=0, ep_size=4, num_nodes=2,
              num_physical_experts=4)
