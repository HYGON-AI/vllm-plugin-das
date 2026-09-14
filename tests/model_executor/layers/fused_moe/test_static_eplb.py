"""Static plans must fail closed before any checkpoint write."""

from dataclasses import FrozenInstanceError
import hashlib
import importlib
import importlib.util
import json
import os

import pytest
import torch


@pytest.fixture
def api():
    class LazyAPI:
        def __getattr__(self, attr):
            name = "vllm_hcu.model_executor.layers.fused_moe.static_eplb"
            assert importlib.util.find_spec(name) is not None, "static plan implementation missing"
            return getattr(importlib.import_module(name), attr)
    return LazyAPI()


def write_plan(path, rows=None, **metadata):
    rows = [[0, 1, 2, 1], [2, 0, 1, 2]] if rows is None else rows
    path.write_text(json.dumps({"model_maps": {"Model": {
        "physical_to_logical_map": rows, **metadata}}}))
    return path


def load(api, path, **kwargs):
    args = dict(model_key="Model", expected_shape=(2, 4),
                num_logical_experts=3, num_redundant_experts=1)
    return api.load_static_eplb_plan(path, **(args | kwargs))


def test_plan_is_immutable_and_fingerprints_file_bytes(api, tmp_path):
    path = write_plan(tmp_path / "map.json")
    plan = load(api, path)
    assert plan.physical_to_logical_map.tolist() == [[0, 1, 2, 1], [2, 0, 1, 2]]
    assert plan.source_path == str(path.resolve())
    assert plan.fingerprint() == ("Model", hashlib.sha256(path.read_bytes()).hexdigest(),
                                  (2, 4), 3, 4, 1)
    with pytest.raises(FrozenInstanceError):
        plan.num_logical_experts = 7
    plan.physical_to_logical_map.fill_(9)
    assert plan.layer_map(1) == (2, 0, 1, 2)


def test_cache_invalidates_even_same_size_same_mtime(api, tmp_path):
    path = write_plan(tmp_path / "map.json")
    stat = path.stat()
    first = load(api, path)
    write_plan(path, [[2, 0, 1, 2], [0, 1, 2, 1]])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = load(api, path)
    assert second.source_sha256 != first.source_sha256
    assert second.layer_map(0) == (2, 0, 1, 2)


@pytest.mark.parametrize("rows", [[], [[0, 1]], [[0, 1, 2, 1], [0]],
    [[True, 1, 2, 1]] * 2, [[0.0, 1, 2, 1]] * 2, [[-1, 1, 2, 1]] * 2,
    [[0, 1, 3, 1]] * 2, [[0, 1, 1, 1]] * 2])
def test_invalid_rows_rejected(api, tmp_path, rows):
    with pytest.raises(ValueError):
        load(api, write_plan(tmp_path / "map.json", rows))


@pytest.mark.parametrize("kwargs", [dict(model_key="Missing"), dict(model_key=""),
    dict(expected_shape=(1, 4)), dict(expected_shape=(True, 4)),
    dict(num_logical_experts=True), dict(num_logical_experts=0),
    dict(num_redundant_experts=-1), dict(num_redundant_experts=2)])
def test_invalid_request_rejected(api, tmp_path, kwargs):
    with pytest.raises(ValueError):
        load(api, write_plan(tmp_path / "map.json"), **kwargs)


@pytest.mark.parametrize("field,value", [("num_logical_experts", 4),
    ("num_redundant_experts", 0), ("num_physical_experts", 3),
    ("num_moe_layers", 1), ("model_class", "Other")])
def test_declared_metadata_must_match(api, tmp_path, field, value):
    with pytest.raises(ValueError):
        load(api, write_plan(tmp_path / "map.json", **{field: value}))


def test_constructor_cannot_publish_mutable_or_invalid_map(api):
    with pytest.raises(ValueError):
        api.StaticEplbPlan("Model", "/tmp/map", "a" * 64, [[0, 1]], 2, 2, 0)


def test_direct_loader_duplicates_physical_slots_and_preserves_return(api):
    from types import SimpleNamespace
    owner = SimpleNamespace(_vllm_hcu_static_eplb_row=(2, 0, 1, 2))
    values = torch.zeros(2, 2)
    def original(self, *, param, loaded_weight, weight_name, shard_id,
                 expert_id, return_success):
        assert self is owner and weight_name == "w" and shard_id == "w2"
        if expert_id < 2:
            return False
        param[expert_id - 2].copy_(loaded_weight)
        return True
    kwargs = dict(param=values, loaded_weight=torch.tensor([7., 9.]),
                  weight_name="w", shard_id="w2", logical_expert_id=2)
    assert api.load_static_logical_expert(owner, original, **kwargs, return_success=True)
    assert values.tolist() == [[0., 0.], [7., 9.]]
    assert api.load_static_logical_expert(owner, original, **kwargs, return_success=False) is None
    kwargs["logical_expert_id"] = 0
    assert api.load_static_logical_expert(owner, original, **kwargs, return_success=True) is False


@pytest.mark.parametrize("expert_id", [-1, 3, True, 1.0])
def test_direct_loader_rejects_invalid_logical_id_before_write(api, expert_id):
    from types import SimpleNamespace
    owner = SimpleNamespace(_vllm_hcu_static_eplb_row=(2, 0, 1, 2))
    def must_not_write(*args, **kwargs):
        pytest.fail("invalid expert ID reached weight loader")
    with pytest.raises(ValueError):
        api.load_static_logical_expert(owner, must_not_write, param=None,
            loaded_weight=None, weight_name="w", shard_id="w1",
            logical_expert_id=expert_id, return_success=True)
