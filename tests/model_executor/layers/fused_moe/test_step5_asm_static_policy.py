"""CPU admission contracts, independent of an installed accelerator runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def policy():
    source = Path(__file__).resolve().parents[4] / "vllm_hcu/ops/step5/prefill_asm_g1.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    func = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == "_expert_layout_eligible")
    namespace = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[func.name]


def fixture():
    context = dict(enable_eplb=True, static_expert_map_path="/model/placement.json",
                   expert_map_record_path=None, eplb_disable_rearrange=True,
                   static_dispatch_policy="token_hash", num_redundant_experts=16,
                   eplb_use_async=False, enable_elastic_ep=False, enable_ep_weight_filter=False)
    # Deliberately noncanonical replica order: admission must not reorder experts.
    row = tuple(reversed(range(352))) + tuple(range(16))
    layer = SimpleNamespace(_vllm_hcu_static_eplb_row=row)
    moe = SimpleNamespace(num_experts=368, num_logical_experts=352)
    return context, moe, layer


def test_static_replica_slots_admitted_without_reordering():
    context, moe, layer = fixture()
    original = layer._vllm_hcu_static_eplb_row
    assert policy()(context, moe, layer, 23)
    assert layer._vllm_hcu_static_eplb_row is original


@pytest.mark.parametrize("key,value", [
    ("enable_eplb", False), ("static_expert_map_path", None),
    ("expert_map_record_path", "/tmp/record"), ("eplb_disable_rearrange", False),
    ("static_dispatch_policy", "nearest"), ("num_redundant_experts", 32),
    ("eplb_use_async", True), ("enable_elastic_ep", True), ("enable_ep_weight_filter", True),
])
def test_dynamic_or_unsupported_replica_lifecycle_rejected(key, value):
    context, moe, layer = fixture()
    context[key] = value
    assert not policy()(context, moe, layer, 23)


@pytest.mark.parametrize("row", [None, list(range(368)), tuple(range(368)),
                                 tuple(range(351)) + (0,) * 17,
                                 tuple(range(352)) + (True,) * 16])
def test_unbound_mutable_or_incomplete_plan_rejected(row):
    context, moe, layer = fixture()
    layer._vllm_hcu_static_eplb_row = row
    assert not policy()(context, moe, layer, 23)


def test_counts_and_legacy_e22_admission_are_separate():
    check = policy()
    context, moe, layer = fixture()
    moe.num_experts = 352
    assert not check(context, moe, layer, 23)
    assert check(dict(enable_eplb=False), moe, SimpleNamespace(), 22)
    assert not check(dict(enable_eplb=True), moe, SimpleNamespace(), 22)
    for count in (1, 16, 24, 32):
        assert not check(context, moe, layer, count)
