# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU tests for the A1 FP8 prefill detour's MODEL scope gate (review P1).

The rewrite carries K3-only validation (unit descales, e4m3 noise), so it
must be scoped by the served model's architectures -- never by call
geometry, which DeepSeek-family MLA shares (qk=192 / v=128).
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest

MOD = "vllm_hcu.patch.worker.core_fix.patch_mla_prefill_fp8_a1"


@pytest.fixture()
def patch_mod():
    mod = importlib.import_module(MOD)
    mod._A1_SCOPE.update(resolved=False, ok=False)
    yield mod
    mod._A1_SCOPE.update(resolved=False, ok=False)


def _set_config(monkeypatch, architectures):
    import vllm.config as vcfg

    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architectures=list(architectures))
    )
    monkeypatch.setattr(
        vcfg, "get_current_vllm_config", lambda: config, raising=False
    )


def test_allowlist_names_the_k3_registry_architecture(patch_mod):
    """Pin the allowlist to the real deployed K3 architectures.

    Ground truth (nmz20 lwc-kimik3-plugin-281-qua): the serving
    checkpoint config.json carries architectures =
    ['KimiK3ForConditionalGeneration'] (model_type kimi_k3); the wheel
    registry additionally maps the inner KimiLinearForCausalLM variant
    used by MTP paths.  Gating on the inner name alone would refuse the
    main serve -- the dangerous false negative this gate must never do.
    """
    assert patch_mod._A1_ARCHITECTURES == (
        "KimiK3ForConditionalGeneration",
        "KimiLinearForCausalLM",
    )


def test_allowlist_names_resolve_in_the_deployed_registry(patch_mod):
    """Every allowlist name must resolve against the serving wheel."""
    vllm = pytest.importorskip("vllm")
    from vllm.model_executor.models.registry import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    missing = set(patch_mod._A1_ARCHITECTURES) - supported
    assert not missing, f"allowlist names absent from the wheel: {missing}"


def test_k3_main_serve_architecture_is_in_scope(patch_mod, monkeypatch):
    """The exact architectures the deployed K3 config.json declares."""
    _set_config(monkeypatch, ["KimiK3ForConditionalGeneration"])
    assert patch_mod._model_in_scope() is True


def test_k3_architecture_is_in_scope_and_latches(patch_mod, monkeypatch):
    _set_config(monkeypatch, ["KimiLinearForCausalLM"])
    assert patch_mod._model_in_scope() is True
    assert patch_mod._A1_SCOPE == {"resolved": True, "ok": True}
    # Latch: architectures never change at runtime; a later different
    # config must not flip the verdict.
    _set_config(monkeypatch, ["DeepseekV3ForCausalLM"])
    assert patch_mod._model_in_scope() is True


def test_same_geometry_non_k3_model_is_out_of_scope(patch_mod, monkeypatch):
    """The P1 case: DeepSeek-family MLA shares qk=192/v=128 but carries no
    K3 validation -- the rewrite must stay stock."""
    _set_config(monkeypatch, ["DeepseekV3ForCausalLM"])
    assert patch_mod._model_in_scope() is False
    assert patch_mod._A1_SCOPE == {"resolved": True, "ok": False}


def test_unresolved_config_stays_stock_and_retries_later(patch_mod, monkeypatch):
    import vllm.config as vcfg

    def _raise():
        raise AssertionError("Current vLLM config is not set")

    monkeypatch.setattr(
        vcfg, "get_current_vllm_config", _raise, raising=False
    )
    # Fail-closed: stock now, and do NOT latch on the unresolved context.
    assert patch_mod._model_in_scope() is False
    assert patch_mod._A1_SCOPE["resolved"] is False
    # The engine configures itself later: same process resolves on retry.
    _set_config(monkeypatch, ["KimiLinearForCausalLM"])
    assert patch_mod._model_in_scope() is True


def test_missing_architecture_field_refuses(patch_mod, monkeypatch):
    _set_config(monkeypatch, [])
    assert patch_mod._model_in_scope() is False
    # Degenerate config object without model_config at all.
    import vllm.config as vcfg

    monkeypatch.setattr(
        vcfg,
        "get_current_vllm_config",
        lambda: types.SimpleNamespace(),
        raising=False,
    )
    assert patch_mod._model_in_scope() is False


def test_scope_is_the_first_semantic_gate_in_execution_order(patch_mod):
    """AST check on the REAL call order (docstrings stripped).

    The scope verdict must be consulted before every geometry, shape,
    capture and device condition; only the module kill/capability knobs
    may precede it.
    """
    import ast

    source = inspect.getsource(patch_mod._a1_rewrites_call)
    fn = ast.parse(source).body[0]
    conditions = [
        ast.unparse(node.test)
        for node in fn.body
        if isinstance(node, ast.If)
    ]
    assert conditions, "predicate lost its gate structure"
    scope_idx = next(
        (i for i, c in enumerate(conditions) if "_model_in_scope" in c),
        None,
    )
    assert scope_idx is not None, "scope gate missing from the predicate"
    allowed_before = ("_A1_DISABLE", "_A1_CAPABLE")
    for c in conditions[:scope_idx]:
        assert any(knob in c for knob in allowed_before), c
    # And everything after it exists (geometry/stream/device gates follow).
    assert scope_idx < len(conditions) - 1


def test_out_of_scope_short_circuits_before_any_later_gate(
    patch_mod, monkeypatch
):
    """Behavioral isolation: with the scope refusing, NO later gate is
    even consulted -- the later gates are booby-trapped to raise."""
    capture_probe = "capture check must not be consulted out of scope"
    monkeypatch.setattr(
        patch_mod.torch.cuda,
        "is_current_stream_capturing",
        lambda: (_ for _ in ()).throw(AssertionError(capture_probe)),
    )
    monkeypatch.setattr(patch_mod, "_A1_DISABLE", False)
    monkeypatch.setattr(patch_mod, "_A1_CAPABLE", True)
    monkeypatch.setattr(patch_mod, "_model_in_scope", lambda: False)
    backend = types.SimpleNamespace(
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )
    q = patch_mod.torch.zeros(2, 192)
    k = patch_mod.torch.zeros(1, 192)
    v = patch_mod.torch.zeros(1, 128)
    assert patch_mod._a1_rewrites_call(backend, q, k, v) is False

    # Symmetric proof the booby-traps sit after the scope gate: with the
    # scope allowing, the predicate walks past dtype/geometry and reaches
    # the capture probe (raises) -- the gates really are ordered scope-
    # first.  dtypes must be supported for the walk to get that far.
    q = q.to(patch_mod.torch.bfloat16)
    k = k.to(patch_mod.torch.bfloat16)
    monkeypatch.setattr(patch_mod, "_model_in_scope", lambda: True)
    with pytest.raises(AssertionError, match=capture_probe):
        patch_mod._a1_rewrites_call(backend, q, k, v)
