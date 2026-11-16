# SPDX-License-Identifier: Apache-2.0
"""Real Core consumer contracts plus CPU routing/lifecycle fault injection.

The provider doubles establish call semantics, not native numerical parity.
"""

import inspect
import os
from dataclasses import replace
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest
import torch
from hcu_vllm_das import gdn
from hcu_vllm_das import gdn_activation as activation
from hcu_vllm_das.environment import gdn_policy
from hcu_vllm_das.gdn_contracts import BOLTOPS, OFFICIAL, OWNERS


@pytest.fixture
def state(monkeypatch):
    from vllm import envs

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    for stage in ("PREFILL", "DECODE"):
        monkeypatch.setenv("VLLM_HCU_GDN_" + stage + "_BACKEND", "hcu-boltops")
    monkeypatch.setattr(activation, "_published", None)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(activation, "_publishing", False)
    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    monkeypatch.delitem(envs.environment_variables, activation.FACTOR, raising=False)
    originals = {}
    for name in OFFICIAL:
        consumer = import_module(
            activation.CHUNK if name in gdn.PREFILL else activation.QWEN
        )
        originals[name] = getattr(consumer, name)
        # Register restoration before activation mutates the genuine module.
        monkeypatch.setattr(consumer, name, originals[name])
    return SimpleNamespace(
        context=SimpleNamespace(
            vllm_version="0.33.0+test", requested_capabilities={activation.CAPABILITY}
        ),
        originals=originals,
        envs=envs,
    )


def providers(monkeypatch, result=None):
    """Generate strict-signature doubles rather than permissive *args mocks."""
    calls = []
    modules = {}
    from hcu_vllm_das.gdn_contracts import AITER

    for selection, contracts in (("hcu-boltops", BOLTOPS), ("hcu-aiter", AITER)):
        module = ModuleType(gdn.MODULES[selection])
        module.__file__ = __file__
        modules[module.__name__] = module
        for name, contract in contracts.items():
            if selection == "hcu-aiter":
                attribute, names, defaults = contract
            else:
                attribute, (names, defaults) = name, contract
            parameters = ", ".join(
                n + ("=" + repr(defaults[n]) if n in defaults else "") for n in names
            )
            namespace = {
                "record": lambda a, s=selection, n=name: (
                    calls.append((s, n, dict(a))),
                    result(a) if callable(result) else result,
                )[1]
            }
            exec(
                compile(
                    "def kernel(" + parameters + "):\n    return record(locals())\n",
                    __file__,
                    "exec",
                ),
                namespace,
            )
            setattr(module, attribute, namespace["kernel"])
    real_import = import_module

    def load(name):
        if name in modules:
            return modules[name]
        return real_import(name)

    monkeypatch.setattr(gdn, "import_module", load)
    monkeypatch.setattr(activation, "import_module", load)
    return calls, modules


def test_real_core_alias_and_constructor_ownership(state, monkeypatch):
    providers(monkeypatch)
    qwen = import_module(activation.QWEN)
    chunk = import_module(activation.CHUNK)
    before = (
        qwen.ChunkGatedDeltaRule.__init__,
        qwen.ChunkGatedDeltaRule.forward_native,
        qwen._resolve_gdn_prefill_backend,
        chunk.chunk_gated_delta_rule_fwd,
        chunk.chunk_gated_delta_rule,
    )
    activation.activate(state.context)
    activation.activate(state.context)
    activation.check_health()
    for module, name, owner, original, wrapper in activation._published.bindings:
        assert getattr(module, name) is wrapper
        assert wrapper.__wrapped__ is original is state.originals[name]
        assert getattr(owner, name) is original
        assert inspect.signature(wrapper) == inspect.signature(original)
    assert before == (
        qwen.ChunkGatedDeltaRule.__init__,
        qwen.ChunkGatedDeltaRule.forward_native,
        qwen._resolve_gdn_prefill_backend,
        chunk.chunk_gated_delta_rule_fwd,
        chunk.chunk_gated_delta_rule,
    )


def test_real_constructor_still_selects_official_native(state, monkeypatch):
    providers(monkeypatch)
    from vllm.config import VllmConfig, set_current_vllm_config

    qwen = import_module(activation.QWEN)
    activation.activate(state.context)
    config = VllmConfig()
    config.model_config = SimpleNamespace(hf_text_config=SimpleNamespace())
    config.additional_config = {"gdn_prefill_backend": "triton"}
    with set_current_vllm_config(config):
        operation = qwen.ChunkGatedDeltaRule()
    assert operation._forward_method.__func__ is qwen.ChunkGatedDeltaRule.forward_native


def test_reentry_swallowed_by_dependency_still_poisoned(state, monkeypatch):
    providers(monkeypatch)
    prepare = activation.prepare_routes

    def reenter(policy):
        from hcu_adapter.extensions import ExtensionCommitError

        with pytest.raises(ExtensionCommitError, match="recursive"):
            activation.activate(state.context)
        return prepare(policy)

    monkeypatch.setattr(activation, "prepare_routes", reenter)
    from hcu_adapter.extensions import ExtensionCommitError

    with pytest.raises(ExtensionCommitError):
        activation.activate(state.context)
    assert activation.FACTOR not in state.envs.environment_variables


def test_fork_during_preparation_cannot_publish_in_child(state, monkeypatch):
    providers(monkeypatch)
    prepare = activation.prepare_routes
    original_pid = os.getpid()

    def fork_once(policy):
        pid = os.fork()
        if pid:
            assert os.waitpid(pid, 0)[1] == 0
        return prepare(policy)

    monkeypatch.setattr(activation, "prepare_routes", fork_once)
    try:
        activation.activate(state.context)
    except BaseException:
        if os.getpid() != original_pid:
            os._exit(
                0
                if activation._failure
                and activation.FACTOR not in state.envs.environment_variables
                else 31
            )
        raise
    if os.getpid() != original_pid:
        os._exit(32)
    activation.check_health()


@pytest.mark.parametrize(
    "master,prefill,decode,expected",
    [
        (None, None, None, ("triton", "triton")),
        ("0", "hcu-aiter", "hcu-boltops", ("triton", "triton")),
        ("1", None, None, ("triton", "triton")),
        ("1", "hcu-boltops", "hcu-aiter", ("hcu-boltops", "hcu-aiter")),
    ],
)
def test_policy(monkeypatch, master, prefill, decode, expected):
    for name, value in (
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
        ("VLLM_HCU_GDN_PREFILL_BACKEND", prefill),
        ("VLLM_HCU_GDN_DECODE_BACKEND", decode),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert gdn_policy() == expected


def test_invalid_policy_refused_even_with_master_off(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_GDN_PREFILL_BACKEND", "invalid")
    with pytest.raises(ValueError, match="PREFILL"):
        gdn_policy()


def test_foreign_consumer_refused_before_any_publication(state, monkeypatch):
    chunk = import_module(activation.CHUNK)
    monkeypatch.setattr(chunk, gdn.PREFILL[-1], lambda: None)
    with pytest.raises(RuntimeError, match="another owner"):
        activation.activate(state.context)
    assert activation.FACTOR not in state.envs.environment_variables
    assert chunk.chunk_fwd_o is state.originals["chunk_fwd_o"]
    assert activation._published is None and activation._failure is None


@pytest.mark.parametrize(
    "drift", ["alias", "canonical", "provider", "factor", "policy"]
)
def test_late_drift_latches_and_cannot_recover(state, monkeypatch, drift):
    _, modules = providers(monkeypatch)
    activation.activate(state.context)
    name = "chunk_fwd_o"
    consumer = import_module(activation.CHUNK)
    if drift == "alias":
        target, key = consumer, name
    elif drift == "canonical":
        target, key = import_module(OWNERS[name]), name
    elif drift == "provider":
        target, key = modules[gdn.MODULES["hcu-boltops"]], name
    elif drift == "factor":
        target, key = state.envs.environment_variables, activation.FACTOR
    else:
        target, key = None, None
    if drift == "policy":
        monkeypatch.setenv("VLLM_HCU_GDN_DECODE_BACKEND", "triton")
    elif isinstance(target, dict):
        monkeypatch.setitem(target, key, lambda: None)
    else:
        monkeypatch.setattr(target, key, lambda: None)
    from hcu_adapter.extensions import ExtensionCommitError

    with pytest.raises(ExtensionCommitError, match="owner changed"):
        activation.check_health()
    # A caller retaining an earlier wrapper must also reject the drift.
    with pytest.raises(ExtensionCommitError):
        activation._published.bindings[0][-1]()
    with pytest.raises(ExtensionCommitError):
        activation.activate(state.context)


def test_partial_publication_is_fatal(state, monkeypatch):
    providers(monkeypatch)
    chunk = import_module(activation.CHUNK)
    original_setattr = setattr

    def fail(owner, name, value):
        if owner is chunk and name == "chunk_fwd_o":
            raise RuntimeError("injected publication failure")
        original_setattr(owner, name, value)

    monkeypatch.setattr(activation, "setattr", fail, raising=False)
    from hcu_adapter.extensions import ExtensionCommitError

    with pytest.raises(ExtensionCommitError, match="publication failed"):
        activation.activate(state.context)
    with pytest.raises(ExtensionCommitError):
        chunk.chunk_gated_delta_rule_fwd_h()
    with pytest.raises(ExtensionCommitError):
        activation.activate(state.context)


def test_present_incompatible_provider_is_not_absence(state, monkeypatch):
    _, modules = providers(monkeypatch)
    module = modules[gdn.MODULES["hcu-boltops"]]
    module.chunk_fwd_o = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="signature"):
        activation.activate(state.context)
    assert activation.FACTOR not in state.envs.environment_variables
    assert activation._failure is None


@pytest.mark.parametrize("full_chunks_default", [None, True, False])
def test_boltops_full_chunks_extension_requires_exact_safe_default(
    state, monkeypatch, full_chunks_default
):
    _, modules = providers(monkeypatch)
    module = modules[gdn.MODULES["hcu-boltops"]]
    function = module.chunk_fwd_o
    signature = inspect.signature(function)
    extended = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "full_chunks",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=full_chunks_default,
            ),
        ]
    )
    monkeypatch.setattr(function, "__signature__", extended, raising=False)
    if full_chunks_default is None:
        assert gdn.resolve("hcu-boltops", "chunk_fwd_o")[2] is function
    else:
        with pytest.raises(RuntimeError, match="signature"):
            gdn.resolve("hcu-boltops", "chunk_fwd_o")


@pytest.mark.parametrize(
    "missing", ["boltops", "boltops.fla.gdn", "internal_broken_dependency"]
)
def test_missing_optional_and_broken_dependency_are_distinct(monkeypatch, missing):
    def absent(_):
        raise ModuleNotFoundError("injected", name=missing)

    monkeypatch.setattr(gdn, "import_module", absent)
    if missing == "internal_broken_dependency":
        with pytest.raises(ModuleNotFoundError):
            gdn.resolve("hcu-boltops", "chunk_fwd_o")
    else:
        assert gdn.resolve("hcu-boltops", "chunk_fwd_o") is None


def test_source_drift_refuses_cache_identity(state, monkeypatch, tmp_path):
    providers(monkeypatch)
    activation.activate(state.context)
    assert len(activation.compile_identity()) == 64
    assert (
        state.envs.compile_factors()[activation.FACTOR] == activation.compile_identity()
    )
    file = tmp_path / "provider.py"
    file.write_text("changed source")
    monkeypatch.setattr(
        activation,
        "_published",
        replace(activation._published, files=((str(file), "bad-digest"),)),
    )
    from hcu_adapter.extensions import ExtensionCommitError

    with pytest.raises(ExtensionCommitError, match="source changed"):
        activation.compile_identity()


def test_fork_requires_child_activation_then_checks_inherited_owners(
    state, monkeypatch
):
    providers(monkeypatch)
    activation.activate(state.context)
    pid = os.fork()
    if pid == 0:
        try:
            with pytest.raises(RuntimeError, match="after fork"):
                activation.check_health()
            activation.activate(state.context)
            activation.check_health()
            assert activation._published.pid == os.getpid()
        except BaseException:
            os._exit(21)
        os._exit(0)
    assert os.waitpid(pid, 0)[1] == 0
    activation.check_health()


def leaf(name, candidates):
    names, defaults = OFFICIAL[name]
    parameters = ", ".join(
        n + ("=" + repr(defaults[n]) if n in defaults else "") for n in names
    )
    namespace = {}
    exec("def original(" + parameters + "):\n    return 'official'\n", namespace)
    return gdn.make_wrapper(name, namespace["original"], candidates, lambda: None)


def test_chunk_h_preserves_metadata_state_layout_and_null(monkeypatch):
    calls, modules = providers(monkeypatch, result="native")
    name = "chunk_gated_delta_rule_fwd_h"
    module = modules[gdn.MODULES["hcu-boltops"]]
    wrapped = leaf(name, (("hcu-boltops", module, name, getattr(module, name)),))
    k = torch.empty((1, 17, 4, 128), dtype=torch.bfloat16)
    u = torch.empty((1, 17, 8, 128), dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    initial = torch.ones((1, 8, 128, 128), dtype=torch.float32)
    assert (
        wrapped(
            k,
            k,
            u,
            initial_state=initial,
            chunk_offsets=offsets,
            output_final_state=True,
        )
        == "native"
    )
    a = calls[-1][2]
    assert a["k"] is k and a["u"] is u
    assert a["initial_state"] is initial and a["initial_state_indices"] is None
    assert a["null_state_index"] == -1 and a["transpose_state_layout"] is True
    assert a["inplace_final_state"] is False and a["output_final_state"] is True
    assert "chunk_offsets" not in a
    # Exact earlier official-faster BF16 head ratio; no vendor launch.
    assert wrapped(k, k, torch.empty((1, 17, 12, 128), dtype=k.dtype)) == "official"
    assert len(calls) == 1


def test_chunk_o_flat_backing_padding_and_prelaunch_rejection(monkeypatch):
    calls, modules = providers(monkeypatch, result=lambda a: a["v"] + 1)
    name = "chunk_fwd_o"
    module = modules[gdn.MODULES["hcu-boltops"]]
    wrapped = leaf(name, (("hcu-boltops", module, name, getattr(module, name)),))
    v = torch.randn(1, 17, 8, 128)
    backing = torch.full((v.numel() + 19,), -99.0)
    result = wrapped(v, v, v, v, core_attn_out=backing)
    assert result.data_ptr() == backing.data_ptr() and result.shape == v.shape
    torch.testing.assert_close(result, v + 1)
    assert torch.all(backing[v.numel() :] == -99)
    with pytest.raises(ValueError, match="capacity"):
        wrapped(v, v, v, v, core_attn_out=backing[:2])
    assert len(calls) == 1


def test_gated_chunk_o_does_not_select_unqualified_aiter(monkeypatch):
    calls, modules = providers(monkeypatch, result="native")
    name = "chunk_fwd_o"
    candidates = tuple(
        (
            p,
            modules[gdn.MODULES[p]],
            (name if p == "hcu-boltops" else "chunk_fwd_o_vllm_hip_blockdim64"),
            getattr(
                modules[gdn.MODULES[p]],
                name if p == "hcu-boltops" else "chunk_fwd_o_vllm_hip_blockdim64",
            ),
        )
        for p in ("hcu-aiter", "hcu-boltops")
    )
    wrapped = leaf(name, candidates)
    x = torch.empty((1, 17, 8, 128), dtype=torch.bfloat16)
    assert wrapped(x, x, x, x, g=torch.zeros((1, 17, 8))) == "native"
    assert calls[-1][0] == "hcu-boltops"
    assert wrapped(x, x, x, x) == "native"
    assert calls[-1][0] == "hcu-aiter"


@pytest.mark.parametrize("provider", ["hcu-boltops", "hcu-aiter"])
def test_packed_decode_null_indices_and_out_identity(monkeypatch, provider):
    calls, modules = providers(monkeypatch, result="native")
    name = "fused_recurrent_gated_delta_rule_packed_decode"
    attribute = name if provider == "hcu-boltops" else "aiter_" + name
    module = modules[gdn.MODULES[provider]]
    wrapped = leaf(name, ((provider, module, attribute, getattr(module, attribute)),))
    x = torch.randn(3, 128)
    state = torch.randn(4, 2, 128, 128)
    out = torch.empty_like(x)
    indices = torch.tensor([0, 2, -2], dtype=torch.int32)
    assert wrapped(x, x, x, x, x, 0.125, state, out, indices) == "native"
    a = calls[-1][2]
    assert a["out"] is out and a["initial_state"] is state
    if provider == "hcu-boltops":
        assert a["null_state_index"] == 0 and a["ssm_state_indices"] is indices
    else:
        assert a["ssm_state_indices"].tolist() == [-1, 2, -1]
    assert indices.tolist() == [0, 2, -2]


def test_selected_native_error_is_not_swallowed(monkeypatch):
    def fail(_):
        raise RuntimeError("native launch failed")

    _, modules = providers(monkeypatch, result=fail)
    name = "recompute_w_u_fwd"
    module = modules[gdn.MODULES["hcu-boltops"]]
    wrapped = leaf(name, (("hcu-boltops", module, name, getattr(module, name)),))
    with pytest.raises(RuntimeError, match="native launch failed"):
        wrapped(None, None, None, None, None, None)


@pytest.mark.parametrize("provider", ["hcu-boltops", "hcu-aiter"])
def test_sigmoid_spec_metadata_and_state_sentinel_preserved(monkeypatch, provider):
    calls, modules = providers(monkeypatch, result="native")
    name = "fused_sigmoid_gating_delta_rule_update"
    attribute = name if provider == "hcu-boltops" else "vllm_" + name
    module = modules[gdn.MODULES[provider]]
    wrapped = leaf(name, ((provider, module, attribute, getattr(module, attribute)),))
    x = torch.empty((1, 3, 4, 128), dtype=torch.bfloat16)
    gates = torch.empty((1, 3, 4), dtype=x.dtype)
    state = torch.empty((4, 4, 128, 128), dtype=torch.float32)
    indices = torch.tensor([[0, 2, -3]], dtype=torch.int32)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    accepted = torch.tensor([2], dtype=torch.int32)
    assert (
        wrapped(
            torch.empty(4),
            gates,
            gates,
            torch.empty(4, dtype=x.dtype),
            x,
            x,
            x,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        == "native"
    )
    a = calls[-1][2]
    assert a["num_accepted_tokens"] is accepted and a["cu_seqlens"] is cu
    assert a["initial_state"] is state and a["use_qk_l2norm_in_kernel"] is True
    assert a["is_kda"] is False
    if provider == "hcu-boltops":
        assert a["null_state_index"] == 0 and a["ssm_state_indices"] is indices
    else:
        assert a["ssm_state_indices"].tolist() == [[-1, 2, -1]]


def test_aiter_chunk_h_uses_int64_offsets_without_mutating_metadata(monkeypatch):
    calls, modules = providers(monkeypatch, result="native")
    name = "chunk_gated_delta_rule_fwd_h"
    attribute = "chunk_gated_delta_rule_fwd_vllm_hip_blockdim64"
    module = modules[gdn.MODULES["hcu-aiter"]]
    wrapped = leaf(
        name, (("hcu-aiter", module, attribute, getattr(module, attribute)),)
    )
    k = torch.empty(1, 17, 4, 128, dtype=torch.bfloat16)
    u = torch.empty(1, 17, 8, 128, dtype=k.dtype)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    assert wrapped(k, k, u, chunk_offsets=offsets) == "native"
    a = calls[-1][2]
    assert a["chunk_offsets"].dtype == torch.int64 and offsets.dtype == torch.int32
    assert a["chunk_offsets"].tolist() == offsets.tolist()
    assert a["transpose_state_layout"] is True
