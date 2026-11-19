# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the optional in-place PCP query rotation."""
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import hcu_config
from hcu_adapter.kernels import pcp_fork
from hcu_vllm_das import inverse_rope_hcu, pcp_q_rope, v41_pcp, v41_pcp_activation


@pytest.fixture
def isolated_q_rope(monkeypatch):
    for name in ("_q_rope_provider", "_q_rope_binding", "_q_rope_failure"):
        monkeypatch.setattr(pcp_fork, name, None)
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    monkeypatch.setenv("HCU_DISABLE_OPS", "")


def test_q_rope_policy(monkeypatch, isolated_q_rope):
    ops = {}
    monkeypatch.setattr(hcu_config, "load", lambda: {"ops": ops, "overrides": {}})
    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    assert not pcp_fork.q_rope_selected()
    ops[pcp_fork.Q_ROPE_OP] = "hcu"
    assert pcp_fork.q_rope_selected()
    for op in (pcp_fork.Q_ROPE_OP, pcp_fork.OP_NAME):
        monkeypatch.setenv("HCU_DISABLE_OPS", op)
        assert not pcp_fork.q_rope_selected()


def test_compile_factor_distinguishes_the_two_rope_policies(
        monkeypatch, isolated_q_rope):
    activation = v41_pcp_activation
    ops = {}
    monkeypatch.setattr(hcu_config, "load", lambda: {"ops": ops, "overrides": {}})
    monkeypatch.setattr(activation, "pcp_enabled", lambda: True)
    monkeypatch.setattr(activation, "host_enabled", lambda: False)
    monkeypatch.setattr(activation, "fp4_selected", lambda: False)
    monkeypatch.setattr(activation, "_profile_policy", lambda: True)

    stock = (True, False, True)
    assert activation.policy_factor() == stock
    ops[pcp_fork.Q_ROPE_OP] = "hcu"
    assert activation.policy_factor() == (*stock, "q_rope_hcu")
    ops["vllm_inverse_rope"] = "hcu"
    assert activation.policy_factor() == (*stock, "hcu", "q_rope_hcu")
    monkeypatch.setenv("HCU_DISABLE_OPS", pcp_fork.Q_ROPE_OP)
    assert activation.policy_factor() == (*stock, "hcu")


def test_initializer_warms_once_and_reuses_only_the_original_table(monkeypatch):
    original = torch.arange(3 * 64, dtype=torch.float32).view(3, 64)
    baseline = original.clone()
    calls = []
    load_count = []

    def operator(*args):
        calls.append(args)

    def load_operator():
        load_count.append(True)
        return operator

    monkeypatch.setattr(pcp_q_rope, "_supports_table", lambda _: True)
    monkeypatch.setattr(pcp_q_rope, "load_operator", load_operator)
    try:
        prepare = pcp_q_rope.initialize(original)
        assert pcp_q_rope.initialize(original) is prepare
        assert load_count == [True]

        def unexpected_init(*args):
            pytest.fail("real prefill retried startup preparation")

        monkeypatch.setattr(pcp_q_rope, "load_operator", unexpected_init)
        monkeypatch.setattr(pcp_q_rope, "forward_table", unexpected_init)
        q = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
        positions = torch.tensor([0, 1])
        with torch.inference_mode():
            assert prepare(q, positions, original.clone()) is None
            rotate = prepare(q, positions, original)
            assert rotate is not None
            rotate()
        q_view, used_positions, table = calls[0]
        assert q_view.data_ptr() == q[..., -64:].data_ptr()
        assert used_positions is positions
        torch.testing.assert_close(original, baseline, rtol=0, atol=0)
        torch.testing.assert_close(table[:, :32], baseline[:, :32], rtol=0, atol=0)
        torch.testing.assert_close(table[:, 32:], -baseline[:, 32:], rtol=0, atol=0)
    finally:
        pcp_q_rope.initialize.cache_clear()


def test_query_and_inverse_rope_share_the_operator(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    loads = []
    operator = object()
    monkeypatch.setattr(torch.version, "hip", "cpu-contract")
    monkeypatch.setattr(cpp_extension, "_get_build_directory", lambda *args: str(tmp_path))
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: loads.append(kwargs))
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    monkeypatch.setattr(torch.ops.hcu_vllm_das, "deepseek_v4_inv_rope", operator, raising=False)
    monkeypatch.setattr(pcp_q_rope.registry, "_RECORDS", {})
    monkeypatch.setattr(pcp_q_rope.registry, "_BOUND", {})
    inverse_rope_hcu.load_operator.cache_clear()
    pcp_q_rope.load_operator.cache_clear()
    try:
        assert inverse_rope_hcu.load_operator() is operator
        assert pcp_q_rope.load_operator() is operator
        assert pcp_q_rope.registry.get(pcp_fork.Q_ROPE_OP)["implementation"] == "hcu"
        assert len(loads) == 1
    finally:
        pcp_q_rope.load_operator.cache_clear()
        inverse_rope_hcu.load_operator.cache_clear()


@pytest.mark.parametrize("eager", [True, False])
def test_plugin_binds_provider_for_owned_eager_attention(monkeypatch, isolated_q_rope, eager):
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    monkeypatch.setattr(v41_pcp, "_bound", False)
    for field in ("_reference", "_engram_validator", "_forward_context_resolver"):
        monkeypatch.setattr(pcp_fork, field, None)
    monkeypatch.setattr(hcu_config, "load", lambda: {
        "ops": {pcp_fork.Q_ROPE_OP: "hcu"}, "overrides": {},
    })
    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    monkeypatch.setattr(pcp_fork, "validate_config", lambda cfg: True)
    monkeypatch.setattr(
        DeepseekV41ROCMAiterMLAAttention, "__init__",
        lambda self, *args, **kwargs: torch.nn.Module.__init__(self),
    )

    cls = v41_pcp.attention_class()
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=eager))
    attn = cls(vllm_config=config)
    assert attn._initialize_q_rope is (pcp_q_rope.initialize if eager else None)
    assert attn._prepare_q_rope is None


@pytest.mark.parametrize("route", [
    "prefill", "stock", "padding", "mixed", "profile", "unsupported", "cold",
    "drift-before-profile", "drift-after-profile",
])
def test_owned_prefill_routes_q_and_keeps_native_kv(monkeypatch, isolated_q_rope, route):
    from vllm import distributed
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    calls = []
    native_q = object()

    def native(self, q, kv, positions, metadata):
        calls.append("native")
        return native_q

    monkeypatch.setattr(DeepseekV41ROCMAiterMLAAttention, "_fused_qnorm_rope_kv_insert", native)
    cls = pcp_fork.build_classes()["attention"]
    attn = cls.__new__(cls)
    torch.nn.Module.__init__(attn)
    q = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1])
    kv = torch.zeros(2, 512, dtype=torch.bfloat16)
    gathered = torch.cat((kv, kv))
    slots = torch.arange(4) + 64
    cache_positions = torch.arange(4) + 256
    cos_sin = torch.zeros(300, 64, dtype=torch.float32)
    cache = torch.empty(1, 64, 584, dtype=torch.uint8)
    metadata = {"swa": SimpleNamespace(
        num_prefill_tokens=2, num_decode_tokens=int(route == "mixed"),
        cache_slot_mapping=slots, cache_positions=cache_positions, block_size=64,
    )}
    if route == "profile":
        metadata = None

    def prepare(query, pos, table):
        assert query is q and pos is positions and table is cos_sin
        calls.append("prepare")
        if route == "unsupported":
            return None

        def rotate():
            calls.append("rotate")
            query[..., -64:].fill_(1)

        return rotate

    def gather(arg, dim):
        assert arg is kv and arg.is_contiguous() and dim == 0
        calls.append("gather")
        return gathered

    def insert(*args):
        assert all(x is y for x, y in zip(args[:5], (
            gathered, cache, slots, cache_positions, cos_sin,
        ), strict=True))
        assert args[5:] == (64, None, False)
        calls.append("kv_insert")

    monkeypatch.setattr(distributed, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather))
    monkeypatch.setattr(torch.ops._C, "fused_deepseek_v4_kv_rope_insert", insert, raising=False)
    def initialize(table):
        assert table is cos_sin
        calls.append("initialize")
        return prepare

    attn._initialize_q_rope = None if route == "stock" else initialize
    attn._prepare_q_rope = None
    attn.use_pcp = True
    monkeypatch.setattr(cls, "accepts_unnormed_unroped_query", property(lambda self: False))
    attn.n_local_heads = 64
    attn.padded_heads = 128 if route == "padding" else 64
    attn.rope_head_dim = 64
    attn.rotary_emb = SimpleNamespace(is_neox_style=False, cos_sin_cache=cos_sin)
    attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=cache)
    attn.kv_mxfp8 = False
    attn.kv_cache_torch_dtype = torch.uint8

    drift = route.startswith("drift-")
    if drift:
        from hcu_adapter.bridges import vllm_pcp

        monkeypatch.setattr(hcu_config, "load", lambda: {
            "ops": {pcp_fork.Q_ROPE_OP: "hcu"}, "overrides": {},
        })
        monkeypatch.setattr(vllm_pcp, "_failure", None)
        monkeypatch.setattr(vllm_pcp, "_publication", None)
        monkeypatch.setattr(vllm_pcp, "_lifecycle", vllm_pcp.ProcessLocalLifecycle(
            runtime_action="vllm-pcp", error_type=vllm_pcp.PcpCompatibilityError))
        monkeypatch.setattr(vllm_pcp.registry, "register", lambda **kwargs: None)
        assert vllm_pcp.install()
        pcp_fork.set_q_rope_provider(initialize)
        vllm_pcp.require_healthy()

    if route in ("prefill", "mixed", "profile", "unsupported", "drift-after-profile"):
        assert attn._fused_qnorm_rope_kv_insert(q, kv, positions, None) is native_q
        assert calls == ["initialize", "native"]
        assert torch.count_nonzero(q) == 0
        calls.clear()

    if drift:
        monkeypatch.setenv("HCU_DISABLE_OPS", pcp_fork.Q_ROPE_OP)
        with pytest.raises(vllm_pcp.PcpCompatibilityError, match="restart process"):
            vllm_pcp.require_healthy()
        monkeypatch.setenv("HCU_DISABLE_OPS", "")
        with pytest.raises(vllm_pcp.PcpCompatibilityError, match="restart process"):
            vllm_pcp.require_healthy()
        with pytest.raises(pcp_fork.PcpAttentionError, match="restart process"):
            attn._fused_qnorm_rope_kv_insert(q, kv, positions, metadata)
        assert calls == []
        return

    result = attn._fused_qnorm_rope_kv_insert(q, kv, positions, metadata)
    if route == "prefill":
        assert result is q
        assert calls == ["prepare", "gather", "kv_insert", "rotate"]
        assert torch.count_nonzero(q[..., :-64]) == 0
        assert torch.all(q[..., -64:] == 1)
    else:
        assert result is native_q
        assert calls == (["prepare", "native"] if route == "unsupported" else ["native"])
        assert torch.count_nonzero(q) == 0


@pytest.mark.parametrize("change", ["policy", "provider"])
def test_plugin_owner_latches_q_rope_binding_drift(monkeypatch, isolated_q_rope, change):
    from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
    from hcu_vllm_das import v41_pcp_activation as activation
    from vllm import envs
    from vllm.model_executor.models import ModelRegistry

    monkeypatch.setattr(v41_pcp, "_bound", False)
    for field in ("_reference", "_engram_validator", "_forward_context_resolver"):
        monkeypatch.setattr(pcp_fork, field, None)
    for field, value in (("_state", None), ("_failure", None), ("_publishing", False)):
        monkeypatch.setattr(activation, field, value)
    monkeypatch.setattr(activation, "_is_hcu", lambda: True)
    monkeypatch.setattr(activation, "_install_profile_worker", lambda: None)
    monkeypatch.setattr(hcu_config, "load", lambda: {
        "ops": {pcp_fork.Q_ROPE_OP: "hcu"}, "overrides": {},
    })
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_ENGRAM_HOST", "0")
    monkeypatch.setenv("HCU_VLLM_FP4_INDEXER_IMPL", "stock")
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    envs.environment_variables.pop(activation._FACTOR, None)

    context = ExtensionContext("0.5.22", "0.33.0")
    activation.activate_extension(context)
    q_rope_factor = activation.policy_factor()
    assert q_rope_factor[-1] == "q_rope_hcu"
    assert pcp_fork._q_rope_provider is pcp_q_rope.initialize
    if change == "policy":
        monkeypatch.setenv("HCU_DISABLE_OPS", pcp_fork.Q_ROPE_OP)
        assert activation.policy_factor() == q_rope_factor[:-1]
    else:
        monkeypatch.setattr(pcp_fork, "_q_rope_provider", lambda cache: None)
    with pytest.raises(ExtensionCommitError, match="restart process"):
        activation.check_health()

    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    monkeypatch.setattr(pcp_fork, "_q_rope_provider", pcp_q_rope.initialize)
    with pytest.raises(ExtensionCommitError, match="restart process"):
        activation.activate_extension(context)
    with pytest.raises(pcp_fork.PcpAttentionError, match="restart process"):
        pcp_fork.require_q_rope_healthy()
