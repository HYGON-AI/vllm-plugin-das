# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the optional in-place PCP query rotation."""
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import hcu_config
from hcu_adapter.kernels import pcp_fork
from hcu_vllm_das import pcp_q_rope, v41_pcp


def test_q_rope_policy(monkeypatch):
    ops = {}
    monkeypatch.setattr(hcu_config, "load", lambda: {"ops": ops, "overrides": {}})
    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    assert not pcp_fork.q_rope_selected()
    ops[pcp_fork.Q_ROPE_OP] = "hcu"
    assert pcp_fork.q_rope_selected()
    for op in (pcp_fork.Q_ROPE_OP, pcp_fork.OP_NAME):
        monkeypatch.setenv("HCU_DISABLE_OPS", op)
        assert not pcp_fork.q_rope_selected()


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


@pytest.mark.parametrize("eager", [True, False])
def test_plugin_binds_provider_for_owned_eager_attention(monkeypatch, eager):
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    monkeypatch.setattr(v41_pcp, "_bound", False)
    for field in ("_reference", "_engram_validator", "_forward_context_resolver", "_q_rope_provider"):
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
])
def test_owned_prefill_routes_q_and_keeps_native_kv(monkeypatch, route):
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

    if route in ("prefill", "mixed", "profile", "unsupported"):
        assert attn._fused_qnorm_rope_kv_insert(q, kv, positions, None) is native_q
        assert calls == ["initialize", "native"]
        assert torch.count_nonzero(q) == 0
        calls.clear()

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


@pytest.mark.parametrize("tokens", [1, 128, 4096])
def test_device_matches_native_q_only(tokens):
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires gfx938 and the vLLM native Q-only operator")
    device = torch.device("cuda", torch.cuda.current_device())
    if not pcp_q_rope._supported_device(device):
        pytest.skip("requires gfx938")

    from vllm.platforms import current_platform

    current_platform.import_kernels()
    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(33)
        q = torch.randn(tokens, 64, 512, dtype=torch.bfloat16, device=device, generator=generator)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        angles = torch.randn(tokens, 32, device=device, generator=generator)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        kv = torch.empty(tokens, 512, dtype=q.dtype, device=device)
        cache = torch.empty(1, 64 * 584, dtype=torch.uint8, device=device)
        slots = positions[:0]
        native = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, cache, slots, positions, cos_sin, 64, 1e-6, 64,
            False, False, True, False,
        )
        prepare = pcp_q_rope.initialize(cos_sin)
        assert prepare is not None
        rotate = prepare(q, positions, cos_sin)
        assert rotate is not None
        rotate()
        torch.testing.assert_close(q, native, rtol=0, atol=0)
