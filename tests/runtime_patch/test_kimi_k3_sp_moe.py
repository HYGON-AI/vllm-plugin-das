# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3 SP-MoE patch contracts (CPU only, no CUDA, no distributed init).

Binding contract: for every N (tokens) x TP (shards) x rank combination the
SP forward must produce the rows of the replicated-domain reference
computation. Collectives and the attn_res kernel are replaced by stubs with
exact semantics, so equality is bit-exact (torch.equal).

Review-contract tests (chatgpt-codex-connector P1 round):
- P1-1: the phase-routed communicator selection must survive torch.compile;
  the collectives are registered custom ops (opaque to dynamo, fake impl for
  shapes), and the module must not fall back "while compiling".
- P1-2: the SP forward must be branch-free in the token count (dropped-guard
  hazard), and its window arithmetic must never slice past N.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_sp_moe as sp


# ---------------------------------------------------------------------------
# Stubs with exact collective / kernel semantics.
# ---------------------------------------------------------------------------


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _stub_attn_res(prefix, residual, proj, norm, num_valid_blocks):
    """Row-wise stand-in for the boltops attn_res kernel.

    Mirrors the real wrapper's static gate (num_valid_blocks <= 0 short-
    circuits), then pins the kernel contract under test: padding rows
    (softmax of zeroed prefix) must never be fed in, so every live call
    must carry a positive batch of finite rows whose sizes match.
    """
    if num_valid_blocks <= 0:
        return prefix
    assert prefix.shape[0] > 0, "padding rows reached the attn_res kernel"
    assert residual.shape[0] == prefix.shape[0]
    return prefix + residual.sum(dim=1)


def _make_layer(*, tp_rank: int, tp_size: int, block_write: bool, prev_blocks: int, H: int = 6):
    """Stub KimiDecoderLayer exposing exactly the attributes the SP forward uses."""
    layer = types.SimpleNamespace()
    res_proj = types.SimpleNamespace(weight=torch.eye(H))
    res_norm = types.SimpleNamespace(weight=torch.ones(H), variance_epsilon=1e-6)
    layer.self_attention_res_proj = res_proj
    layer.self_attention_res_norm = res_norm
    layer.mlp_res_proj = res_proj
    layer.mlp_res_norm = res_norm
    layer.is_block_write_layer = block_write
    layer.block_write_idx = 1
    layer.prev_valid_blocks = prev_blocks
    layer.input_layernorm = _Identity()
    layer.post_attention_layernorm = _Identity()
    layer._run_self_attn = lambda positions, h: h * 3.0 + 1.0
    layer.mlp = lambda h: h - 2.0
    setattr(layer, sp._LAYER_STATE, sp._SpState(tp_size, tp_rank, runner=None))
    state = getattr(layer, sp._LAYER_STATE)
    state.checked = True  # skip the real MoERunner introspection
    return layer


def _reference_full_domain(hs, br, *, block_write: bool, prev_blocks: int):
    """Replicated-domain computation of the same layer math (AR == identity).

    Written from the layer equations with the same stubs, never from the SP
    code path. ``br`` is mutated in place exactly like the layer does (block
    write), so ``br.sum(dim=1)`` after the write is the residual stream the
    MLP-side kernel consumes.
    """
    pre = hs if prev_blocks <= 0 else hs + br.sum(dim=1)   # entry kernel; a no-op gate at 0 blocks
    attn = pre * 3.0 + 1.0
    if block_write:
        br[:, 1, :] = hs                        # block write (in place)
        prefix_mid = attn                       # residual stream restarts at attn
    else:
        prefix_mid = hs + attn                  # upstream :  prefix_sum + attn_out
    if prev_blocks + (1 if block_write else 0) > 0:
        hidden_mid = prefix_mid + br.sum(dim=1)  # br reflects a done write
    else:
        hidden_mid = prefix_mid
    return prefix_mid + (hidden_mid - 2.0)


def _run_sp_rank(
    hs,
    br,
    reference,
    *,
    tp_rank: int,
    tp_size: int,
    block_write: bool,
    prev_blocks: int,
    observe: dict,
):
    """Run one rank's SP forward with exact collective stubs.

    RS: rank r receives rows [r*S, (r+1)*S) of the summed padded partials.
    With deterministic stubs every rank's padded partial is identical, so the
    summed tensor is the padded input itself and the window is a slice.
    AG: the gathered stream is the replicated full-domain result, taken from
    the independently derived ``reference`` -- this asserts the window each
    rank contributes sits at its own slot.
    """
    layer = _make_layer(
        tp_rank=tp_rank, tp_size=tp_size, block_write=block_write, prev_blocks=prev_blocks
    )
    state = getattr(layer, sp._LAYER_STATE)
    S = -(-hs.shape[0] // tp_size)
    padded = S * tp_size

    def rs_stub(x, tp):
        # All ranks hold the same padded partial (deterministic stubs), so
        # the reduction is a no-op over identical windows and rank r's
        # received window is the slice of x itself.
        assert x.shape[0] == padded
        observe.setdefault("rs_windows", []).append(
            (tp_rank, x[tp_rank * S : (tp_rank + 1) * S].clone())
        )
        return x[tp_rank * S : (tp_rank + 1) * S].clone()

    def ag_stub(x, tp):
        observe.setdefault("ag_locals", []).append((tp_rank, x.clone()))
        # AG semantics: each rank's window placed at its own slot. Slots of
        # other ranks are reconstructed independently from the reference so
        # the equality below asserts the placement and the final slice, not
        # a passthrough of the reference object. Reference windows past N
        # don't exist -- they stay zero (they get sliced away anyway).
        full = torch.zeros(S * tp_size, x.shape[1])
        for r in range(tp_size):
            g = r * S
            v = max(0, min(S, reference.shape[0] - g))
            if v:
                full[g : g + v] = reference[g : g + v]
        full[tp_rank * S : (tp_rank + 1) * S] = x
        return full

    with pytest.MonkeyPatch.context() as mp:
        # Patch the op namespace attributes the forward calls through. This
        # keeps the numerics contract independent of dispatcher backends:
        # on a CPU-only test box the custom ops may have no backend impl.
        mp.setattr(
            torch.ops.vllm, "hcu_k3_sp_reduce_scatter", rs_stub, raising=False
        )
        mp.setattr(
            torch.ops.vllm, "hcu_k3_sp_all_gather", ag_stub, raising=False
        )
        mp.setattr(sp, "_attn_res", _stub_attn_res)
        prefix, br_out = sp._sp_forward_attn_residual(
            layer, state, torch.zeros(hs.shape[0]), hs, br
        )
    assert br_out is br
    return prefix


# ---------------------------------------------------------------------------
# Numerics contract: SP rows == replicated rows for every N x TP x rank.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("block_write", [False, True])
@pytest.mark.parametrize("prev_blocks", [0, 2])
def test_sp_rows_match_replicated_domain(tp_size, num_tokens, block_write, prev_blocks):
    H, B = 6, 3
    gen = torch.Generator().manual_seed(1234 + num_tokens)
    hs = torch.randn(num_tokens, H, generator=gen)
    br = torch.randn(num_tokens, B, H, generator=gen)
    reference = _reference_full_domain(
        hs.clone(), br.clone(), block_write=block_write, prev_blocks=prev_blocks
    )

    observe: dict = {}
    for r in range(tp_size):
        prefix = _run_sp_rank(
            hs.clone(), br.clone(), reference,
            tp_rank=r, tp_size=tp_size,
            block_write=block_write, prev_blocks=prev_blocks,
            observe=observe,
        )
        assert prefix.shape == (num_tokens, H)
        # Every rank must land on the same replicated-domain stream.
        assert torch.equal(prefix, reference), (
            f"rank {r} mismatch N={num_tokens} TP={tp_size} "
            f"bw={block_write} prev={prev_blocks}"
        )

    # Each rank contributed exactly its own window to the gather. Window
    # rows beyond N are padding rows, which this patch leaves at the mlp
    # stub's bias (-2.0) before the final slice drops them.
    S = -(-num_tokens // tp_size)
    H = hs.shape[1]
    for r, local in observe["ag_locals"]:
        g = r * S
        v = max(0, min(S, num_tokens - g))
        assert torch.equal(local[:v], reference[g : g + v])
        if v < S:
            assert torch.equal(local[v:], torch.full((S - v, H), -2.0))
    # And the reduce-scatter window is the summed padded partial's window.
    padded = S * tp_size
    pre = hs if prev_blocks <= 0 else hs + br.sum(dim=1)
    attn_partial = pre * 3.0 + 1.0
    padded_partial = torch.zeros(padded, H)
    padded_partial[:num_tokens] = attn_partial
    for r, window in observe["rs_windows"]:
        assert torch.equal(window, padded_partial[r * S : (r + 1) * S])


# ---------------------------------------------------------------------------
# Router-side windowing (the REAL router_runtime helper): rank slices,
# short-mask skip, no-window no-op.
# ---------------------------------------------------------------------------

from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    window_sp_padding_mask,
)


def test_router_windows_the_global_suffix_mask():
    N, TP, S = 9, 4, 3  # padded 12; rank 3 owns window rows [9, 12) -> all pad
    global_mask = torch.tensor([i >= N for i in range(N)])
    for rank in range(TP):
        topk_ids = torch.zeros(S, 3, dtype=torch.long)
        out = window_sp_padding_mask(global_mask, topk_ids, (rank, TP))
        assert out.shape == (S,)
        v = max(0, min(S, N - rank * S))
        assert not bool(out[:v].any()), f"rank {rank} real rows must stay unmasked"
        if v < S:
            assert bool(out[v:].all()), f"rank {rank} padding rows must be masked"


def test_router_windows_below_window_start_as_padding():
    """N < rank*S is legal (small batches): those windows are all padding.

    Review repro N=1, TP=4, S=1: rank 0 keeps its real row; ranks 1-3 must
    get all-True windows and dispatch nothing -- NOT keep a fake real row
    behind the global end.
    """
    mask = torch.tensor([False])  # exactly one real token in the batch
    for rank in range(4):
        out = window_sp_padding_mask(
            mask, torch.zeros(1, 2, dtype=torch.long), (rank, 4)
        )
        assert out.tolist() == ([False] if rank == 0 else [True])


def test_router_total_kept_rows_match_n():
    """Aggregate kept routing rows across ranks == N (review's table)."""
    for N, TP in [(1, 4), (1, 16), (2, 8), (5, 4)]:
        shard = -(-N // TP)  # SP republishes ceil(N/TP) rows per rank
        mask = torch.zeros(N, dtype=torch.bool)  # N real rows, none padding
        kept = 0
        for rank in range(TP):
            out = window_sp_padding_mask(
                mask, torch.zeros(shard, 2, dtype=torch.long), (rank, TP)
            )
            kept += int((~out).sum())
        assert kept == N, f"expected {N} dispatched rows, got {kept}"


def test_router_total_kept_rows_match_real_rows_when_graph_padded():
    """Graph-padded local batch (len(pad) > N): kept rows track real rows,
    the padding-tail rows never dispatch."""
    real, padded_len, TP = 2, 8, 4
    shard = -(-padded_len // TP)  # the MoE sees P rows here, not N
    mask = torch.tensor([i >= real for i in range(padded_len)])
    kept = 0
    for rank in range(TP):
        out = window_sp_padding_mask(
            mask, torch.zeros(shard, 2, dtype=torch.long), (rank, TP)
        )
        kept += int((~out).sum())
    assert kept == real


def test_router_treats_uncovered_window_rows_as_padding():
    """Window coordinates past the bitmap's end True-pad (never fabricate
    real rows). E.g. N=2, TP=4: rank 3's window [3, 4) starts past the
    domain end, so its row is covered-but-unmapped -> True, no dispatch."""
    out = window_sp_padding_mask(
        torch.tensor([False, True]),
        torch.zeros(1, 2, dtype=torch.long),
        (3, 4),
    )
    assert out.tolist() == [True]


def test_router_windows_a_full_exactly_n_bitmap():
    """len(mask) == N == rank*S boundary: the fully-padding last window."""
    mask = torch.tensor([False] * 9)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, (3, 4))
    assert out.shape == (3,) and bool(out.all())


def test_router_without_window_attribute_is_noop():
    mask = torch.tensor([False, True])
    topk_ids = torch.zeros(2, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, None)
    assert out is mask


def test_router_window_major_mask_is_cropped():
    """A mask longer than padded (graph-padded local batches) crops cleanly."""
    N, S, TP = 8, 4, 2
    mask = torch.tensor([i >= 6 for i in range(N + 4)])  # longer than padded
    topk_ids = torch.zeros(S, 2, dtype=torch.long)
    out = window_sp_padding_mask(mask, topk_ids, (1, TP))
    assert out.shape == (S,)
    assert out.tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# Enablement: the window must be attached to the router that masks (P0).
# ---------------------------------------------------------------------------


def test_attach_padding_window_targets_the_router():
    moe = types.SimpleNamespace(
        gate=object(),  # GateLinear: no routing hook (review P0)
        experts=types.SimpleNamespace(
            router=types.SimpleNamespace(_compute_routing=lambda *a, **k: None)
        ),
    )
    sp._attach_padding_window(moe, 1, 4)
    assert moe.experts.router._hcu_sp_padding_window == (1, 4)
    # Arbitrary attributes on the gate must stay untouched.
    assert not hasattr(moe.gate, "_hcu_sp_padding_window")


def test_attach_padding_window_refuses_routerless_moe():
    moe = types.SimpleNamespace(experts=types.SimpleNamespace(router=None))
    with pytest.raises(sp.PatchCompatibilityError, match="_compute_routing"):
        sp._attach_padding_window(moe, 0, 2)


# ---------------------------------------------------------------------------
# P1-2 regression: no branch in the SP body may depend on the token count.
# ---------------------------------------------------------------------------


def _sp_forward_source() -> str:
    src = open(sp.__file__, encoding="utf-8").read()
    return ast.get_source_segment(src, sp._sp_forward_attn_residual) or src


def test_sp_forward_is_branch_free_in_token_count():
    """Dynamo specializes shape branches; vLLM drops the guards (Qwen precedent)."""
    tree = ast.parse(_sp_forward_source())
    banned = {"padded", "valid", "num_tokens", "rows"}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            overlap = banned & names
            assert not overlap, (
                f"SP branch on shape-derived value(s) {sorted(overlap)}: "
                f"{ast.unparse(node.test)[:120]}"
            )


def test_phase_comm_has_no_compiling_fallback():
    """P1-1: tracing must not pick a comm; routing happens per invocation."""
    src = open(sp.__file__, encoding="utf-8").read()
    assert "is_compiling" not in src, (
        "_phase_comm must not treat compilation as a stock-API fast path; "
        "routing happens at invocation time inside custom ops"
    )


# ---------------------------------------------------------------------------
# P1-1 regression: registered collectives with shape fakes.
# ---------------------------------------------------------------------------


def _ensure_ops():
    sp._ensure_custom_ops_registered()


def test_collective_ops_are_registered_with_shape_fakes():
    _ensure_ops()
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.randn(8, 3)
        rs = torch.ops.vllm.hcu_k3_sp_reduce_scatter(x, 4)
        assert tuple(rs.shape) == (2, 3)
        ag = torch.ops.vllm.hcu_k3_sp_all_gather(x, 4)
        assert tuple(ag.shape) == (32, 3)


def test_registration_is_idempotent():
    _ensure_ops()
    _ensure_ops()  # registry guard: no schema redefinition


def test_reduce_scatter_impl_uses_the_phase_comm(monkeypatch):
    calls = []

    class FakeComm:
        def reduce_scatter(self, out, x):
            calls.append(("rs", tuple(out.shape)))
            out.copy_(x[: out.shape[0]])

    monkeypatch.setattr(sp, "_phase_comm", lambda: FakeComm())
    x = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    out = sp._sp_reduce_scatter(x, 4)
    assert calls == [("rs", (2, 3))]
    assert torch.equal(out, x[:2])


def test_all_gather_impl_uses_the_phase_comm(monkeypatch):
    calls = []

    class FakeComm:
        def all_gather(self, out, x):
            calls.append(("ag", tuple(out.shape)))
            out.zero_()
            out[3:6] = x

    monkeypatch.setattr(sp, "_phase_comm", lambda: FakeComm())
    x = torch.ones(3, 2)
    out = sp._sp_all_gather(x, 3)
    assert calls == [("ag", (9, 2))]
    assert torch.equal(out[3:6], x)


def test_impls_fall_back_to_stock_group_api(monkeypatch):
    monkeypatch.setattr(sp, "_phase_comm", lambda: None)

    import vllm.distributed as vd

    # The real stock APIs (TP > 1) always return freshly allocated tensors,
    # not views -- the alias contract of the custom ops depends on it.
    def fake_rs(x, dim=0):
        return x[: x.shape[0] // 2].clone()

    def fake_ag(x, dim=0):
        return x.repeat(2, 1)
    monkeypatch.setattr(
        vd, "tensor_model_parallel_reduce_scatter", fake_rs
    )
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_gather", fake_ag
    )
    x = torch.arange(4 * 2, dtype=torch.float32).view(4, 2)
    assert torch.equal(sp._sp_reduce_scatter(x, 2), x[:2])
    assert torch.equal(sp._sp_all_gather(x, 2), x.repeat(2, 1))


def test_phase_comm_selection_table(monkeypatch):
    """capturing -> TP pynccl comm; eager -> None (stock group API).

    Role-separated eager collectives are deliberately out of scope for this
    PR: there is no accessor, no latch, and no eager aux-comm branch to
    mis-route with. A capturing invocation with an unusable TP pynccl comm
    degrades to None -- the enablement preflight already refuses SP when
    that comm cannot move data, so this is a defensive fallback only.
    """
    role_comm = types.SimpleNamespace(disabled=False, available=True)

    monkeypatch.setattr(
        sp.torch.cuda, "is_current_stream_capturing", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: types.SimpleNamespace(
            device_communicator=types.SimpleNamespace(pynccl_comm=role_comm)
        ),
        raising=False,
    )
    # Capturing: the captured-role comm.
    assert sp._phase_comm() is role_comm

    # Capturing with an unusable (disabled) comm: defensive fallback to None.
    role_comm.disabled = True
    assert sp._phase_comm() is None
    role_comm.disabled = False

    # Eager: None -> the stock group API, whatever the comm looks like.
    monkeypatch.setattr(sp.torch.cuda, "is_current_stream_capturing", lambda: False)
    assert sp._phase_comm() is None


def test_sp_patch_carries_no_role_separation():
    """The role-separation workstream owns alias communicator routing.

    This PR's SP patch must not parse VLLM_ROLE_SEP, resolve any
    role-separation module, or keep a resolution latch (the latch's
    publish-before-assign window was exactly the reload race the review
    rejected). Pins the removal so it cannot quietly creep back in here.
    """
    src = open(sp.__file__, encoding="utf-8").read()
    assert "ROLE_SEP" not in src
    assert "role_sep" not in src
    assert "_get_aux_ar_comm" not in src
    assert "_role_separation" not in src


def test_op_entries_resolve_the_live_module(monkeypatch):
    """A reload must not strand the registered ops on stale globals (P1-1)."""
    sent_x, sent_tp = object(), 7
    seen = {}

    class FakeModule:
        def _sp_reduce_scatter(self, x, tp):
            seen["args"] = (x, tp)
            return "rs-done"

    monkeypatch.setitem(sys.modules, sp.__name__, FakeModule())
    assert sp._sp_reduce_scatter_op_entry(sent_x, sent_tp) == "rs-done"
    assert seen["args"] == (sent_x, sent_tp)


# ---------------------------------------------------------------------------
# P2: the stock reduce-scatter fallback has no c10d path -- enablement must
# refuse SP when the TP pynccl comm cannot move data.
# ---------------------------------------------------------------------------


def test_tp_pynccl_probe_matches_utilization_gate(monkeypatch):
    stub = types.SimpleNamespace(pynccl_comm=None)
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: types.SimpleNamespace(device_communicator=stub),
        raising=False,
    )
    assert sp._tp_pynccl_usable() is False  # comm absent

    # disabled=True exactly mirrors the PyNcclCommunicator no-op gate: the
    # reduced output would be returned unwritten.
    stub.pynccl_comm = types.SimpleNamespace(disabled=True, available=False)
    assert sp._tp_pynccl_usable() is False

    stub.pynccl_comm = types.SimpleNamespace(disabled=False, available=True)
    assert sp._tp_pynccl_usable() is True


# ---------------------------------------------------------------------------
# Config-hash partitioning: the SP switch must share ONE parser with the
# persisted HcuFeatureConfig policy, or enablement and the persistent
# compilation-cache key can drift apart across restarts (review P1).
# ---------------------------------------------------------------------------


def test_switch_enabled_is_sourced_from_the_envs_parser(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    # Effective policy (master AND child); pin the master so the matrix
    # below reads exactly instead of depending on the master default.
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_K3_SP_MOE", "1")
    assert sp._switch_enabled() is True
    monkeypatch.setenv("VLLM_HCU_K3_SP_MOE", "0")
    assert sp._switch_enabled() is False

    # Rollback matrix (review P2): the child switch stays configured while
    # the operator turns the custom-ops master off to fall back. The
    # shared envs helper must disable the install gate exactly like the
    # persisted HcuFeatureConfig policy does, or the wrapper would stay
    # installed and every SP-eligible layer would die at construction.
    monkeypatch.setenv("VLLM_HCU_K3_SP_MOE", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert sp._switch_enabled() is False

    # Prove delegation, not a second parse: whatever the effective
    # authority returns, the switch repeats it verbatim. The persisted
    # cache policy calls this same helper, so a future re-parse cannot
    # reintroduce the drift.
    sentinel = object()
    monkeypatch.setattr(henvs, "k3_sp_moe_enabled", lambda: sentinel)
    assert sp._switch_enabled() is sentinel


def _make_target_module():
    """Minimal vllm.models.kimi_k3.amd.linear stand-in for apply tests."""
    module = types.ModuleType(sp.TARGET_MODULE)

    class KimiDecoderLayer:
        def __init__(self):
            self.built = True

        def forward_attn_residual(self, positions, hidden_states, block_residual):
            return hidden_states

        def _run_self_attn(self, positions, hidden_states):
            return hidden_states

    module.KimiDecoderLayer = KimiDecoderLayer
    return module, KimiDecoderLayer


def test_master_off_child_on_skips_sp_installation(monkeypatch):
    """Full apply+construct under the rollback matrix: master off with the
    child switch still set must NOT install the SP wrapper -- marker
    disabled, original methods intact, construction and forward take the
    stock path instead of raising from _maybe_enable_layer (review P2).
    """
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_K3_SP_MOE", "1")

    module, decoder = _make_target_module()
    original_init = decoder.__init__
    original_forward = decoder.forward_attn_residual

    assert sp.apply_to_module(module) is True
    assert getattr(module, sp._MARKER) == "disabled"
    assert getattr(
        vars(decoder).get("__init__"), sp._INIT_WRAPPER, False
    ) is False
    assert getattr(
        vars(decoder).get("forward_attn_residual"), sp._FORWARD_WRAPPER, False
    ) is False
    assert decoder.__init__ is original_init
    assert decoder.forward_attn_residual is original_forward

    layer = decoder()
    assert layer.built is True
    assert not hasattr(layer, sp._LAYER_STATE)
    assert layer.forward_attn_residual(None, "hs", None) == "hs"

    # Re-application sees the disabled marker and stays out of the way.
    assert sp.apply_to_module(module) is False


def test_master_on_child_on_installs_wrapper_and_constructs_clean(monkeypatch):
    """The positive triplet of the rollback test: master and child on wraps
    the decoder, and constructing a non-SP-shaped layer leaves it on the
    stock path (no state, no RuntimeError) instead of refusing it.
    """
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_K3_SP_MOE", "1")

    module, decoder = _make_target_module()
    assert sp.apply_to_module(module) is True
    assert getattr(module, sp._MARKER) == "enabled"
    assert getattr(
        vars(decoder).get("__init__"), sp._INIT_WRAPPER, False
    ) is True

    layer = decoder()
    assert layer.built is True
    # The stub layer carries none of the SP shape markers, so
    # _maybe_enable_layer declines it before any precondition check.
    assert getattr(layer, sp._LAYER_STATE, None) is None
    assert layer.forward_attn_residual(None, "hs", None) == "hs"


def test_env_surface_declares_the_sp_switch():
    from vllm_hcu.platforms import envs as henvs

    # Worker env propagation and tooling iterate this registry; a missing
    # entry would make the resolved policy engine-only.
    assert "VLLM_HCU_K3_SP_MOE" in henvs.hcu_vllm_environment_variables


def test_feature_config_partitions_the_sp_switch():
    from vllm_hcu.patch.config import HcuFeatureConfig

    defaults = HcuFeatureConfig()
    assert defaults.k3_sp_moe is False
    flipped = defaults.with_updates(k3_sp_moe=True)
    assert flipped.k3_sp_moe is True
    assert flipped != defaults  # bool-affecting fields change the hash input


# ---------------------------------------------------------------------------
# End-to-end-ish subprocess check: patch applies with the switch on and the
# module compiles clean after the changes.
# ---------------------------------------------------------------------------


def test_module_imports_and_compiles_clean():
    script = textwrap.dedent(
        """
        import ast
        import vllm_hcu.patch.worker.op_opt.patch_kimi_k3_sp_moe as sp
        ast.parse(open(sp.__file__).read())  # file parses
        assert hasattr(sp, "_ensure_custom_ops_registered")
        assert hasattr(sp, "_sp_reduce_scatter_fake")
        assert hasattr(sp, "_sp_all_gather_fake")
        print("IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "IMPORT_OK" in result.stdout
