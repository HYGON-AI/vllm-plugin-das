# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression: non-causal flattening must stay LOCAL to the forward call.

Mutating the shared ``attn_metadata.decode`` rows re-flattens on every
attention layer over the same metadata and aliases sibling requests onto the
first request's KV blocks (bs>=2 draft acceptance collapse; serial immune).
Fork 877e6996 parity: flatten with LOCAL variables only.
"""

from __future__ import annotations

import re

import torch

from vllm_hcu.v1.attention.backends.mla.triton_mla import (
    NON_CAUSAL_DRAFT_STAMP,
    HcuTritonMLABackend,
    HcuTritonMLAMetadataBuilder,
    HcuTritonMLAImpl,
    expand_causal_rows,
    flatten_non_causal_rows,
)
from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAMetadataBuilder,
    TritonMLAImpl,
)


def test_backend_declares_non_causal_capability():
    assert HcuTritonMLABackend.supports_non_causal() is True
    assert TritonMLABackend.supports_non_causal() is not True


def test_builder_opts_into_spec_aware_reorder_threshold():
    assert (
        HcuTritonMLAMetadataBuilder.query_len_support
        is not QueryLenSupport.SINGLE_ONLY
    )
    assert HcuTritonMLAMetadataBuilder.supports_non_causal_multi_token_decode is True
    assert issubclass(HcuTritonMLAMetadataBuilder, TritonMLAMetadataBuilder)


def test_backend_wires_hcu_builder_and_impl():
    assert HcuTritonMLABackend.get_builder_cls() is HcuTritonMLAMetadataBuilder
    assert HcuTritonMLABackend.get_impl_cls() is HcuTritonMLAImpl
    assert issubclass(HcuTritonMLAImpl, TritonMLAImpl)


def test_flatten_maps_every_row_to_its_own_request():
    block_table = torch.tensor([[5], [9]])
    seq_lens = torch.tensor([10, 20])
    bt, sl = flatten_non_causal_rows(block_table, seq_lens, 14, 2)
    assert bt.shape[0] == 14 and sl.shape[0] == 14
    # rows 0-6 (request 0's 7 draft tokens) -> request 0; rows 7-13 -> request 1
    assert torch.equal(bt[:, 0], torch.tensor([5] * 7 + [9] * 7))
    assert torch.equal(sl, torch.tensor([10] * 7 + [20] * 7))


def test_flatten_is_noop_for_causal_single_token_rows():
    block_table = torch.tensor([[5], [9]])
    seq_lens = torch.tensor([10, 20])
    bt, sl = flatten_non_causal_rows(block_table, seq_lens, 2, 2)
    assert bt is block_table and sl is seq_lens


def test_flatten_never_mutates_inputs():
    block_table = torch.tensor([[5], [9]])
    seq_lens = torch.tensor([10, 20])
    bt_snapshot = block_table.clone()
    sl_snapshot = seq_lens.clone()
    for _ in range(3):
        bt, sl = flatten_non_causal_rows(block_table, seq_lens, 14, 2)
    # repeated "layers" over the same metadata get identical results and the
    # original per-request rows stay intact
    assert torch.equal(block_table, bt_snapshot)
    assert torch.equal(seq_lens, sl_snapshot)
    assert bt.shape[0] == 14 and sl.shape[0] == 14


def test_impl_overrides_forward_with_local_flatten_source():
    """The override must carry the fork's local-flatten implementation."""

    import inspect

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    assert "flatten_non_causal_rows" in src
    # no attribute write-back on the shared metadata
    banned = [ln for ln in src.splitlines() if "decode_meta." in ln and "=" in ln]
    assert not banned, banned


def test_expand_causal_rows_reviewer_scalars():
    """Review 2026-10-12 P1 #1: causal verify must NOT read future KV.

    Prefix 2 + 2 query tokens, seq_lens 4 -> per-row lengths [3, 4] (the
    early query may read up to its own token only), never the flattened
    [4, 4].  With uniform V=[0, 0, 0, 100] the first row's output must then
    be 0 (it cannot see the last value), not the flattened 25.
    """

    block_table = torch.tensor([[5], [5]])
    seq_lens = torch.tensor([4, 4])
    bt, sl = expand_causal_rows(block_table, seq_lens, 4, 2)
    assert bt.shape[0] == 4 and sl.shape[0] == 4
    assert torch.equal(sl, torch.tensor([3, 4, 3, 4]))
    assert torch.equal(
        bt[:, 0], torch.tensor([5, 5, 5, 5])
    )


def test_expand_causal_rows_is_pure():
    block_table = torch.tensor([[5], [5]])
    seq_lens = torch.tensor([4, 4])
    bt_snapshot = block_table.clone()
    sl_snapshot = seq_lens.clone()
    for _ in range(2):
        bt, sl = expand_causal_rows(block_table, seq_lens, 4, 2)
    assert torch.equal(block_table, bt_snapshot)
    assert torch.equal(seq_lens, sl_snapshot)
    assert bt.shape[0] == 4 and sl.shape[0] == 4


def test_expand_causal_rows_passthrough_single_token():
    block_table = torch.tensor([[5], [9]])
    seq_lens = torch.tensor([10, 20])
    bt, sl = expand_causal_rows(block_table, seq_lens, 2, 2)
    assert bt is block_table and sl is seq_lens
    bt, sl = expand_causal_rows(block_table, seq_lens, 3, 2)
    # 3 tokens over 2 decodes is not a uniform verify pack: no expand is
    # triggered at this size (query_len <= 1 passthrough branch).
    assert bt is block_table and sl is seq_lens


def test_expand_causal_rows_rejects_non_divisible_multi_token_pack():
    """Review 2026-10-14 P1 discussion (defensive guard): forward_mqa
    launches one kernel row per query, so the expansion must produce
    exactly ``num_decode_tokens`` rows.  A multi-token pack whose token
    count is NOT a multiple of the request count (query_len > 1) would
    mis-map queries to requests -- impossible on every caller path today
    (uniform-split contract + graph-descriptor divisibility), so fail
    loudly instead of drifting silently."""

    block_table = torch.tensor([[5], [9]])
    seq_lens = torch.tensor([10, 20])
    try:
        expand_causal_rows(block_table, seq_lens, 7, 2)
    except RuntimeError as exc:
        assert "not a multiple of" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for 7 tokens / 2 decodes")
    # Divisible multi-token packs still expand fine.
    bt, sl = expand_causal_rows(block_table, seq_lens, 6, 2)
    assert sl.shape[0] == 6


def test_forward_keys_row_semantics_on_the_producer_stamp():
    """Stamped multi-token rows flatten; unstamped rows keep causal lengths."""

    import inspect

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    # forward_mqa references the stamp by NAME (imported constant), so assert
    # the name in the source and pin the constant's VALUE separately.
    # ANCHOR ON STATEMENTS, not first occurrence: the DCP-reject comment
    # (latest review P1) also mentions the stamp, so searching the bare
    # name would land inside prose and break the ordering contract.
    assert NON_CAUSAL_DRAFT_STAMP == "_vllm_hcu_non_causal_draft"
    assert "NON_CAUSAL_DRAFT_STAMP, False" in src
    assert "expand_causal_rows(" in src
    stamp_pos = src.index("NON_CAUSAL_DRAFT_STAMP, False")
    flatten_pos = src.index("flatten_non_causal_rows(\n")
    expand_pos = src.index("expand_causal_rows(\n")
    assert stamp_pos < flatten_pos < expand_pos


def test_producer_stamp_shares_the_backend_constant():
    """Single source of truth: the stamp patch imports the backend symbol."""

    from vllm_hcu.patch.worker.framework_opt import patch_dspark_non_causal_stamp

    assert patch_dspark_non_causal_stamp._STAMP == NON_CAUSAL_DRAFT_STAMP
    assert patch_dspark_non_causal_stamp.TARGET_MODULE == (
        "vllm.v1.worker.gpu.spec_decode.speculator"
    )


def test_decode_descales_both_sides_with_k_scale():
    """Latest review 2026-10-15 P2: the MLA combined c_kv cache is written
    with _k_scale only; the kernel's reachable descale paths must use the
    same factor.  The grouped MLA path ignores v_scale (v = trans(k)); the
    normal MHA fallback (kv_group_num == 1) loads this very cache as V, so
    a separately calibrated _v_scale would dequantize with the wrong
    factor.  Fork 877e6996 passes _k_scale for both sides, as does
    upstream.  Source-contract style: no GPU needed."""

    import inspect
    import re

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    assert re.search(r"\bv_scale=layer\._k_scale\b", src), src[-400:]
    assert not re.search(r"\bv_scale=layer\._v_scale\b", src)


def test_forward_rejects_causal_multi_token_under_dcp():
    """Latest review 2026-10-15 P1: under DCP>1 the query is all-gathered
    while each rank owns only its interleaved KV slice, so global-lens
    causal expansion cannot yield per-shard masks -- the lse reduction then
    folds wrong per-rank partitions.  expand_causal_rows is our own
    addition (fork has no causal expansion), so REJECT loudly instead of
    silently corrupting logits.  Stamped non-causal batches stay allowed
    (repeat_interleave is rank-symmetric).  Source-contract style."""

    import inspect

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    assert "dcp_world_size > 1" in src
    assert "RuntimeError" in src
    # The guard must sit BEFORE the row-semantics split so no path can
    # expand global causal lengths under DCP.
    guard_pos = src.index("dcp_world_size > 1")
    expand_pos = src.index("expand_causal_rows(\n")
    assert guard_pos < expand_pos
    # ...and must not fire for stamped non-causal (flatten) batches: the
    # guard predicate must include the stamp negation, not just row counts
    # (round-3 audit B1: a shape-only predicate would crash stamped DSpark
    # draft batches under DCP>1).
    assert re.search(
        r"unstamped_multi_token\s*=\s*\(\s*not is_non_causal",
        src,
    ), src[:1200]
    assert re.search(r"is_non_causal\s*=\s*getattr\(", src)


def test_forward_dcp_guard_rejects_only_unstamped_multi_token(
    monkeypatch, tmp_path
):
    """CPU behavior test for the DCP>1 guard's input domain.

    (a) stamped multi-token  -> no raise, flatten path taken;
    (b) unstamped multi-token + dcp_world_size=2 -> RuntimeError;
    (c) unstamped single-token + dcp_world_size=2 -> no raise.
    ``forward_mqa`` is driven to the launch boundary only: the real kernel
    is monkeypatched out, so no GPU is touched.
    """

    import types

    import vllm_hcu.v1.attention.backends.mla.triton_mla as tmla

    impl = object.__new__(HcuTritonMLAImpl)
    impl.kv_lora_rank = 4
    impl.scale = 1.0
    impl.dcp_world_size = 2

    decode = types.SimpleNamespace(
        block_table=torch.tensor([[7], [9]]),
        seq_lens=torch.tensor([8, 12], dtype=torch.int32),
    )
    metadata = types.SimpleNamespace(
        decode=decode,
        num_decode_tokens=4,
        num_decodes=2,
        max_seq_len=12,
    )
    impl._sm_count = 8

    calls = {}

    class _Layer:
        _k_scale = torch.tensor(1.0)
        _v_scale = torch.tensor(1.0)

    def fake_launcher(*args, **kwargs):
        calls["kwargs"] = kwargs
        return torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(tmla, "decode_attention_fwd", fake_launcher)
    # workspace path is optional; force the fallback allocation branch by
    # ensuring the manager reports uninitialized.
    monkeypatch.setattr(tmla, "is_workspace_manager_initialized", lambda: False)

    q = torch.zeros(4, 1, 8)
    kv = torch.zeros(1, 1, 16)

    # (b) unstamped multi-token under DCP>1: the guard must fire.
    try:
        impl.forward_mqa(q, kv, metadata, _Layer())
    except RuntimeError as exc:
        assert "decode_context_parallel_size > 1" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unstamped DCP batch")

    # (a) stamped multi-token: must NOT raise and must flatten (rows = 4).
    setattr(metadata, NON_CAUSAL_DRAFT_STAMP, True)
    consumed = {}

    real_flatten = tmla.flatten_non_causal_rows

    def capture_flatten(block_table, seq_lens, num_decode_tokens, num_decodes):
        bt, sl = real_flatten(
            block_table, seq_lens, num_decode_tokens, num_decodes
        )
        consumed["rows"] = int(sl.shape[0])
        return bt, sl

    monkeypatch.setattr(tmla, "flatten_non_causal_rows", capture_flatten)
    impl.forward_mqa(q, kv, metadata, _Layer())
    assert consumed["rows"] == 4

    # (c) unstamped single-token: no raise, rows equal requests (2).
    metadata2 = types.SimpleNamespace(
        decode=decode, num_decode_tokens=2, num_decodes=2, max_seq_len=12
    )
    # return the stamped flag off
    metadata2.__setattr__(NON_CAUSAL_DRAFT_STAMP, False)
    monkeypatch.setattr(
        tmla, "expand_causal_rows", tmla.expand_causal_rows
    )
    impl.forward_mqa(q[:2], kv, metadata2, _Layer())


def test_builder_build_rejects_multi_token_under_dcp(monkeypatch):
    """Review 2026-10-14 P2: under DCP>1 the wheel resets
    ``reorder_batch_threshold`` to 1 (``supports_dcp_with_varlen=False``),
    so a multi-token DRAFT batch is routed to the PREFILL branch of the
    inherited build and dies on ``assert seq_lens_cpu is not None`` (draft
    metadata is built with ``seq_lens_cpu_upper_bound=None``, attn_utils
    ``build_attn_metadata`` default).  The builder must reject that
    combination loudly BEFORE the wheel split.  Scope (round-4 audit B1):
    target/plain PREFILL under DCP>1 carries a real
    ``seq_lens_cpu_upper_bound`` and MUST stay on the wheel's supported
    chunked-prefill path -- only draft batches with a query len > 1 AND a
    missing bound are rejected.

    Source-level control flow (codex reproduction, numpy stand-in):
    DCP=1 (threshold=15): (decodes,prefills)=(2,0); DCP=2 (threshold=1):
    (0,2) then AssertionError in the wheel build paths.
    """

    import types

    calls = []

    def spy_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        calls.append(
            (
                common_attn_metadata.max_query_len,
                getattr(
                    common_attn_metadata, "seq_lens_cpu_upper_bound", "<absent>"
                ),
            )
        )
        return "built"

    monkeypatch.setattr(TritonMLAMetadataBuilder, "build", spy_build)

    builder = object.__new__(HcuTritonMLAMetadataBuilder)
    builder.dcp_world_size = 2

    def meta(q_len, bound):
        return types.SimpleNamespace(
            max_query_len=q_len, seq_lens_cpu_upper_bound=bound
        )

    # (a) DCP>1 + draft shape (q>1, no bound): loud reject before the
    # wheel split.
    try:
        builder.build(0, meta(7, None))
    except RuntimeError as exc:
        assert "decode_context_parallel_size > 1" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for DCP draft build")
    assert calls == []

    # (b) DCP>1 + plain prefill (q>1 but WITH a real bound): must stay on
    # the wheel's supported chunked-prefill path (round-4 audit B1).
    calls.clear()
    assert builder.build(0, meta(37, torch.tensor([50, 60]))) == "built"
    assert len(calls) == 1
    assert calls[0][0] == 37
    assert calls[0][1] is not None

    # (b2) DCP>1 + q>1 WITHOUT the field at all (sentinel path, round-4
    # audit WARN): a bare stub must fail OPEN -- the guard only matches a
    # genuine ``None`` bound, never an absent field.
    calls.clear()
    assert builder.build(0, types.SimpleNamespace(max_query_len=7)) == "built"
    assert len(calls) == 1 and calls[0][0] == 7

    # (c) DCP>1 + pure single-token decode: keep the inherited wheel path.
    calls.clear()
    assert builder.build(0, meta(1, None)) == "built"
    assert len(calls) == 1
    assert calls[0][0] == 1

    # (d) DCP=1 + draft shape (normal non-DCP DSpark verify on this
    # backend): unchanged.
    builder.dcp_world_size = 1
    calls.clear()
    assert builder.build(0, meta(7, None)) == "built"
    assert len(calls) == 1 and calls[0][0] == 7


def test_builder_build_guard_sits_before_super_delegation():
    """Source contract: the DCP guard is the first thing the HCU build
    override does, so no path reaches the wheel split first."""

    import inspect

    src = inspect.getsource(HcuTritonMLAMetadataBuilder.build)
    guard_pos = src.index("dcp_world_size > 1")
    super_pos = src.index("super().build(")
    assert guard_pos < super_pos
