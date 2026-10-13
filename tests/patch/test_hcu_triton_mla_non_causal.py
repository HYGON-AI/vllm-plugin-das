# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression: non-causal flattening must stay LOCAL to the forward call.

Mutating the shared ``attn_metadata.decode`` rows re-flattens on every
attention layer over the same metadata and aliases sibling requests onto the
first request's KV blocks (bs>=2 draft acceptance collapse; serial immune).
Fork 877e6996 parity: flatten with LOCAL variables only.
"""

from __future__ import annotations

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
    # 3 tokens over 2 decodes is not a uniform verify pack: leave untouched.
    assert bt is block_table and sl is seq_lens


def test_forward_keys_row_semantics_on_the_producer_stamp():
    """Stamped multi-token rows flatten; unstamped rows keep causal lengths."""

    import inspect

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    # forward_mqa references the stamp by NAME (imported constant), so assert
    # the name in the source and pin the constant's VALUE separately.
    assert NON_CAUSAL_DRAFT_STAMP == "_vllm_hcu_non_causal_draft"
    assert "NON_CAUSAL_DRAFT_STAMP" in src
    assert "expand_causal_rows" in src
    stamp_pos = src.index("NON_CAUSAL_DRAFT_STAMP")
    flatten_pos = src.index("flatten_non_causal_rows")
    expand_pos = src.index("expand_causal_rows")
    assert stamp_pos < flatten_pos < expand_pos


def test_producer_stamp_shares_the_backend_constant():
    """Single source of truth: the stamp patch imports the backend symbol."""

    from vllm_hcu.patch.worker.framework_opt import patch_dspark_non_causal_stamp

    assert patch_dspark_non_causal_stamp._STAMP == NON_CAUSAL_DRAFT_STAMP
    assert patch_dspark_non_causal_stamp.TARGET_MODULE == (
        "vllm.v1.worker.gpu.spec_decode.speculator"
    )


def test_decode_descales_v_with_the_layer_v_scale():
    """Review 2026-10-15 round 8 P3: the ported upstream call fed k's
    calibration to v_scale (wheel triton_mla.py:253-254 has the same
    substitution); ``_v_scale`` is part of the wheel's attention-layer
    contract (vllm/v1/attention/backend.py:752-753), so pass it through.
    Source-contract style: no GPU needed."""

    import inspect
    import re

    src = inspect.getsource(HcuTritonMLAImpl.forward_mqa)
    assert re.search(r"\bv_scale=layer\._v_scale\b", src), src[-400:]
    assert not re.search(r"\bv_scale=layer\._k_scale\b", src)
