# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP cache ownership contracts for decode graph padding."""

from types import SimpleNamespace

import torch

from vllm_hcu.model_executor.layers.attention import pcp


def _decode_only_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        pcp_world_size=1,
        num_decode_tokens=4,
        num_prefills=0,
        num_actual_tokens=4,
        pcp_has_global_prefill=False,
    )


def test_mla_decode_only_cache_inputs_drop_graph_padding(monkeypatch) -> None:
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: (_ for _ in ()).throw(AssertionError("must not gather")),
    )
    kv_c = torch.arange(10).reshape(5, 2)
    k_pe = torch.arange(15).reshape(5, 1, 3)
    slots = torch.arange(5)

    cache_kv_c, cache_k_pe, cache_slots = (
        pcp.maybe_gather_mla_latent_cache_inputs(
            kv_c,
            k_pe,
            slots,
            _decode_only_metadata(),
        )
    )

    torch.testing.assert_close(cache_kv_c, kv_c[:4])
    torch.testing.assert_close(cache_k_pe, k_pe[:4])
    torch.testing.assert_close(cache_slots, slots[:4])


def test_indexer_decode_only_cache_inputs_drop_graph_padding(monkeypatch) -> None:
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: (_ for _ in ()).throw(AssertionError("must not gather")),
    )
    indexer_k = torch.arange(15).reshape(5, 3)
    slots = torch.arange(5)

    cache_k, cache_slots = pcp.maybe_gather_indexer_k(
        indexer_k,
        slots,
        _decode_only_metadata(),
    )

    torch.testing.assert_close(cache_k, indexer_k[:4])
    torch.testing.assert_close(cache_slots, slots[:4])


def test_uniform_fallback_prefill_does_not_change_pcp_cache_ownership(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: (_ for _ in ()).throw(AssertionError("must not gather")),
    )
    kv_c = torch.arange(10).reshape(5, 2)
    k_pe = torch.arange(15).reshape(5, 1, 3)
    slots = torch.arange(5)
    metadata = SimpleNamespace(
        pcp_world_size=8,
        num_decodes=1,
        num_decode_tokens=4,
        num_prefills=1,
        num_actual_tokens=5,
        pcp_has_global_prefill=False,
    )

    cache_kv_c, cache_k_pe, cache_slots = (
        pcp.maybe_gather_mla_latent_cache_inputs(
            kv_c,
            k_pe,
            slots,
            metadata,
        )
    )

    torch.testing.assert_close(cache_kv_c, kv_c)
    torch.testing.assert_close(cache_k_pe, k_pe)
    torch.testing.assert_close(cache_slots, slots)
