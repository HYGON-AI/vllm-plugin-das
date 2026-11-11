# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unit tests for DSA cache layer-split (SGLang #29421 port)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.dsa_cache_layer_split import (
    LayerSplitState,
    broadcast_dsa_cache_from_owner,
    clear_broadcast_scratch_for_tests,
    effective_num_layers,
    get_layer_owner,
    get_layer_shard_range,
    get_runtime_layer_split_state,
    invalidate_dsa_cache_broadcast,
    is_dsa_cache_layer_split_requested,
    local_packed_slot_for_family_layer,
    local_packed_slot_for_layer,
    max_owned_count_for_layers,
    parse_layer_index,
    phys_slots_for_layer_family,
    remap_packed_kv_tensor_offsets,
    set_active_dsa_broadcast_block_ids,
    should_commit_durable_dsa_cache,
    validate_dsa_cache_layer_split,
)
import vllm_hcu.v1.dsa_cache_layer_split as layer_split_mod


def _set_layer_split(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Force the sidecar switch for tests whose configs carry no sidecar."""
    monkeypatch.setattr(
        layer_split_mod,
        "is_dsa_cache_layer_split_requested",
        lambda vllm_config=None: enabled,
    )


def test_requested_only_via_hcu_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HCU_ENABLE_DSA_CACHE_LAYER_SPLIT", "1")
    off = SimpleNamespace(additional_config={"hcu": {}})
    on = SimpleNamespace(
        additional_config={"hcu": {"enable_dsa_cache_layer_split": True}}
    )
    assert not is_dsa_cache_layer_split_requested(None)
    assert not is_dsa_cache_layer_split_requested(off)
    assert is_dsa_cache_layer_split_requested(on)


@pytest.mark.parametrize(
    ("total_layers", "shard_size"),
    [
        (61, 4),
        (61, 8),
        (48, 4),
        (78, 4),
        (8, 8),
        (7, 4),
        (1, 4),
    ],
)
def test_layer_shard_ranges_cover_exactly_once(
    total_layers: int, shard_size: int
) -> None:
    covered: list[int | None] = [None] * total_layers
    owned_counts: list[int] = []
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        assert 0 <= start <= end <= total_layers
        owned_counts.append(end - start)
        for idx in range(start, end):
            assert covered[idx] is None
            covered[idx] = rank
    assert all(owner is not None for owner in covered)
    # Contiguous even split: sizes differ by at most 1.
    assert max(owned_counts) - min(owned_counts) <= 1
    for local_idx, owner in enumerate(covered):
        assert get_layer_owner(local_idx, shard_size, total_layers) == owner


def test_effective_num_layers_matches_sglang() -> None:
    # ceil(61/8)+1 = 8+1 = 9
    assert effective_num_layers(61, 8) == 9
    # ceil(78/4)+1 = 20+1 = 21
    assert effective_num_layers(78, 4) == 21
    assert effective_num_layers(61, 1) == 61


def test_validate_requires_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        )
    )
    with pytest.raises(ValueError, match="prefill_context_parallel_size"):
        validate_dsa_cache_layer_split(cfg)


def test_validate_rejects_pp(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            pipeline_parallel_size=2,
        )
    )
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        validate_dsa_cache_layer_split(cfg)


def test_validate_rejects_pcp_greater_than_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty owned ranges cannot register Mooncake KV tensors."""
    _set_layer_split(monkeypatch, True)
    cfg = _pd_prefill_cfg()
    cfg.parallel_config.prefill_context_parallel_size = 8
    cfg.model_config.hf_config.num_hidden_layers = 3
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_dsa_cache_layer_split(cfg)


def _pd_prefill_cfg(
    architectures=("HYV4ForCausalLM",),
    use_mla=True,
    kv_role="kv_producer",
    index_topk=64,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            architectures=list(architectures),
            use_mla=use_mla,
            hf_config=SimpleNamespace(index_topk=index_topk),
        ),
        kv_transfer_config=(
            None if kv_role is None else SimpleNamespace(kv_role=kv_role)
        ),
        additional_config={"hcu": {"enable_dsa_cache_layer_split": True}},
    )


def test_validate_ok_with_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = _pd_prefill_cfg()
    validate_dsa_cache_layer_split(cfg)
    assert is_dsa_cache_layer_split_requested(cfg)


@pytest.mark.parametrize(
    "architecture",
    [
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "HYV4ForCausalLM",
    ],
)
def test_validate_accepts_registered_dsa_families(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    _set_layer_split(monkeypatch, True)
    validate_dsa_cache_layer_split(_pd_prefill_cfg((architecture,)))


def test_validate_rejects_dsa_without_index_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    with pytest.raises(ValueError, match="index_topk"):
        validate_dsa_cache_layer_split(_pd_prefill_cfg(index_topk=None))


def test_validate_rejects_ordinary_mla_even_with_index_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    with pytest.raises(ValueError, match="architecture"):
        validate_dsa_cache_layer_split(
            _pd_prefill_cfg(("LlamaForCausalLM",), use_mla=True)
        )


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"],
)
def test_validate_rejects_when_dsa_is_disabled(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    _set_layer_split(monkeypatch, True)
    monkeypatch.setenv("VLLM_HCU_DISABLE_DSA", "1")
    with pytest.raises(ValueError, match="VLLM_HCU_DISABLE_DSA"):
        validate_dsa_cache_layer_split(_pd_prefill_cfg((architecture,)))


def test_hyv4_ignores_dsa_env_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYV4 always builds its indexer; the DeepSeek-V2 opt-out must not apply."""
    _set_layer_split(monkeypatch, True)
    monkeypatch.setenv("VLLM_HCU_DISABLE_DSA", "1")
    validate_dsa_cache_layer_split(_pd_prefill_cfg(("HYV4ForCausalLM",)))


@pytest.mark.parametrize("kv_role", [None, "kv_consumer", "kv_both"])
def test_validate_requires_pd_prefill_worker(
    monkeypatch: pytest.MonkeyPatch, kv_role: str | None
) -> None:
    """SGLang model_hook: layer split only on PD prefill workers."""
    _set_layer_split(monkeypatch, True)
    with pytest.raises(ValueError, match="kv_producer"):
        validate_dsa_cache_layer_split(_pd_prefill_cfg(kv_role=kv_role))


def test_validate_rejects_non_dsa(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = _pd_prefill_cfg(architectures=("LlamaForCausalLM",), use_mla=False)
    with pytest.raises(ValueError, match="DSA MLA"):
        validate_dsa_cache_layer_split(cfg)


def test_parse_layer_index_from_prefix() -> None:
    assert parse_layer_index("model.layers.7.self_attn.attn") == 7
    assert (
        parse_layer_index("model.layers.41.self_attn.indexer.k_cache") == 41
    )


@pytest.mark.parametrize(
    ("rank", "layer_idx", "owned"),
    [
        (0, 0, True),
        (0, 15, True),
        (0, 16, False),
        (1, 16, True),
        (1, 30, True),
        (1, 31, False),
        (1, 15, False),
        (2, 31, True),
        (3, 60, True),
        (2, 60, False),
    ],
)
def test_should_commit_only_on_owner(
    monkeypatch: pytest.MonkeyPatch, rank: int, layer_idx: int, owned: bool
) -> None:
    _set_layer_split(monkeypatch, True)
    # 61 layers / 4 ranks -> [0,16), [16,31), [31,46), [46,61)
    layer_name = f"model.layers.{layer_idx}.self_attn.attn"
    assert (
        should_commit_durable_dsa_cache(
            layer_name,
            shard_rank=rank,
            shard_size=4,
            total_layers=61,
        )
        is owned
    )


def test_should_commit_true_when_feature_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Legacy PCP: every rank commits durable cache.
    _set_layer_split(monkeypatch, False)
    assert should_commit_durable_dsa_cache(
        "model.layers.0.self_attn.attn",
        shard_rank=1,
        shard_size=4,
        total_layers=61,
    )


def test_mtp_layers_remain_fully_replicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    # layer_idx >= total_layers is treated as replicated MTP.
    assert should_commit_durable_dsa_cache(
        "model.layers.61.self_attn.attn",
        shard_rank=2,
        shard_size=4,
        total_layers=61,
    )


def test_layer_split_state_ownership() -> None:
    state = LayerSplitState(shard_rank=1, shard_size=4, total_layers=61)
    assert state.is_owned(20)
    assert not state.is_owned(0)
    assert state.owner_of(0) == 0
    assert state.owner_of(20) == 1
    assert state.covers(60)
    assert not state.covers(61)


def test_runtime_state_none_when_pcp_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=61)
        ),
    )
    assert get_runtime_layer_split_state(cfg, shard_rank=0) is None


def test_broadcast_from_owner_fills_non_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()

    owner_cache = torch.arange(8, dtype=torch.float32)
    non_owner_cache = torch.zeros(8, dtype=torch.float32)

    class _Group:
        def __init__(self, rank: int) -> None:
            self.rank_in_group = rank
            self._payload: torch.Tensor | None = None

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            if self.rank_in_group == 0:
                self._payload = tensor.clone()
                _Group.last_payload = self._payload
            else:
                assert _Group.last_payload is not None
                tensor.copy_(_Group.last_payload)
            return tensor

    _Group.last_payload = None

    # Owner (rank 0) seeds the payload.
    broadcast_dsa_cache_from_owner(
        owner_cache,
        "model.layers.0.self_attn.attn",
        shard_rank=0,
        shard_size=4,
        total_layers=61,
        pcp_group=_Group(0),
    )
    # Ranks are separate processes; drop rank 0's remote_layer_id dedup state.
    clear_broadcast_scratch_for_tests()
    # Non-owner (rank 1) receives into its local buffer.
    broadcast_dsa_cache_from_owner(
        non_owner_cache,
        "model.layers.0.self_attn.attn",
        shard_rank=1,
        shard_size=4,
        total_layers=61,
        pcp_group=_Group(1),
    )
    assert torch.equal(non_owner_cache, owner_cache)


def test_remote_layer_id_skips_repeat_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGLang remote_layer_id: second get_broadcastable is a no-op collective."""
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()
    cache = torch.arange(8, dtype=torch.float32)
    calls = {"n": 0}

    class _Group:
        rank_in_group = 0

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            calls["n"] += 1
            return tensor

    layer = "model.layers.0.self_attn.attn"
    broadcast_dsa_cache_from_owner(
        cache, layer, shard_rank=0, shard_size=2, total_layers=8, pcp_group=_Group()
    )
    broadcast_dsa_cache_from_owner(
        cache, layer, shard_rank=0, shard_size=2, total_layers=8, pcp_group=_Group()
    )
    assert calls["n"] == 1
    # Write-path invalidate forces the next read to rebroadcast.
    invalidate_dsa_cache_broadcast(cache, layer)
    broadcast_dsa_cache_from_owner(
        cache, layer, shard_rank=0, shard_size=2, total_layers=8, pcp_group=_Group()
    )
    assert calls["n"] == 2


def test_sparse_selection_change_forces_same_layer_rebroadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new active block set cannot reuse the old layer-only dedup state."""
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()
    owner = torch.arange(8 * 2, dtype=torch.float32).view(8, 2)
    non_owner = torch.full_like(owner, -1.0)
    active = torch.tensor([1])
    calls = {"n": 0}

    class _Group:
        rank_in_group = 1

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            calls["n"] += 1
            tensor.copy_(owner.index_select(0, active))
            return tensor

    layer = "model.layers.0.self_attn.attn"
    set_active_dsa_broadcast_block_ids(active)
    broadcast_dsa_cache_from_owner(
        non_owner,
        layer,
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(),
    )
    assert torch.equal(non_owner[1], owner[1])

    active = torch.tensor([4])
    set_active_dsa_broadcast_block_ids(active)
    broadcast_dsa_cache_from_owner(
        non_owner,
        layer,
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(),
    )
    assert calls["n"] == 2
    assert torch.equal(non_owner[4], owner[4])


def test_sparse_selection_superset_broadcasts_only_new_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active-set superset incrementally synchronizes only added rows."""
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()
    owner = torch.arange(8 * 2, dtype=torch.float32).view(8, 2)
    non_owner = torch.full_like(owner, -1.0)
    active = torch.tensor([1, 2, 3])
    payload_sizes: list[int] = []

    class _Group:
        rank_in_group = 1

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            payload_sizes.append(int(tensor.shape[0]))
            tensor.copy_(owner.index_select(0, active[-tensor.shape[0] :]))
            return tensor

    layer = "model.layers.0.self_attn.attn"
    set_active_dsa_broadcast_block_ids(active)
    broadcast_dsa_cache_from_owner(
        non_owner,
        layer,
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(),
    )
    active = torch.tensor([1, 2, 3, 4])
    set_active_dsa_broadcast_block_ids(active)
    broadcast_dsa_cache_from_owner(
        non_owner,
        layer,
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(),
    )
    assert payload_sizes == [3, 1]
    assert torch.equal(non_owner[1:5], owner[1:5])
    clear_broadcast_scratch_for_tests()


def test_broadcast_noop_when_feature_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, False)
    cache = torch.ones(4)
    out = broadcast_dsa_cache_from_owner(
        cache,
        "model.layers.0.self_attn.attn",
        shard_rank=1,
        shard_size=4,
        total_layers=61,
    )
    assert out is cache
    assert torch.equal(cache, torch.ones(4))


def test_broadcast_noncontiguous_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()

    # Simulate a packed stride view: every other element.
    backing = torch.arange(16, dtype=torch.float32)
    owner_view = backing[::2]
    assert not owner_view.is_contiguous()
    non_owner_backing = torch.zeros(16, dtype=torch.float32)
    non_owner_view = non_owner_backing[::2]

    class _Group:
        last_payload: torch.Tensor | None = None

        def __init__(self, rank: int) -> None:
            self.rank_in_group = rank

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            assert tensor.is_contiguous()
            if self.rank_in_group == 0:
                _Group.last_payload = tensor.clone()
            else:
                assert _Group.last_payload is not None
                tensor.copy_(_Group.last_payload)
            return tensor

    broadcast_dsa_cache_from_owner(
        owner_view,
        "model.layers.0.self_attn.attn",
        shard_rank=0,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(0),
    )
    # Ranks are separate processes; drop rank 0's remote_layer_id dedup state.
    clear_broadcast_scratch_for_tests()
    broadcast_dsa_cache_from_owner(
        non_owner_view,
        "model.layers.0.self_attn.attn",
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(1),
    )
    assert torch.equal(non_owner_view, owner_view)


def test_sparse_broadcast_only_selected_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active block ids shrink the NCCL payload to used rows only."""
    _set_layer_split(monkeypatch, True)
    clear_broadcast_scratch_for_tests()

    # Pool of 8 blocks; only blocks 1 and 4 are live in the batch.
    owner = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    non_owner = torch.full((8, 4), -1.0)
    sizes: list[int] = []

    class _Group:
        last_payload: torch.Tensor | None = None

        def __init__(self, rank: int) -> None:
            self.rank_in_group = rank

        def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            del src
            sizes.append(int(tensor.numel()))
            assert tensor.is_contiguous()
            assert tensor.shape[0] == 2
            if self.rank_in_group == 0:
                _Group.last_payload = tensor.clone()
            else:
                assert _Group.last_payload is not None
                tensor.copy_(_Group.last_payload)
            return tensor

    set_active_dsa_broadcast_block_ids(torch.tensor([4, 1]))
    broadcast_dsa_cache_from_owner(
        owner,
        "model.layers.0.self_attn.attn",
        shard_rank=0,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(0),
    )
    clear_broadcast_scratch_for_tests()
    set_active_dsa_broadcast_block_ids(torch.tensor([4, 1]))
    broadcast_dsa_cache_from_owner(
        non_owner,
        "model.layers.0.self_attn.attn",
        shard_rank=1,
        shard_size=2,
        total_layers=8,
        pcp_group=_Group(1),
    )
    assert sizes == [8, 8]  # 2 blocks * 4 elems, not full pool (32)
    assert torch.equal(non_owner[1], owner[1])
    assert torch.equal(non_owner[4], owner[4])
    # Untouched rows stay as-is on the non-owner scratch.
    assert torch.all(non_owner[0] == -1.0)
    assert torch.all(non_owner[2] == -1.0)


def test_scratch_allocated_once_for_varying_payloads() -> None:
    """Per-step sparse payload sizes reuse one pool-sized scratch buffer."""
    clear_broadcast_scratch_for_tests()
    cache = torch.zeros(8, 4)
    first = layer_split_mod._scratch_for(cache, numel=2 * 4, shape=(2, 4))
    base_ptr = first.data_ptr()
    for n_rows in (5, 1, 7):
        payload = layer_split_mod._scratch_for(
            cache, numel=n_rows * 4, shape=(n_rows, 4)
        )
        assert payload.shape == (n_rows, 4)
        assert payload.data_ptr() == base_ptr
    full = layer_split_mod._scratch_for(cache)
    assert full.shape == cache.shape
    assert full.data_ptr() == base_ptr
    assert len(layer_split_mod._broadcast_scratch) == 1


def test_local_packed_slot_owned_vs_scratch() -> None:
    # 8 layers / CP=4 -> owned upper=2, slots=3 (2 owned + scratch)
    state = LayerSplitState(shard_rank=0, shard_size=4, total_layers=8)
    assert effective_num_layers(8, 4) == 3
    assert local_packed_slot_for_layer(state, 0) == 0
    assert local_packed_slot_for_layer(state, 1) == 1
    assert local_packed_slot_for_layer(state, 2) == 2  # scratch
    assert local_packed_slot_for_layer(state, 7) == 2


def test_phys_slots_mixed_indexer_and_mtp() -> None:
    # Hy4-like: sparse indexer on a subset + one MTP extra.
    total_layers = 8
    pcp = 4
    indexer = [0, 1, 5, 7]  # sparse backbone
    assert max_owned_count_for_layers(set(indexer), total_layers, pcp) == 2
    assert phys_slots_for_layer_family(indexer, total_layers, pcp) == 3  # +scratch
    assert phys_slots_for_layer_family(indexer + [8], total_layers, pcp) == 4  # +MTP

    state = LayerSplitState(shard_rank=0, shard_size=pcp, total_layers=total_layers)
    # Rank0 owns [0,2) -> indexer layers 0,1 -> dense 0,1; others scratch.
    assert local_packed_slot_for_family_layer(state, 0, indexer) == 0
    assert local_packed_slot_for_family_layer(state, 1, indexer) == 1
    assert local_packed_slot_for_family_layer(state, 5, indexer) == 2
    family_with_mtp = indexer + [8]
    assert local_packed_slot_for_family_layer(state, 8, family_with_mtp) == 3


def test_remap_packed_kv_tensor_offsets_aliases_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    page = 16
    total_layers = 8
    pcp = 4
    eff = effective_num_layers(total_layers, pcp)  # 3
    block_stride = page * eff
    num_blocks = 2
    size = block_stride * num_blocks

    class _Tensor:
        def __init__(self, size, shared_by, offset=0, block_stride=0):
            self.size = size
            self.shared_by = shared_by
            self.offset = offset
            self.block_stride = block_stride

    # Provisional clamp offsets (same as config patch).
    tensors = [
        _Tensor(
            size,
            [f"model.layers.{i}.self_attn.attn"],
            offset=min(i, eff - 1) * page,
            block_stride=block_stride,
        )
        for i in range(total_layers)
    ]
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=total_layers)
        ),
    )
    kv_cfg = SimpleNamespace(
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[t.shared_by[0] for t in tensors],
                kv_cache_spec=SimpleNamespace(page_size_bytes=page),
            )
        ],
    )

    assert remap_packed_kv_tensor_offsets(kv_cfg, cfg) is True
    # Rank 0 owns [0,2); scratch at slot 2.
    by_layer = {
        (getattr(t, "shared_by", None) or t.layers)[0]: t.offset
        for t in kv_cfg.kv_cache_tensors
    }
    assert by_layer["model.layers.0.self_attn.attn"] == 0
    assert by_layer["model.layers.1.self_attn.attn"] == page
    scratch = 2 * page
    for i in range(2, total_layers):
        assert by_layer[f"model.layers.{i}.self_attn.attn"] == scratch


class _RemapTensor:
    def __init__(self, size, shared_by, offset=0, block_stride=0):
        self.size = size
        self.shared_by = shared_by
        self.offset = offset
        self.block_stride = block_stride


def _remap_cfg(total_layers: int, pcp: int) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=total_layers)
        ),
    )


def test_remap_skips_no_shrink_when_phys_slots_exceed_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner keeps the native stride; worker must not invent owned+scratch."""
    _set_layer_split(monkeypatch, True)
    page, total_layers, pcp = 16, 1, 2
    # Native full layout: one slot. Owned+scratch would want two.
    assert phys_slots_for_layer_family([0], total_layers, pcp) == 2
    block_stride = page * total_layers
    tensors = [
        _RemapTensor(
            size=block_stride * 2,
            shared_by=["model.layers.0.self_attn.attn"],
            offset=0,
            block_stride=block_stride,
        )
    ]
    kv_cfg = SimpleNamespace(
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.self_attn.attn"],
                kv_cache_spec=SimpleNamespace(page_size_bytes=page),
            )
        ],
    )
    assert remap_packed_kv_tensor_offsets(kv_cfg, _remap_cfg(total_layers, pcp)) is False
    assert tensors[0].offset == 0
    assert tensors[0].block_stride == block_stride


def test_remap_raises_on_block_stride_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped remap leaves owned layers aliased onto one slot."""
    _set_layer_split(monkeypatch, True)
    page, total_layers, pcp = 16, 8, 4
    eff = effective_num_layers(total_layers, pcp)
    wrong_stride = page * (eff + 1)
    tensors = [
        _RemapTensor(
            wrong_stride * 2,
            [f"model.layers.{i}.self_attn.attn"],
            offset=min(i, eff - 1) * page,
            block_stride=wrong_stride,
        )
        for i in range(total_layers)
    ]
    kv_cfg = SimpleNamespace(
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[t.shared_by[0] for t in tensors],
                kv_cache_spec=SimpleNamespace(page_size_bytes=page),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="block_stride"):
        remap_packed_kv_tensor_offsets(kv_cfg, _remap_cfg(total_layers, pcp))


def test_remap_raises_when_page_size_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    tensors = [
        _RemapTensor(
            96, ["model.layers.0.self_attn.attn"], offset=0, block_stride=48
        )
    ]
    kv_cfg = SimpleNamespace(kv_cache_tensors=tensors, kv_cache_groups=[])
    with pytest.raises(RuntimeError, match="page sizes"):
        remap_packed_kv_tensor_offsets(kv_cfg, _remap_cfg(8, 4))


def test_make_packed_kv_tensor_v2_layers_api() -> None:
    from dataclasses import dataclass

    from vllm_hcu.v1.dsa_cache_layer_split import make_packed_kv_tensor

    @dataclass
    class _V2Tensor:
        size: int
        layers: list[str]
        layer_stride: int
        block_stride: int
        offset: int = 0

    t = make_packed_kv_tensor(
        _V2Tensor,
        size=100,
        layer_names=["model.layers.0.self_attn.attn"],
        page_size=16,
        offset=32,
        block_stride=48,
    )
    assert t.layers == ["model.layers.0.self_attn.attn"]
    assert t.layer_stride == 16
    assert t.offset == 32
    assert t.block_stride == 48


def test_remap_mixed_page_sizes_hy4_like(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLA (all layers+MTP) + sparse indexer families must both shrink."""
    _set_layer_split(monkeypatch, True)
    total_layers = 8
    pcp = 4
    mla_page = 100
    idx_page = 10
    mla_layers = list(range(total_layers)) + [8]  # +MTP
    indexer_layers = [0, 1, 5, 7]
    mla_phys = phys_slots_for_layer_family(mla_layers, total_layers, pcp)
    idx_phys = phys_slots_for_layer_family(indexer_layers, total_layers, pcp)
    assert mla_phys == effective_num_layers(total_layers, pcp) + 1  # +MTP
    assert idx_phys == 3
    block_stride = mla_page * mla_phys + idx_page * idx_phys
    size = block_stride * 2

    class _Tensor:
        def __init__(self, size, shared_by, offset=0, block_stride=0):
            self.size = size
            self.shared_by = shared_by
            self.offset = offset
            self.block_stride = block_stride

    tensors = []
    # Emit in sorted page-size order (idx_page then mla_page) like the planner.
    base = 0
    for i in indexer_layers:
        tensors.append(
            _Tensor(
                size,
                [f"model.layers.{i}.self_attn.indexer"],
                offset=base + min(indexer_layers.index(i), idx_phys - 1) * idx_page,
                block_stride=block_stride,
            )
        )
    base = idx_page * idx_phys
    for i in mla_layers:
        tensors.append(
            _Tensor(
                size,
                [f"model.layers.{i}.self_attn.attn"],
                offset=base + min(mla_layers.index(i), mla_phys - 1) * mla_page,
                block_stride=block_stride,
            )
        )

    def _spec_for(name: str):
        return SimpleNamespace(
            page_size_bytes=idx_page if "indexer" in name else mla_page
        )

    all_names = [t.shared_by[0] for t in tensors]
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=total_layers)
        ),
    )
    kv_cfg = SimpleNamespace(
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=all_names,
                kv_cache_spec=SimpleNamespace(
                    page_size_bytes=mla_page,
                    kv_cache_specs={n: _spec_for(n) for n in all_names},
                ),
            )
        ],
    )

    assert remap_packed_kv_tensor_offsets(kv_cfg, cfg) is True
    by_layer = {
        (getattr(t, "shared_by", None) or t.layers)[0]: t.offset
        for t in kv_cfg.kv_cache_tensors
    }
    # Indexer family at base 0; rank0 owns 0,1.
    assert by_layer["model.layers.0.self_attn.indexer"] == 0
    assert by_layer["model.layers.1.self_attn.indexer"] == idx_page
    assert by_layer["model.layers.5.self_attn.indexer"] == 2 * idx_page
    # MLA family after indexer phys region.
    mla_base = idx_page * idx_phys
    assert by_layer["model.layers.0.self_attn.attn"] == mla_base
    assert by_layer["model.layers.2.self_attn.attn"] == mla_base + 2 * mla_page  # scratch
    assert by_layer["model.layers.8.self_attn.attn"] == mla_base + 3 * mla_page  # MTP


def test_remap_accepts_row_aligned_block_stride(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker remap must pad the same way the planner does.

    Hy4 PCP8 startup failed when the planner wrote 562848 and remap still
    expected the unpadded 554496.
    """
    from vllm_hcu.v1.dsa_cache_layer_split import align_packed_block_stride

    _set_layer_split(monkeypatch, True)
    block = 64
    idx_page, mla_page = 8448, 41984
    indexer_layers = list(range(22))
    mla_layers = list(range(78)) + [78]
    total_layers, pcp = 78, 8
    idx_phys = phys_slots_for_layer_family(indexer_layers, total_layers, pcp)
    mla_phys = phys_slots_for_layer_family(mla_layers, total_layers, pcp)
    payload = idx_page * idx_phys + mla_page * mla_phys

    def _spec(page: int):
        return SimpleNamespace(page_size_bytes=page, block_size=block)

    names = [
        f"model.layers.{i}.self_attn.indexer" for i in indexer_layers
    ] + [f"model.layers.{i}.self_attn.attn" for i in mla_layers]
    groups = [
        SimpleNamespace(
            layer_names=names,
            kv_cache_spec=SimpleNamespace(
                page_size_bytes=mla_page,
                block_size=block,
                kv_cache_specs={
                    name: _spec(idx_page if "indexer" in name else mla_page)
                    for name in names
                },
            ),
        )
    ]
    stride = align_packed_block_stride(payload, groups)
    assert stride > payload
    assert align_packed_block_stride(
        554496,
        [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    kv_cache_specs={
                        "indexer": SimpleNamespace(
                            page_size_bytes=8448, block_size=64
                        ),
                        "mla": SimpleNamespace(
                            page_size_bytes=41984, block_size=64
                        ),
                    }
                )
            )
        ],
    ) == 562848

    class _Tensor:
        def __init__(self, size, shared_by, offset=0, block_stride=0):
            self.size = size
            self.shared_by = shared_by
            self.offset = offset
            self.block_stride = block_stride

    tensors = []
    for name in names:
        page = idx_page if "indexer" in name else mla_page
        tensors.append(_Tensor(stride, [name], offset=0, block_stride=stride))
        del page
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=total_layers)
        ),
    )
    kv_cfg = SimpleNamespace(kv_cache_tensors=tensors, kv_cache_groups=groups)
    assert remap_packed_kv_tensor_offsets(kv_cfg, cfg) is True
    assert kv_cfg.kv_cache_tensors[0].block_stride == stride


def test_admission_sizing_reserves_broadcast_bounce_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KV admission includes both persistent broadcast scratch pools."""
    import types

    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    total_layers, pcp, page = 8, 4, 16
    full_bytes = page * total_layers
    shrunk_bytes = page * phys_slots_for_layer_family(
        list(range(total_layers)), total_layers, pcp
    )
    bounce_bytes = 2 * page
    accounted_bytes = shrunk_bytes + bounce_bytes

    mod = types.ModuleType(kv_patch.TARGET_MODULE)

    def get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        raise AssertionError("unused")

    def _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups):
        return full_bytes * 1000

    def _pool_bytes_per_block(vllm_config, kv_cache_groups):
        return full_bytes

    mod.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    mod._max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups
    mod._pool_bytes_per_block = _pool_bytes_per_block
    mod.may_override_num_blocks = lambda cfg, n: n
    mod.KVCacheConfig = SimpleNamespace
    mod.KVCacheTensor = SimpleNamespace
    assert kv_patch.apply_to_module(mod) is True
    assert kv_patch.apply_to_module(mod) is False

    cfg = _remap_cfg(total_layers, pcp)
    groups = [
        SimpleNamespace(
            layer_names=[
                f"model.layers.{i}.self_attn.attn" for i in range(total_layers)
            ],
            kv_cache_spec=SimpleNamespace(page_size_bytes=page),
        )
    ]
    assert mod._pool_bytes_per_block(cfg, groups) == accounted_bytes
    assert mod._max_memory_usage_bytes_from_groups(cfg, groups) == (
        accounted_bytes * 1000
    )


def _planner_module(kv_patch, *, override=None, seen=None):
    import types

    mod = types.ModuleType(kv_patch.TARGET_MODULE)

    def get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        if seen is not None:
            seen.append(available_memory)
        return "original"

    mod.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    mod.may_override_num_blocks = lambda cfg, n: n if override is None else override
    mod.KVCacheConfig = SimpleNamespace
    mod.KVCacheTensor = SimpleNamespace
    assert kv_patch.apply_to_module(mod) is True
    return mod


def _single_family_groups(total_layers: int, page: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            layer_names=[
                f"model.layers.{i}.self_attn.attn" for i in range(total_layers)
            ],
            kv_cache_spec=SimpleNamespace(page_size_bytes=page),
        )
    ]


def test_no_shrink_fallback_still_reserves_bounce_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2 layers / PCP2 keeps every slot, but broadcasts still need scratch."""
    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    monkeypatch.setattr(kv_patch, "_layer_split_runtime_reserve_bytes", lambda: 0)
    total_layers, pcp, page = 2, 2, 16
    assert phys_slots_for_layer_family([0, 1], total_layers, pcp) == total_layers
    seen: list[int] = []
    mod = _planner_module(kv_patch, seen=seen)
    full_bytes = page * total_layers
    available = (full_bytes + 2 * page) * 10
    out = mod.get_kv_cache_config_from_groups(
        _remap_cfg(total_layers, pcp),
        _single_family_groups(total_layers, page),
        available,
    )
    assert out == "original"
    # Original planner sees only the share that fits packed KV + bounce.
    assert seen == [full_bytes * 10]


def test_bounce_budget_shares_scratch_across_same_dtype_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-dtype families share one scratch per pool; distinct dtypes add up."""
    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    monkeypatch.setattr(kv_patch, "_layer_split_runtime_reserve_bytes", lambda: 0)
    total_layers, pcp = 8, 4
    mla_page, idx_page = 100, 10

    def groups(idx_dtype):
        return [
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{i}.self_attn.attn"
                    for i in range(total_layers)
                ],
                kv_cache_spec=SimpleNamespace(
                    page_size_bytes=mla_page, dtype="uint8"
                ),
            ),
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{i}.self_attn.indexer"
                    for i in range(total_layers)
                ],
                kv_cache_spec=SimpleNamespace(
                    page_size_bytes=idx_page, dtype=idx_dtype
                ),
            ),
        ]

    shrunk = (mla_page + idx_page) * phys_slots_for_layer_family(
        list(range(total_layers)), total_layers, pcp
    )
    cfg = _remap_cfg(total_layers, pcp)
    # HY4 fp8: both views are uint8 -> one scratch per pool, sized to max.
    shared = shrunk + 2 * mla_page
    config = _planner_module(kv_patch).get_kv_cache_config_from_groups(
        cfg, groups("uint8"), shared * 7
    )
    assert config.num_blocks == 7
    # GLM bf16 indexer: distinct dtypes -> separate scratches, sum.
    separate = shrunk + 2 * (mla_page + idx_page)
    config = _planner_module(kv_patch).get_kv_cache_config_from_groups(
        cfg, groups("bfloat16"), separate * 7
    )
    assert config.num_blocks == 7
    # Unknown dtype falls back to the conservative standalone sum.
    config = _planner_module(kv_patch).get_kv_cache_config_from_groups(
        cfg, groups(None), separate * 7
    )
    assert config.num_blocks == 7


def test_block_override_cannot_exceed_bounce_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    monkeypatch.setattr(kv_patch, "_layer_split_runtime_reserve_bytes", lambda: 0)
    total_layers, pcp, page = 8, 4, 16
    shrunk_bytes = page * phys_slots_for_layer_family(
        list(range(total_layers)), total_layers, pcp
    )
    accounted = shrunk_bytes + 2 * page
    available = accounted * 10
    cfg = _remap_cfg(total_layers, pcp)
    groups = _single_family_groups(total_layers, page)

    within = _planner_module(kv_patch, override=10)
    config = within.get_kv_cache_config_from_groups(cfg, groups, available)
    assert config.num_blocks == 10

    over = _planner_module(kv_patch, override=11)
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        over.get_kv_cache_config_from_groups(cfg, groups, available)


def test_layer_split_runtime_reserve_reduces_num_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime peak headroom is deducted once in get_kv_cache_configs."""
    import types

    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    total_layers, pcp, page = 8, 4, 16
    shrunk_bytes = page * phys_slots_for_layer_family(
        list(range(total_layers)), total_layers, pcp
    )
    accounted = shrunk_bytes + 2 * page
    available = accounted * 20
    reserve = accounted * 5
    monkeypatch.setattr(
        kv_patch, "_layer_split_runtime_reserve_bytes", lambda: reserve
    )
    cfg = _remap_cfg(total_layers, pcp)
    layer_names = [
        f"model.layers.{i}.self_attn.attn" for i in range(total_layers)
    ]
    spec = {n: SimpleNamespace(page_size_bytes=page) for n in layer_names}

    # Minimal upstream get_kv_cache_configs that forwards one worker budget.
    mod = types.ModuleType(kv_patch.TARGET_MODULE)
    seen: list[int] = []

    def get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        raise AssertionError("from_groups unused in this test")

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        seen.extend(int(m) for m in available_memory)
        return [
            SimpleNamespace(num_blocks=int(m) // accounted, kv_cache_groups=[])
            for m in available_memory
        ]

    mod.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    mod.get_kv_cache_configs = get_kv_cache_configs
    mod.may_override_num_blocks = lambda c, n: n
    mod.KVCacheConfig = SimpleNamespace
    mod.KVCacheTensor = SimpleNamespace
    assert kv_patch.apply_to_module(mod) is True

    configs = mod.get_kv_cache_configs(cfg, [spec], [available, available])
    assert seen == [available - reserve, available - reserve]
    assert all(c.num_blocks == 15 for c in configs)


def test_make_packed_kv_tensor_modern_shared_by_fallback() -> None:
    from dataclasses import dataclass

    from vllm_hcu.v1.dsa_cache_layer_split import make_packed_kv_tensor

    @dataclass
    class _ModernTensor:
        size: int
        shared_by: list[str]

    t = make_packed_kv_tensor(
        _ModernTensor,
        size=128,
        layer_names=["model.layers.3.self_attn.attn"],
        page_size=32,
        offset=64,
        block_stride=96,
    )
    assert t.shared_by == ["model.layers.3.self_attn.attn"]
    assert t.layers == t.shared_by
    assert t.offset == 64
    assert t.block_stride == 96
    assert t.layer_stride == 32


def test_family_key_unifies_allocator_view_and_mla_squeeze() -> None:
    from vllm_hcu.v1.dsa_cache_layer_split import _family_key

    registered = torch.zeros(4, 1, 8, 16)
    bound = registered.squeeze(1)
    name = "model.layers.0.self_attn.attn"
    assert tuple(bound.shape) == (4, 8, 16)
    assert _family_key(registered, name) == _family_key(bound, name)
    other_layer = torch.zeros(4, 1, 8, 16)
    assert _family_key(other_layer, name) == _family_key(registered, name)
    indexer = "model.layers.0.self_attn.indexer"
    assert _family_key(registered, indexer) != _family_key(registered, name)


def test_validate_rejects_splittable_manager_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_layer_split(monkeypatch, True)
    cfg = _pd_prefill_cfg()
    cfg.cache_config = SimpleNamespace(block_size=64)
    validate_dsa_cache_layer_split(cfg)
    cfg.cache_config = SimpleNamespace(block_size=128)
    with pytest.raises(ValueError, match="kernel blocks"):
        validate_dsa_cache_layer_split(cfg)


def test_packed_stride_is_a_whole_number_of_bf16_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """61-layer PCP4 BF16 MLA + uint8 indexer, the review's remainder case."""
    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    monkeypatch.setattr(kv_patch, "_layer_split_runtime_reserve_bytes", lambda: 0)
    total_layers, pcp, block = 61, 4, 64
    mla_row, idx_row = 576 * 2, 132
    mla_page, idx_page = mla_row * block, idx_row * block
    groups = [
        SimpleNamespace(
            layer_names=[
                f"model.layers.{i}.self_attn.attn" for i in range(total_layers)
            ],
            kv_cache_spec=SimpleNamespace(
                kv_cache_specs={
                    **{
                        f"model.layers.{i}.self_attn.attn": SimpleNamespace(
                            page_size_bytes=mla_page,
                            block_size=block,
                            dtype=torch.bfloat16,
                        )
                        for i in range(total_layers)
                    },
                    **{
                        f"model.layers.{i}.self_attn.indexer": SimpleNamespace(
                            page_size_bytes=idx_page,
                            block_size=block,
                            dtype=torch.uint8,
                        )
                        for i in range(total_layers)
                    },
                }
            ),
        )
    ]
    mod = _planner_module(kv_patch)
    available = (mla_page + idx_page) * total_layers * 4
    plan = mod.get_kv_cache_config_from_groups(
        _remap_cfg(total_layers, pcp),
        groups,
        available,
    )
    stride = plan.kv_cache_tensors[0].block_stride
    assert stride % mla_row == 0
    assert stride % idx_row == 0
    assert (stride // 2) % 576 == 0


def test_max_usage_applies_layer_split_ratio_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native max-usage multiplies the wrapped pool; do not shrink again."""
    import types

    from vllm_hcu.patch.platform.framework_opt import (
        patch_dsa_layer_split_kv_config as kv_patch,
    )

    monkeypatch.setattr(kv_patch, "_layer_split_active", lambda cfg: True)
    total_layers, pcp, page, blocks = 8, 4, 512, 1000
    full_bytes = page * total_layers
    accounted = page * phys_slots_for_layer_family(
        list(range(total_layers)), total_layers, pcp
    ) + 2 * page

    mod = types.ModuleType(kv_patch.TARGET_MODULE)

    def get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        del vllm_config, kv_cache_groups, available_memory
        raise AssertionError("unused")

    def _pool_bytes_per_block(vllm_config, kv_cache_groups):
        del vllm_config, kv_cache_groups
        return full_bytes

    def _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups):
        return mod._pool_bytes_per_block(vllm_config, kv_cache_groups) * blocks

    mod.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    mod._max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups
    mod._pool_bytes_per_block = _pool_bytes_per_block
    mod.may_override_num_blocks = lambda cfg, n: n
    mod.KVCacheConfig = SimpleNamespace
    mod.KVCacheTensor = SimpleNamespace
    assert kv_patch.apply_to_module(mod) is True

    groups = _single_family_groups(total_layers, page)
    cfg = _remap_cfg(total_layers, pcp)
    assert mod._pool_bytes_per_block(cfg, groups) == accounted
    assert mod._max_memory_usage_bytes_from_groups(cfg, groups) == accounted * blocks


def test_native_positive_block_stride_uses_original_offload() -> None:
    import types

    from vllm_hcu.patch.platform.framework_opt import patch_packed_kv_offload

    calls: list[int] = []
    mod = types.ModuleType(patch_packed_kv_offload.TARGET_MODULE)

    class Scheduler:
        @staticmethod
        def _derive_cpu_config(gpu_config, cpu_capacity_bytes):
            del gpu_config
            calls.append(cpu_capacity_bytes)
            return "native"

    mod.SimpleCPUOffloadScheduler = Scheduler
    assert patch_packed_kv_offload.apply_to_module(mod) is True
    gpu = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[
            SimpleNamespace(size=64, block_stride=32, layers=["layer.0"]),
            SimpleNamespace(size=64, block_stride=32, layers=["layer.1"]),
        ],
    )
    assert (
        mod.SimpleCPUOffloadScheduler._derive_cpu_config(gpu, 128) == "native"
    )
    assert calls == [128]
