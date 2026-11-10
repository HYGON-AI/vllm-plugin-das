# SPDX-License-Identifier: Apache-2.0
"""Real CPU PCP layout contracts for token-aligned multimodal inputs."""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import make_dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def pcp_helpers(monkeypatch):
    """Reuse manager fixtures; isolate only unavailable vLLM import boundaries."""
    root = Path(__file__).resolve().parents[3]
    if importlib.util.find_spec("vllm") is not None:
        from tests.runtime_patch import test_glm52_pcp_manager as helpers

        yield helpers
        return

    fixture_path = root / "tests/runtime_patch/test_glm52_pcp_manager.py"
    tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
    # The fixture's actual InputBatch constructor defines its CPU data carrier.
    # Production layout/partition/gather math is loaded unchanged below.
    batch_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InputBatch"
    )
    batch_type = make_dataclass(
        "InputBatch", [(kw.arg, object) for kw in batch_call.keywords]
    )

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module(
        "vllm.distributed.parallel_state",
        get_dcp_group=lambda: None,
        get_pcp_group=lambda: None,
    )
    module(
        "vllm.v1.attention.backends.utils",
        PAD_SLOT_ID=-1,
        get_dcp_local_seq_lens=lambda *args: None,
    )
    module("vllm.v1.worker.gpu.input_batch", InputBatch=batch_type)
    module(
        "vllm_hcu.patch.platform.core_fix._common",
        PatchCompatibilityError=type("PatchCompatibilityError", (RuntimeError,), {}),
    )
    module(
        "vllm_hcu.patch.platform.core_fix.patch_vllm_config",
        _validate_hcu_pcp_scope=lambda config: True,
    )
    name = "_pcp_multimodal_cpu_manager"
    spec = importlib.util.spec_from_file_location(
        name, root / "vllm_hcu/v1/pcp_manager.py"
    )
    manager_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, manager_module)
    spec.loader.exec_module(manager_module)
    import numpy as np

    namespace = {
        "np": np,
        "torch": torch,
        "SimpleNamespace": SimpleNamespace,
        "InputBatch": batch_type,
        "HcuPCPManager": manager_module.HcuPCPManager,
        "HcuFeatureConfig": lambda: SimpleNamespace(to_dict=dict),
    }
    names = {"_InMemoryPCPGroup", "_make_config", "_make_batch", "_make_managers"}
    helpers_tree = ast.Module(
        body=[node for node in tree.body if getattr(node, "name", None) in names],
        type_ignores=[],
    )
    exec(compile(helpers_tree, str(fixture_path), "exec"), namespace)  # noqa: S102
    yield SimpleNamespace(**namespace)


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("token_dim", [0, 1])
def test_pcp16_gather_restores_each_token_and_zeros_padding(
    pcp_helpers, length, token_dim
):
    """Wrong rank slices, second chunks or padding masks corrupt exact restoration."""
    h = pcp_helpers
    batch = h._make_batch([("image", list(range(1, length + 1)), length + 19, True)])
    canonical = batch.positions.clone()
    source = torch.arange(1, length * 3 + 1).reshape(length, 3)
    if token_dim == 1:
        source = source.T
    managers, groups = h._make_managers(16)
    gathered = []
    for manager in managers:
        local = manager.partition_batch(batch)
        local_positions = local.positions.clone()
        width = local.num_tokens_after_padding
        storage = torch.full((width + 2, 3) if token_dim == 0 else (3, width + 2), -123)
        out = storage[:width] if token_dim == 0 else storage[:, :width]
        ptr = out.data_ptr()
        assert manager.get_global_batch(local) is batch
        indices, valid = manager.get_local_token_mapping(local)
        assert indices.shape == valid.shape == (width,)
        assert indices.dtype == torch.int64 and valid.dtype == torch.bool
        assert torch.equal(valid, ~local.is_padding)
        assert (
            manager.gather_global_token_tensor(
                source, local, token_dim=token_dim, out=out
            )
            is out
        )
        assert out.data_ptr() == ptr
        token_major = out if token_dim == 0 else out.T
        # Expected token identity comes from partitioned IDs, not the new map.
        expected = (local.input_ids.to(torch.int64) - 1).unsqueeze(
            1
        ) * 3 + torch.tensor([1, 2, 3])
        expected[local.is_padding] = 0
        assert torch.equal(token_major, expected)
        assert torch.equal(local.positions, local_positions)
        gathered.append(token_major.contiguous())
    for manager, group, values in zip(managers, groups, gathered):
        group.gathered = gathered
        assert torch.equal(
            manager.restore_hidden_states(values),
            source if token_dim == 0 else source.T,
        )
    assert torch.equal(batch.positions, canonical)


def test_mixed_reordered_requests_preserve_nonowner_decode_and_chunk_offsets(
    pcp_helpers,
):
    h = pcp_helpers
    requests = [
        ("fresh", list(range(1, 66)), 65, True),
        ("continued", list(range(66, 99)), 134, True),
        ("decode", [99], 17, False),
    ]
    batch = h._make_batch(requests)
    managers, _ = h._make_managers(16)
    source = torch.stack(
        (
            batch.positions[:99] + 100,
            batch.positions[:99] + 200,
            batch.positions[:99] + 300,
        )
    )
    for manager in managers:
        local = manager.partition_batch(batch)
        out = torch.empty((3, local.num_tokens_after_padding), dtype=source.dtype)
        manager.gather_global_token_tensor(source, local, token_dim=1, out=out)
        valid = ~local.is_padding
        expected = source[:, local.input_ids[valid].long() - 1]
        assert torch.equal(out[:, valid], expected)
        decode = local.input_ids == 99
        assert torch.equal(out[:, decode], torch.tensor([[116], [216], [316]]))
        if manager.pcp_rank > 0:
            start = manager.pcp_rank * manager._padded_num_tokens
            assert not manager._gathered_kv_write_mask[start : start + len(decode)][
                decode
            ].any()
        assert torch.count_nonzero(out[:, local.is_padding]) == 0
    # Request order changes next step; no previous layout may leak.
    second = h._make_batch(
        [("decode", [1], 18, False), ("other", list(range(2, 19)), 57, True)]
    )
    local = managers[5].partition_batch(second)
    out = torch.empty((local.num_tokens_after_padding, 1))
    managers[5].gather_global_token_tensor(
        second.input_ids[:18, None].float(), local, token_dim=0, out=out
    )
    assert torch.equal(out[:, 0], local.input_ids.float())


def test_empty_layout_and_collective_tail_are_zero(pcp_helpers):
    h = pcp_helpers
    for use_mla in [True, False]:
        batch = h._make_batch([("empty", [], 0, True)])
        manager = h._make_managers(16)[0][15]
        manager._use_mla = use_mla
        local = manager.partition_batch(batch)
        out = torch.empty((0, 3))
        assert (
            manager.gather_global_token_tensor(
                torch.empty((0, 3)), local, token_dim=0, out=out
            )
            is out
        )
        batch = h._make_batch([("short", [1], 1, True)])
        local = manager.partition_batch(batch)
        out = torch.full((local.num_tokens_after_padding, 3), -1.0)
        manager.gather_global_token_tensor(
            torch.ones((1, 3)), local, token_dim=0, out=out
        )
        assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize(
    "api", ["get_global_batch", "get_local_token_mapping", "gather_global_token_tensor"]
)
def test_stale_or_forged_batch_is_rejected(pcp_helpers, api):
    h = pcp_helpers
    manager = h._make_managers(16)[0][0]
    batch = h._make_batch([("a", list(range(1, 34)), 33, True)])
    first = manager.partition_batch(batch)
    manager.partition_batch(h._make_batch([("b", [1], 1, True)]))
    for stale in (first, replace(manager._local_batch)):
        with pytest.raises(ValueError, match="current"):
            if api == "gather_global_token_tensor":
                getattr(manager, api)(
                    torch.ones((33, 3)),
                    stale,
                    token_dim=0,
                    out=torch.empty((stale.num_tokens_after_padding, 3)),
                )
            else:
                getattr(manager, api)(stale)


@pytest.mark.parametrize(
    "bad",
    [
        "source_short",
        "output_shape",
        "dtype",
        "device",
        "dimension",
        "alias",
        "canonical_alias",
        "mapping_dtype",
        "mapping_shape",
        "padding_dtype",
        "padding_shape",
        "local_width",
    ],
)
def test_invalid_gather_contract_is_rejected_before_writing(pcp_helpers, bad):
    h = pcp_helpers
    batch = h._make_batch([("a", list(range(1, 34)), 33, True)])
    manager = h._make_managers(16)[0][0]
    local = manager.partition_batch(batch)
    source = torch.ones((33, 3))
    out = torch.empty((local.num_tokens_after_padding, 3))
    dim = 0
    if bad == "source_short":
        source = source[:1]
    elif bad == "output_shape":
        out = out[:1]
    elif bad == "dtype":
        out = out.long()
    elif bad == "device":
        out = torch.empty(out.shape, device="meta")
    elif bad == "dimension":
        dim = 2
    elif bad == "alias":
        out = source[: len(out)]
    elif bad == "canonical_alias":
        source = torch.ones(33, dtype=torch.int64)
        out = local.positions
    elif bad == "mapping_dtype":
        manager._padded_gather_idx = manager._padded_gather_idx.int()
    elif bad == "mapping_shape":
        manager._padded_gather_idx = manager._padded_gather_idx[:-1]
    elif bad == "padding_dtype":
        local.is_padding = local.is_padding.int()
    elif bad == "padding_shape":
        local.is_padding = local.is_padding[:-1]
    elif bad == "local_width":
        local.num_tokens_after_padding += 1
    with pytest.raises((ValueError, TypeError)):
        manager.gather_global_token_tensor(source, local, token_dim=dim, out=out)


def test_source_capacity_and_noncontiguous_output_need_no_scalar_sync(
    pcp_helpers, monkeypatch
):
    h = pcp_helpers
    batch = h._make_batch([("a", list(range(1, 34)), 33, True)], pad_to=40)
    manager = h._make_managers(16)[0][7]
    local = manager.partition_batch(batch)
    source = torch.arange(3 * 40).reshape(3, 40)
    out = torch.empty((3, local.num_tokens_after_padding + 1), dtype=source.dtype)[
        :, :-1
    ]
    expected = source[:, local.input_ids[~local.is_padding].long() - 1].clone()

    def scalar_sync_forbidden(*args, **kwargs):
        raise AssertionError("gather validation synchronized a device scalar")

    monkeypatch.setattr(torch.Tensor, "item", scalar_sync_forbidden)
    manager.gather_global_token_tensor(source, local, token_dim=1, out=out)
    assert torch.equal(out[:, ~local.is_padding], expected)


def test_failed_partition_cannot_expose_an_unprepared_local_batch(pcp_helpers):
    h = pcp_helpers
    manager = h._make_managers(16)[0][0]
    batch = h._make_batch([("a", list(range(4097)), 4097, True)])
    with pytest.raises(RuntimeError, match="buffer"):
        manager.partition_batch(batch)
    with pytest.raises(ValueError, match="current"):
        manager.get_global_batch(None)


def test_unknown_overlapping_output_is_rejected_before_writing(pcp_helpers):
    h = pcp_helpers
    manager = h._make_managers(16)[0][0]
    batch = h._make_batch([("a", list(range(1, 66)), 65, True)])
    local = manager.partition_batch(batch)
    source = torch.arange(65 * 3).reshape(65, 3)
    backing = torch.full((local.num_tokens_after_padding + 2,), -77, dtype=source.dtype)
    out = backing.as_strided((local.num_tokens_after_padding, 3), (1, 1))
    assert torch._debug_has_internal_overlap(out) == 2
    before = backing.clone()
    with pytest.raises(ValueError, match="overlap"):
        manager.gather_global_token_tensor(source, local, token_dim=0, out=out)
    assert torch.equal(backing, before)


@pytest.mark.parametrize(
    "layout", ["rotary_capacity_gap", "transposed", "strided_columns"]
)
def test_safe_noncontiguous_output_preserves_gather_values(pcp_helpers, layout):
    h = pcp_helpers
    manager = h._make_managers(16)[0][0]
    batch = h._make_batch([("a", list(range(1, 66)), 65, True)])
    local = manager.partition_batch(batch)
    width = local.num_tokens_after_padding
    source = torch.arange(3 * 65).reshape(3, 65)
    if layout == "rotary_capacity_gap":
        out = torch.empty((3, 66), dtype=source.dtype)[:, :width]
    elif layout == "transposed":
        out = torch.empty((width, 3), dtype=source.dtype).T
    else:
        out = torch.empty((3, 2 * width), dtype=source.dtype)[:, ::2]
    assert not out.is_contiguous()
    if layout != "transposed":
        assert torch._debug_has_internal_overlap(out) == 2
    manager.gather_global_token_tensor(source, local, token_dim=1, out=out)
    valid = ~local.is_padding
    assert torch.equal(out[:, valid], source[:, local.input_ids[valid].long() - 1])
    assert torch.count_nonzero(out[:, local.is_padding]) == 0
