# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real Core033 view/bind methods + real Torch CPU replacement tensors.

No GPU constructors, allocators, streams, native operators or model instances.
The runtime/group are actual Core types constructed via __new__: only the CPU
fields consumed by this HOT-ONLY slice are initialized. This is not full Core
allocation/binding, host registration, DFly slot wiring or joint qualification.
"""
from __future__ import annotations

import ast
from importlib import import_module, util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "adapter_extension/src/hcu_vllm_das/hisparse_owner_contract.py"
spec = util.spec_from_file_location("_hisparse_owner_contract_slice", MODULE)
provider_module = util.module_from_spec(spec)
sys.modules[spec.name] = provider_module
spec.loader.exec_module(provider_module)
Error = provider_module.OwnerContractError
STATE = ("lru_init", "device_global_indices", "lru_slots", "swap_stats")


@pytest.fixture(scope="module")
def core():
    return import_module("vllm.v1.hisparse.runtime")


@pytest.fixture
def provider(core, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU owner contract must not initialize an accelerator")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    assert not torch.cuda.is_initialized()
    yield provider_module.get_owner_provider()
    assert not torch.cuda.is_initialized()


def runtime(core, stride=4096, union=128, requests=2, width=8):
    obj = core.HiSparseRuntime.__new__(core.HiSparseRuntime)
    obj.max_num_reqs, obj.region_stride = requests, stride
    obj.max_union_rows, obj.row_width = union, width
    obj.kv_dtype, obj.device = torch.float16, torch.device("cpu")
    obj.is_group_leader, obj._swap_staged = True, False
    obj._layer_ready_event, obj._swap_step = None, 0
    group = core.HiSparseIndexGroup.__new__(core.HiSparseIndexGroup)
    group.leader, group.followers = obj, []
    group.lru_init = torch.arange(stride, dtype=torch.int16)
    group.device_global_indices = torch.full((requests, stride), -1, dtype=torch.int32)
    group.lru_slots = group.lru_init.repeat(requests, 1).contiguous()
    group.swap_stats = torch.zeros(2, dtype=torch.uint64)
    group.shared_topk = core._create_shared_topk_state(obj.device, requests, 128)
    group.copy_stream = None  # No fake stream, nor use of Core's GPU reset on CPU.
    obj.index_group = group
    return obj


def slab(owner, *, offset=64, padding=2, block_size=16):
    row_bytes = owner.row_width * owner.kv_dtype.itemsize
    block_stride = (1 + padding) * block_size * row_bytes
    blocks = (owner.region_stride + block_size - 1) // block_size + 1
    # Extra bytes remain block-aligned after byte_offset for Core's flat view.
    raw = torch.zeros(offset + blocks * block_stride, dtype=torch.uint8)
    table = torch.arange(blocks - 1, dtype=torch.int32).repeat(owner.max_num_reqs, 1)
    return raw, dict(byte_offset=offset, block_stride=block_stride,
                     num_blocks=blocks, block_size=block_size, block_table=table)


def bind(provider, owner, **kwargs):
    raw, layout = slab(owner, **kwargs)
    provider.bind_owner(owner, raw, **layout, quiescent=True)
    return raw, layout


def state(obj):
    return tuple(getattr(obj.index_group, name) for name in STATE)


def assert_shared(owner, reader):
    assert reader.region_stride == owner.region_stride
    assert all(a is b for a, b in zip(state(owner), state(reader)))
    assert reader.index_group is not owner.index_group
    assert reader.index_group.shared_topk is not owner.index_group.shared_topk
    assert reader.hot_backing is owner.hot_backing
    assert reader.hot_block_table is owner.hot_block_table
    assert reader.hot.cache.untyped_storage().data_ptr() == owner.hot.cache.untyped_storage().data_ptr()
    assert reader.hot.cache.storage_offset() == owner.hot.cache.storage_offset()
    assert reader.hot.cache.stride() == owner.hot.cache.stride()
    assert reader.hot.attention_block_stride == owner.hot.attention_block_stride
    assert reader.index_group.lru_init.dtype == torch.int16
    assert reader.index_group.lru_slots.dtype == torch.int16


@pytest.mark.parametrize("capacity,union,accepted", [
    (4096, 2176, True), (2048, 2176, False), (2176, 2176, True),
    (4096, 4096, True), (4096, 4097, False), (32768, 32768, True),
])
def test_real_torch_capacity_and_view(core, provider, capacity, union, accepted):
    owner, reader = runtime(core, capacity), runtime(core, max(union, 128), union)
    raw, layout = bind(provider, owner)
    own_state, own_stride = state(reader), reader.region_stride
    if not accepted:
        with pytest.raises(ValueError, match=f"max_union_rows={union}.*device_buffer_size={capacity}"):
            provider.bind_reader(reader, owner=owner, quiescent=True)
        assert not hasattr(reader, "hot")
        assert reader.region_stride == own_stride
        assert all(a is b for a, b in zip(state(reader), own_state))
        return
    provider.bind_reader(reader, owner=owner, quiescent=True)
    assert_shared(owner, reader)
    assert reader.hot.cache.storage_offset() == layout["byte_offset"] // 2
    assert reader.hot.cache.stride() == (layout["block_stride"] // 2, 8, 1)
    reader.hot.cache[1, 2, 3] = 19
    assert owner.hot.cache[1, 2, 3].item() == 19
    assert raw.dtype == torch.uint8


def test_original_core_rejects_owner_keyword(core):
    obj = runtime(core)
    raw, layout = slab(obj)
    with pytest.raises(TypeError, match="owner"):
        obj.bind_hot_cache(raw, **layout, owner=obj)
    assert not hasattr(obj, "hot")


def test_multireader_request_isolation_reset_and_idempotence(core, provider):
    owner, other = runtime(core), runtime(core)
    readers = [runtime(core, 2176, 2176), runtime(core, 1024, 1024)]
    raw, layout = bind(provider, owner)
    bind(provider, other)
    for reader in readers:
        provider.bind_reader(reader, owner=owner, quiescent=True)
        assert_shared(owner, reader)
    readers[0].index_group.device_global_indices[0, 3] = 71
    readers[0].index_group.lru_slots[0].copy_(torch.arange(4095, -1, -1, dtype=torch.int16))
    owner.index_group.swap_stats.fill_(9)
    assert readers[1].index_group.device_global_indices[0, 3].item() == 71
    assert owner.index_group.device_global_indices[1, 3].item() == -1
    assert other.index_group.device_global_indices[0, 3].item() == -1
    for a, b in zip(state(owner), state(other)):
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    provider.bind_owner(owner, raw, **layout, quiescent=True)
    provider.bind_reader(readers[0], owner=owner, quiescent=True)
    assert owner.index_group.device_global_indices[0, 3].item() == 71
    assert owner.index_group.swap_stats.tolist() == [9, 9]
    identities = state(owner)
    provider.reset(readers[1], quiescent=True)
    for member in (owner, *readers):
        assert all(a is b for a, b in zip(state(member), identities))
        assert member.index_group.device_global_indices.eq(-1).all().item()
        assert member.index_group.swap_stats.tolist() == [0, 0]
        torch.testing.assert_close(member.index_group.lru_slots,
                                   torch.arange(4096, dtype=torch.int16).repeat(2, 1))
    assert other.index_group.swap_stats.tolist() == [0, 0]


def test_detach_restores_private_state_and_allows_new_owner(core, provider):
    first, second, reader = runtime(core), runtime(core, 8192), runtime(core, 2176, 2176)
    initial = state(reader)
    bind(provider, first)
    bind(provider, second)
    provider.bind_reader(reader, owner=first, quiescent=True)
    with pytest.raises(Error, match="detach all readers"):
        provider.unbind_owner(first, quiescent=True)
    with pytest.raises(Error, match="already bound"):
        provider.bind_reader(reader, owner=second, quiescent=True)
    provider.unbind_reader(reader, quiescent=True)
    assert not hasattr(reader, "hot")
    assert reader.region_stride == 2176
    assert all(a is b for a, b in zip(state(reader), initial))
    provider.bind_reader(reader, owner=second, quiescent=True)
    assert_shared(second, reader)
    reader.index_group.device_global_indices[0, 0] = 11
    assert first.index_group.device_global_indices[0, 0].item() == -1
    provider.unbind_reader(reader, quiescent=True)
    provider.unbind_owner(first, quiescent=True)
    provider.unbind_owner(second, quiescent=True)
    assert not hasattr(first, "hot_backing")
    bind(provider, first)
    provider.bind_reader(reader, owner=first, quiescent=True)
    assert_shared(first, reader)


def test_atomic_rebind_all_members_invalidates_old_cache(core, provider):
    owner, reader = runtime(core), runtime(core, 2176, 2176)
    old, layout = bind(provider, owner)
    provider.bind_reader(reader, owner=owner, quiescent=True)
    owner.index_group.device_global_indices[0, 0] = 7
    replacement = torch.zeros_like(old)
    table = layout["block_table"].clone()
    provider.rebind_owner(owner, replacement, block_table=table, quiescent=True)
    assert_shared(owner, reader)
    assert reader.hot_backing is replacement
    assert reader.hot_block_table is table
    assert reader.index_group.device_global_indices.eq(-1).all().item()
    owner.hot.cache[0, 0, 0] = 5
    assert old.eq(0).all().item()
    assert reader.hot.cache[0, 0, 0].item() == 5
    owner.index_group.device_global_indices[0, 0] = 8
    provider.rebind_owner(owner, replacement, block_table=table, quiescent=True)
    assert reader.index_group.device_global_indices[0, 0].item() == 8
    with pytest.raises(Error, match="resize"):
        provider.rebind_owner(owner, replacement[:-16], block_table=table, quiescent=True)
    assert reader.hot_backing is replacement
    with pytest.raises(Error, match="only the direct owner"):
        provider.rebind_owner(reader, replacement, block_table=table, quiescent=True)


@pytest.mark.parametrize("bad", ["followers", "nonleader", "staged", "dtype", "template", "capacity"])
def test_fail_closed_group_abi(core, provider, bad):
    obj = runtime(core)
    if bad == "followers":
        obj.index_group.followers.append(object())
    elif bad == "nonleader":
        obj.is_group_leader = False
    elif bad == "staged":
        obj._swap_staged = True
    elif bad == "dtype":
        obj.index_group.lru_slots = obj.index_group.lru_slots.int()
    elif bad == "template":
        obj.index_group.lru_init[0] = 1
    else:
        obj.region_stride = 32769
    raw, layout = slab(obj)
    with pytest.raises(Error):
        provider.bind_owner(obj, raw, **layout, quiescent=True)
    assert not hasattr(obj, "hot")


@pytest.mark.parametrize("field,value", [("row_width", 16), ("max_num_reqs", 3),
                                         ("kv_dtype", torch.bfloat16)])
def test_reject_incompatible_reader(core, provider, field, value):
    owner = runtime(core)
    reader = runtime(core, requests=3 if field == "max_num_reqs" else 2)
    setattr(reader, field, value)
    bind(provider, owner)
    with pytest.raises(Error, match="mismatch"):
        provider.bind_reader(reader, owner=owner, quiescent=True)
    assert not hasattr(reader, "hot")


@pytest.mark.parametrize("case", ["unaligned", "offset", "table_dtype", "table_capacity", "truncated", "raw_view"])
def test_reject_invalid_slab_before_binding(core, provider, case):
    obj = runtime(core)
    raw, layout = slab(obj)
    if case == "unaligned":
        layout["block_stride"] += 2
    elif case == "offset":
        layout["byte_offset"] = 1
    elif case == "table_dtype":
        layout["block_table"] = layout["block_table"].long()
    elif case == "table_capacity":
        layout["block_table"] = layout["block_table"][:, :1].contiguous()
    elif case == "truncated":
        raw = raw[:32]
    else:
        raw = raw[16:]
    with pytest.raises((Error, RuntimeError)):
        provider.bind_owner(obj, raw, **layout, quiescent=True)
    assert not hasattr(obj, "hot")


def test_quiescence_unbound_chain_and_takeover_rejected(core, provider):
    owner, reader = runtime(core), runtime(core)
    raw, layout = slab(owner)
    with pytest.raises(Error, match="quiescence"):
        provider.bind_owner(owner, raw, **layout)
    with pytest.raises(Error, match="before its readers"):
        provider.bind_reader(reader, owner=owner, quiescent=True)
    bind(provider, owner)
    with pytest.raises(Error, match="cannot own itself"):
        provider.bind_reader(owner, owner=owner, quiescent=True)
    provider.bind_reader(reader, owner=owner, quiescent=True)
    with pytest.raises(Error, match="before its readers"):
        provider.bind_reader(runtime(core), owner=reader, quiescent=True)
    owner._swap_staged = True
    with pytest.raises(Error, match="staged"):
        provider.reset(reader, quiescent=True)
    owner._swap_staged = False
    # An independent provider must not silently take over bound Core objects.
    with pytest.raises(Error, match="already bound"):
        provider_module.get_owner_provider().bind_owner(owner, raw, **layout, quiescent=True)


def test_fork_and_binding_drift_fail_closed(core, provider, monkeypatch):
    owner, reader = runtime(core), runtime(core)
    raw, layout = bind(provider, owner)
    provider.bind_reader(reader, owner=owner, quiescent=True)
    with monkeypatch.context() as patch:
        patch.setattr(provider_module.os, "getpid", lambda: provider._pid + 1)
        with pytest.raises(Error, match="fork"):
            provider.reset(owner, quiescent=True)
    reader.hot_block_table = layout["block_table"].clone()
    with pytest.raises(Error, match="mutation"):
        provider.reset(owner, quiescent=True)
    assert owner.hot_backing is raw


def test_native_owner_abi_not_silently_accepted(core):
    class NewRuntime:
        def bind_hot_cache(self, raw_tensor, *, byte_offset, block_stride,
                           num_blocks, block_size, block_table, owner=None):
            pass
    with pytest.raises(Error, match="exact Core033"):
        provider_module.Core033OwnerProvider(NS(HiSparseRuntime=NewRuntime), torch)


def test_import_is_inert():
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('inert_owner', {str(MODULE)!r}); "
            "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; "
            "s.loader.exec_module(m); "
            "assert not any(n == 'torch' or n.startswith(('vllm', 'hcu_adapter')) "
            "for n in sys.modules)")
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("method", ["dfly", "dflash"])
def test_existing_joint_admission_remains_closed(method):
    path = ROOT / "vllm_hcu/models/hy_v4_flash/hisparse.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "validate_flash_hisparse")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    cfg = NS(kv_transfer_config=NS(kv_connector="HiSparseConnector"),
             attention_config=NS(hisparse_config=NS()),
             speculative_config=NS(method=method), model_config=NS(enforce_eager=True))
    with pytest.raises(NotImplementedError, match="shared-hot-cache ABI"):
        namespace["validate_flash_hisparse"](cfg)


def test_new_owner_bind_invalidates_old_residency_after_unbind(core, provider):
    owner = runtime(core)
    bind(provider, owner)
    owner.index_group.device_global_indices[0, 0] = 55
    owner.index_group.swap_stats.fill_(9)
    provider.unbind_owner(owner, quiescent=True)
    bind(provider, owner)
    assert owner.index_group.device_global_indices.eq(-1).all().item()
    assert owner.index_group.swap_stats.tolist() == [0, 0]


@pytest.mark.parametrize("name", STATE)
def test_external_owner_state_replacement_is_detected(core, provider, name):
    owner = runtime(core)
    bind(provider, owner)
    setattr(owner.index_group, name, getattr(owner.index_group, name).clone())
    with pytest.raises(Error, match="mutation"):
        provider.reset(owner, quiescent=True)


def test_external_owner_view_replacement_is_detected(core, provider):
    owner = runtime(core)
    raw, layout = bind(provider, owner)
    owner.bind_hot_cache(raw, **layout)  # Outside provider: even same-layout replacement is rejected.
    with pytest.raises(Error, match="mutation"):
        provider.reset(owner, quiescent=True)


def test_independent_owners_cannot_alias_state_or_hot_ranges(core, provider):
    owner, other = runtime(core), runtime(core)
    raw, layout = bind(provider, owner)
    with pytest.raises(Error, match="overlap"):
        provider.bind_owner(other, raw, **layout, quiescent=True)
    other.index_group.swap_stats = owner.index_group.swap_stats
    with pytest.raises(Error, match="aliases another owner"):
        bind(provider, other)
    assert not hasattr(other, "hot")


def test_partial_rebind_failure_rolls_back_all_views(core, provider, monkeypatch):
    owner, reader = runtime(core), runtime(core, 2176, 2176)
    old, layout = bind(provider, owner)
    provider.bind_reader(reader, owner=owner, quiescent=True)
    old_views = (owner.hot, reader.hot)
    owner.index_group.device_global_indices[0, 0] = 3
    original = provider._bind
    def failing_bind(member, raw, **kwargs):
        original(member, raw, **kwargs)  # Real Core method, not a view double.
        if member is reader:
            raise RuntimeError("injected failure after second real Core bind")
    with monkeypatch.context() as patch:
        patch.setattr(provider, "_bind", failing_bind)
        with pytest.raises(RuntimeError, match="injected failure"):
            provider.rebind_owner(owner, torch.zeros_like(old),
                                  block_table=layout["block_table"].clone(), quiescent=True)
    assert owner.hot is old_views[0] and reader.hot is old_views[1]
    assert owner.hot_backing is old and reader.hot_backing is old
    assert owner.index_group.device_global_indices[0, 0].item() == 3
    provider.reset(owner, quiescent=True)
    assert owner.index_group.device_global_indices[0, 0].item() == -1


def test_reset_failure_latches_instead_of_silent_retry(core, provider, monkeypatch):
    owner = runtime(core)
    bind(provider, owner)
    original = torch.Tensor.zero_
    def failing_zero(tensor, *args, **kwargs):
        if tensor is owner.index_group.swap_stats:
            raise RuntimeError("injected reset failure")
        return original(tensor, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "zero_", failing_zero)
        with pytest.raises(RuntimeError, match="injected reset failure"):
            provider.reset(owner, quiescent=True)
    with pytest.raises(Error, match="restart with fresh runtimes"):
        provider.reset(owner, quiescent=True)


def test_serialized_parallel_reader_bind_is_idempotent(core, provider):
    from concurrent.futures import ThreadPoolExecutor
    owner = runtime(core)
    bind(provider, owner)
    readers = [runtime(core, 2176, 2176) for _ in range(4)]
    def attach(reader):
        provider.bind_reader(reader, owner=owner, quiescent=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(attach, readers + readers))
    assert len(provider._owners[owner].readers) == 4
    for reader in readers:
        assert_shared(owner, reader)
        provider.unbind_reader(reader, quiescent=True)
    provider.unbind_owner(owner, quiescent=True)


@pytest.mark.parametrize("connector", ["HiSparseConnector", "MooncakeConnector"])
def test_dfly_constructor_transfer_guard_remains_closed(connector):
    path = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/dfly_speculator.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4DFlySpeculator")
    ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    namespace = {"VllmConfig": object, "torch": torch}
    exec(compile(ast.Module(body=[ctor], type_ignores=[]), str(path), "exec"), namespace)
    cfg = NS(cache_config=NS(cache_dtype="auto"),
             speculative_config=NS(kv_cache_dtype=None),
             kv_transfer_config=NS(kv_connector=connector))
    with pytest.raises(ValueError, match="does not support PD disaggregation"):
        namespace["__init__"](NS(), cfg, torch.device("cpu"))


@pytest.mark.parametrize("role", ["owner", "reader"])
def test_failed_fresh_bind_has_no_partial_publication(core, provider, monkeypatch, role):
    owner, reader = runtime(core), runtime(core, 2176, 2176)
    raw, layout = slab(owner)
    if role == "reader":
        provider.bind_owner(owner, raw, **layout, quiescent=True)
    target = owner if role == "owner" else reader
    initial = state(target)
    original = provider._bind
    def fail_after_real_bind(member, backing, **kwargs):
        original(member, backing, **kwargs)
        raise RuntimeError("injected fresh bind failure")
    with monkeypatch.context() as patch:
        patch.setattr(provider, "_bind", fail_after_real_bind)
        with pytest.raises(RuntimeError, match="injected fresh bind failure"):
            if role == "owner":
                provider.bind_owner(owner, raw, **layout, quiescent=True)
            else:
                provider.bind_reader(reader, owner=owner, quiescent=True)
    assert not any(hasattr(target, name) for name in ("hot", "hot_backing", "hot_block_table"))
    assert all(a is b for a, b in zip(state(target), initial))
    if role == "owner":
        provider.bind_owner(owner, raw, **layout, quiescent=True)
    else:
        provider.bind_reader(reader, owner=owner, quiescent=True)
        assert_shared(owner, reader)
