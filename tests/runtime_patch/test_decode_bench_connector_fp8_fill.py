# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU tests for the DecodeBenchConnector FP8 fill patch.

Exercises the REAL interface shapes and the REAL call chains:

* ``KVCacheGroupSpec``-style groups whose ``kv_cache_spec`` is a single spec,
  and ``UniformTypeKVCacheSpecs``-style wrappers carrying a
  ``kv_cache_specs`` mapping (expanded like
  ``vllm.v1.kv_cache_interface.iter_layer_specs``);
* per-layer ``cache_dtype_str`` classification (scalar whitelist vs packed
  flavours vs flavour-less fallback to the global ``cache_dtype`` field);
* entry through ``_fill_blocks`` only (never helpers directly), covering both
  the block-indexed tensor branch and the list/tuple state branch, plus the
  circular-buffer zero-fill group;
* scalar-FP8 byte caches get e4m3fn bytes (0.015 -> 0x08 / 0.015625), int8
  negatives stay bit-exact, packed/mixed/unknown flavours refuse clearly;
* signature preservation, idempotent arming, non-contiguous views.
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest
import torch

from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

FN = torch.float8_e4m3fn
MODULE_REF = "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"


def _load_patched_connector():
    connector = importlib.import_module(MODULE_REF)
    patch = importlib.import_module(
        "vllm_hcu.patch.platform.core_fix.patch_decode_bench_connector"
    )
    patch.apply_to_module(connector)
    return connector, patch


# --- interface fakes mirroring the real vllm.v1.kv_cache_interface shapes ---


class _LayerSpec:
    """A per-layer spec; flavourless when cache_dtype_str is empty (real
    FullAttentionSpec base classes carry ``cache_dtype_str=None``)."""

    def __init__(self, cache_dtype_str=None):
        self.cache_dtype_str = cache_dtype_str


class _UniformSpec:
    """``UniformTypeKVCacheSpecs``-style wrapper: one spec per layer."""

    def __init__(self, per_layer):
        self.kv_cache_specs = {
            name: spec for name, spec in per_layer.items()
        }


def _vcache_config(vllm_config, groups):
    return types.SimpleNamespace(kv_cache_groups=groups)


def _make_worker(connector, cache_dtype, groups, layer_caches,
                 zero_fill_group_ids=frozenset(), fill_mean=0.015,
                 fill_std=0.0):
    """Build a worker through the REAL constructor + register_kv_caches."""
    kv_transfer_config = types.SimpleNamespace(
        get_from_extra_config=lambda k, d: {
            "fill_mean": fill_mean, "fill_std": fill_std}.get(k, d)
    )
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=types.SimpleNamespace(cache_dtype=cache_dtype),
    )
    kv_cache_config = _vcache_config(vllm_config, groups)
    w = connector.DecodeBenchConnectorWorker(vllm_config, kv_cache_config)
    # mirror the deployed zero-fill derivation for circular-buffer groups
    w._zero_fill_group_ids = set(zero_fill_group_ids)
    w.register_kv_caches(layer_caches)
    return w


def _groups_from(specs_by_group, layer_names_by_group):
    groups = []
    for specs, layer_names in zip(specs_by_group, layer_names_by_group):
        if len(specs) == 1:
            group_spec = specs[0]
        else:
            group_spec = _UniformSpec(
                {name: spec for name, spec in zip(layer_names, specs)}
            )
        group_spec = _AttachNames(group_spec, layer_names)
        groups.append(types.SimpleNamespace(
            layer_names=layer_names, kv_cache_spec=group_spec))
    return groups


class _AttachNames:
    def __init__(self, inner, layer_names):
        self._inner = inner
        self.layer_names = layer_names

    def __getattr__(self, item):
        return getattr(self._inner, item)


# --------------------------------------------------------------- signature ---


def test_arm_is_idempotent_and_signatures_preserved():
    connector, patch = _load_patched_connector()
    assert patch.apply_to_module(connector) is False  # second arm is a no-op
    worker_cls = connector.DecodeBenchConnectorWorker
    assert getattr(worker_cls, patch._PATCHED_FLAG, False)
    for name, params in (
        ("__init__", ["self", "vllm_config", "kv_cache_config"]),
        ("_fill_blocks", ["self", "group_idx", "block_ids", "num_tokens"]),
        ("_fill_block_tensor",
         ["self", "kv_cache", "block_ids", "fill_mean", "fill_std"]),
        ("_fill_state_tensor", ["self", "kv_cache", "fill_mean", "fill_std"]),
    ):
        real = inspect.signature(getattr(worker_cls, name).__wrapped__)
        assert list(real.parameters) == params, name


# ------------------------------------------------------ uniform scalar fp8 ---
#

def test_uniform_scalar_fp8_group_encodes_bytes_real_chain():
    connector, _ = _load_patched_connector()
    layers = ["model.layers.0.kv"]
    groups = _groups_from([[_LayerSpec("fp8_e4m3")]], [layers])
    cache = torch.zeros((10, 2, 16), dtype=torch.uint8)
    w = _make_worker(connector, "fp8_e4m3", groups, {layers[0]: cache})
    w._fill_blocks(0, [2, 5], 32)
    assert cache.view(FN)[2, 0, 0].item() == 0.015625
    assert cache.view(FN)[5, 1, 3].item() == 0.015625
    assert cache[0].sum().item() == 0 and cache[7].sum().item() == 0


def test_flavourless_layers_fall_back_to_global_dtype():
    connector, _ = _load_patched_connector()
    layers = ["model.layers.1.kv"]
    groups = _groups_from([[_LayerSpec()]], [layers])
    cache = torch.zeros((4, 4), dtype=torch.uint8)
    w = _make_worker(connector, "fp8", groups, {layers[0]: cache})
    w._fill_blocks(0, [1], 4)
    assert cache.view(FN)[1].float()[0].item() == 0.015625


def test_int8_negative_mean_is_bit_exact_real_chain():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec("fp8_e4m3")]], [layers])
    c8 = torch.zeros((4, 8), dtype=torch.int8)
    # REAL chain semantics: fill_mean comes from the worker's extra config,
    # not from the _fill_blocks call, so configure the negative mean there.
    w = _make_worker(
        connector, "fp8_e4m3", groups, {"l0": c8},
        fill_mean=-0.02,
    )
    w._fill_blocks(0, [1], 8)
    ref_unsigned = int(
        torch.tensor([-0.02], dtype=torch.float32).to(FN).view(torch.uint8)[0]
    )
    ref_signed = torch.tensor([ref_unsigned], dtype=torch.uint8).view(torch.int8)[0].item()
    assert set(c8[1].tolist()) == {ref_signed}
    assert c8[0].sum().item() == 0


def test_random_fill_deterministic_reference_real_chain():
    """Real random branch (fill_std>0 via extra_config), CPU-seeded, compared
    byte-exactly against a float32->e4m3fn reference, unselected block kept."""
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec("fp8_e4m3")]], [layers])
    cr = torch.zeros((4, 64), dtype=torch.uint8)
    w = _make_worker(connector, "fp8_e4m3", groups, {"l0": cr},
                     fill_mean=0.015, fill_std=0.02)
    torch.manual_seed(1234)
    w._fill_blocks(0, [0, 1, 2], 192)
    torch.manual_seed(1234)
    reference = torch.normal(
        mean=0.015, std=0.02, size=(3, 64), dtype=torch.float32,
        device=cr.device,
    ).to(FN).view(torch.uint8)
    assert torch.equal(cr[0:3], reference)
    assert cr[3].sum().item() == 0  # unselected block untouched
    vals = cr[0:3].view(FN).float()
    assert torch.isfinite(vals).all()


# ------------------------------------------------------------ packed guard ---


def test_packed_uniform_group_refused():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec("fp8_ds_mla")]], [layers])
    cache = torch.zeros((2, 8), dtype=torch.uint8)
    w = _make_worker(connector, "fp8_ds_mla", groups, {"l0": cache})
    with pytest.raises(Exception, match="packed FP8"):
        w._fill_blocks(0, [0], 8)
    assert cache.max().item() == 0  # nothing written


def test_packed_global_with_flavourless_layers_refused():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec()]], [layers])
    cache = torch.zeros((2, 8), dtype=torch.uint8)
    w = _make_worker(connector, "fp8_ds_mla", groups, {"l0": cache})
    with pytest.raises(Exception):
        w._fill_blocks(0, [0], 8)
    assert cache.max().item() == 0


def test_uniform_group_packed_layer_in_scalargroup_refused_mixed():
    connector, _ = _load_patched_connector()
    layers = ["l0", "l1", "l2"]
    groups = _groups_from(
        [[_LayerSpec("fp8_e4m3"), _LayerSpec("fp8_ds_mla"), _LayerSpec("fp8_e4m3")]],
        [layers])
    c0 = torch.zeros((2, 4), dtype=torch.uint8)
    c1 = torch.zeros((2, 4), dtype=torch.uint8)
    native = torch.zeros((2, 4), dtype=FN)
    w = _make_worker(connector, "fp8_e4m3", groups,
                     {"l0": c0, "l1": c1, "l2": native})
    with pytest.raises(Exception, match="packed FP8"):
        w._fill_blocks(0, [0], 4)
    assert c0.max().item() == 0 and c1.max().item() == 0
    assert native.float().abs().sum().item() == 0


def _snapshot(caches):
    """Byte-exact clones of every cache (tensors and list/tuple states)."""
    clones = {}
    for name, value in caches.items():
        if isinstance(value, (list, tuple)):
            clones[name] = [t.clone() for t in value]
        else:
            clones[name] = value.clone()
    return clones


def _assert_unchanged(caches, clones):
    for name, value in caches.items():
        if isinstance(value, (list, tuple)):
            for tensor, clone in zip(value, clones[name]):
                assert torch.equal(tensor, clone), name
        else:
            assert torch.equal(value, clones[name]), name


NON_BYTE_FIRST_CASES = [
    # (label, layer flavours in group order; a CASE with cache_kind list in
    #  the same order: byte | native | float | state)
    ("native-first-packed",
     ["fp8_e4m3", "fp8_ds_mla"],
     ["native", "byte"]),
    ("float-first-mixed",
     [None, "fp8_e4m3"],
     ["float", "byte"]),
    ("state-first-packed",
     ["kimi_linear", "fp8_ds_mla"],
     ["state", "byte"]),
]

BYTE_FIRST_CASES = [
    ("byte-first-packed",
     ["fp8_ds_mla", "fp8_e4m3"],
     ["byte", "native"]),
    ("byte-first-mixed",
     ["fp8_e4m3", None],
     ["byte", "float"]),
]


@pytest.mark.parametrize("label,flavours,kinds", NON_BYTE_CASES := NON_BYTE_FIRST_CASES)
def test_group_refusal_zero_writes_when_non_byte_layers_come_first(
        label, flavours, kinds):
    """r12 WRITE-12-1: the early group refusal must guard EVERY layer order.

    With native/float/state layers placed BEFORE the offending byte layer,
    a naive implementation (no early refusal in the wrapper) writes those
    layers first and only then raises; these snapshots catch that mutation.
    """
    connector, _ = _load_patched_connector()
    layers = [f"l{i}" for i in range(len(flavours))]
    groups = _groups_from(
        [[_LayerSpec(f) for f in flavours]], [layers])
    caches = {}
    for name, kind in zip(layers, kinds):
        if kind == "byte":
            caches[name] = torch.zeros((2, 4), dtype=torch.uint8)
        elif kind == "native":
            caches[name] = torch.zeros((2, 4), dtype=FN)
        elif kind == "float":
            caches[name] = torch.zeros((2, 4), dtype=torch.float32)
        else:
            caches[name] = [torch.zeros((2, 4), dtype=torch.bfloat16)]
    w = _make_worker(
        connector,
        "fp8_e4m3" if "packed" in label else "auto",
        groups,
        caches,
    )
    clones = _snapshot(caches)
    with pytest.raises(
        PatchCompatibilityError,
        match="packed FP8" if "packed" in label else "mixed",
    ):
        w._fill_blocks(0, [0], 4)
    _assert_unchanged(caches, clones)


@pytest.mark.parametrize("label,flavours,kinds", BYTE_FIRST_CASES)
def test_group_refusal_zero_writes_when_byte_layer_comes_first(
        label, flavours, kinds):
    connector, _ = _load_patched_connector()
    layers = [f"l{i}" for i in range(len(flavours))]
    groups = _groups_from(
        [[_LayerSpec(f) for f in flavours]], [layers])
    caches = {}
    for name, kind in zip(layers, kinds):
        if kind == "byte":
            caches[name] = torch.zeros((2, 4), dtype=torch.uint8)
        else:
            caches[name] = torch.zeros((2, 4), dtype=FN)
    w = _make_worker(
        connector,
        "fp8_ds_mla" if "packed" in label else "auto",
        groups,
        caches,
    )
    clones = _snapshot(caches)
    with pytest.raises(
        PatchCompatibilityError,
        match="packed FP8" if "packed" in label else "mixed",
    ):
        w._fill_blocks(0, [0], 4)
    _assert_unchanged(caches, clones)


def test_mixed_scalar_fp8_and_float_group_refused_zero_writes():
    """scalar-FP8 + float layers in one group must NOT silently truncate the
    byte layer (r11 P1): refuse the whole group with zero writes."""
    connector, _ = _load_patched_connector()
    layers = ["l_byte", "l_float"]
    groups = _groups_from(
        [[_LayerSpec("fp8_e4m3"), _LayerSpec(None)]], [layers])
    byte_cache = torch.zeros((2, 4), dtype=torch.uint8)
    float_cache = torch.zeros((2, 4), dtype=torch.float32)
    w = _make_worker(connector, "auto", groups,
                     {"l_byte": byte_cache, "l_float": float_cache})
    with pytest.raises(PatchCompatibilityError, match="mixed"):
        w._fill_blocks(0, [1], 4)
    assert byte_cache.max().item() == 0
    assert float_cache.abs().sum().item() == 0


def test_unknown_flavour_refused_clearly():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec("somefuture_fp9")]], [layers])
    cache = torch.zeros((2, 8), dtype=torch.uint8)
    w = _make_worker(connector, "fp8_e4m3", groups, {"l0": cache})
    with pytest.raises(Exception, match="unrecognised"):
        w._fill_blocks(0, [0], 8)
    assert cache.max().item() == 0


# ----------------------------------------------------- float / state / cb ----


def test_float_cache_group_keeps_upstream_semantics():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec()]], [layers])
    for dt in (torch.bfloat16, torch.float32):
        cf = torch.zeros((2, 4), dtype=dt)
        w = _make_worker(connector, "auto", groups, {"l0": cf})
        w._fill_blocks(0, [1], 4)
        assert torch.allclose(cf[1], torch.full((4,), 0.015, dtype=dt), atol=1e-6)
        assert cf[0].abs().sum().item() == 0


def test_state_list_branch_via_fill_blocks_keeps_upstream():
    connector, _ = _load_patched_connector()
    layers = ["l_gdn"]
    groups = _groups_from([[_LayerSpec("kimi_linear")]], [layers])
    st = [torch.zeros((2, 4), dtype=torch.bfloat16)]
    w = _make_worker(connector, "fp8_e4m3", groups, {"l_gdn": st})
    # upstream guard: an empty block_ids list returns before ANY layer loop,
    # so a state-only request must still carry at least one block id.
    w._fill_blocks(0, [1], 0)
    assert abs(st[0].float().mean().item() - 0.015) < 1e-3


def test_zero_fill_circular_group_stays_zero_bytes():
    connector, _ = _load_patched_connector()
    layers = ["l_cb"]
    groups = _groups_from([[_LayerSpec("fp8_e4m3")]], [layers])
    cache = torch.full((2, 4), 7, dtype=torch.uint8)
    w = _make_worker(connector, "fp8_e4m3", groups, {"l_cb": cache},
                     zero_fill_group_ids={0})
    w._fill_blocks(0, [1], 4)
    assert cache[1].max().item() == 0  # forced (0.0, 0.0) -> zeros
    assert cache[0].max().item() == 7


# ---------------------------------------------------------- view placement ---


def test_non_contiguous_view_written_in_place_real_chain():
    connector, _ = _load_patched_connector()
    layers = ["l0"]
    groups = _groups_from([[_LayerSpec("fp8_e4m3")]], [layers])
    base = torch.zeros((6, 2, 8), dtype=torch.uint8)
    view = base[::2]
    assert not view.is_contiguous()
    w = _make_worker(connector, "fp8_e4m3", groups, {"l0": view})
    w._fill_blocks(0, [1], 16)
    assert base[0].sum().item() == 0
    assert base[2].sum().item() == 0x08 * 16
    assert base[4].sum().item() == 0
    assert base[1].sum().item() == 0 and base[3].sum().item() == 0


# ------------------------------------------------------------ registration ---


def test_registration_identity_and_targets():
    patch = importlib.import_module(
        "vllm_hcu.patch.platform.core_fix.patch_decode_bench_connector"
    )
    core_fix = importlib.import_module("vllm_hcu.patch.platform.core_fix")
    assert hasattr(core_fix, "register_platform_core_callbacks")
    assert (patch.PATCH_ID, patch.TARGET_MODULE) == (
        "platform.core.decode_bench_fp8_fill",
        MODULE_REF,
    )
    assert patch.TARGETS[1].endswith("DecodeBenchConnectorWorker._fill_blocks")
    for target in patch.TARGETS:
        assert target.startswith(MODULE_REF)
