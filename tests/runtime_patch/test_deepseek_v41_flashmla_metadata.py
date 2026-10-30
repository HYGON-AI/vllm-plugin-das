# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable ownership/build contracts; graph/speculation remain gated off."""

import ast
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.fixtures.vllm_source import resolve_target_vllm_root
from tests.runtime_patch.test_deepseek_v41_flashmla_decode import inputs, model
from vllm_hcu.patch.worker.op_opt import (
    patch_deepseek_v41_rocm_flashmla_decode as decode_patch,
    patch_deepseek_v41_rocm_flashmla_metadata as patch,
)
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import (
    SCHEDULERS_ATTR,
    DecodeScheduler,
)


def forbidden(*args, **kwargs):
    pytest.fail("disabled/unsupported path must not allocate native schedulers")


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "1")
    monkeypatch.setattr(runtime, "_supported_device", lambda q: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def builder():
    calls = []

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            calls.append((common_prefix_len, common_attn_metadata, fast_build))
            return common_attn_metadata

    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV4ROCMAiterSparseSWAMetadataBuilder = Builder
    instance = Builder()
    instance.layer_names = ["target.layer0.swa_cache", "target.layer1.swa_cache"]
    instance.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
        ), speculative_config=None,
    )
    metadata = SimpleNamespace(
        num_decode_tokens=3, max_decode_query_len=1,
        decode_swa_indices=torch.zeros(3, 128, dtype=torch.int32),
        decode_swa_ragged_indices=object(), decode_swa_ragged_indptr=object(),
        tile_sched_swaonly=object(), tile_sched_c1a=object(), tile_sched_c2a=object(),
    )
    return module, instance, metadata, calls


def test_builder_owns_isolated_plans_and_preserves_fallback(enabled, monkeypatch):
    module, instance, metadata, calls = builder()
    plans = []

    def factory():
        plan = SimpleNamespace(have_initialized=False)
        plans.append(plan)
        return plan, None

    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (forbidden, forbidden, factory))
    original_fields = dict(vars(metadata))
    original = type(instance).build
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    assert instance.build(17, metadata, fast_build=True) is metadata
    assert calls == [(17, metadata, True)]
    for name, value in original_fields.items():
        assert getattr(metadata, name) is value
    leases = getattr(metadata, SCHEDULERS_ATTR)
    assert set(leases) == set(instance.layer_names)
    assert len(plans) == 2
    first = leases[instance.layer_names[0]].take()
    second = leases[instance.layer_names[1]].take()
    assert first is plans[0] and second is plans[1] and first is not second
    first.have_initialized = True
    # Same metadata forwarded twice must not reuse the initialized first plan.
    assert leases[instance.layer_names[0]].take() is plans[2]
    assert not plans[2].have_initialized
    old_leases = leases
    instance.build(0, metadata)
    assert getattr(metadata, SCHEDULERS_ATTR) is not old_leases
    assert len(plans) == 5
    type(instance).build = original
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)


@pytest.mark.parametrize("reason", [
    "disabled", "master", "pcp", "dcp", "speculation", "prefill",
    "multi_query", "indices", "device", "library",
])
def test_builder_gates_clear_old_state(enabled, monkeypatch, reason):
    module, instance, metadata, calls = builder()
    setattr(metadata, SCHEDULERS_ATTR, {"old": object()})
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    if reason == "disabled":
        monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "0")
    elif reason == "master":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    elif reason == "pcp":
        instance.vllm_config.parallel_config.prefill_context_parallel_size = 8
    elif reason == "dcp":
        instance.vllm_config.parallel_config.decode_context_parallel_size = 2
    elif reason == "speculation":
        instance.vllm_config.speculative_config = object()
    elif reason == "prefill":
        metadata.num_decode_tokens = 0
    elif reason == "multi_query":
        metadata.max_decode_query_len = 5
    elif reason == "indices":
        metadata.decode_swa_indices = None
    elif reason == "device":
        monkeypatch.setattr(runtime, "_supported_device", lambda q: False)
    elif reason == "library":
        monkeypatch.setattr(runtime, "_decode_bindings", lambda: None)
    patch.apply_to_module(module)
    assert instance.build(0, metadata) is metadata and len(calls) == 1
    assert getattr(metadata, SCHEDULERS_ATTR) is None


def test_builder_signature_drift_is_atomic():
    module, instance, _, _ = builder()
    type(instance).build = lambda self, metadata: None
    with pytest.raises(PatchCompatibilityError, match="signature"):
        patch.apply_to_module(module)
    assert not hasattr(type(instance), patch._MARKER)
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch.apply_to_module(ModuleType("vllm.models.deepseek_v4.amd.rocm"))


def test_forwards_replay_start_via_parent_when_rocm_omits_it(enabled, monkeypatch):
    """Installed ROCm build omits replay_start; parent already accepts it."""
    parent_calls = []

    class Parent:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
            replay_start=None,
        ):
            parent_calls.append(
                (common_prefix_len, common_attn_metadata, fast_build, replay_start)
            )
            return SimpleNamespace(
                num_decode_tokens=0,
                max_decode_query_len=1,
                decode_swa_indices=None,
                decode_swa_lens=None,
                decode_swa_width=0,
                replay_start=replay_start,
            )

    class Child(Parent):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            raise AssertionError("ROCm build must not be used when replay_start is set")

    class RocmMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV4ROCMAiterSparseSWAMetadataBuilder = Child
    module.DeepseekV4ROCMAiterSparseSWAMetadata = RocmMeta
    module.build_ragged_indices_from_dense = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no decode tokens")
    )
    module._copy_ragged_to_graph_buffers = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no decode tokens")
    )
    instance = Child()
    instance.layer_names = ["target.layer0.swa_cache"]
    instance.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: None)
    assert patch.apply_to_module(module)
    replay = torch.tensor([3, 0], dtype=torch.int32)
    out = instance.build(0, object(), replay_start=replay)
    assert parent_calls == [(0, parent_calls[0][1], False, replay)]
    assert out.replay_start is replay
    assert getattr(out, SCHEDULERS_ATTR) is None


def test_builder_instances_and_steps_do_not_share_state(enabled, monkeypatch):
    module, target, target_metadata, _ = builder()
    draft = type(target)()
    draft.layer_names = ["draft.layer0.swa_cache"]
    draft.vllm_config = target.vllm_config
    draft_metadata = SimpleNamespace(**vars(target_metadata))
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (
        forbidden, forbidden, lambda: (object(), None),
    ))
    patch.apply_to_module(module)
    target.build(0, target_metadata)
    first = getattr(target_metadata, SCHEDULERS_ATTR)
    draft.build(0, draft_metadata)
    second = getattr(draft_metadata, SCHEDULERS_ATTR)
    assert set(first) == set(target.layer_names)
    assert set(second) == set(draft.layer_names)
    old_plan = first[target.layer_names[0]].take()
    next_metadata = SimpleNamespace(**vars(target_metadata))
    target.build(0, next_metadata)
    third = getattr(next_metadata, SCHEDULERS_ATTR)
    assert third is not first and first is not second
    assert third[target.layer_names[0]].take() is not old_plan
    assert getattr(target_metadata, SCHEDULERS_ATTR) is first


def test_adapter_consumes_builder_plan_only_on_selection(enabled, monkeypatch):
    plans = []
    launched = []

    def factory():
        plan = object()
        plans.append(plan)
        return plan, None

    def kernel(**kwargs):
        launched.append(kwargs["tile_scheduler_metadata"])
        return torch.zeros_like(kwargs["q"]), None

    lease = DecodeScheduler(factory)
    args = inputs()
    args["scheduler"] = lease
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (kernel, forbidden, forbidden))
    # Unsupported mode and capture leave the builder's first plan unconsumed.
    assert not runtime.try_flashmla_decode(**dict(args, max_decode_query_len=5))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not runtime.try_flashmla_decode(**args)
    assert not launched and len(plans) == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert runtime.try_flashmla_decode(**args)
    args["swa_lens"].fill_(17)
    assert runtime.try_flashmla_decode(**args)
    assert len(plans) == 2 and launched == plans


def test_native_failure_consumes_lease(enabled, monkeypatch):
    plans = []

    def factory():
        plan = object()
        plans.append(plan)
        return plan, None

    def fail(**kwargs):
        assert kwargs["tile_scheduler_metadata"] is plans[-1]
        raise RuntimeError("HIP launch failed")

    args = inputs()
    args["scheduler"] = DecodeScheduler(factory)
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (fail, forbidden, forbidden))
    with pytest.raises(RuntimeError, match="HIP launch failed"):
        runtime.try_flashmla_decode(**args)
    assert args["scheduler"].take() is plans[1]
    assert plans[0] is not plans[1]


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_model_resolves_cache_prefix_not_shared_type(enabled, monkeypatch, ratio):
    module, attention, swa, calls = model()
    attention.compress_ratio = ratio
    attention.swa_cache_layer.prefix = "target.layer1.swa_cache"
    lease = DecodeScheduler(lambda: (object(), None))
    setattr(swa, SCHEDULERS_ATTR, {attention.swa_cache_layer.prefix: lease})
    decode_patch.apply_to_module(module)
    monkeypatch.setattr(decode_patch, "_context_supported", lambda self: True)
    seen = []
    monkeypatch.setattr(runtime, "try_flashmla_decode", lambda **kw: seen.append(kw) or True)
    args = inputs(page=None if ratio == 0 else 64 // ratio)
    metadata = None if ratio == 0 else SimpleNamespace(block_size=64, block_table=args["block_table"])
    attention._forward_decode(args["q"], args["kv_cache"], swa, metadata, ratio == 0, args["output"])
    assert seen[0]["scheduler"] is lease and not calls
    attention.swa_cache_layer.prefix = "unregistered.swa_cache"
    attention._forward_decode(args["q"], args["kv_cache"], swa, metadata, ratio == 0, args["output"])
    assert seen[1]["scheduler"] is None  # Adapter must fall back, not use a shared plan.


def test_real_builder_boundary_and_registration():
    root = resolve_target_vllm_root()
    tree = ast.parse((root / "vllm/models/deepseek_v41/amd/rocm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ROCMAiterSparseSWAMetadataBuilder")
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    assert [a.arg for a in build.args.args] == ["self", "common_prefix_len", "common_attn_metadata", "fast_build"]
    assert len(build.args.defaults) == 1 and build.args.defaults[0].value is False
    assert not build.args.kwonlyargs
    assert cls.bases[0].id == "DeepseekV41SparseSWAMetadataBuilder"
    from vllm_hcu.patch.worker import worker_callback_names
    names = worker_callback_names()
    assert (patch.PATCH_ID, patch.TARGET_MODULE) in names
    assert names.index((patch.PATCH_ID, patch.TARGET_MODULE)) < names.index((decode_patch.PATCH_ID, decode_patch.TARGET_MODULE))
