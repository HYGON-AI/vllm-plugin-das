# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the real-vision profiling lifetime (no GPU claims)."""

import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def load_module():
    path = Path(__file__).parents[3] / "vllm_hcu/v1/hyv4_vl_profile.py"
    assert path.is_file(), "HYV4-VL real encoder profiling is missing"
    spec = importlib.util.spec_from_file_location("hyv4_vl_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profile():
    return load_module().profile_hyv4_vl_vision


@pytest.fixture
def fixture(monkeypatch):
    # CUDA and the native registry/encoder boundary are unavailable on CPU.
    # All allocations and output validation below use real CPU torch tensors.
    allocations = []
    original_empty = torch.empty

    def empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        allocations.append(weakref.ref(value))
        return value

    monkeypatch.setattr(torch, "empty", empty)
    syncs = []
    monkeypatch.setattr(torch, "accelerator", NS(synchronize=lambda: syncs.append(True)), raising=False)
    budget = NS(
        mm_max_toks_per_item={"image": 6},
        mm_max_items_per_batch={"image": 2},
        encoder_cache_size=20,
        cache=None,
        get_encoder_budget=lambda: 12,
        reset_cache=lambda: None,
    )
    native_budget = types.ModuleType("vllm.multimodal.encoder_budget")
    native_budget.MultiModalBudget = lambda config, registry: budget
    monkeypatch.setitem(sys.modules, native_budget.__name__, native_budget)
    item = object()
    placeholder = NS(get_num_embeds=lambda: 6)

    class Registry:
        def get_dummy_mm_inputs(self, config, *, mm_counts, cache):
            assert mm_counts == {"image": 1}
            return {"mm_kwargs": {"image": [item]}, "mm_placeholders": {"image": [placeholder]}}

    output_refs = []
    events = []
    encoder = NS()

    def encode(items):
        assert items == [("image", item), ("image", item)]
        events.append(sum(ref().numel() for ref in allocations if ref() is not None))
        # The actual HYV4 merger returns transposed views with exact storage.
        outputs = [torch.ones(4, 6).t(), torch.ones(4, 6).t()]
        output_refs.extend(weakref.ref(value) for value in outputs)
        return outputs

    encoder.execute_mm_encoder = encode
    config = NS(
        architectures=["HYV4VLForConditionalGeneration"],
        multimodal_config=NS(skip_mm_profiling=False, enable_mm_embeds=False),
        get_hidden_size=lambda: 4,
        dtype=torch.float32,
    )
    class State:
        encoder_runner = encoder
        _hcu_vl_persistent_memory_bytes = 0

    runner = NS(
        model_config=config,
        vllm_config=NS(model_config=config, speculative_config=None),
        mm_registry=Registry(),
        model_state=State(),
        encoder_cache=NS(encoder_outputs={}, mm_features={"live-request": [object()]}),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_num_tokens=1,
        model_memory_usage=100,
    )
    load_module().account_hyv4_vl_state_memory(runner)
    return NS(runner=runner, budget=budget, allocations=allocations, output_refs=output_refs,
              events=events, syncs=syncs, placeholder=placeholder)


@pytest.mark.parametrize("pcp", [1, 16])
def test_full_per_rank_encoder_and_exact_cache_reserve_survive_decoder(fixture, pcp):
    f = fixture
    f.runner.vllm_config.parallel_config = NS(prefill_context_parallel_size=pcp)
    profile = load_profile()
    with profile(f.runner):
        # 20 cache tokens minus 12 newly encoded tokens = 8 resident tokens.
        assert f.events == [32]
        assert sum(ref().numel() for ref in f.allocations if ref() is not None) == 32
        assert all(ref() is not None for ref in f.output_refs)
        assert f.runner.max_num_tokens == 1
    gc.collect()
    assert all(ref() is None for ref in f.allocations + f.output_refs)
    assert len(f.syncs) >= 2


def test_preprofile_live_cache_does_not_reduce_measured_increment(fixture):
    f = fixture
    live = torch.ones(2, 4)
    features = f.runner.encoder_cache.mm_features.copy()
    f.runner.encoder_cache.encoder_outputs["live"] = live
    with load_profile()(f.runner):
        # The worker's torch baseline already contains the two live tokens.
        # Incremental profiling must still observe all 20 configured tokens.
        assert f.events == [32]
        assert f.runner.encoder_cache.encoder_outputs == {"live": live}
    assert f.runner.encoder_cache.encoder_outputs == {"live": live}
    assert f.runner.encoder_cache.mm_features == features


@pytest.mark.parametrize("failure", ["encoder", "decoder"])
def test_exception_releases_only_profile_allocations(fixture, failure):
    f = fixture
    live = torch.ones(2, 4)
    f.runner.encoder_cache.encoder_outputs["live"] = live
    if failure == "encoder":
        def fail(items):
            raise RuntimeError("encoder failed")
        f.runner.model_state.encoder_runner.execute_mm_encoder = fail
    with pytest.raises(RuntimeError, match=failure), load_profile()(f.runner):
        raise RuntimeError("decoder failed")
    gc.collect()
    assert all(ref() is None for ref in f.allocations + f.output_refs)
    assert f.runner.encoder_cache.encoder_outputs == {"live": live}
    assert f.syncs


@pytest.mark.parametrize("bad", ["count", "tokens", "hidden", "dtype", "storage"])
def test_malformed_encoder_outputs_fail_before_decoder(fixture, bad):
    f = fixture
    outputs = {
        "count": [torch.ones(6, 4)],
        "tokens": [torch.ones(5, 4), torch.ones(6, 4)],
        "hidden": [torch.ones(6, 3), torch.ones(6, 4)],
        "dtype": [torch.ones(6, 4, dtype=torch.float64), torch.ones(6, 4)],
        "storage": [torch.ones(7, 4)[:6], torch.ones(6, 4)],
    }[bad]
    f.runner.model_state.encoder_runner.execute_mm_encoder = lambda items: outputs
    with pytest.raises(ValueError, match="encoder output"):
        with load_profile()(f.runner):
            pytest.fail("malformed output reached decoder")
    gc.collect()
    assert all(ref() is None for ref in f.allocations)


def test_overfull_live_cache_fails_before_real_encoder(fixture):
    f = fixture
    f.runner.encoder_cache.encoder_outputs["live"] = torch.ones(9, 4)
    with pytest.raises(ValueError, match="cache budget"), load_profile()(f.runner):
        pytest.fail("overfull profile reached decoder")
    assert f.events == []
    assert f.allocations == []


def test_dummy_token_count_mismatch_fails_before_allocating(fixture):
    f = fixture
    f.placeholder.get_num_embeds = lambda: 7
    with pytest.raises(ValueError, match="dummy.*tokens"), load_profile()(f.runner):
        pytest.fail("unbounded dummy reached decoder")
    assert f.events == []
    assert f.allocations == []


def test_skip_profiling_does_not_silently_bypass_vision(fixture):
    f = fixture
    f.runner.model_config.multimodal_config.skip_mm_profiling = True
    with pytest.raises(ValueError, match="skip_mm_profiling"):
        with load_profile()(f.runner):
            pytest.fail("unsafe skip reached decoder")


def test_unrelated_architecture_needs_no_multimodal_runtime():
    runner = NS(model_config=NS(architectures=["HYV4ForCausalLM"]))
    with load_profile()(runner):
        assert vars(runner) == {"model_config": runner.model_config}


def test_actual_state_bytes_are_accounted_before_profile_and_only_once():
    class State:
        _hcu_vl_persistent_memory_bytes = 48

    runner = NS(model_config=NS(architectures=["HYV4VLForConditionalGeneration"]),
                model_state=State(), model_memory_usage=100)
    account = load_module().account_hyv4_vl_state_memory
    account(runner)
    account(runner)
    assert runner.model_memory_usage == 148


@pytest.mark.parametrize("value", [None, -1, 1.5, True])
def test_unbounded_or_invalid_state_memory_ledger_is_rejected(value):
    class State:
        pass

    if value is not None:
        State._hcu_vl_persistent_memory_bytes = value
    runner = NS(model_config=NS(architectures=["HYV4VLForConditionalGeneration"]),
                model_state=State(), model_memory_usage=100)
    with pytest.raises(ValueError, match="persistent.*memory"):
        load_module().account_hyv4_vl_state_memory(runner)
    assert runner.model_memory_usage == 100


def test_zero_cpu_state_ledger_does_not_add_memory():
    class State:
        _hcu_vl_persistent_memory_bytes = 0

    runner = NS(model_config=NS(architectures=["HYV4VLForConditionalGeneration"]),
                model_state=State(), model_memory_usage=100)
    load_module().account_hyv4_vl_state_memory(runner)
    assert runner.model_memory_usage == 100


def test_state_accounting_noop_for_text_models():
    runner = NS(model_config=NS(architectures=["HYV4ForCausalLM"]))
    load_module().account_hyv4_vl_state_memory(runner)
    assert vars(runner) == {"model_config": runner.model_config}


def test_profile_rejects_state_not_accounted_before_worker_snapshot(fixture):
    f = fixture
    class Unaccounted:
        _hcu_vl_persistent_memory_bytes = 0
    f.runner.model_state = Unaccounted()
    with pytest.raises(ValueError, match="account.*before"):
        with load_profile()(f.runner):
            pytest.fail("unaccounted state reached profile")
