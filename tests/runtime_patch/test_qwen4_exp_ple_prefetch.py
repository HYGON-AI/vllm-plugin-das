# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-side contract tests for Qwen4Exp PLE prefetch orchestration."""

from contextlib import nullcontext
import importlib
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_ple_int8 as int8_patch
from vllm_hcu.patch.worker.core_fix import (
    patch_qwen4_exp_ple_cudagraph as graph_patch,
)
from vllm_hcu.patch.worker.core_fix import (
    patch_qwen4_exp_ple_prefetch as model_patch,
)
from vllm_hcu.platforms import envs as hcu_envs


@pytest.fixture(scope="module")
def ple_layer():
    return importlib.import_module("vllm_hcu.models.qwen4_exp.amd.ple_layer")


def _capable_method():
    return SimpleNamespace(
        supports_prefetch=True,
        prefetch_output_dtype=lambda layer: torch.bfloat16,
        prefetch_lookup_into=lambda layer, ids, output: None,
        finalize_prefetched=lambda layer, rows: rows,
    )


def test_prefetch_environment_is_lazy_and_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_PLE_PREFETCH_STREAM", raising=False)
    assert hcu_envs.VLLM_HCU_PLE_PREFETCH_STREAM is False
    assert hcu_envs.is_set("VLLM_HCU_PLE_PREFETCH_STREAM") is False
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "1")
    assert hcu_envs.VLLM_HCU_PLE_PREFETCH_STREAM is True
    assert hcu_envs.is_set("VLLM_HCU_PLE_PREFETCH_STREAM") is True


def test_prefetch_gate_is_int8_uva_only(monkeypatch, ple_layer):
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "1")
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")

    assert ple_layer._prefetch_method_enabled(_capable_method()) is True
    assert ple_layer._prefetch_method_enabled(SimpleNamespace()) is False
    assert int8_patch.HcuQwen4ExpPLEInt8EmbeddingMethod.supports_prefetch is False
    assert (
        int8_patch.HcuQwen4ExpPLEInt8OffloadEmbeddingMethod.supports_prefetch is False
    )
    assert int8_patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod.supports_prefetch is True

    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "0")
    assert ple_layer._prefetch_method_enabled(_capable_method()) is False
    monkeypatch.setenv("VLLM_HCU_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "0")
    assert ple_layer._prefetch_method_enabled(_capable_method()) is False


def test_incomplete_claimed_capability_fails_fast(ple_layer):
    method = SimpleNamespace(
        supports_prefetch=True,
        prefetch_output_dtype=lambda layer: torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="prefetch_lookup_into"):
        ple_layer._prefetch_output_dtype(method, SimpleNamespace())


def test_uva_lookup_into_buffer_matches_embedding():
    weight = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, 0]], dtype=torch.long)
    output = torch.empty((1, 2, 3), dtype=torch.bfloat16)
    layer = SimpleNamespace(params_dtype=torch.bfloat16, tp_size=1)
    method = int8_patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
    method._uva_view = lambda unused: (weight, scale)

    method.prefetch_lookup_into(layer, ids, output)
    expected = torch.nn.functional.embedding(ids, weight).to(torch.bfloat16)
    expected *= torch.nn.functional.embedding(ids, scale)

    assert torch.equal(output, expected)
    assert method.finalize_prefetched(layer, output.flatten(-2)) is not None


def test_uva_post_process_invalidates_view_from_previous_storage():
    method = int8_patch.HcuQwen4ExpPLEInt8UVAEmbeddingMethod()
    layer = SimpleNamespace(
        weight=torch.empty((2, 3), dtype=torch.int8),
        weight_scale=torch.empty((2, 1), dtype=torch.bfloat16),
        _hcu_uva_views=(object(), object()),
    )

    method.process_weights_after_loading(layer)

    assert layer._hcu_uva_views is None
    assert method.is_prefetch_prepared(layer) is False


def test_start_and_consume_use_side_stream_event_and_static_buffers(
    monkeypatch, ple_layer
):
    events = []

    class Stream:
        def wait_event(self, event):
            events.append(("side-wait", event))

    class Event:
        def record(self, side):
            events.append(("record", side))

    class Main:
        def record_event(self, event):
            events.append(("fork-record", event))

        def wait_event(self, event):
            events.append(("main-wait", event))

    side = Stream()
    event = Event()
    main = Main()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda unused: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class Method:
        def prefetch_lookup_into(self, layer, ids, output):
            events.append(("lookup", ids.clone()))
            output.copy_(ids.unsqueeze(-1).expand_as(output))

        def finalize_prefetched(self, layer, rows):
            events.append(("finalize", rows.shape))
            return rows.clone()

    ngram_ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.long)
    embedding = SimpleNamespace(tp_size=1, quant_method=Method())
    owner = SimpleNamespace(
        _hcu_prefetch_enabled=True,
        _hcu_prefetch_pending=False,
        _hcu_prefetch_prepared=True,
        _hcu_prefetch_stream=side,
        _hcu_prefetch_event=event,
        _hcu_prefetch_fork_event=Event(),
        _hcu_prefetch_ids_buffer=torch.empty((4, 2), dtype=torch.long),
        _hcu_prefetch_rows_buffer=torch.empty((4, 6), dtype=torch.long),
        max_total_tokens=4,
        ngram_heads=2,
        head_dim=3,
        layer_name="model.layers.1.ple",
        ngram_embedding=embedding,
        compute_ngram_ids=lambda input_ids, query_start_loc, ngram_context: ngram_ids,
        prepare_prefetch=lambda: None,
    )

    started = ple_layer.Qwen4ExpNGramEmbedding._start_prefetch_impl(
        owner,
        torch.tensor([10, 11]),
        torch.tensor([0, 2]),
        torch.zeros((1, 2), dtype=torch.long),
    )
    assert started is True
    assert owner._hcu_prefetch_pending is True
    assert torch.equal(owner._hcu_prefetch_ids_buffer[:2], ngram_ids)
    with pytest.raises(RuntimeError, match="duplicate"):
        ple_layer.Qwen4ExpNGramEmbedding._start_prefetch_impl(
            owner,
            torch.tensor([10, 11]),
            torch.tensor([0, 2]),
            torch.zeros((1, 2), dtype=torch.long),
        )

    result = ple_layer.Qwen4ExpNGramEmbedding._consume_prefetched_impl(owner, 2)
    assert result.shape == (2, 6)
    assert owner._hcu_prefetch_pending is False
    assert [item[0] for item in events] == [
        "fork-record",
        "side-wait",
        "lookup",
        "record",
        "main-wait",
        "finalize",
    ]
    with pytest.raises(RuntimeError, match="miss"):
        ple_layer.Qwen4ExpNGramEmbedding._consume_prefetched_impl(owner, 2)


def test_graph_capture_boundary_joins_pending_prefetches_only_for_capture(
    monkeypatch,
):
    calls = []

    class Mode:
        NONE = "none"

    class Wrapper:
        def __init__(self):
            self.runtime_mode = "piecewise"
            self.concrete_cudagraph_entries = {}
            self.runnable = lambda: "runnable-output"

        def __call__(self, *args, **kwargs):
            del args, kwargs
            calls.append("call")
            return "output"

    module = ModuleType(graph_patch.TARGET_MODULE)
    module.CUDAGraphWrapper = Wrapper
    module.CUDAGraphMode = Mode
    module.is_forward_context_available = lambda: True
    context = SimpleNamespace(
        cudagraph_runtime_mode="piecewise",
        batch_descriptor="batch",
    )
    module.get_forward_context = lambda: context
    original = Wrapper.__call__
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "1")
    monkeypatch.setattr(
        graph_patch,
        "_will_capture",
        lambda cudagraph_module, wrapper: (
            wrapper.concrete_cudagraph_entries.get("batch") is None
        ),
    )
    monkeypatch.setattr(
        model_patch,
        "join_pending_prefetches",
        lambda: calls.append("join"),
    )
    monkeypatch.setattr(graph_patch, "_validate_call", lambda call: None)

    assert graph_patch.apply_to_module(module) is True
    wrapper = Wrapper()
    assert wrapper() == "output"
    assert calls == ["join", "call"]

    wrapper.concrete_cudagraph_entries["batch"] = SimpleNamespace(
        cudagraph=object()
    )
    assert wrapper() == "output"
    assert calls == ["join", "call", "call"]


def test_graph_capture_proxy_joins_after_runnable_before_capture_exit(monkeypatch):
    calls = []

    class Mode:
        NONE = "none"

    class Wrapper:
        def __init__(self):
            self.runtime_mode = "piecewise"
            self.concrete_cudagraph_entries = {}
            self.runnable = lambda: calls.append("runnable") or "output"

        def __call__(self, *args, **kwargs):
            del args, kwargs
            calls.append("call")
            self.runnable()
            return "graph-output"

    module = ModuleType(graph_patch.TARGET_MODULE)
    module.CUDAGraphWrapper = Wrapper
    module.CUDAGraphMode = Mode
    module.is_forward_context_available = lambda: True
    module.get_forward_context = lambda: SimpleNamespace(
        cudagraph_runtime_mode="piecewise",
        batch_descriptor="batch",
    )
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "1")
    monkeypatch.setattr(graph_patch, "_validate_call", lambda call: None)
    monkeypatch.setattr(model_patch, "join_pending_prefetches", lambda: calls.append("join"))

    assert graph_patch.apply_to_module(module) is True
    wrapper = Wrapper()
    assert wrapper() == "graph-output"
    assert calls == ["join", "call", "runnable", "join"]
    assert wrapper.runnable() == "output"
    assert calls == ["join", "call", "runnable", "join", "runnable"]


def test_join_pending_prefetches_waits_only_live_pending_owners(monkeypatch):
    calls = []

    class Main:
        def wait_event(self, event):
            calls.append(event)

    class Owner:
        pass

    pending = Owner()
    pending.layer_name = "model.layers.1.ple"
    pending._hcu_prefetch_pending = True
    pending._hcu_prefetch_event = "pending-event"
    complete = Owner()
    complete.layer_name = "model.layers.2.ple"
    complete._hcu_prefetch_pending = False
    complete._hcu_prefetch_event = "complete-event"
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Main())
    monkeypatch.setattr(model_patch, "_PREFETCH_OWNERS", {pending, complete})

    model_patch.join_pending_prefetches()

    assert calls == ["pending-event"]


def test_layer_fires_successor_before_consuming_current(ple_layer):
    calls = []
    successor = SimpleNamespace(
        _start_prefetch_impl=lambda *args: calls.append("successor-fire")
    )
    owner = SimpleNamespace(
        _hcu_prefetch_successor=successor,
        _hcu_prefetch_enabled=True,
        _consume_prefetched_impl=lambda num_tokens: (
            calls.append("current-consume") or torch.zeros((num_tokens, 4))
        ),
    )
    result = ple_layer.Qwen4ExpNGramEmbedding.forward(
        owner,
        torch.tensor([1, 2]),
        torch.tensor([0, 2]),
        torch.zeros((1, 2), dtype=torch.long),
    )
    assert result.shape == (2, 4)
    assert calls == ["successor-fire", "current-consume"]


def _ngram(name: str, calls: list[str]):
    class NGram:
        pass

    ngram = NGram()
    ngram.name = name
    ngram._hcu_prefetch_enabled = True
    ngram._hcu_prefetch_ids_buffer = torch.empty(0)
    ngram._hcu_prefetch_rows_buffer = torch.empty(0)
    ngram._start_prefetch_impl = lambda *args: None
    ngram.prepare_prefetch = lambda: calls.append(f"prepare:{name}")
    ngram._hcu_prefetch_successor = None
    return ngram


def test_model_callback_binds_isolated_chains_and_fires_first(monkeypatch):
    calls = []

    class Model:
        def __init__(self, *args, vllm_config=None, prefix="", **kwargs):
            del args, vllm_config, prefix, kwargs
            self.start_layer = 0
            self.end_layer = 3
            first = _ngram("first", calls)
            second = _ngram("second", calls)
            self.layers = [
                SimpleNamespace(ple=SimpleNamespace(ple_embedding=first)),
                SimpleNamespace(ple=None),
                SimpleNamespace(ple=SimpleNamespace(ple_embedding=second)),
            ]

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
            query_start_loc=None,
            ngram_context=None,
            deepstack_input_embeds=None,
        ):
            del (
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                query_start_loc,
                ngram_context,
                deepstack_input_embeds,
            )
            calls.append("forward")
            return "output"

        def load_weights(self, weights):
            calls.append("load")
            return set(weights)

    class CausalModel:
        def __init__(self, model):
            self.model = model

    class ConditionalModel:
        def __init__(self, model):
            self.language_model = CausalModel(model)

    module = ModuleType(model_patch.TARGET_MODULE)
    module.Qwen4ExpModel = Model
    module.Qwen4ExpForCausalLM = CausalModel
    module.Qwen4ExpForConditionalGeneration = ConditionalModel
    monkeypatch.setenv("VLLM_HCU_PLE_PREFETCH_STREAM", "1")
    monkeypatch.setattr(model_patch, "_ensure_custom_op_registered", lambda: None)
    monkeypatch.setattr(
        model_patch,
        "_run_prefetch_custom_op",
        lambda input_ids, query_start_loc, ngram_context, ngram: calls.append(
            f"fire:{ngram.name}"
        ),
    )

    assert model_patch.apply_to_module(module) is True
    assert model_patch.apply_to_module(module) is False
    first_model = Model(vllm_config=object())
    second_model = Model(vllm_config=object())

    first_chain = first_model._vllm_hcu_local_ple_chain
    second_chain = second_model._vllm_hcu_local_ple_chain
    assert first_chain[0]._hcu_prefetch_successor is first_chain[1]
    assert second_chain[0]._hcu_prefetch_successor is second_chain[1]
    assert first_chain[0] is not second_chain[0]

    result = first_model.forward(
        torch.tensor([1]),
        torch.tensor([0]),
        query_start_loc=torch.tensor([0, 1]),
        ngram_context=torch.zeros((1, 2), dtype=torch.long),
    )
    assert result == "output"
    assert calls[-2:] == ["fire:first", "forward"]

    CausalModel(first_model).process_weights_after_loading()
    assert calls[-2:] == ["prepare:first", "prepare:second"]
    ConditionalModel(second_model).process_weights_after_loading()
    assert calls[-2:] == ["prepare:first", "prepare:second"]


def test_model_callback_is_noop_when_prefetch_is_disabled(monkeypatch):
    class Model:
        def __init__(self, *args, vllm_config=None, prefix="", **kwargs):
            pass

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
            query_start_loc=None,
            ngram_context=None,
            deepstack_input_embeds=None,
        ):
            pass

        def load_weights(self, weights):
            pass

    module = ModuleType(model_patch.TARGET_MODULE)
    module.Qwen4ExpModel = Model
    monkeypatch.delenv("VLLM_HCU_PLE_PREFETCH_STREAM", raising=False)
    original = Model.forward

    assert model_patch.apply_to_module(module) is False
    assert Model.forward is original
