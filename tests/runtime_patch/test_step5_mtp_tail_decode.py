# SPDX-License-Identifier: Apache-2.0
"""MTP prefill trimming must preserve native speculative input preparation.

Execute the installed native Triton kernel body using CPU arrays and checked
pointers. Only Triton loads/stores and program IDs are substituted; rejection
handling and every native input-buffer write remain the original function.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.models.step5_config import Step5Config
from vllm_hcu.patch.worker.framework_opt import patch_mtp_prefill_tail as tail


class _Value(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(_Value)


class _Pointer:
    def __init__(self, name, values, offset=0):
        self.name, self.values, self.offset = name, values, offset

    def __add__(self, offset):
        return _Pointer(self.name, self.values, self.offset + offset)

    def __sub__(self, offset):
        return self + -offset


class _CPUOps:
    int32, int64 = np.int32, np.int64

    def __init__(self, count, starts):
        self.count, self.starts, self.req = count, starts, 0

    def program_id(self, axis):
        assert axis == 0
        return self.req

    def num_programs(self, axis):
        assert axis == 0
        return self.count

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop).view(_Value)

    @staticmethod
    def where(condition, left, right):
        return np.asarray(np.where(condition, left, right)).view(_Value)

    def _indices(self, pointer, mask, operation):
        idx = np.asarray(pointer.offset)
        active = np.ones(idx.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        idx, active = np.broadcast_arrays(idx, active)
        selected = idx[active]
        bad = selected[(selected < 0) | (selected >= len(pointer.values))]
        if len(bad):
            raise AssertionError(f"{operation} OOB at {bad[0]} in {pointer.name}")
        if operation == "store" and pointer.name in ("draft_input_ids", "draft_positions"):
            start, stop = self.starts[self.req:self.req + 2]
            assert np.all((selected >= start) & (selected < stop)), (
                f"native write crossed request {self.req} boundary"
            )
        return idx.astype(np.intp), active

    def load(self, pointer, mask=None):
        idx, active = self._indices(pointer, mask, "load")
        value = np.zeros(idx.shape, dtype=pointer.values.dtype)
        value[active] = pointer.values[idx[active]]
        return value.view(_Value)

    def store(self, pointer, value, mask=None):
        idx, active = self._indices(pointer, mask, "store")
        value = np.broadcast_to(np.asarray(value), idx.shape)
        pointer.values[idx[active]] = value[active]


def _native_prepare(batch, rejected, sampled=None):
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp import speculator

    kernel = speculator._prepare_input_buffers_kernel
    # Triton's CPU fallback may leave jit as an identity decorator.
    # Both forms must execute the same original Python kernel body below.
    fn = getattr(kernel, "fn", kernel)
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    starts = batch.query_start_loc_np.astype(np.int64).copy()
    count, tokens = batch.num_reqs, int(starts[-1])
    ops = _CPUOps(count, starts)
    namespace = {"tl": ops}
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), function,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<native MTP input kernel>", "exec"), namespace)
    kernel = namespace[function.name]
    last_indices = np.full(count, -99, dtype=np.int64)
    arrays = {
        "last_token_indices": last_indices,
        "draft_input_ids": np.full(tokens, -99, dtype=np.int64),
        "draft_positions": np.full(tokens, -99, dtype=np.int64),
        "draft_seq_lens": np.zeros(count, dtype=np.int32),
        "target_input_ids": batch.input_ids.numpy().copy(),
        "target_positions": batch.positions.numpy().copy(),
        "cached_draft_input_ids": np.arange(count * 2, dtype=np.int64),
        "draft_input_id_overrides": np.zeros(count * 2, dtype=np.int64),
        "idx_mapping": np.arange(count, dtype=np.int32),
        "last_sampled": np.arange(count, dtype=np.int64) + 1000,
        "next_prefill_tokens": np.arange(count * 3, dtype=np.int64) + 2000,
        "num_sampled": (np.asarray(sampled, dtype=np.int32) if sampled is not None
                        else np.where(batch.is_prefilling_np, 0, 1).astype(np.int32)),
        "num_rejected": np.asarray(rejected, dtype=np.int32),
        "target_seq_lens": batch.seq_lens.numpy().copy(),
        "query_start_loc": starts,
    }
    pointers = {name: _Pointer(name, values) for name, values in arrays.items()}
    for req in range(count):
        ops.req = req
        kernel(
            pointers["last_token_indices"], pointers["draft_input_ids"],
            pointers["draft_positions"], pointers["draft_seq_lens"],
            pointers["target_input_ids"], pointers["target_positions"],
            pointers["cached_draft_input_ids"], 2,
            pointers["draft_input_id_overrides"], 2,
            pointers["idx_mapping"], pointers["last_sampled"],
            pointers["next_prefill_tokens"], count, pointers["num_sampled"],
            pointers["num_rejected"], pointers["target_seq_lens"],
            pointers["query_start_loc"], count, 3, 8,
        )
    return last_indices


def _batch(queries, prefilling):
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    count, tokens = len(queries), sum(queries)
    batch = InputBatch.make_dummy(count, tokens, InputBuffers(count, tokens, torch.device("cpu")))
    starts = np.array([0, *np.cumsum(queries)], dtype=np.int32)
    return dataclasses.replace(
        batch, num_scheduled_tokens=np.array(queries, dtype=np.int32),
        query_start_loc_np=starts, query_start_loc=torch.from_numpy(starts),
        seq_lens=torch.tensor(queries, dtype=torch.int32) + 32,
        is_prefilling_np=np.array(prefilling, dtype=np.bool_),
        has_prefill=any(prefilling),
        input_ids=torch.arange(tokens, dtype=torch.int32),
        positions=torch.arange(tokens, dtype=torch.int64),
        max_query_len=max(queries),
    )


def _wrapped_propose(monkeypatch, batch, rejected):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(tail, "_TAIL", 1)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    config = Step5Config(
        num_hidden_layers=2, num_nextn_predict_layers=3, sliding_window=1,
        layer_types=["full_attention", "sliding_attention"] + ["sliding_attention"] * 3,
    )
    layers = {str(i): SimpleNamespace(mtp_block=SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(sliding_window=1)))) for i in range(2, 5)}
    module = ModuleType(tail.TARGET_MODULE)
    received = []

    class MultiModuleMTPSpeculator:
        def propose(self, input_batch, attn_metadata, slot_mappings, hidden, auxiliary,
                    num_sampled, num_rejected, last_sampled, next_prefill_tokens,
                    temperature, seeds, **kwargs):
            received.append((input_batch, hidden, num_rejected))
            indices = _native_prepare(input_batch, num_rejected.numpy(), num_sampled.numpy())
            return input_batch, hidden, indices

    module.MultiModuleMTPSpeculator = MultiModuleMTPSpeculator
    tail.apply_to_module(module)
    runner = MultiModuleMTPSpeculator()
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=SimpleNamespace(enable_prefix_caching=False))
    runner.draft_model_config = SimpleNamespace(hf_text_config=config)
    runner.model = SimpleNamespace(model=SimpleNamespace(
        mtp_start_layer_idx=2, num_mtp_layers=3, layers=layers))
    runner.num_speculative_steps = 3
    runner.dp_size, runner.device = 1, torch.device("cpu")
    assert tail._tail_enabled(runner)
    hidden = torch.arange(batch.num_tokens * 2).reshape(batch.num_tokens, 2)
    rejections = torch.tensor(rejected, dtype=torch.int32)
    sampled = torch.from_numpy(np.where(batch.is_prefilling_np, 0, 1).astype(np.int32))
    result = runner.propose(
        batch, None, None, hidden, None, sampled, rejections,
        torch.zeros(batch.num_reqs, dtype=torch.int64),
        torch.zeros((3, batch.num_reqs), dtype=torch.int64),
        torch.zeros(batch.num_reqs), torch.zeros(batch.num_reqs, dtype=torch.int64),
    )
    assert received[0][2] is rejections
    torch.testing.assert_close(rejections, torch.tensor(rejected, dtype=torch.int32))
    return hidden, result


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_speculative_decode_query_reaches_native_preparation_unchanged(monkeypatch, rejected):
    batch = _batch([4], [False])
    hidden, (forwarded, forwarded_hidden, indices) = _wrapped_propose(monkeypatch, batch, [rejected])
    assert forwarded is batch and forwarded_hidden is hidden
    assert forwarded.num_scheduled_tokens.tolist() == [4]
    assert indices.tolist() == [3 if rejected == 0 else 2]


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_mixed_batch_trims_prefill_only_before_native_input_preparation(monkeypatch, rejected):
    batch = _batch([4, 12, 4], [False, True, False])
    hidden, (forwarded, forwarded_hidden, indices) = _wrapped_propose(
        monkeypatch, batch, [rejected, 0, rejected])
    assert forwarded.num_scheduled_tokens.tolist() == [4, 1, 4]
    assert forwarded.query_start_loc_np.tolist() == [0, 4, 5, 9]
    assert forwarded.positions.tolist() == [0, 1, 2, 3, 15, 16, 17, 18, 19]
    torch.testing.assert_close(forwarded_hidden, hidden[[0, 1, 2, 3, 15, 16, 17, 18, 19]])
    assert indices.tolist() == ([3, 4, 8] if rejected == 0 else [2, 4, 7])


@pytest.mark.parametrize("rejected", [1, 2, 3])
def test_native_cpu_checker_detects_the_original_negative_write(rejected):
    unsafe = _batch([1], [False])
    with pytest.raises(AssertionError, match="store OOB at -1 in draft_input_ids"):
        _native_prepare(unsafe, [rejected])


@pytest.mark.parametrize("wrapped", [False, True], ids=["python", "jit"])
def test_native_cpu_checker_accepts_both_kernel_forms(monkeypatch, wrapped):
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp import speculator

    kernel = speculator._prepare_input_buffers_kernel
    function = getattr(kernel, "fn", kernel)
    monkeypatch.setattr(
        speculator, "_prepare_input_buffers_kernel",
        SimpleNamespace(fn=function) if wrapped else function,
    )
    safe = _batch([4], [False])
    for rejected in (0, 1, 2, 3):
        indices = _native_prepare(safe, [rejected])
        assert indices.tolist() == [3 if rejected == 0 else 2]
    for rejected in (1, 2, 3):
        with pytest.raises(AssertionError, match="store OOB at -1 in draft_input_ids"):
            _native_prepare(_batch([1], [False]), [rejected])


@pytest.mark.parametrize("phase", [None, [True], np.array([1], dtype=np.int32),
                                   np.array([], dtype=np.bool_), np.array([[True]])])
def test_missing_or_invalid_request_phase_keeps_full_batch(monkeypatch, phase):
    monkeypatch.setattr(tail, "_TAIL", 1)
    batch = _batch([12], [True])
    batch.is_prefilling_np = phase
    assert tail._trim(SimpleNamespace(dp_size=1), batch, None, None) is None
