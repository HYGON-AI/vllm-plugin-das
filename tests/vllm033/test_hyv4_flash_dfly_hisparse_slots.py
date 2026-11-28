# SPDX-License-Identifier: Apache-2.0
"""Test the installed model's DFly query-slot ownership with Core boundary doubles."""

from __future__ import annotations

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / "adapter_extension/src/hcu_vllm_das"
          / "models/hy_v4_flash/dfly_speculator.py")


class SlotMappings:
    ndim = 2
    dtype = "int64"

    def __init__(self, groups, rows):
        self.shape = (groups, rows)
        self.rows = [object() for _ in range(groups)]

    def __getitem__(self, group):
        return self.rows[group]

    def new_full(self, shape, value):
        tensor = SlotMappings(*shape)
        tensor.fill_value = value
        return tensor


class HiSparseResidentSpec:
    uses_slot_mapping = True


class DraftModelSpeculator:
    def __init__(self, config, device):
        self.vllm_config = config
        self.max_num_tokens = config.scheduler_config.max_num_batched_tokens
        self.device = device

    def set_attn(self, state, config, block_tables, input_buffers, groups):
        self.kv_cache_config = config
        self.block_tables = block_tables
        self.attn_groups = groups


def make_speculator(enabled, *, scheduler=None):
    calls = []
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    target = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "HYV4DFlySpeculator")
    target.body = [node for node in target.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"__init__", "set_attn", "_write_query_slots"}]
    constructor = next(node for node in target.body if node.name == "__init__")
    # Execute the real validation and draft scheduler sizing through the Core
    # constructor. The rest allocates model buffers outside this contract.
    tail = next(i for i, node in enumerate(constructor.body)
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == "_hisparse_config_owner"
                    for t in node.targets))
    constructor.body = constructor.body[:tail]
    tree.body = [target]
    scope = dict(DraftModelSpeculator=DraftModelSpeculator,
                 torch=SimpleNamespace(int64="int64", full=lambda shape, *_a, **_k: SlotMappings(*shape)),
                 copy=copy, _admits_hisparse_graph=lambda _: enabled,
                 PAD_SLOT_ID=-1, write_dfly_query_slots=lambda dst, *_: calls.append(dst))
    exec(compile(tree, str(SOURCE), "exec"), scope)
    if scheduler is None:
        scheduler = SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=384)
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object() if enabled else None),
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        speculative_config=SimpleNamespace(kv_cache_dtype=None, num_speculative_tokens=5),
        compilation_config=SimpleNamespace(cudagraph_mode=object()),
        parallel_config=SimpleNamespace(use_ubatching=False),
        scheduler_config=scheduler, kv_transfer_config=None, lora_config=None,
    )
    instance = scope[target.name](vllm_config, "device")
    instance.reuse = SimpleNamespace(validate_groups=lambda _: None)
    instance.input_buffers = SimpleNamespace(positions=object())
    instance.active_requests = object()
    instance.max_model_len_gpu = object()
    instance.num_query_per_req = 6
    regular = SimpleNamespace(uses_slot_mapping=True)
    raw = SimpleNamespace(uses_slot_mapping=False)
    groups = [SimpleNamespace(kv_cache_spec=spec)
              for spec in (regular, HiSparseResidentSpec(), raw, regular)]
    config = SimpleNamespace(kv_cache_groups=groups)
    tables = SimpleNamespace(slot_mappings=SlotMappings(4, scheduler.max_num_batched_tokens),
                             max_num_batched_tokens=scheduler.max_num_batched_tokens,
                             input_block_tables=[object() for _ in groups], block_sizes=[64] * len(groups))
    return instance, config, tables, calls


class TestDFlyHiSparseSlots(unittest.TestCase):
    def test_hisparse_writes_bound_resident_group_even_without_a_draft_layer(self):
        speculator, config, tables, calls = make_speculator(True)
        original = tables.slot_mappings
        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        self.assertIs(tables.slot_mappings, original)
        self.assertIs(speculator._slot_mappings, original)
        speculator._write_query_slots(4)
        self.assertEqual(calls, [tables.slot_mappings[i] for i in (0, 1, 3)])

    def test_gpu_only_draft_retains_private_rows_and_skips_unowned_groups(self):
        speculator, config, tables, calls = make_speculator(False)
        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        self.assertIsNot(speculator._slot_mappings, tables.slot_mappings)
        speculator._write_query_slots(4)
        self.assertEqual(calls, [speculator._slot_mappings[0]])

    def test_hisparse_extends_shared_slots_from_the_target_scheduler_budget(self):
        scheduler = SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=256,
                                    enable_chunked_prefill=True)
        speculator, config, tables, calls = make_speculator(True, scheduler=scheduler)
        original = tables.slot_mappings
        self.assertEqual(original.shape, (4, 256))
        self.assertEqual(speculator.max_num_tokens, 384)
        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        shared = tables.slot_mappings
        self.assertIsNot(shared, original)
        self.assertIs(speculator._slot_mappings, shared)
        self.assertEqual(shared.shape, (4, 384))
        self.assertEqual(shared.fill_value, -1)
        self.assertEqual(scheduler.max_num_batched_tokens, 256)
        self.assertEqual(tables.max_num_batched_tokens, 256)
        speculator._write_query_slots(64)
        self.assertEqual(calls, [shared[i] for i in (0, 1, 3)])

        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        self.assertIs(tables.slot_mappings, shared)

    def test_hisparse_rejects_an_invalid_slot_mapping_layout(self):
        speculator, config, tables, _ = make_speculator(True)
        tables.slot_mappings = SlotMappings(3, 256)
        with self.assertRaisesRegex(ValueError, "HiSparse DFly slot mapping .*layout"):
            speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
