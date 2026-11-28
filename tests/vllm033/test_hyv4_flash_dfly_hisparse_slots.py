# SPDX-License-Identifier: Apache-2.0
"""Test the installed model's DFly query-slot ownership with Core boundary doubles."""

from __future__ import annotations

import ast
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


class HiSparseResidentSpec:
    uses_slot_mapping = True


class DraftModelSpeculator:
    def set_attn(self, state, config, block_tables, input_buffers, groups):
        self.kv_cache_config = config
        self.block_tables = block_tables
        self.attn_groups = groups


def make_speculator(enabled, *, slot_capacity=384):
    calls = []
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    target = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "HYV4DFlySpeculator")
    target.body = [node for node in target.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"set_attn", "_write_query_slots"}]
    tree.body = [target]
    scope = dict(DraftModelSpeculator=DraftModelSpeculator,
                 torch=SimpleNamespace(int64="int64", full=lambda shape, *_a, **_k: SlotMappings(*shape)),
                 PAD_SLOT_ID=-1, write_dfly_query_slots=lambda dst, *_: calls.append(dst))
    exec(compile(tree, str(SOURCE), "exec"), scope)
    instance = scope[target.name]()
    instance.reuse = SimpleNamespace(validate_groups=lambda _: None)
    instance.vllm_config = SimpleNamespace(attention_config=SimpleNamespace(
        hisparse_config=object() if enabled else None))
    instance.max_num_tokens = 384
    instance.device = "device"
    instance.input_buffers = SimpleNamespace(positions=object())
    instance.active_requests = object()
    instance.max_model_len_gpu = object()
    instance.num_query_per_req = 6
    regular = SimpleNamespace(uses_slot_mapping=True)
    raw = SimpleNamespace(uses_slot_mapping=False)
    groups = [SimpleNamespace(kv_cache_spec=spec)
              for spec in (regular, HiSparseResidentSpec(), raw, regular)]
    config = SimpleNamespace(kv_cache_groups=groups)
    tables = SimpleNamespace(slot_mappings=SlotMappings(4, slot_capacity),
                             input_block_tables=[object() for _ in groups], block_sizes=[64] * len(groups))
    return instance, config, tables, calls


class TestDFlyHiSparseSlots(unittest.TestCase):
    def test_hisparse_writes_bound_resident_group_even_without_a_draft_layer(self):
        speculator, config, tables, calls = make_speculator(True)
        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        self.assertIs(speculator._slot_mappings, tables.slot_mappings)
        speculator._write_query_slots(4)
        self.assertEqual(calls, [tables.slot_mappings[i] for i in (0, 1, 3)])

    def test_gpu_only_draft_retains_private_rows_and_skips_unowned_groups(self):
        speculator, config, tables, calls = make_speculator(False)
        speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
        self.assertIsNot(speculator._slot_mappings, tables.slot_mappings)
        speculator._write_query_slots(4)
        self.assertEqual(calls, [speculator._slot_mappings[0]])

    def test_hisparse_rejects_insufficient_bound_slot_mapping_capacity(self):
        speculator, config, tables, _ = make_speculator(True, slot_capacity=383)
        with self.assertRaisesRegex(ValueError, "HiSparse DFly slot mapping capacity"):
            speculator.set_attn(None, config, tables, None, [[object()], [], [], []])
