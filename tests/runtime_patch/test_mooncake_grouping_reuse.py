# SPDX-License-Identifier: Apache-2.0
"""CPU unit coverage of the real builder without importing GPU/framework packages.

Only selected source definitions are compiled. Mamba type identity and the
unchanged BlockTable arithmetic boundary use small CPU stand-ins; this does not
claim complete module import, runtime Mamba support or native transport coverage.
"""
from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / (
    "vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)


def load_builder(path=SOURCE, module_name="_mooncake_grouping_unit"):
    """Load the complete async builder and its actual pure source helpers."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_get_tp_ratio", "_compute_sender_transfer_plan",
        "_can_coalesce_block_transfers", "_validate_nonnegative_block_ids",
        "group_concurrent_contiguous", "TransferRegion",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "MooncakeConnectorWorker")
    methods = {"_build_transfer_params", "_logical_to_kernel_block_ids",
               "_get_sender_transfer_plan", "_producer_cache_is_replicated"}
    selected = [node for node in owner.body if getattr(node, "name", None) in methods]
    assert len(nodes) == len(names) and len(selected) == len(methods)
    nodes.append(ast.ClassDef(name="Worker", bases=[], keywords=[], body=selected,
                              decorator_list=[]))
    nodes.insert(0, ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ))
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update(
        np=np, dataclass=dataclass, logger=logging.getLogger(module_name),
        MambaSpec=type("MambaSpec", (), {}), NULL_BLOCK_ID=0,
        BlockTable=SimpleNamespace(map_to_kernel_blocks=lambda ids, ratio, arange:
            (ids.reshape(-1, 1) * ratio + arange).reshape(-1)),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), module.__dict__)
    return module


class MooncakeGroupingReuseTests(unittest.TestCase):
    def setUp(self):
        self.m = load_builder()
        self.m.logger.setLevel(logging.CRITICAL)

    def worker(self, groups=1, *, mamba=(), ratio=1):
        worker = self.m.Worker()
        worker.tp_rank = 0
        worker.tp_size = 1
        worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)
        worker._physical_blocks_per_logical_kv_block = ratio
        worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=self.m.MambaSpec() if i in mamba else object()
            )
            for i in range(groups)
        ])
        return worker

    def regions(self, group_indices=(0,), *, stride=8, payload=8):
        return ([self.m.TransferRegion(str(i), i, 1000*(i+1), stride, payload, g)
                 for i, g in enumerate(group_indices)],
                [self.m.TransferRegion(str(i), i, 10000*(i+1), stride, payload, g)
                 for i, g in enumerate(group_indices)])

    def build(self, worker, src, dst, regions=None, *, remote_tp_size=1,
              remote_tp_rank=0):
        regions = regions or self.regions()
        send = SimpleNamespace(local_block_ids=src)
        metadata = SimpleNamespace(remote_hostname="test", remote_port=1,
            remote_tp_size=remote_tp_size, remote_tp_rank=remote_tp_rank,
            req_blocks={"d": ("x", dst)})
        return asyncio.run(worker._build_transfer_params(
            [("d", send)], metadata, *regions
        ))

    def test_groups_once_per_request_and_kv_group(self):
        regions = self.regions((0, 1, 0, 1))
        with patch.object(
            self.m, "group_concurrent_contiguous",
            wraps=self.m.group_concurrent_contiguous,
        ) as grouped:
            result = self.build(
                self.worker(2), [[2, 3], [8, 9]], [[5, 6], [10, 12]], regions
            )
        self.assertEqual(result, ([1016, 2064, 2072, 3016, 4064, 4072],
                                 [10040, 20080, 20096, 30040, 40080, 40096],
                                 [16, 8, 8, 16, 8, 8], [], None))
        self.assertEqual(grouped.call_count, 2)

    def test_no_transfer_rank_does_not_group(self):
        worker = self.worker()
        worker.tp_size = 16
        worker.tp_rank = 1
        worker.transfer_topo.local_replicates_kv_cache = True
        with patch.object(
            self.m, "group_concurrent_contiguous",
            wraps=self.m.group_concurrent_contiguous,
        ) as grouped:
            result = self.build(worker, [[2, 3]], [[5, 6]], self.regions((0, 0, 0)))
        self.assertEqual(result, ([], [], [], [], None))
        self.assertEqual(grouped.call_count, 0)

    def test_cache_does_not_cross_requests_or_invocations(self):
        worker = self.worker()
        first = SimpleNamespace(local_block_ids=[[1, 2]])
        second = SimpleNamespace(local_block_ids=[[20, 21]])
        metadata = SimpleNamespace(remote_hostname="test", remote_port=1,
            remote_tp_size=1, remote_tp_rank=0,
            req_blocks={"a": ("x", [[3, 4]]), "b": ("y", [[30, 31]])})
        result = asyncio.run(worker._build_transfer_params(
            [("a", first), ("b", second)], metadata, *self.regions()))
        self.assertEqual(result, ([1008, 1160], [10024, 10240], [16, 16], [], None))
        self.assertEqual(self.build(worker, [[90, 91]], [[100, 101]]),
                         ([1720], [10800], [16], [], None))

    def test_dense_and_padded_regions_share_groups_without_changing_descriptors(self):
        local, remote = self.regions((0, 0))
        local[1] = self.m.TransferRegion("1", 1, 2000, 16, 8, 0)
        remote[1] = self.m.TransferRegion("1", 1, 20000, 24, 8, 0)
        result = self.build(self.worker(), [[2, 3]], [[5, 6]], (local, remote))
        self.assertEqual(result, (
            [1016, 2032, 2048], [10040, 20120, 20144], [16, 8, 8], [], None
        ))

    def test_partial_prefix_and_empty_group_preserve_inputs(self):
        src, dst = [[1, 2, 9, 11], [3]], [[30, 31], []]
        result = self.build(self.worker(2), src, dst, self.regions((0, 1)))
        self.assertEqual(result, ([1072, 1088], [10240, 10248], [8, 8], [], None))
        self.assertEqual(src, [[1, 2, 9, 11], [3]])
        self.assertEqual(dst, [[30, 31], []])

    def test_attention_expansion_and_mamba_nulls_keep_group_identity(self):
        result = self.build(self.worker(2, mamba=(1,), ratio=2),
            [[0], [0, 3, 0, 4]], [[1], [0, 10, 0, 11]], self.regions((0, 1)))
        self.assertEqual(result, ([1000, 2024], [10016, 20080], [16, 16], [], None))

    def test_heterogeneous_tp_slice_keeps_offsets_and_padding(self):
        worker = self.worker()
        worker.tp_size = 2
        local, remote = self.regions(stride=32, payload=8)
        remote[0] = self.m.TransferRegion("0", 0, 10000, 32, 4, 0)
        result = self.build(
            worker, [[0, 1]], [[0, 1]], (local, remote),
            remote_tp_size=4, remote_tp_rank=1,
        )
        self.assertEqual(result, ([1004, 1036], [10000, 10032], [4, 4], [], None))

    def test_invalid_block_tables_still_fail_for_no_transfer_rank(self):
        for src, dst, message in (([[0]], [[0], [1]], "KV group count mismatch"),
                                  ([[0]], [[0, 1]], "P num blocks less than D"),
                                  ([[-1]], [[0]], "producer block ID -1"),
                                  ([[0]], [[-1]], "consumer block ID -1")):
            with self.subTest(src=src, dst=dst):
                worker = self.worker()
                worker.tp_rank = 1
                worker.tp_size = 16
                worker.transfer_topo.local_replicates_kv_cache = True
                result = self.build(worker, src, dst)
                self.assertEqual(result[:4], ([], [], [], ["d"]))
                self.assertIn(message, result[4])


if __name__ == "__main__":
    unittest.main()
