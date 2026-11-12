# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise the installed upstream chunk planner, including empty PCP rows."""
import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_mla_indexer as adapter


def _native_split():
    root = Path(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))
    path = root / "v1/attention/backends/mla/indexer.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "split_indexer_prefill_chunks")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["split_indexer_prefill_chunks"]


def _patched(native):
    def split_batch(common_attn_metadata, decode_threshold=1, require_uniform=False,
                    treat_short_extends_as_decodes=True):
        return ()

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return SimpleNamespace()

    module = ModuleType(adapter.TARGET_MODULE)
    module.split_indexer_prefill_chunks = native
    module.split_decodes_and_prefills = split_batch
    module.DeepseekV32IndexerMetadataBuilder = Builder
    adapter.apply_to_module(module)
    return module.split_indexer_prefill_chunks


@pytest.mark.parametrize("seq,query,workspace,logits,expected", [
    ([0], [0], 1, 64, []),
    ([0, 0], [0, 0], 1, 64, []),
    ([0, 8], [0, 2], 4, 64, [(slice(12, 13), slice(0, 2))]),
    ([8, 0, 9], [2, 0, 3], 4, 256,
     [(slice(11, 12), slice(0, 2)), (slice(13, 14), slice(0, 3))]),
    ([0, 4], [0, 3], 100, 16,
     [(slice(12, 13), slice(0, 1)), (slice(12, 13), slice(1, 2)),
      (slice(12, 13), slice(2, 3))]),
])
def test_zero_query_chunks_are_skipped_before_native_range(
    seq, query, workspace, logits, expected,
):
    split = _patched(_native_split())
    assert split(torch.tensor(seq), torch.tensor(query), workspace, logits, 11) == expected


def test_nonempty_native_chunks_remain_exact():
    native = _native_split()
    split = _patched(native)
    for seq, query in [([4], [4]), ([3, 7], [1, 2]), ([10, 4], [0, 2]),
                       ([3, 0, 4], [2, 0, 2]), ([0, 4], [0, 3])]:
        for workspace in (1, 4, 16):
            for logits in (16, 64, 256):
                tensors = (torch.tensor(seq), torch.tensor(query), workspace, logits, 7)
                try:
                    expected = native(*tensors)
                except ValueError as exc:
                    assert str(exc) == "range() arg 3 must not be zero"
                    continue
                assert split(*tensors) == [
                    (r, q) for r, q in expected if q.stop > q.start
                ]


def test_unrelated_native_value_error_is_preserved():
    def native(seq_lens_cpu, query_lens_cpu, workspace_size, max_logits_bytes,
               request_offset=0):
        raise ValueError("unrelated input problem")
    split = _patched(native)
    with pytest.raises(ValueError, match="unrelated input problem"):
        split(torch.tensor([0]), torch.tensor([0]), 1, 64)
