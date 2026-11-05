# SPDX-License-Identifier: Apache-2.0
"""The transfer plan preserves bytes, including fragmented and reversed pages."""
import ast
import random
from pathlib import Path

import pytest

source = Path(__file__).parents[2] / 'vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/step5_aux.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_coalesced_block_runs')
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
coalesce = scope['_coalesced_block_runs']


@pytest.mark.parametrize('source,target,expected', [
    ([], [], []),
    ([7], [2], [(7, 2, 1)]),
    ([7, 8, 9], [2, 3, 4], [(7, 2, 3)]),
    ([9, 8, 7], [4, 3, 2], [(7, 2, 3)]),
    ([7, 8, 9], [4, 3, 2], [(7, 4, 1), (8, 3, 1), (9, 2, 1)]),
    ([7, 8, 2, 3], [2, 3, 8, 9], [(7, 2, 2), (2, 8, 2)]),
    ([7, 8, 7], [2, 3, 2], [(7, 2, 2), (7, 2, 1)]),
    ([7, 7], [2, 3], [(7, 2, 1), (7, 3, 1)]),
])
def test_runs_merge_only_matching_adjacent_mappings(source, target, expected):
    assert coalesce(source, target) == expected


def test_128k_summary_becomes_one_run_without_changing_length():
    runs = coalesce(list(range(2048)), list(range(3072, 5120)))
    assert runs == [(0, 3072, 2048)]
    assert sum(count * 2048 for _, _, count in runs) == 4 * 1024**2


def test_random_mappings_copy_exactly_the_same_bytes():
    rng = random.Random(73)
    page_bytes = 11
    payload = bytes(rng.randrange(256) for _ in range(64 * page_bytes))
    for _ in range(200):
        count = rng.randrange(1, 65)
        source = rng.choices(range(64), k=count)
        target = rng.sample(range(64), k=count)
        reference = bytearray([211] * len(payload))
        actual = bytearray(reference)
        for src, dst in zip(source, target):
            reference[dst*page_bytes:(dst+1)*page_bytes] = payload[src*page_bytes:(src+1)*page_bytes]
        for src, dst, n in coalesce(source, target):
            actual[dst*page_bytes:(dst+n)*page_bytes] = payload[src*page_bytes:(src+n)*page_bytes]
        assert actual == reference
