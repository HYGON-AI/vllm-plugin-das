# SPDX-License-Identifier: Apache-2.0
"""Exercise the companion patch on the actual core helper, without GPU imports."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

import pytest


PATCH = (Path(__file__).resolve().parents[2] / "patches"
         / "vllm-0.25.1-indexer-zero-chunk.patch")
TARGET = Path("vllm/v1/attention/backends/mla/indexer.py")


class _Scalar(int):
    def item(self):
        return int(self)


class _CPUValues(list):
    """Only implement the CPU scalar access used by this pure Python helper."""

    def __getitem__(self, index):
        return _Scalar(super().__getitem__(index))


def _load_helper(source: str):
    tree = ast.parse(source)
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "split_indexer_prefill_chunks")
    scope = {"torch": SimpleNamespace(Tensor=_CPUValues)}
    exec(compile(ast.Module(body=[helper], type_ignores=[]),
                 str(TARGET), "exec"), scope)
    return scope[helper.name]


@pytest.fixture(scope="module")
def helpers(tmp_path_factory):
    explicit_root = os.environ.get("VLLM_SOURCE")
    if explicit_root:
        source = Path(explicit_root) / TARGET
        assert source.is_file(), f"VLLM_SOURCE does not contain {TARGET}"
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            pytest.skip("set VLLM_SOURCE to a matching vLLM core checkout")
        source = Path(spec.origin).parent / Path(*TARGET.parts[1:])
        if not source.is_file():
            pytest.skip("installed vLLM does not provide the matching indexer")
    git = os.environ.get("GIT_EXECUTABLE") or shutil.which("git")
    if not git:
        pytest.skip("git is required to verify and apply the companion patch")
    root = tmp_path_factory.mktemp("indexer-core")
    target = root / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text(encoding="utf-8"),
                      encoding="utf-8", newline="\n")

    def apply(*options):
        return subprocess.run([git, "-C", str(root), "apply", *options,
                               str(PATCH)], capture_output=True, text=True)

    check = apply("--check")
    if check.returncode:
        # An already-patched installation can still exercise both versions;
        # reverse only the temporary copy, never the user's checkout.
        reverse = apply("--reverse", "--check")
        if reverse.returncode and not explicit_root:
            pytest.skip("indexer source is incompatible with this v0.25.1 patch")
        assert reverse.returncode == 0, check.stderr + reverse.stderr
        assert apply("--reverse").returncode == 0
    original = target.read_text(encoding="utf-8")
    assert apply("--check").returncode == 0
    changed = apply()
    assert changed.returncode == 0, changed.stderr
    fixed = target.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8").replace("\r\n", "\n") in (
        original, fixed)
    return _load_helper(original), _load_helper(fixed)


def _run(helper, seq, query, *, workspace=100000,
         budget=512 * 1024 * 1024, offset=7):
    chunks = helper(_CPUValues(seq), _CPUValues(query),
                    workspace, budget, offset)
    return [(req.start, req.stop, q.start, q.stop) for req, q in chunks]


@pytest.mark.parametrize("seq,query", [([], []), ([0], [0]),
                                     ([0, 0, 0], [0, 0, 0]),
                                     ([85000], [0])])
def test_empty_query_work_returns_no_chunks(helpers, seq, query):
    assert _run(helpers[1], seq, query) == []


def test_over_budget_prefill_then_zero_padding_preserves_offsets(helpers):
    original, fixed = helpers
    with pytest.raises(ValueError, match=r"range\(\) arg 3 must not be zero"):
        _run(original, [85000, 0], [4096, 0])
    assert _run(fixed, [85000, 0], [4096, 0]) == [
        (7, 8, 0, 1579), (7, 8, 1579, 3158), (7, 8, 3158, 4096)]


@pytest.mark.parametrize("seq,query", [([0], [0]), ([0, 0], [0, 0])])
def test_original_zero_only_failure_is_reproduced(helpers, seq, query):
    with pytest.raises(ValueError, match=r"range\(\) arg 3 must not be zero"):
        _run(helpers[0], seq, query)


@pytest.mark.parametrize("seq,query", [([100, 200], [3, 4]),
                                     ([100, 0, 200], [3, 0, 4]),
                                     ([85000], [4096])])
def test_nonempty_chunk_boundaries_are_unchanged(helpers, seq, query):
    assert _run(helpers[0], seq, query) == _run(helpers[1], seq, query)


def test_randomized_query_coverage_and_unchanged_nonerror_results(helpers):
    original, fixed = helpers
    rng = random.Random(20261110)
    compared = recovered = 0
    for _ in range(2000):
        seq = [rng.choice([0, 100, 4096, 65536, 85000])
               for _ in range(rng.randrange(9))]
        query = [rng.randrange(min(length, 8192) + 1) for length in seq]
        workspace = rng.choice([1024, 98304, 200000])
        budget = rng.choice([64, 512]) * 1024 * 1024
        result = _run(fixed, seq, query, workspace=workspace, budget=budget)
        assert sum(stop - start for _, _, start, stop in result) == sum(query)
        for req_start, req_stop, start, stop in result:
            assert 7 <= req_start < req_stop <= 7 + len(seq)
            assert 0 <= start < stop <= sum(query[req_start - 7:req_stop - 7])
            assert (stop - start) * sum(seq[req_start - 7:req_stop - 7]) * 4 <= budget
        try:
            reference = _run(original, seq, query,
                             workspace=workspace, budget=budget)
        except ValueError as error:
            assert str(error) == "range() arg 3 must not be zero"
            recovered += 1
        else:
            assert result == reference
            compared += 1
    assert compared > 0 and recovered > 0
