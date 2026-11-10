# SPDX-License-Identifier: Apache-2.0
"""QA resume identity checks without a dataset download or serving requests."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/hyv4_realworldqa_eval.py"


def _evaluator_scope():
    # Keep the actual entry point, manifest/resume, scoring and result logic.
    # Only optional image/parquet/HTTP imports are excluded at this boundary.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    scope = {
        "argparse": argparse,
        "hashlib": hashlib,
        "importlib": importlib,
        "io": io,
        "json": json,
        "math": math,
        "Path": Path,
        "statistics": statistics,
        "subprocess": subprocess,
        "time": time,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "__file__": str(SOURCE),
    }
    exec(  # noqa: S102 - Execute only the checked-in evaluator functions.
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope
    )
    return scope


@pytest.fixture
def run(tmp_path, monkeypatch):
    scope = _evaluator_scope()
    data = tmp_path / "data"
    data.mkdir()
    (data / "test-0.parquet").write_bytes(b"controlled fake parquet boundary")
    scorer = tmp_path / "scorer.py"
    scorer.write_text("def extract_mcq_answer(value, choices): return value\n")
    rows = [{"answer": "A", "question": str(i)} for i in range(765)]
    scope["pq"] = SimpleNamespace(
        read_table=lambda path: SimpleNamespace(to_pylist=lambda: rows)
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "fixed-revision\n")
    scope["as_completed"] = lambda futures: iter(futures)
    calls = []
    interrupt = [True]

    def evaluate(index, row, args, extract_mcq):
        calls.append((index, args.base_url))
        if interrupt[0] and index == 1:
            raise KeyboardInterrupt("controlled interruption after one saved result")
        return {
            "index": index,
            "status": 200,
            "correct": True,
            "strict_correct": True,
            "prediction": "A",
            "answer": "A",
            "finish_reason": "stop",
            "seconds": 0.01,
        }

    scope["evaluate"] = evaluate

    def invoke(endpoint):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SOURCE),
                "--data-dir",
                str(data),
                "--output-dir",
                str(tmp_path / "run"),
                "--mcq-extractor",
                str(scorer),
                "--concurrency",
                "1",
                "--base-url",
                endpoint,
            ],
        )
        return scope["main"]()

    with pytest.raises(KeyboardInterrupt):
        invoke("http://reference:8010")
    interrupt[0] = False
    calls.clear()
    return invoke, calls, tmp_path / "run"


def test_qa_resume_rejects_another_endpoint_before_requests(run):
    invoke, calls, output = run
    with pytest.raises(ValueError, match="Resume settings or data differ"):
        invoke("http://candidate:8010")
    assert calls == []
    assert len((output / "results.jsonl").read_text().splitlines()) == 1


def test_qa_resume_accepts_same_endpoint_with_trailing_slash(run):
    invoke, calls, output = run
    invoke("http://reference:8010/")
    assert len(calls) == 764
    assert {endpoint for _, endpoint in calls} == {"http://reference:8010"}
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    assert (
        json.loads((output / "manifest.json").read_text())["base_url"]
        == "http://reference:8010"
    )


def test_qa_resume_rejects_legacy_manifest_without_endpoint(run):
    invoke, calls, output = run
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("base_url", None)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Resume settings or data differ"):
        invoke("http://reference:8010")
    assert calls == []


def test_qa_result_records_its_endpoint():
    scope = _evaluator_scope()

    class Image:
        size = (2, 2)

        def convert(self, mode):
            return self

        def save(self, stream, format):
            stream.write(b"controlled encoded image")

    import base64

    scope["base64"] = base64
    scope["requests"] = SimpleNamespace(
        RequestException=OSError,
        post=lambda *a, **kw: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]
            },
        ),
    )
    args = SimpleNamespace(
        model="alias", base_url="http://reference:8010/", timeout=1, max_tokens=16
    )
    result = scope["evaluate"](
        0,
        {"image": Image(), "question": "q", "answer": "A"},
        args,
        lambda text, choices: text,
    )
    assert result["base_url"] == "http://reference:8010"
    assert result["correct"] is True
