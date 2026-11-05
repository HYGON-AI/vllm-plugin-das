# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete concurrent evaluation flow without a model or network."""
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_complete_evaluation_saves_incremental_answers(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/eval_kimi_k3_humaneval_full.py"
    spec = importlib.util.spec_from_file_location("kimi_humaneval_full", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(task_id=f"HumanEval/{i}", prompt="def f(): pass", test="", entry_point="f")
            for i in range(164)]
    datasets = ModuleType("datasets")
    datasets.load_from_disk = lambda path: rows
    adapter = ModuleType("evalscope.benchmarks.humaneval.humaneval_adapter")
    adapter.HumanevalAdapter = SimpleNamespace(_postprocess=lambda value: value)
    utils = ModuleType("evalscope.benchmarks.humaneval.utils")
    utils.check_correctness = lambda **kwargs: {"passed": True}
    for name, value in {"datasets": datasets,
                        "evalscope.benchmarks.humaneval.humaneval_adapter": adapter,
                        "evalscope.benchmarks.humaneval.utils": utils}.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    lock = threading.Lock()
    active, peak = 0, 0

    def fake_request(base, path, payload=None):
        nonlocal active, peak
        if path == "/v1/models":
            return {"data": [{"id": "fake"}]}
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return {"choices": [{"message": {"content": "def f(): return 1"}}],
                "usage": {"completion_tokens": 4}}

    monkeypatch.setattr(module, "request", fake_request)
    monkeypatch.setattr(sys, "argv", ["eval", "--api-base", "http://unused",
                                      "--dataset", "unused", "--output", str(tmp_path)])
    assert module.main() == 0
    assert 1 < peak <= 32
    assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 164
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 164
    assert json.loads((tmp_path / "config.json").read_text())["concurrency"] == 32
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] == summary["total"] == 164
    assert summary["pass_at_1"] == 1.0
