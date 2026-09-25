# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reproducible Hy4 MRV2 launch and HumanEval/0-7 request contract.

Run ``python this_file.py command {target,mtp,fp8}`` to print a shell-safe
server command, or ``python this_file.py request OUTPUT.json`` to save the
same eight deterministic OpenAI API responses for each server mode.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shlex
import sys
import urllib.request
from pathlib import Path

import pytest


MODEL = "/models/Hy4-preview-Channel-FP8-w8a8-v2"
DATASET = Path("/models/datasets/humaneval/HumanEval.jsonl.gz")
PROMPT_TEMPLATE = (
    "Read the following function signature and docstring, and fully "
    "implement the function described. Your response should only contain "
    "the code for this function.\n\n{question}"
)
BASE_ARGS = [
    "--host", "127.0.0.1", "--port", "8000", "--trust-remote-code",
    "--tensor-parallel-size", "8", "--moe-backend", "aiter",
    "--enable-prefix-caching", "--max-model-len", "4096",
    "--block-size", "64",  # FLASHMLA_SPARSE requires 64-token KV pages.
    "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
    "--default-chat-template-kwargs", '{"reasoning_effort":"no_think"}',
    "--seed", "0",
]
MTP_ARGS = [
    "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
]
FP8_KV_ARGS = ["--kv-cache-dtype", "fp8_e4m3"]


def server_args(mode: str) -> list[str]:
    if mode not in {"target", "mtp", "fp8"}:
        raise ValueError(f"Unknown Hy4 validation mode: {mode}")
    result = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
    result.extend(BASE_ARGS)
    if mode in {"mtp", "fp8"}:
        result.extend(MTP_ARGS)
    if mode == "fp8":
        result.extend(FP8_KV_ARGS)
    return result


def server_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTHONNOUSERSITE": "1",
    })
    return env


def humaneval8() -> list[dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    selected = [row for row in rows if row["task_id"] in {
        f"HumanEval/{i}" for i in range(8)
    }]
    if [row["task_id"] for row in selected] != [
        f"HumanEval/{i}" for i in range(8)
    ]:
        raise RuntimeError("HumanEval source must contain exactly ordered 0-7")
    return selected


def request_payload(problem: dict) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                question=problem["prompt"]
            )},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": 0,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }


def request_humaneval8(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior result: {output_path}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for problem in humaneval8():
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(request_payload(problem)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=1800) as response:
            body = json.load(response)
        choices = body.get("choices", [])
        if len(choices) != 1 or not isinstance(
            choices[0].get("message", {}).get("content"), str
        ):
            raise RuntimeError(f"Invalid completion for {problem['task_id']}")
        records.append({
            "task_id": problem["task_id"],
            "completion": choices[0]["message"]["content"],
            "finish_reason": choices[0].get("finish_reason"),
            "usage": body.get("usage"),
        })
        # Preserve completed requests if a later sample or server fails.
        output_path.write_text(json.dumps(records, indent=2) + "\n")
        if records[-1]["finish_reason"] != "stop":
            raise RuntimeError(
                f"{problem['task_id']} finished with "
                f"{records[-1]['finish_reason']!r}; refusing accuracy claim"
            )


def normalize_completion(completion: str) -> str:
    opening = re.search(
        r"(?m)^[ \t]*```(?:python|py)?[ \t]*\r?\n", completion
    )
    if opening is None:
        return completion.strip()
    code = completion[opening.end():]
    closing = re.search(r"(?m)^[ \t]*```[ \t]*$", code)
    return (code[:closing.start()] if closing else code).strip()


def validate_prediction_records(records: list[dict], problems: list[dict]) -> None:
    if [record.get("task_id") for record in records] != [
        problem["task_id"] for problem in problems
    ]:
        raise RuntimeError("Prediction file must contain HumanEval/0-7 in order")
    for record in records:
        if record.get("finish_reason") != "stop":
            raise RuntimeError(
                f"{record['task_id']} finished with "
                f"{record.get('finish_reason')!r}; refusing accuracy claim"
            )


def score_humaneval8(predictions_path: Path, report_path: Path) -> None:
    if os.environ.get("VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC") != "1":
        raise RuntimeError(
            "HumanEval runs generated code. Set "
            "VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC=1 after reviewing outputs."
        )
    records = json.loads(predictions_path.read_text())
    problems = humaneval8()
    validate_prediction_records(records, problems)
    from evalscope.benchmarks.humaneval.utils import check_correctness
    results = []
    for problem, record in zip(problems, records):
        result = check_correctness(
            problem,
            normalize_completion(record["completion"]),
            timeout=4,
        )
        results.append(result)
        report_path.write_text(json.dumps({
            "num": len(results),
            "passed": sum(bool(item["passed"]) for item in results),
            "results": results,
        }, indent=2) + "\n")


def test_args_mode_deltas_are_only_mtp_and_fp8_kv():
    target, mtp, fp8 = (server_args(mode) for mode in ("target", "mtp", "fp8"))
    assert mtp == target + MTP_ARGS
    assert fp8 == mtp + FP8_KV_ARGS
    assert target.count("--tensor-parallel-size") == 1
    assert target[target.index("--tensor-parallel-size") + 1] == "8"
    assert target[target.index("--block-size") + 1] == "64"


def test_args_pin_mrv2_and_eight_cards():
    env = server_env()
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "VLLM_PLUGINS" not in env


def test_humaneval_subset_has_eight_unique_ids():
    assert [row["task_id"] for row in humaneval8()] == [
        f"HumanEval/{i}" for i in range(8)
    ]


def test_request_payload_is_fixed_across_modes():
    payload = request_payload(humaneval8()[0])
    assert payload["messages"][0]["content"].endswith(
        humaneval8()[0]["prompt"]
    )
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 2048
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "no_think"}


def test_completion_normalization_removes_single_markdown_fence():
    assert normalize_completion("```python\ndef f():\n    return 1\n```\n") == (
        "def f():\n    return 1"
    )
    assert normalize_completion("  return 2  ") == "return 2"


def test_score_gate_rejects_truncated_prediction():
    problems = humaneval8()
    records = [
        {"task_id": problem["task_id"], "completion": "pass", "finish_reason": "stop"}
        for problem in problems
    ]
    records[3]["finish_reason"] = "length"
    with pytest.raises(RuntimeError, match="HumanEval/3.*length"):
        validate_prediction_records(records, problems)


def test_request_refuses_existing_result_file(tmp_path, monkeypatch):
    output_path = tmp_path / "prior-run.json"
    output_path.write_text("prior result")
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *_args, **_kwargs: pytest.fail("must reject before HTTP"),
    )
    with pytest.raises(FileExistsError, match="prior-run.json"):
        request_humaneval8(output_path)
    assert output_path.read_text() == "prior result"


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"] and len(sys.argv) == 4:
        score_humaneval8(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) != 3 or sys.argv[1] not in {"command", "request"}:
        raise SystemExit(
            "Usage: test_evalscope_hy4_humaneval.py "
            "command target|mtp|fp8 OR request OUTPUT.json OR "
            "score PREDICTIONS.json REPORT.json"
        )
    if sys.argv[1] == "request":
        request_humaneval8(Path(sys.argv[2]))
    else:
        env = server_env()
        assigned = [
            f"{name}={shlex.quote(env[name])}"
            for name in (
                "VLLM_USE_V2_MODEL_RUNNER", "HIP_VISIBLE_DEVICES",
                "NO_PROXY", "PYTHONNOUSERSITE",
            )
        ]
        print(" ".join(assigned + ["env", "-u", "VLLM_PLUGINS"] + [
            shlex.quote(arg) for arg in server_args(sys.argv[2])
        ]))
