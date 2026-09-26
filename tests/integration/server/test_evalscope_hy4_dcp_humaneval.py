# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Hy4 FP8 DCP2 launch and deterministic HumanEval validation harness."""

from __future__ import annotations

import gzip
import json
import os
import re
import shlex
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


MODEL = "/models/Hy4-preview-Channel-FP8-w8a8-v2"
DATASET = Path("/models/datasets/humaneval/HumanEval.jsonl.gz")
MODES = frozenset(
    {
        "control_target",
        "control_mtp3",
        "dcp_eager_target",
        "dcp_graph_target",
        "dcp_graph_mtp3",
        "dcp_dp4_graph_mtp3",
    }
)
PROMPT_TEMPLATE = (
    "Read the following function signature and docstring, and fully "
    "implement the function described. Your response should only contain "
    "the code for this function.\n\n{question}"
)
BASE_ARGS = [
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
    "--trust-remote-code",
    "--tensor-parallel-size",
    "8",
    "--enable-expert-parallel",
    "--moe-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--enable-prefix-caching",
    "--max-model-len",
    "4096",
    "--block-size",
    "64",
    "--max-num-seqs",
    "16",
    "--max-num-batched-tokens",
    "4096",
    "--default-chat-template-kwargs",
    '{"reasoning_effort":"no_think"}',
    "--seed",
    "0",
]
DCP_ARGS = [
    "--decode-context-parallel-size",
    "2",
    "--dcp-comm-backend",
    "ag_rs",
    "--cp-kv-cache-interleave-size",
    "1",
]
MTP_ARGS = [
    "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":3}',
]


def server_args(mode: str) -> list[str]:
    """Return the exact server argv for one validation mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown Hy4 DCP validation mode: {mode}")
    result = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
    result.extend(BASE_ARGS)
    if mode == "dcp_dp4_graph_mtp3":
        result[result.index("--tensor-parallel-size") + 1] = "2"
        result.extend(["--data-parallel-size", "4"])
    if mode.startswith("dcp_"):
        result.extend(DCP_ARGS)
    if mode == "dcp_eager_target":
        result.append("--enforce-eager")
    if mode in {
        "control_mtp3",
        "dcp_graph_mtp3",
        "dcp_dp4_graph_mtp3",
    }:
        result.extend(MTP_ARGS)
    return result


def server_env() -> dict[str, str]:
    """Return an isolated eight-HCU environment for the launcher."""
    env = os.environ.copy()
    for name in (
        "VLLM_PLUGINS",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(name, None)
    env.update(
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_NN": "1",
            "VLLM_DCP_Q_REPLICATE": "0",
            "VLLM_HCU_USE_CUSTOM_OPS": "1",
            "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM": "1",
            "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def humaneval(count: int) -> list[dict]:
    if count not in {8, 32}:
        raise ValueError("Hy4 DCP validation supports HumanEval counts 8 or 32")
    with gzip.open(DATASET, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    wanted = {f"HumanEval/{index}" for index in range(count)}
    selected = [row for row in rows if row["task_id"] in wanted]
    expected = [f"HumanEval/{index}" for index in range(count)]
    if [row["task_id"] for row in selected] != expected:
        raise RuntimeError(f"HumanEval source must contain ordered 0-{count - 1}")
    return selected


def request_payload(problem: dict) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(question=problem["prompt"]),
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": 0,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }


def request_humaneval(output_path: Path, count: int) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior result: {output_path}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for problem in humaneval(count):
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
        records.append(
            {
                "task_id": problem["task_id"],
                "completion": choices[0]["message"]["content"],
                "finish_reason": choices[0].get("finish_reason"),
                "usage": body.get("usage"),
            }
        )
        output_path.write_text(json.dumps(records, indent=2) + "\n")
        if records[-1]["finish_reason"] != "stop":
            raise RuntimeError(
                f"{problem['task_id']} finished with "
                f"{records[-1]['finish_reason']!r}; refusing accuracy claim"
            )


def validate_concurrent_records(records: list[dict]) -> None:
    """Require different request-local draft patterns, not global step IDs."""
    if [row.get("task_id") for row in records] != [
        "HumanEval/0", "HumanEval/3", "HumanEval/7"
    ]:
        raise RuntimeError("Concurrent task ids or order do not match")
    acceptances = []
    for row in records:
        if row.get("finish_reason") != "stop":
            raise RuntimeError(f"{row['task_id']} did not stop normally")
        spec_stats = (row.get("metrics") or {}).get("speculative_decoding") or {}
        steps = spec_stats.get("per_step_accepted")
        if not isinstance(steps, list) or not steps or not all(
            isinstance(step, int) and 0 <= step <= 3 for step in steps
        ):
            raise RuntimeError(f"{row['task_id']} lacks detailed MTP3 metrics")
        acceptances.append(steps)
    if not any(len(set(step_counts)) > 1 for step_counts in zip(*acceptances)):
        raise RuntimeError(
            "Concurrent requests did not show different request-local "
            "draft acceptance"
        )


def request_concurrent(output_path: Path) -> None:
    """Exercise concurrent MTP3 target verification with detailed metrics."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior result: {output_path}")
    problems = humaneval(8)
    selected = [problems[index] for index in (0, 3, 7)]
    start = threading.Barrier(len(selected))

    def submit(problem: dict) -> dict:
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(request_payload(problem)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start.wait(timeout=30)
        with opener.open(request, timeout=1800) as response:
            body = json.load(response)
        choice = body["choices"][0]
        return {
            "task_id": problem["task_id"],
            "completion": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "usage": body.get("usage"),
            "metrics": body.get("metrics"),
        }

    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        records = list(pool.map(submit, selected))
    output_path.write_text(json.dumps(records, indent=2) + "\n")
    validate_concurrent_records(records)


def normalize_completion(completion: str) -> str:
    opening = re.search(
        r"(?m)^[ \t]*```(?:python|py)?[ \t]*\r?\n", completion
    )
    if opening is None:
        return completion.strip()
    code = completion[opening.end() :]
    closing = re.search(r"(?m)^[ \t]*```[ \t]*$", code)
    return (code[: closing.start()] if closing else code).strip()


def validate_prediction_records(records: list[dict], problems: list[dict]) -> None:
    expected = [problem["task_id"] for problem in problems]
    if [record.get("task_id") for record in records] != expected:
        raise RuntimeError("Prediction file task ids or order do not match")
    for record in records:
        if record.get("finish_reason") != "stop":
            raise RuntimeError(
                f"{record['task_id']} finished with "
                f"{record.get('finish_reason')!r}; refusing accuracy claim"
            )


def score_humaneval(predictions_path: Path, report_path: Path) -> None:
    if os.environ.get("VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC") != "1":
        raise RuntimeError(
            "HumanEval runs generated code. Set "
            "VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC=1 after reviewing outputs."
        )
    records = json.loads(predictions_path.read_text())
    problems = humaneval(len(records))
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
        report_path.write_text(
            json.dumps(
                {
                    "num": len(results),
                    "passed": sum(bool(item["passed"]) for item in results),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


def test_dcp_launcher_keeps_graph_and_eager_distinct():
    eager = server_args("dcp_eager_target")
    graph = server_args("dcp_graph_target")
    control = server_args("control_target")

    assert "--enforce-eager" in eager
    assert "--enforce-eager" not in graph
    assert "--decode-context-parallel-size" not in control
    assert graph[graph.index("--decode-context-parallel-size") + 1] == "2"
    for args in (eager, graph, control):
        assert args.count("--kv-cache-dtype") == 1
        assert args.count("--tensor-parallel-size") == 1


def test_dcp_launcher_mtp3_is_the_only_graph_delta():
    target = server_args("dcp_graph_target")
    mtp = server_args("dcp_graph_mtp3")

    assert mtp == target + [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":3}',
    ]


def test_dcp_dp4_launcher_pins_requested_graph_topology():
    args = server_args("dcp_dp4_graph_mtp3")

    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--decode-context-parallel-size") + 1] == "2"
    assert args[args.index("--data-parallel-size") + 1] == "4"
    assert "--enable-expert-parallel" in args
    assert args[args.index("--speculative-config") + 1] == (
        '{"method":"mtp","num_speculative_tokens":3}'
    )
    assert "--enforce-eager" not in args
    for option in (
        "--tensor-parallel-size",
        "--decode-context-parallel-size",
        "--data-parallel-size",
        "--speculative-config",
    ):
        assert args.count(option) == 1


def test_dcp_launcher_pins_mrv2_query_sharding_and_eight_cards():
    env = server_env()

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert env["VLLM_USE_NN"] == "1"
    assert env["VLLM_DCP_Q_REPLICATE"] == "0"
    assert env["VLLM_HCU_USE_CUSTOM_OPS"] == "1"
    assert env["VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "VLLM_PLUGINS" not in env
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in env
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        assert name not in env


def test_dcp_launcher_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown Hy4 DCP validation mode"):
        server_args("dcp_magic")


def test_humaneval_subsets_are_ordered_and_unique():
    assert [row["task_id"] for row in humaneval(8)] == [
        f"HumanEval/{index}" for index in range(8)
    ]
    assert [row["task_id"] for row in humaneval(32)] == [
        f"HumanEval/{index}" for index in range(32)
    ]


def test_request_payload_is_deterministic():
    problem = humaneval(8)[0]
    payload = request_payload(problem)

    assert payload["messages"][0]["content"].endswith(problem["prompt"])
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 2048
    assert payload["seed"] == 0
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "no_think"}


def test_completion_normalization_removes_one_markdown_fence():
    assert normalize_completion("```python\ndef f():\n    return 1\n```\n") == (
        "def f():\n    return 1"
    )
    assert normalize_completion("  return 2  ") == "return 2"


def test_prediction_gate_rejects_wrong_order_and_truncation():
    problems = humaneval(8)
    records = [
        {
            "task_id": problem["task_id"],
            "completion": "pass",
            "finish_reason": "stop",
        }
        for problem in problems
    ]
    records[0], records[1] = records[1], records[0]
    with pytest.raises(RuntimeError, match="task ids or order"):
        validate_prediction_records(records, problems)

    records[0], records[1] = records[1], records[0]
    records[3]["finish_reason"] = "length"
    with pytest.raises(RuntimeError, match="HumanEval/3.*length"):
        validate_prediction_records(records, problems)


def test_concurrent_gate_requires_different_request_local_draft_acceptance():
    records = [
        {
            "task_id": f"HumanEval/{index}",
            "finish_reason": "stop",
            "metrics": {
                "speculative_decoding": {"per_step_accepted": accepted}
            },
        }
        for index, accepted in zip((0, 3, 7), ([3, 3, 2], [3, 1, 3], [3, 3, 3]))
    ]
    validate_concurrent_records(records)

    records[1]["metrics"]["speculative_decoding"]["per_step_accepted"] = [3, 3, 2]
    records[2]["metrics"]["speculative_decoding"]["per_step_accepted"] = [3, 3, 2]
    with pytest.raises(
        RuntimeError, match="different request-local draft acceptance"
    ):
        validate_concurrent_records(records)


def test_request_refuses_existing_result_file(tmp_path):
    output_path = tmp_path / "prior-run.json"
    output_path.write_text("prior result")

    with pytest.raises(FileExistsError, match="prior-run.json"):
        request_humaneval(output_path, 8)
    assert output_path.read_text() == "prior result"


def _print_server_command(mode: str) -> None:
    env = server_env()
    assigned = [
        f"{name}={shlex.quote(env[name])}"
        for name in (
            "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_USE_NN",
            "VLLM_DCP_Q_REPLICATE",
            "VLLM_HCU_USE_CUSTOM_OPS",
            "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
            "HIP_VISIBLE_DEVICES",
            "NO_PROXY",
            "no_proxy",
            "PYTHONNOUSERSITE",
        )
    ]
    command = assigned + [
        "env",
        "-u",
        "VLLM_PLUGINS",
        "-u",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "-u",
        "HTTP_PROXY",
        "-u",
        "HTTPS_PROXY",
        "-u",
        "ALL_PROXY",
        "-u",
        "http_proxy",
        "-u",
        "https_proxy",
        "-u",
        "all_proxy",
    ]
    command.extend(shlex.quote(argument) for argument in server_args(mode))
    print(" ".join(command))


if __name__ == "__main__":
    if sys.argv[1:2] == ["command"] and len(sys.argv) == 3:
        _print_server_command(sys.argv[2])
        raise SystemExit(0)
    if sys.argv[1:2] == ["request"] and len(sys.argv) == 4:
        request_humaneval(Path(sys.argv[2]), int(sys.argv[3]))
        raise SystemExit(0)
    if sys.argv[1:2] == ["score"] and len(sys.argv) == 4:
        score_humaneval(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    if sys.argv[1:2] == ["concurrent"] and len(sys.argv) == 3:
        request_concurrent(Path(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(
        "Usage: test_evalscope_hy4_dcp_humaneval.py "
        "command MODE | request OUTPUT.json 8|32 | "
        "score PREDICTIONS.json REPORT.json | concurrent OUTPUT.json"
    )
