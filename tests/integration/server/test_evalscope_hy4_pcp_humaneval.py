# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Hy4 eager PCP launch contract and reproducible HumanEval entrypoint."""

from __future__ import annotations

import gzip
import json
import os
import shlex
import sys
import urllib.request
from pathlib import Path

import pytest

from tests.integration.server.test_evalscope_hy4_humaneval import (
    BASE_ARGS,
    DATASET,
    MODEL,
    normalize_completion,
    request_humaneval8,
    request_payload,
    score_humaneval8,
)


_MODES = {"tp4_target", "tp4_mtp1", "tp4_mtp2", "tp4_mtp3", "pp2_target", "pp2_mtp2"}


def _replace_option(args: list[str], name: str, value: str) -> None:
    index = args.index(name)
    args[index + 1] = value


def server_args(mode: str) -> list[str]:
    if mode not in _MODES:
        raise ValueError(f"Unknown Hy4 PCP validation mode: {mode}")
    args = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
    args.extend(BASE_ARGS)
    pp2 = mode.startswith("pp2_")
    _replace_option(args, "--tensor-parallel-size", "1" if pp2 else "4")
    if pp2:
        _replace_option(args, "--moe-backend", "deep_gemm")
        args.extend([
            "--pipeline-parallel-size", "2",
            "--prefill-context-parallel-size", "4",
            "--kv-cache-dtype", "fp8_e4m3",
            "--all2all-backend", "deepep_high_throughput",
        ])
    else:
        args.extend(["--prefill-context-parallel-size", "2"])
    args.extend([
        "--enable-expert-parallel", "--enforce-eager",
        "--attention-backend", "FLASHMLA_SPARSE",
    ])
    if "mtp" in mode:
        depth = mode.rsplit("mtp", 1)[1]
        args.extend([
            "--speculative-config",
            f'{{"method":"mtp","num_speculative_tokens":{depth}}}',
        ])
    return args


def server_env(mode: str) -> dict[str, str]:
    if mode not in _MODES:
        raise ValueError(f"Unknown Hy4 PCP validation mode: {mode}")
    env = os.environ.copy()
    for name in (
        "VLLM_PLUGINS", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "VLLM_PP_LAYER_PARTITION",
    ):
        env.pop(name, None)
    env.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTHONNOUSERSITE": "1",
    })
    if mode.startswith("pp2_"):
        env["VLLM_PP_LAYER_PARTITION"] = "41,37"
    return env


def humaneval32() -> list[dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    wanted = {f"HumanEval/{index}" for index in range(32)}
    selected = [row for row in rows if row["task_id"] in wanted]
    if [row["task_id"] for row in selected] != [
        f"HumanEval/{index}" for index in range(32)
    ]:
        raise RuntimeError("HumanEval source must contain exactly ordered 0-31")
    return selected


def request_humaneval32(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior result: {output_path}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for problem in humaneval32():
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
        output_path.write_text(json.dumps(records, indent=2) + "\n")
        if records[-1]["finish_reason"] != "stop":
            raise RuntimeError(
                f"{problem['task_id']} finished with "
                f"{records[-1]['finish_reason']!r}; refusing accuracy claim"
            )


def score_humaneval32(predictions_path: Path, report_path: Path) -> None:
    if os.environ.get("VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC") != "1":
        raise RuntimeError("Set VLLM_HCU_ALLOW_HUMANEVAL_CODE_EXEC=1 after reviewing outputs")
    records = json.loads(predictions_path.read_text())
    problems = humaneval32()
    if [record.get("task_id") for record in records] != [
        problem["task_id"] for problem in problems
    ]:
        raise RuntimeError("Prediction file must contain HumanEval/0-31 in order")
    from evalscope.benchmarks.humaneval.utils import check_correctness

    results = []
    for problem, record in zip(problems, records):
        if record.get("finish_reason") != "stop":
            raise RuntimeError(f"{problem['task_id']} did not stop normally")
        results.append(check_correctness(
            problem, normalize_completion(record["completion"]), timeout=4
        ))
        report_path.write_text(json.dumps({
            "num": len(results),
            "passed": sum(bool(item["passed"]) for item in results),
            "results": results,
        }, indent=2) + "\n")


def test_pcp_server_args_have_single_parallel_and_moe_selection():
    for mode in ("tp4_target", "tp4_mtp3", "pp2_target", "pp2_mtp2"):
        args = server_args(mode)
        assert args.count("--tensor-parallel-size") == 1
        assert args.count("--moe-backend") == 1
        assert args.count("--prefill-context-parallel-size") == 1
        assert "--enforce-eager" in args
        assert "--enable-expert-parallel" in args


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_pcp_server_args_pin_topology_and_mtp_depth(mode):
    args = server_args(mode)
    env = server_env(mode)
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "VLLM_PLUGINS" not in env
    assert args[args.index("--attention-backend") + 1] == "FLASHMLA_SPARSE"
    if mode.startswith("pp2_"):
        assert args[args.index("--tensor-parallel-size") + 1] == "1"
        assert args[args.index("--prefill-context-parallel-size") + 1] == "4"
        assert args[args.index("--pipeline-parallel-size") + 1] == "2"
        assert args[args.index("--moe-backend") + 1] == "deep_gemm"
        assert args[args.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert env["VLLM_PP_LAYER_PARTITION"] == "41,37"
    else:
        assert args[args.index("--tensor-parallel-size") + 1] == "4"
        assert args[args.index("--prefill-context-parallel-size") + 1] == "2"
        assert args[args.index("--moe-backend") + 1] == "aiter"
        assert "VLLM_PP_LAYER_PARTITION" not in env
    assert ("--speculative-config" in args) is ("mtp" in mode)


def test_humaneval32_has_ordered_0_to_31():
    assert [row["task_id"] for row in humaneval32()] == [
        f"HumanEval/{index}" for index in range(32)
    ]


if __name__ == "__main__":
    if sys.argv[1:2] == ["command"] and len(sys.argv) == 3:
        mode = sys.argv[2]
        env = server_env(mode)
        keys = [
            "VLLM_USE_V2_MODEL_RUNNER", "HIP_VISIBLE_DEVICES", "NO_PROXY",
            "no_proxy", "PYTHONNOUSERSITE",
        ]
        if mode.startswith("pp2_"):
            keys.append("VLLM_PP_LAYER_PARTITION")
        assignments = [f"{name}={shlex.quote(env[name])}" for name in keys]
        print(" ".join([
            "env", "-u", "VLLM_PLUGINS", "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY",
            "-u", "ALL_PROXY", "-u", "http_proxy", "-u", "https_proxy",
            "-u", "all_proxy", *assignments,
            *(shlex.quote(arg) for arg in server_args(mode)),
        ]))
    elif sys.argv[1:2] == ["request"] and len(sys.argv) == 3:
        request_humaneval8(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["score"] and len(sys.argv) == 4:
        score_humaneval8(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1:2] == ["request32"] and len(sys.argv) == 3:
        request_humaneval32(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["score32"] and len(sys.argv) == 4:
        score_humaneval32(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(
            "Usage: this_file.py command MODE | request OUT.json | "
            "score IN.json OUT.json | request32 OUT.json | "
            "score32 IN.json OUT.json"
        )
