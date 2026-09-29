#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Run and score deterministic HumanEval/0-7 against a Qwen3 API server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/model/Qwen3-8B")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    sys.path.insert(0, str(args.human_eval_root))
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: args.output_dir / name
        for name in (
            "problems.jsonl",
            "responses.jsonl",
            "samples.jsonl",
            "summary.json",
        )
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prior evidence: {existing}")

    problems = list(read_problems().values())[:8]
    expected_tasks = [f"HumanEval/{index}" for index in range(8)]
    if [problem["task_id"] for problem in problems] != expected_tasks:
        raise RuntimeError("HumanEval source must contain ordered tasks 0 through 7")
    write_jsonl(str(outputs["problems.jsonl"]), problems)

    opener = build_opener(ProxyHandler({}))
    samples = []
    elapsed_total = 0.0
    completion_tokens = 0
    with outputs["responses.jsonl"].open("w", encoding="utf-8") as raw:
        for problem in problems:
            body = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Complete the following Python function. Return only "
                            "the complete Python code, including required imports "
                            "and the function definition.\n\n" + problem["prompt"]
                        ),
                    }
                ],
                "temperature": 0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.monotonic()
            request = Request(
                f"{args.base_url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(request, timeout=600) as response:
                result = json.load(response)
            elapsed = time.monotonic() - started
            elapsed_total += elapsed
            usage = result.get("usage") or {}
            completion_tokens += int(usage.get("completion_tokens") or 0)
            record = {
                "task_id": problem["task_id"],
                "request": body,
                "response": result,
                "elapsed_s": elapsed,
            }
            raw.write(json.dumps(record) + "\n")
            raw.flush()

            choices = result.get("choices") or []
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise RuntimeError(f"invalid completion for {problem['task_id']}")
            content = choices[0].get("message", {}).get("content") or ""
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
            code = blocks[0] if blocks else content
            samples.append(
                {
                    "task_id": problem["task_id"],
                    "completion": "\n\n" + code.strip() + "\n",
                }
            )
            print(problem["task_id"], f"{elapsed:.2f}s", usage, flush=True)

    write_jsonl(str(outputs["samples.jsonl"]), samples)
    score = evaluate_functional_correctness(
        str(outputs["samples.jsonl"]),
        k=[1],
        n_workers=1,
        timeout=10,
        problem_file=str(outputs["problems.jsonl"]),
    )
    summary = {
        "tasks": expected_tasks,
        "temperature": 0,
        "max_tokens": 2048,
        "enable_thinking": False,
        "elapsed_s": elapsed_total,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed_total,
        **score,
    }
    outputs["summary.json"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
