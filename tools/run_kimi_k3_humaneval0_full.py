#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and score one full-length HumanEval/0 answer against vLLM."""
import argparse
import importlib.metadata
import json
import time
import urllib.request
from pathlib import Path

from datasets import load_from_disk
from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
from evalscope.benchmarks.humaneval.utils import check_correctness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        req = urllib.request.Request(
            args.api_base.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=3600) as response:
            return json.load(response)

    model = request("/v1/models")["data"][0]["id"]
    dataset = load_from_disk(str(args.dataset))
    problem = next(row for row in dataset if row["task_id"] == "HumanEval/0")
    template = (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for "
        "this function.\n{question}"
    )
    payload = dict(
        model=model,
        messages=[dict(role="user", content=template.format(question=problem["prompt"]))],
        temperature=0,
        top_p=1,
        seed=20261026,
        max_tokens=4096,
        chat_template_kwargs={"enable_thinking": False},
    )
    started = time.monotonic()
    response = request("/v1/chat/completions", payload)
    elapsed = time.monotonic() - started
    prediction = response["choices"][0]["message"]["content"] or ""
    completion = HumanevalAdapter._postprocess(prediction)
    outcome = check_correctness(problem=problem, completion=completion, timeout=4)
    result = dict(
        task_id="HumanEval/0",
        max_tokens=4096,
        elapsed_seconds=elapsed,
        evalscope=importlib.metadata.version("evalscope"),
        problem=problem,
        request=payload,
        response=response,
        prediction=prediction,
        extracted_code=completion,
        result=outcome,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(
        task_id=result["task_id"],
        finish_reason=response["choices"][0].get("finish_reason"),
        completion_tokens=response.get("usage", {}).get("completion_tokens"),
        elapsed_seconds=elapsed,
        result=outcome,
        output=str(args.output),
    ), ensure_ascii=False, indent=2))
    return 0 if outcome.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
