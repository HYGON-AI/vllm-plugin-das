#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One deterministic candidate per HumanEval/0..7 using EvalScope's tests."""
import argparse
import hashlib
import importlib.metadata
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_from_disk
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
    from evalscope.benchmarks.humaneval.utils import check_correctness

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "samples.jsonl").exists():
        raise SystemExit("Refusing to overwrite existing candidates; use a new evaluation directory")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        req = urllib.request.Request(
            args.api_base.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=3600) as response:
            return json.load(response)

    model = request("/v1/models")["data"][0]["id"]
    template = ("Read the following function signature and docstring, and fully implement "
                "the function described. Your response should only contain the code for "
                "this function.\n{question}")
    generation = dict(temperature=0, top_p=1, seed=20261026, max_tokens=4096,
                      chat_template_kwargs={"enable_thinking": False})
    config = dict(model=model, prompt_template=template, generation=generation,
                  candidates_per_task=1, task_ids=[f"HumanEval/{i}" for i in range(8)],
                  review_timeout=4, evalscope=importlib.metadata.version("evalscope"),
                  datasets=importlib.metadata.version("datasets"))
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset = load_from_disk(str(args.dataset))
    tasks = {row["task_id"]: row for row in dataset}
    results = []
    for task_id in config["task_ids"]:
        task = tasks[task_id]
        payload = dict(model=model, messages=[dict(
            role="user", content=template.format(question=task["prompt"]))], **generation)
        started = time.monotonic()
        response = request("/v1/chat/completions", payload)
        prediction = response["choices"][0]["message"]["content"] or ""
        completion = HumanevalAdapter._postprocess(prediction)
        outcome = check_correctness(problem=task, completion=completion, timeout=4)
        result = dict(task_id=task_id, elapsed_seconds=time.monotonic() - started,
                      problem=task, request=payload, response=response,
                      prediction=prediction, extracted_code=completion, result=outcome,
                      prompt_sha256=hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest())
        with (args.output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        print(task_id, outcome, flush=True)
    passed = sum(bool(row["result"]["passed"]) for row in results)
    summary = dict(passed=passed, total=8, threshold=5, accepted=passed >= 5,
                   per_task=[dict(task_id=row["task_id"], result=row["result"]) for row in results])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
