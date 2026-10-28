#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one candidate per HumanEval problem with bounded request concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import time
import urllib.request
from pathlib import Path


def request(api_base, path, payload=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        api_base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=3600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    from datasets import load_from_disk
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
    from evalscope.benchmarks.humaneval.utils import check_correctness

    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("responses.jsonl", "samples.jsonl", "config.json", "summary.json")):
        raise SystemExit("Refusing to overwrite evaluation results; use a new output directory")
    dataset = list(load_from_disk(str(args.dataset)))
    ids = [row["task_id"] for row in dataset]
    expected = {f"HumanEval/{i}" for i in range(164)}
    if len(ids) != 164 or set(ids) != expected:
        raise SystemExit("Expected all 164 unique HumanEval tasks; check --dataset")
    dataset.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    model = request(args.api_base, "/v1/models")["data"][0]["id"]
    template = ("Read the following function signature and docstring, and fully implement "
                "the function described. Your response should only contain the code for "
                "this function.\n{question}")
    generation = dict(temperature=0, top_p=1, seed=20261026, max_tokens=4096,
                      chat_template_kwargs={"enable_thinking": False})
    config = dict(model=model, prompt_template=template, generation=generation,
                  candidates_per_task=1, task_ids=ids, concurrency=args.concurrency,
                  review_timeout=4, evalscope=importlib.metadata.version("evalscope"),
                  datasets=importlib.metadata.version("datasets"))
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def generate(task):
        payload = dict(model=model, messages=[dict(
            role="user", content=template.format(question=task["prompt"]))], **generation)
        started = time.monotonic()
        response = request(args.api_base, "/v1/chat/completions", payload)
        return payload, response, time.monotonic() - started

    results = []
    evaluation_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate, task): task for task in dataset}
        with (args.output / "responses.jsonl").open("w") as responses, \
                (args.output / "samples.jsonl").open("w") as stream:
            for future in as_completed(futures):
                task = futures[future]
                task_id = task["task_id"]
                try:
                    payload, response, elapsed = future.result()
                    prediction = response["choices"][0]["message"]["content"] or ""
                    response_record = dict(task_id=task_id, elapsed_seconds=elapsed,
                                           request=payload, response=response,
                                           prediction=prediction)
                    responses.write(json.dumps(response_record, ensure_ascii=False) + "\n")
                    responses.flush()
                    print(f"\n===== {task_id} answer =====\n{prediction}\n"
                          f"===== end {task_id} answer =====", flush=True)
                    completion = HumanevalAdapter._postprocess(prediction)
                    outcome = check_correctness(problem=task, completion=completion, timeout=4)
                    result = dict(task_id=task_id, elapsed_seconds=elapsed,
                                  problem=task, request=payload, response=response,
                                  prediction=prediction, extracted_code=completion, result=outcome,
                                  prompt_sha256=hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest())
                except Exception as exc:
                    result = dict(task_id=task_id, error=f"{type(exc).__name__}: {exc}")
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(result)
                print(task_id, result.get("result", result.get("error")), flush=True)
    results.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    errors = [row for row in results if "error" in row]
    passed = sum(bool(row.get("result", {}).get("passed")) for row in results)
    elapsed = time.monotonic() - evaluation_started
    completion_tokens = sum(row.get("response", {}).get("usage", {}).get("completion_tokens", 0)
                            for row in results)
    summary = dict(passed=passed, total=len(dataset), completed=len(results) - len(errors),
                   errors=len(errors), pass_at_1=passed / len(dataset) if not errors else None,
                   wall_seconds=elapsed, completion_tokens=completion_tokens,
                   completion_tokens_per_wall_second=completion_tokens / elapsed if elapsed else None,
                   per_task=[dict(task_id=row["task_id"], result=row.get("result"),
                                  error=row.get("error")) for row in results])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
