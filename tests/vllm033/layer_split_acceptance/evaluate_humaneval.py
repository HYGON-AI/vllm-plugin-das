#!/usr/bin/env python3
"""Run the existing EvalScope HumanEval protocol against the FP4 HTTP server.

Adapted from v033-static-eplb-20261109/tests/vllm033/evaluate_humaneval.py.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


HUMANEVAL_TASKS = 32


def sample_results(output, expected_count):
    """Verify every judged task and retain HTTP/truncation evidence separately."""
    rows = {}
    files = list((output/'reviews').glob('**/humaneval_*.jsonl'))
    if len(files) != 1:
        raise RuntimeError('Expected exactly one HumanEval review file')
    for line in files[0].read_text().splitlines():
        sample = json.loads(line)['sample_score']
        task = sample['sample_metadata']['task_id']
        score = sample['score']
        passed = score['value']['acc']
        if task in rows or type(passed) is not bool or score['status'] != 'success':
            raise RuntimeError('Duplicate, incomplete or unjudged HumanEval sample')
        rows[task] = passed
    expected = {'HumanEval/'+str(i) for i in range(expected_count)}
    if set(rows) != expected:
        raise RuntimeError('HumanEval review IDs do not cover the complete requested set')
    files = list((output/'predictions').glob('**/humaneval_*.jsonl'))
    if len(files) != 1:
        raise RuntimeError('Expected exactly one HumanEval prediction file')
    seen, truncated, content_hashes = set(), [], {}
    for line in files[0].read_text().splitlines():
        row = json.loads(line)
        task, result = row['metadata']['task_id'], row['model_output']
        if task in seen or result.get('error') or len(result.get('choices', [])) != 1:
            raise RuntimeError('Duplicate, failed or missing HTTP completion: '+task)
        seen.add(task)
        choice = result['choices'][0]
        if choice['stop_reason'] == 'length':
            truncated.append(task)
        elif choice['stop_reason'] != 'stop':
            raise RuntimeError('Unexpected completion stop reason: '+str(choice['stop_reason']))
        content = choice['message']['content']
        if not isinstance(content, str):
            raise RuntimeError('Missing completion text: '+task)
        content_hashes[task] = hashlib.sha256(content.encode()).hexdigest()
    if seen != expected:
        raise RuntimeError('HumanEval prediction IDs do not cover the complete requested set')
    return dict(task_results=rows, truncated_task_ids=sorted(truncated), http_errors=0,
                completion_sha256=content_hashes)


def summarize(output, expected_count):
    reports = list((output / "reports").glob("**/humaneval.json"))
    if len(reports) != 1:
        raise RuntimeError("Expected one HumanEval report, found " + str(len(reports)))
    report = json.loads(reports[0].read_text())
    matches = [metric for metric in report["metrics"]
               if metric.get("identity", {}).get("aggregation") == "pass_at_k"
               and metric["identity"].get("dimensions", {}).get("k") == 1]
    if len(matches) != 1 or matches[0]["num"] != expected_count:
        raise RuntimeError("HumanEval pass@1 report has incomplete sample coverage")
    metric = matches[0]
    score = metric["score"]
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise RuntimeError("Invalid HumanEval pass@1 score")
    for category in metric.get("categories", []):
        for subset in category.get("subsets", []):
            coverage = subset.get("metadata") or {}
            if coverage.get("excluded", 0) or coverage.get("coverage", 1) != 1:
                raise RuntimeError("HumanEval report excluded samples")
    samples = sample_results(output, expected_count)
    if abs(sum(samples['task_results'].values())/expected_count-score) > 1e-9:
        raise RuntimeError('Aggregate HumanEval score differs from per-task results')
    summary = {"benchmark": "HumanEval", "indexer": "FP4 LightOp",
               "num_tasks": expected_count, "samples_per_task": 1,
               "pass_at_1": score, "pass_at_1_percent": score * 100,
               "report": str(reports[0]), "full_suite": expected_count == 164,
               "dataset_sha256": json.loads((output/'manifest.json').read_text())['dataset_sha256'],
               **samples}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate dataset without HTTP requests")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=HUMANEVAL_TASKS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.max_tokens, args.limit) < 1 or args.limit > 164:
        parser.error("Positive parameters required; limit must be at most 164")
    os.environ.update(HCU_ADAPTER_HOST="none", VLLM_PLUGINS="__disabled__",
                      HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
                      NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    data = args.data.resolve()
    parquet = data / "openai_humaneval/test-00000-of-00001.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    from evalscope.config import TaskConfig
    from evalscope.api.registry import get_benchmark
    import evalscope.benchmarks.humaneval.humaneval_adapter  # Register the standard benchmark.

    config = TaskConfig(
        model=args.model, api_url=args.api_url, api_key="EMPTY", eval_type="openai_api",
        datasets=["humaneval"], dataset_args={"humaneval": {"local_path": str(data)}},
        dataset_dir=str(data.parent / "cache"), eval_batch_size=args.batch_size,
        limit=args.limit, repeats=1, seed=42,
        generation_config={"temperature": 0, "max_tokens": args.max_tokens,
                           "timeout": 1800, "stream": True,
                           "extra_body": {"chat_template_kwargs": {"thinking": False}}},
        work_dir=str(args.output), no_timestamp=True, ignore_errors=False,
    )
    adapter = get_benchmark("humaneval", config)
    datasets = adapter.load_dataset()
    samples = [sample for dataset in datasets.values() for sample in dataset]
    ids = [sample.metadata["task_id"] for sample in samples]
    if ids != ["HumanEval/" + str(index) for index in range(args.limit)]:
        raise RuntimeError("Unexpected HumanEval task IDs or task order")
    manifest = {"dataset_source": str(parquet),
                "evalscope_version": importlib.metadata.version("evalscope"),
                "evaluator_python": sys.executable,
                "dataset_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                "task_ids": ids, "config": config.model_dump(mode="json"),
                "input_policy": "Original HumanEval prompts; no artificial padding to 32768 tokens",
                "service_lifecycle": "Owned by layer_split_acceptance/run.py"}
    print("DATASET", json.dumps({"count": len(ids), "sha256": manifest["dataset_sha256"],
          "evalscope_version": manifest["evalscope_version"]}), flush=True)
    if args.check:
        print("CHECK_COMPLETE", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    from evalscope import run_task
    run_task(task_cfg=config)
    print("RESULT", json.dumps(summarize(args.output, args.limit)), flush=True)


if __name__ == "__main__":
    main()
