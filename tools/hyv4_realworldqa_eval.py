# SPDX-License-Identifier: Apache-2.0
"""Evaluate a pinned RealWorldQA snapshot through the OpenAI chat API.

Use the unmodified lmms-eval MCQ extractor via --mcq-extractor; its revision
and checksum are recorded alongside dataset hashes and all raw responses.
"""

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image

DATASET_REVISION = "17e7f75e092e47169732462ea3cdfebe911105dd"
SCORER_REVISION = "7847ad8539889c87533b78c93e280ddfc117d16b"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(answer, prediction, extract_mcq):
    """Follow pinned lmms-eval realworldqa_process_results semantics."""
    target = answer.strip()
    if target.upper() in ("A", "B", "C", "D"):
        normalized = extract_mcq(prediction, choices=list("ABCD"))
        correct = normalized == target.upper()
    else:
        normalized = prediction.lower().strip().rstrip(".")
        correct = normalized == target.lower()
    return normalized, correct


def evaluate(index, row, args, extract_mcq):
    started = time.monotonic()
    image = row["image"]
    if isinstance(image, dict):
        source = io.BytesIO(image["bytes"]) if image.get("bytes") else image["path"]
        image = Image.open(source)
    image = image.convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    image_bytes = encoded.getvalue()
    # The question is passed verbatim; the target is used only for scoring.
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_bytes).decode()
                        },
                    },
                    {"type": "text", "text": row["question"].strip()},
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }
    result = {
        "index": index,
        "question": row["question"],
        "answer": row["answer"],
        "image_size": list(image.size),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    try:
        response = requests.post(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            json=body,
            timeout=args.timeout,
        )
        result["status"] = response.status_code
        result["response"] = response.json()
        choice = result["response"].get("choices", [{}])[0]
        prediction = choice.get("message", {}).get("content") or ""
        result["prediction"] = prediction
        result["finish_reason"] = choice.get("finish_reason")
        normalized, correct = score(row["answer"], prediction, extract_mcq)
        result["normalized_prediction"] = normalized
        result["correct"] = correct and response.status_code == 200
        result["strict_correct"] = (
            prediction.strip() == row["answer"].strip() and response.status_code == 200
        )
    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        result.update(correct=False, strict_correct=False, error=str(error))
    result["seconds"] = time.monotonic() - started
    return result


def summary(results, total, manifest):
    count = len(results)
    correct = sum(row["correct"] for row in results)
    rate = correct / count if count else 0
    z = 1.959963984540054
    denominator = 1 + z * z / count if count else 1
    center = (rate + z * z / (2 * count)) / denominator if count else 0
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
        / denominator
        if count
        else 0
    )
    groups = {}
    for label, selected in (
        (
            "multiple_choice",
            [r for r in results if r["answer"].strip().upper() in ("A", "B", "C", "D")],
        ),
        (
            "open_ended",
            [
                r
                for r in results
                if r["answer"].strip().upper() not in ("A", "B", "C", "D")
            ],
        ),
    ):
        hits = sum(row["correct"] for row in selected)
        groups[label] = {
            "correct": hits,
            "total": len(selected),
            "accuracy": hits / len(selected) if selected else None,
        }
    return {
        "manifest": manifest,
        "dataset_total": total,
        "completed": count,
        "complete": count == total,
        "correct": correct,
        "accuracy": rate,
        "wilson_95_percent_interval": [center - radius, center + radius],
        "strict_correct": sum(row["strict_correct"] for row in results),
        "http_errors": sum(row.get("status") != 200 for row in results),
        "truncated": sum(row.get("finish_reason") == "length" for row in results),
        "empty_answers": sum(not row.get("prediction") for row in results),
        "median_request_seconds": statistics.median([r["seconds"] for r in results])
        if count
        else None,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcq-extractor", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="hy4-vl")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1048576,
        help="Recorded server setting; must match its processor kwargs",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("mcq_extract", args.mcq_extractor)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    files = sorted(args.data_dir.glob("test-*.parquet"))
    if not files:
        raise ValueError("No RealWorldQA parquet shards found")
    rows = [row for path in files for row in pq.read_table(path).to_pylist()]
    if len(rows) != 765:
        raise ValueError(f"Expected the full 765-item test split, found {len(rows)}")
    manifest = {
        "plugin_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "evaluator_sha256": sha256(Path(__file__)),
        "dataset": "xai-org/RealworldQA",
        "revision": DATASET_REVISION,
        "dataset_sha256": {p.name: sha256(p) for p in files},
        "scorer": "EvolvingLMMs-Lab/lmms-eval",
        "scorer_revision": SCORER_REVISION,
        "extractor_sha256": sha256(args.mcq_extractor),
        "model": args.model,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "max_pixels": args.max_pixels,
        "reasoning_effort": "no_think",
        "concurrency": args.concurrency,
        "prompt": "original question verbatim; no added system or answer prompt",
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume settings or data differ from the existing manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    result_path = args.output_dir / "results.jsonl"
    results = (
        [json.loads(line) for line in result_path.read_text().splitlines()]
        if result_path.exists()
        else []
    )
    completed = {row["index"] for row in results}
    if len(completed) != len(results):
        raise ValueError("Duplicate evaluation indices in saved results")
    started = time.monotonic()
    with result_path.open("a") as stream, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [
            pool.submit(evaluate, i, row, args, extractor.extract_mcq_answer)
            for i, row in enumerate(rows)
            if i not in completed
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            report = summary(results, len(rows), manifest)
            (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "correct": report["correct"],
                        "accuracy": report["accuracy"],
                        "http_errors": report["http_errors"],
                        "last_index": result["index"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda row: row["index"])
    (args.output_dir / "wrong.json").write_text(
        json.dumps(
            [row for row in results if not row["correct"]], ensure_ascii=False, indent=2
        )
    )
    if any(row.get("status") != 200 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
