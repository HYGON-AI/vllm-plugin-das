# SPDX-License-Identifier: Apache-2.0
"""Measure streaming TTFT, latency and throughput on fixed RealWorldQA items.

Disable prefix and MM processor caches; the GPU encoder LRU remains enabled.
Inputs are encoded before timing. Measurements start before HTTP POST.
"""

import argparse
import base64
import io
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def request(body, base_url):
    started = time.perf_counter()
    first = last = None
    content = []
    usage = None
    finish = None
    with requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                token = choice.get("delta", {}).get("content")
                if token:
                    last = time.perf_counter()
                    first = last if first is None else first
                    content.append(token)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    ended = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("Missing streamed answer or token usage")
    return {
        "ttft_seconds": first - started,
        "latency_seconds": ended - started,
        # Text deltas can contain multiple tokens, and usage can include EOS.
        # Report the observed content-event interval rather than token latency.
        "content_chunks": len(content),
        "content_chunk_interval_seconds": (last - first) / (len(content) - 1)
        if len(content) > 1
        else None,
        "usage": usage,
        "answer": "".join(content),
        "finish_reason": finish,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="hy4-vl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 8:
        parser.error("--samples must be at least 8 for the two-image suite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for p in sorted(args.data_dir.glob("test-*.parquet"))
        for row in pq.read_table(p).to_pylist()
    ]
    indices = random.Random(338).sample(range(len(rows)), args.samples)
    urls = {}
    for index in indices:
        image = Image.open(io.BytesIO(rows[index]["image"]["bytes"])).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        urls[index] = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )

    def body(index, second=None, long=False):
        content = [{"type": "image_url", "image_url": {"url": urls[index]}}]
        if second is not None:
            content.append({"type": "image_url", "image_url": {"url": urls[second]}})
        prompt = rows[index]["question"].strip()
        if long:
            prompt = (
                "Describe the image in detail."
                if second is None
                else "Describe both images in detail, in their original order."
            )
        content.append({"type": "text", "text": prompt})
        result = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 32 if long else 16,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        }
        if long:
            result.update(min_tokens=32, ignore_eos=True)
        return result

    manifest = {
        "seed": 338,
        "indices": indices,
        "rounds": args.rounds,
        "max_pixels": 1048576,
        "prefix_cache": False,
        "mm_processor_cache_gb": 0,
        "encoder_embedding_cache": "default vLLM LRU; embeddings may be reused",
        "timing": "streaming HTTP; payload prepared before timing",
        "content_chunk_interval": "nonempty text deltas; not time per token",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for index in indices[:2]:
        request(body(index), args.base_url)
    suites = [
        ("qa_c1", 1, [(i, None, False) for _ in range(args.rounds) for i in indices]),
        ("qa_c2", 2, [(i, None, False) for _ in range(args.rounds) for i in indices]),
        ("caption32_c1", 1, [(i, None, True) for i in indices[:4]]),
        ("two_images32_c1", 1, [(indices[i], indices[i + 4], True) for i in range(4)]),
    ]
    reports = {}
    for name, concurrency, cases in suites:
        payloads = [body(i, second=j, long=long) for i, j, long in cases]
        start = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(
                pool.map(lambda payload: request(payload, args.base_url), payloads)
            )
        elapsed = time.perf_counter() - start
        for result, (i, j, _) in zip(results, cases):
            result.update(index=i, second_index=j, reference=rows[i]["answer"])
        intervals = [
            r["content_chunk_interval_seconds"]
            for r in results
            if r["content_chunk_interval_seconds"] is not None
        ]
        report = {
            "requests": len(results),
            "concurrency": concurrency,
            "wall_seconds": elapsed,
            "requests_per_second": len(results) / elapsed,
            "output_tokens_per_second": sum(
                r["usage"]["completion_tokens"] for r in results
            )
            / elapsed,
            "ttft_p50_seconds": statistics.median(r["ttft_seconds"] for r in results),
            "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in results], 0.95),
            "latency_p50_seconds": statistics.median(
                r["latency_seconds"] for r in results
            ),
            "latency_p95_seconds": percentile(
                [r["latency_seconds"] for r in results], 0.95
            ),
            "content_chunk_interval_median_seconds": statistics.median(intervals)
            if intervals
            else None,
        }
        reports[name] = report
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        (args.output_dir / "summary.json").write_text(json.dumps(reports, indent=2))
        print(name, json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
