#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare prefix-hit continuation with a cold request against a live vLLM API.

The server must be dedicated to this test: hit counters are global.
"""

from __future__ import annotations

import json
import argparse
import random
import sys
import time
import uuid

from check_kimi_k3_kda_prefix import compare_logprobs
import urllib.error
import urllib.request
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("api_base")
parser.add_argument("output_json", type=Path)
parser.add_argument("--profile", action="store_true", help="profile requests on a profiler-enabled server")
args = parser.parse_args()
api_base = args.api_base.rstrip("/")
output_path = args.output_json
output_path.parent.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path: str, payload=None, method="POST"):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{api_base}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with opener.open(req, timeout=3600) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} returned {exc.code}: {detail}") from exc
    return raw


def metrics_snippet():
    raw = request("/metrics", method="GET").decode("utf-8", "replace")
    return [line for line in raw.splitlines()
            if "prefix_cache" in line.lower() or "mamba" in line.lower()]


def summarize_response(response):
    choice = response["choices"][0]
    return {
        "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids"),
        "text": choice.get("text"),
        "logprobs": choice.get("logprobs"),
        "usage": response.get("usage"),
    }


model = json.loads(request("/v1/models", method="GET"))["data"][0]["id"]
rng = random.Random(20261027)
prefix = [rng.randrange(1000, 50000) for _ in range(3072)]
producer_tail = [rng.randrange(1000, 50000) for _ in range(1024)]
consumer_tail = [rng.randrange(1000, 50000) for _ in range(1024)]
shared_salt = uuid.uuid4().hex
cold_salt = uuid.uuid4().hex


def completion(name, prompt, salt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 20261027,
        "ignore_eos": True,
        "logprobs": 5,
        "return_token_ids": True,
        "cache_salt": salt,
    }
    started = time.monotonic()
    response = json.loads(request("/v1/completions", payload))
    return {
        "name": name,
        "elapsed_seconds": time.monotonic() - started,
        "prompt_tokens_expected": len(prompt),
        "response": summarize_response(response),
    }


result = {
    "model": model,
    "seed": 20261027,
    "prefix_tokens": len(prefix),
    "prompt_tokens": len(prefix) + len(producer_tail),
    "producer_suffix_sha256_input": "deterministic seed 20261027",
    "metrics_before": metrics_snippet(),
}
profile_active = False
try:
    if args.profile:
        request("/start_profile", {})
        profile_active = True
    result["producer"] = completion(
        "producer_shared_salt", prefix + producer_tail, shared_salt
    )
    result["metrics_before_cached"] = metrics_snippet()
    result["cached_consumer"] = completion(
        "consumer_shared_salt", prefix + consumer_tail, shared_salt
    )
    result["metrics_before_cold"] = metrics_snippet()
    result["cold_consumer"] = completion(
        "consumer_cold_salt", prefix + consumer_tail, cold_salt
    )
    result["metrics_after_cold"] = metrics_snippet()
finally:
    if profile_active:
        request("/stop_profile", {})

cached = result.get("cached_consumer", {}).get("response", {})
cold = result.get("cold_consumer", {}).get("response", {})
result["comparison"] = {
    "token_ids_equal": cached.get("token_ids") == cold.get("token_ids"),
    "token_logprobs_equal": (
        (cached.get("logprobs") or {}).get("token_logprobs")
        == (cold.get("logprobs") or {}).get("token_logprobs")
    ),
    "cached_token_ids": cached.get("token_ids"),
    "cold_token_ids": cold.get("token_ids"),
    "cached_token_logprobs": (cached.get("logprobs") or {}).get("token_logprobs"),
    "cold_token_logprobs": (cold.get("logprobs") or {}).get("token_logprobs"),
}
result["metrics_after"] = metrics_snippet()
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "output": str(output_path),
    "comparison": result["comparison"],
    "producer_usage": result["producer"]["response"].get("usage"),
    "cached_usage": cached.get("usage"),
    "cold_usage": cold.get("usage"),
}, ensure_ascii=False, indent=2))

def hits(lines):
    return sum(float(line.rsplit(None, 1)[-1]) for line in lines
               if line.startswith("vllm:prefix_cache_hits_total{")
               or line.startswith("vllm:prefix_cache_hits_total "))

try:
    before, middle, after = (result[k] for k in
        ("metrics_before_cached", "metrics_before_cold", "metrics_after_cold"))
    if not all(any(line.startswith("vllm:prefix_cache_hits_total") for line in x)
               for x in (before, middle, after)):
        raise ValueError("missing prefix cache hit metrics")
    cached_hits, cold_hits = hits(middle) - hits(before), hits(after) - hits(middle)
    ids = cached.get("token_ids")
    if not isinstance(ids, list) or not ids or ids != cold.get("token_ids"):
        raise ValueError("missing or mismatched token IDs")
    delta = compare_logprobs(cached.get("logprobs"), cold.get("logprobs"))
    if len(ids) != len(cached["logprobs"]["tokens"]):
        raise ValueError("token IDs and logprobs have different lengths")
    if cached_hits < len(prefix) or cold_hits != 0 or delta > 0.005:
        raise ValueError(f"unexpected cache hits or logprob delta: {cached_hits}, {cold_hits}, {delta}")
except (ValueError, TypeError) as exc:
    print(f"Verification failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
