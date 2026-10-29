#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare prefix-hit continuation with a cold request against a live vLLM API."""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit(f"Usage: {sys.argv[0]} <api-base> <output-json>")

api_base = sys.argv[1].rstrip("/")
output_path = Path(sys.argv[2])
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
shared_salt = "kda-fp32-h-shared-20261027"
cold_salt = "kda-fp32-h-cold-20261027"


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
    request("/start_profile", {})
    profile_active = True
    result["producer"] = completion(
        "producer_shared_salt", prefix + producer_tail, shared_salt
    )
    result["cached_consumer"] = completion(
        "consumer_shared_salt", prefix + consumer_tail, shared_salt
    )
    result["cold_consumer"] = completion(
        "consumer_cold_salt", prefix + consumer_tail, cold_salt
    )
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

if not result["comparison"]["token_ids_equal"]:
    raise SystemExit(1)
