#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay the fixed workload and compare cached/uncached prefix continuations."""
import argparse
import json
import random
import math
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-base", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, raw=False):
        req = urllib.request.Request(args.api_base.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=3600) as response:
            content = response.read().decode()
            return content if raw else json.loads(content) if content.strip() else {}

    model = request("/v1/models")["data"][0]["id"]

    def completion(tokens, name, max_tokens=1):
        payload = dict(model=model, prompt=tokens, max_tokens=max_tokens,
                       temperature=0, seed=20261026, logprobs=5)
        started = time.monotonic()
        result = request("/v1/completions", payload)
        elapsed = time.monotonic() - started
        (args.output / f"{name}.json").write_text(json.dumps(dict(
            request=payload, response=result, seconds=elapsed), indent=2) + "\n")
        return result, elapsed

    warmup = json.loads((args.prompts / "warmup_4096.json").read_text())["prompt"]
    completion(warmup, "warmup")
    manifest = json.loads((args.prompts / "manifest.json").read_text())
    latencies = []
    if args.profile:
        request("/start_profile", {})
    try:
        for filename in manifest["serial_order"]:
            data = json.loads((args.prompts / filename).read_text())
            result, elapsed = completion(data["prompt"], Path(filename).stem)
            assert result["usage"]["prompt_tokens"] == len(data["prompt"])
            assert result["usage"]["completion_tokens"] == 1
            latencies.append(dict(request=filename, seconds=elapsed))
            print(filename, elapsed, flush=True)
    finally:
        if args.profile:
            request("/stop_profile", {})
    (args.output / "latencies.json").write_text(json.dumps(latencies, indent=2) + "\n")

    # Same prefix, different tail. Compare the exact request after warming its
    # sibling against a reset-cache reference, retaining top-5 logprobs too.
    generator = random.Random(20261026)
    base = [generator.randrange(100000, 150000) for _ in range(4096)]
    probe = base[:3072] + [generator.randrange(50000, 99999) for _ in range(1024)]
    reference, _ = completion(probe, "prefix_uncached", max_tokens=8)
    completion(base, "prefix_producer")
    before = request("/metrics", raw=True)
    cached, _ = completion(probe, "prefix_cached", max_tokens=8)
    after = request("/metrics", raw=True)
    (args.output / "metrics_before.txt").write_text(before)
    (args.output / "metrics_after.txt").write_text(after)
    ref_choice, choice = reference["choices"][0], cached["choices"][0]
    def hits(metrics):
        return sum(float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                   if line.startswith("vllm:prefix_cache_hits_total{"))
    hit_tokens = hits(after) - hits(before)
    ref_logprobs = ref_choice.get("logprobs") or {}
    cache_logprobs = choice.get("logprobs") or {}
    def max_logprob_delta():
        deltas = []
        for key in ("token_logprobs",):
            for left, right in zip(ref_logprobs.get(key) or [], cache_logprobs.get(key) or []):
                if left is not None and right is not None:
                    deltas.append(abs(left - right))
        for left_row, right_row in zip(ref_logprobs.get("top_logprobs") or [],
                                       cache_logprobs.get("top_logprobs") or []):
            deltas.extend(abs(left_row[token] - right_row[token])
                          for token in left_row.keys() & right_row.keys())
        return max(deltas, default=0.0)
    delta = max_logprob_delta()
    token_pieces_equal = (ref_logprobs.get("tokens") == cache_logprobs.get("tokens"))
    summary = dict(text_equal=ref_choice["text"] == choice["text"],
                   token_pieces_equal=token_pieces_equal,
                   max_common_logprob_delta=delta,
                   logprobs_uncached=ref_choice["logprobs"], logprobs_cached=choice["logprobs"],
                   prefix_hit_tokens=hit_tokens)
    (args.output / "prefix_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert summary["text_equal"], "prefix-hit generation differs from uncached reference"
    assert summary["token_pieces_equal"], "prefix-hit generated token pieces differ from uncached reference"
    assert math.isclose(delta, 0.0, abs_tol=0.005), f"prefix-hit logits diverged by {delta}"
    assert hit_tokens >= 3072, f"expected prefix hit >=3072 tokens, got {hit_tokens}"
    print("Prefix reuse passed:", hit_tokens, "cached tokens", flush=True)


if __name__ == "__main__":
    main()
