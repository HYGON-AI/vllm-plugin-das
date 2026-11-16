#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare cached/uncached continuations; server needs VLLM_SERVER_DEV_MODE=1.

The server must be dedicated to this check: it resets the shared prefix cache.
"""
import argparse
import json
import random
import math
import time
import urllib.request
from pathlib import Path


def prefix_probe(request, completion, base, probe):
    """Compare a cold probe with reuse caused only by its producer."""
    def reset():
        result = request("/reset_prefix_cache", {})
        if result.get("success") is not True:
            raise RuntimeError(f"prefix cache reset failed: {result!r}")

    reset()
    reference, _ = completion(probe, "prefix_uncached", max_tokens=8)
    reset()
    completion(base, "prefix_producer")
    before = request("/metrics", raw=True)
    cached, _ = completion(probe, "prefix_cached", max_tokens=8)
    after = request("/metrics", raw=True)
    return reference, cached, before, after


def compare_logprobs(reference, cached):
    """Validate complete per-token evidence and compare common top candidates."""
    if not isinstance(reference, dict) or not isinstance(cached, dict):
        raise ValueError("missing logprobs")
    tokens = reference.get("tokens")
    other_tokens = cached.get("tokens")
    if not isinstance(tokens, list) or not tokens or tokens != other_tokens:
        raise ValueError("missing or mismatched generated tokens")
    count = len(tokens)
    deltas = []

    def add_delta(left, right):
        if (not isinstance(left, (int, float)) or isinstance(left, bool)
                or not isinstance(right, (int, float)) or isinstance(right, bool)
                or not math.isfinite(left) or not math.isfinite(right)):
            raise ValueError("missing or non-finite logprob")
        deltas.append(abs(left - right))

    for key in ("token_logprobs", "top_logprobs"):
        left_rows, right_rows = reference.get(key), cached.get(key)
        if (not isinstance(left_rows, list) or not isinstance(right_rows, list)
                or len(left_rows) != count or len(right_rows) != count):
            raise ValueError(f"missing or truncated {key}")
        for left, right in zip(left_rows, right_rows):
            if key == "token_logprobs":
                add_delta(left, right)
            else:
                if not isinstance(left, dict) or not isinstance(right, dict):
                    raise ValueError("missing top_logprobs row")
                common = left.keys() & right.keys()
                if not common:
                    raise ValueError("top_logprobs rows have no common tokens")
                for token in common:
                    add_delta(left[token], right[token])
    return max(deltas)


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
    reference, cached, before, after = prefix_probe(request, completion, base, probe)
    (args.output / "metrics_before.txt").write_text(before)
    (args.output / "metrics_after.txt").write_text(after)
    ref_choice, choice = reference["choices"][0], cached["choices"][0]
    def hits(metrics):
        return sum(float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                   if line.startswith("vllm:prefix_cache_hits_total{"))
    hit_tokens = hits(after) - hits(before)
    ref_logprobs = ref_choice.get("logprobs")
    cache_logprobs = choice.get("logprobs")
    delta = compare_logprobs(ref_logprobs, cache_logprobs)
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
