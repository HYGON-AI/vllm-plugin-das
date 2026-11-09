# SPDX-License-Identifier: Apache-2.0
"""Explicit live-server PCP8/Head/MTP prefix-cache smoke probe.

CPU-safe to import; this client never starts/stops a server or resets caches.
Run only against a dedicated, caller-owned GLM5Next PCP8 eager server with
Head mode 0 or 1, MTP3, prefix caching and --enable-prompt-tokens-details enabled.
Exact greedy token replay is a conservative regression check, not a proof of
floating-point identity or a diagnosis of the cause of any observed mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$')
DEFAULT_PREFIX_TOKENS = 10240
DEFAULT_OUTPUT_TOKENS = 256
CASES = ("A_cold", "A_warm", "B_shared", "B_warm", "B_cold")


def counters(text: str) -> dict[str, float]:
    """Read only exact counter names; never count histogram/per-position rows."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.fullmatch(line.strip())
        if match and match[1] in COUNTERS:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise AssertionError(f"invalid counter sample: {line}")
            result[match[1]] = result.get(match[1], 0.0) + value
    missing = set(COUNTERS) - result.keys()
    if missing:
        raise AssertionError(f"required metrics missing: {sorted(missing)}")
    return result


def completion_payload(model: str, prompt: list[int], output_tokens: int,
                       *, cache_salt: str | None = None) -> dict:
    if not model or not prompt or output_tokens < 16:
        raise ValueError("model, tokenized prompt and at least 16 output tokens required")
    payload = {"model": model, "prompt": list(prompt), "max_tokens": output_tokens,
               "temperature": 0, "seed": 42, "ignore_eos": True,
               "return_token_ids": True, "add_special_tokens": False, "stream": False}
    if cache_salt is not None:
        if not isinstance(cache_salt, str) or not cache_salt:
            raise ValueError("cache_salt must be a non-empty string")
        payload["cache_salt"] = cache_salt
    return payload


def check_completion(body: dict, prompt_length: int, output_tokens: int) -> dict:
    if "error" in body:
        raise AssertionError(f"completion error: {body['error']}")
    choices = body.get("choices", [])
    if len(choices) != 1:
        raise AssertionError("exactly one completion choice required")
    choice = choices[0]
    usage = body.get("usage", {})
    if usage.get("prompt_tokens") != prompt_length:
        raise AssertionError("server prompt length differs from supplied token IDs")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if type(cached) is not int or not 0 <= cached < prompt_length:
        raise AssertionError("missing/invalid cached_tokens; enable --enable-prompt-tokens-details")
    token_ids = choice.get("token_ids")
    if (not isinstance(token_ids, list) or len(token_ids) != output_tokens
            or any(type(t) is not int for t in token_ids)
            or usage.get("completion_tokens") != output_tokens):
        raise AssertionError("multi-step decode/verify was not completed with returned token IDs")
    if choice.get("finish_reason") != "length":
        raise AssertionError("expected length finish with ignore_eos=True")
    return {"prompt_tokens": prompt_length, "cached_tokens": cached,
            "completion_tokens": output_tokens, "token_ids": token_ids}


def check_replays(rows: list[dict], minimum_hit: int) -> None:
    if (len(rows) != len(CASES) or minimum_hit <= 0
            or tuple(row.get("label") for row in rows) != CASES):
        raise ValueError("five sequential A-cold/A-warm/B-shared/B-warm/B-cold results required")
    shared_salt = rows[0].get("cache_salt")
    cold_salt = rows[4].get("cache_salt")
    if (not isinstance(shared_salt, str) or not shared_salt
            or any(row.get("cache_salt") != shared_salt for row in rows[:4])
            or not isinstance(cold_salt, str) or not cold_salt or cold_salt == shared_salt):
        raise AssertionError("cold B must use an independent cache_salt; A and cached B must share one")
    if any(rows[index]["cached_tokens"] != 0 for index in (0, 4)):
        raise AssertionError("cold request unexpectedly hit cache; use fresh independent cache salts")
    if any(rows[index]["cached_tokens"] < minimum_hit for index in (1, 2, 3)):
        raise AssertionError("warm/shared-prefix requests did not actually reuse enough cached tokens")
    if rows[0]["token_ids"] != rows[1]["token_ids"]:
        raise AssertionError("cold/warm A token IDs differ; investigate numerics/cache state before accepting")
    for index, mode in ((2, "shared"), (3, "warm")):
        if rows[4]["token_ids"] != rows[index]["token_ids"]:
            raise AssertionError(f"cold/{mode} B token IDs differ; investigate numerics/cache state before accepting")


def validate_prompts(data: Any) -> dict:
    """Validate the exact token IDs and output length shared across Head modes."""
    fields = {"version", "prefix_tokens", "output_tokens", "prompt_a", "prompt_b"}
    if not isinstance(data, dict) or set(data) != fields or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("prompts file must be a version=1 token-ID manifest")
    prefix = data["prefix_tokens"]
    output = data["output_tokens"]
    if type(prefix) is not int or prefix <= 0 or type(output) is not int or output < 16:
        raise ValueError("invalid prefix/output lengths in prompts file")
    for name in ("prompt_a", "prompt_b"):
        tokens = data[name]
        if (not isinstance(tokens, list) or len(tokens) <= prefix
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("prompts file must contain nonnegative integer token IDs and branch suffixes")
    if data["prompt_a"][:prefix] != data["prompt_b"][:prefix] or data["prompt_a"] == data["prompt_b"]:
        raise ValueError("branch prompts must share the declared prefix and have different suffixes")
    return data


def counter_delta(before: dict, after: dict) -> dict:
    delta = {name: after[name] - before[name] for name in COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise AssertionError("metrics reset during probe; server may have restarted")
    return delta


def check_activity(delta: dict) -> None:
    if any(delta[name] <= 0 for name in COUNTERS):
        raise AssertionError("expected positive prefix-hit, drafted-token and accepted-token deltas")
    if delta[COUNTERS[2]] > delta[COUNTERS[1]]:
        raise AssertionError("accepted draft tokens exceed drafted tokens")


def http(base_url: str, path: str, payload: dict | None, timeout: int) -> Any:
    request = Request(base_url.rstrip("/") + path,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                      method="GET" if payload is None else "POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
    except HTTPError as exc:
        raise AssertionError(f"{path}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:1000]}") from exc
    return raw if path == "/metrics" else json.loads(raw)


def run_probe(base_url: str, model: str, output_dir: Path, *, prefix_tokens: int | None = None,
              output_tokens: int | None = None, minimum_hit: int = 1280,
              prompts_file: Path | None = None,
              timeout: int = 600, metrics_timeout: int = 30) -> dict:
    if (type(minimum_hit) is not int or minimum_hit <= 0
            or (prefix_tokens is not None and (type(prefix_tokens) is not int or prefix_tokens <= minimum_hit))
            or (output_tokens is not None and (type(output_tokens) is not int or output_tokens < 16))):
        raise ValueError("prefix must exceed minimum_hit; output must be >=16")
    output_dir.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    manifest = {"base_url": base_url, "model": model, "nonce": nonce,
                "minimum_hit": minimum_hit,
                "scope": "dedicated-server replay/MTP smoke, not accuracy or performance benchmark"}
    rows = []
    try:
        if prompts_file is not None:
            prompts = validate_prompts(json.loads(prompts_file.read_text()))
            for name, requested in (("prefix_tokens", prefix_tokens), ("output_tokens", output_tokens)):
                if requested is not None and requested != prompts[name]:
                    raise ValueError(f"{name} differs from saved prompts; do not change paired-run inputs")
            manifest["prompts_source"] = str(prompts_file.resolve())
        else:
            prefix_tokens = DEFAULT_PREFIX_TOKENS if prefix_tokens is None else prefix_tokens
            output_tokens = DEFAULT_OUTPUT_TOKENS if output_tokens is None else output_tokens
            note = "Reference note: keep the records in their original order. "
            # Cache salts isolate runs; no per-run UUID is inserted into model input.
            text = "Prefix-cache replay reference.\n" + note * (prefix_tokens // 4 + 64)

            def tokenize(value: str) -> list[int]:
                body = http(base_url, "/tokenize", {"model": model, "prompt": value,
                            "add_special_tokens": False}, timeout)
                tokens = body.get("tokens")
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
                    raise AssertionError("/tokenize did not return integer token IDs")
                return tokens

            tokens = tokenize(text)
            if len(tokens) < prefix_tokens:
                raise AssertionError("generated reference text is shorter than requested prefix")
            prefix = tokens[:prefix_tokens]
            prompts = validate_prompts({"version": 1, "prefix_tokens": prefix_tokens,
                "output_tokens": output_tokens,
                "prompt_a": prefix + tokenize("\nTask A: Continue the reference notes in order.\n"),
                "prompt_b": prefix + tokenize("\nTask B: Summarize the reference notes in numbered sentences.\n")})
            manifest["prompts_source"] = "generated"
        prefix_tokens, output_tokens = prompts["prefix_tokens"], prompts["output_tokens"]
        if prefix_tokens <= minimum_hit:
            raise ValueError("saved prefix must exceed minimum_hit")
        (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
        shared_salt, cold_salt = f"glm53-{nonce}-shared", f"glm53-{nonce}-b-cold"
        manifest.update(prefix_tokens=prefix_tokens, output_tokens=output_tokens,
            prompts_sha256=hashlib.sha256(json.dumps(prompts, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            shared_cache_salt=shared_salt, b_cold_cache_salt=cold_salt)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        before = counters(http(base_url, "/metrics", None, timeout))
        (output_dir / "metrics_before.json").write_text(json.dumps(before, indent=2))
        prompt_a, prompt_b = prompts["prompt_a"], prompts["prompt_b"]
        for i, (label, prompt) in enumerate(zip(CASES, (prompt_a, prompt_a, prompt_b, prompt_b, prompt_b))):
            salt = cold_salt if label == "B_cold" else shared_salt
            payload = completion_payload(model, prompt, output_tokens, cache_salt=salt)
            started = time.monotonic()
            body = http(base_url, "/v1/completions", payload, timeout)
            (output_dir / f"{i}_{label}.json").write_text(json.dumps(
                {"request": payload, "response": body, "elapsed_s": time.monotonic()-started}, indent=2))
            row = {"label": label, "cache_salt": salt, **check_completion(body, len(prompt), output_tokens)}
            rows.append(row)
            print(label, "prompt=", len(prompt), "cached=", row["cached_tokens"], flush=True)
        check_replays(rows, minimum_hit)
        deadline = time.monotonic() + metrics_timeout
        while True:
            after = counters(http(base_url, "/metrics", None, timeout))
            delta = counter_delta(before, after)
            try:
                check_activity(delta)
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    (output_dir / "metrics_failed.json").write_text(json.dumps(
                        {"before": before, "after": after, "delta": delta}, indent=2))
                    raise
                time.sleep(1)
        result = {"passed": True, "requests": rows, "metrics_before": before,
                  "metrics_after": after, "metrics_delta": delta, **manifest}
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        (output_dir / "result.json").write_text(json.dumps({"passed": False, "requests": rows,
            "error_type": type(exc).__name__, "error": str(exc), **manifest}, indent=2))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="root URL, without /v1; dedicated caller-owned server only")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="fresh directory; existing paths are never overwritten")
    parser.add_argument("--prefix-tokens", type=int, help=f"shared prefix length; default {DEFAULT_PREFIX_TOKENS}, or saved value with --prompts-file")
    parser.add_argument("--output-tokens", type=int, help=f"decode length; default {DEFAULT_OUTPUT_TOKENS}, or saved value with --prompts-file")
    parser.add_argument("--minimum-hit", type=int, default=1280, help="required cached tokens on A-warm/B-shared/B-warm; never waived")
    parser.add_argument("--prompts-file", type=Path, help="reuse a previous run's prompts.json verbatim, without tokenizing or adding a UUID")
    args = parser.parse_args()
    result = run_probe(args.base_url, args.model, args.output_dir,
        prefix_tokens=args.prefix_tokens, output_tokens=args.output_tokens,
        minimum_hit=args.minimum_hit, prompts_file=args.prompts_file)
    print(json.dumps({"passed": result["passed"], "metrics_delta": result["metrics_delta"]}, indent=2))


if __name__ == "__main__":
    main()
