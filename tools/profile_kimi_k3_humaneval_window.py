#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture a short window of concurrent HumanEval-style requests, not a score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import urllib.request
from pathlib import Path


def request(base, path, payload=None, timeout=900):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        text = response.read().decode()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.delay_seconds < 0:
        parser.error("concurrency/max-tokens must be positive and delay nonnegative")
    from datasets import load_from_disk
    data = sorted(load_from_disk(str(args.dataset)), key=lambda x: int(x["task_id"].split("/")[-1]))
    if len(data) < args.concurrency:
        raise SystemExit("Dataset has fewer tasks than requested concurrency")
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "trace_requests.json"
    if dest.exists():
        raise SystemExit("Refusing to overwrite trace metadata")
    model = request(args.api_base, "/v1/models")["data"][0]["id"]
    template = ("Read the following function signature and docstring, and fully implement "
                "the function described. Your response should only contain the code for "
                "this function.\n{question}")
    def payload(row, max_tokens):
        return dict(model=model, messages=[dict(role="user", content=template.format(question=row["prompt"]))],
                    temperature=0, top_p=1, seed=20261026, max_tokens=max_tokens,
                    chat_template_kwargs={"enable_thinking": False})
    print("Warming up a short request (profiler must be stopped)", flush=True)
    request(args.api_base, "/v1/chat/completions", payload(data[0], 8))
    records, profile_started = [], False
    start_result = stop_result = None
    empty_window = False
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(request, args.api_base, "/v1/chat/completions", payload(row, args.max_tokens)): row["task_id"]
                   for row in data[:args.concurrency]}
        try:
            time.sleep(args.delay_seconds)
            if all(f.done() for f in futures):
                empty_window = True
                raise RuntimeError("All measured requests completed before profiling started")
            print("Starting profiler; server PROFILE_MAX_STEPS bounds worker steps", flush=True)
            start_result = request(args.api_base, "/start_profile", {})
            profile_started = True
            if all(f.done() for f in futures):
                empty_window = True
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    response = future.result()
                    records.append(dict(task_id=task_id, usage=response.get("usage"),
                                        finish_reason=response["choices"][0].get("finish_reason")))
                    print(task_id, "done", flush=True)
                except Exception as exc:
                    records.append(dict(task_id=task_id, error=f"{type(exc).__name__}: {exc}"))
                    print(task_id, "ERROR", exc, flush=True)
        except RuntimeError as exc:
            if not empty_window:
                raise
            start_result = dict(error=str(exc))
        finally:
            if profile_started:
                print("Stopping profiler and waiting for trace export", flush=True)
                try:
                    stop_result = request(args.api_base, "/stop_profile", {})
                except Exception as exc:
                    stop_result = dict(error=f"{type(exc).__name__}: {exc}")
    dest.write_text(json.dumps(dict(concurrency=args.concurrency, max_tokens=args.max_tokens,
        delay_seconds=args.delay_seconds, started_unix=started, finished_unix=time.time(),
        profile_start=start_result, profile_stop=stop_result, empty_window=empty_window, requests=records), indent=2) + "\n")
    return 1 if empty_window or any("error" in x for x in records) or (isinstance(stop_result, dict) and "error" in stop_result) else 0


if __name__ == "__main__":
    raise SystemExit(main())
