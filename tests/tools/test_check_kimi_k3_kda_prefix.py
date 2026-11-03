# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.check_kimi_k3_kda_prefix import compare_logprobs, prefix_probe


def test_prefix_probe_clears_reference_before_producer():
    events = []

    def request(path, payload=None, raw=False):
        events.append((path, payload, raw))
        return {"success": True} if path == "/reset_prefix_cache" else "metrics"

    def completion(tokens, name, max_tokens=1):
        events.append((name, tokens, max_tokens))
        return name, 0

    assert prefix_probe(request, completion, [1], [2]) == (
        "prefix_uncached", "prefix_cached", "metrics", "metrics")
    assert events == [
        ("/reset_prefix_cache", {}, False), ("prefix_uncached", [2], 8),
        ("/reset_prefix_cache", {}, False), ("prefix_producer", [1], 1),
        ("/metrics", None, True), ("prefix_cached", [2], 8),
        ("/metrics", None, True),
    ]


@pytest.mark.parametrize("failure_at", [1, 2])
def test_prefix_probe_stops_when_reset_fails(failure_at):
    resets = 0
    completions = []

    def request(path, payload=None, raw=False):
        nonlocal resets
        assert path == "/reset_prefix_cache"
        resets += 1
        return {"success": resets != failure_at}

    def completion(tokens, name, max_tokens=1):
        completions.append(name)
        return {}, 0

    with pytest.raises(RuntimeError, match="prefix cache reset failed"):
        prefix_probe(request, completion, [1], [2])
    assert completions == ([] if failure_at == 1 else ["prefix_uncached"])


def test_compare_logprobs_requires_complete_comparable_evidence():
    good = {"tokens": ["a", "b"], "token_logprobs": [-0.2, -0.3],
            "top_logprobs": [{"a": -0.2, "z": -1.0}, {"b": -0.3}]}
    assert compare_logprobs(good, good) == 0
    for bad in (
        None,
        {**good, "tokens": None},
        {**good, "tokens": ["a"]},
        {**good, "token_logprobs": [-0.2]},
        {**good, "token_logprobs": [-0.2, None]},
        {**good, "token_logprobs": [-0.2, float("nan")]},
        {**good, "top_logprobs": [{"a": -0.2}]},
        {**good, "top_logprobs": [{"other": -0.2}, {"b": -0.3}]},
    ):
        with pytest.raises(ValueError):
            compare_logprobs(good, bad)
