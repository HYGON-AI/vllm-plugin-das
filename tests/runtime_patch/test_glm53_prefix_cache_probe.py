# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit GLM5Next live prefix-cache probe.

These tests use synthetic HTTP responses, not a real model or HCU device.
"""
import json

import pytest

from tests.integration.server import glm53_pcp_prefix_cache as probe


def response(prompt=4103, cached=3840, output=128, token=17):
    return {"choices": [{"token_ids": [token] * output, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def metrics(hit=0, draft=0, accepted=0):
    return "\n".join(f'{name}{{model_name="test"}} {value}'
                     for name, value in zip(probe.COUNTERS, (hit, draft, accepted)))


def rows():
    return [probe.check_completion(response(cached=c, token=t), 4103, 128)
            for c, t in [(0, 17), (3840, 17), (3840, 18), (3840, 18)]]


def test_payload_requires_decode_and_requests_exact_token_ids():
    payload = probe.completion_payload("test", [1, 2, 3], 128)
    assert payload["ignore_eos"] is True
    assert payload["temperature"] == 0
    assert payload["return_token_ids"] is True
    assert payload["add_special_tokens"] is False
    assert payload["max_tokens"] == 128
    with pytest.raises(ValueError):
        probe.completion_payload("test", [1], 1)


def test_counters_do_not_double_count_histograms_or_per_position_metrics():
    text = metrics(100, 20, 10) + '\n' + (
        'vllm:prefix_cache_hits_total{model_name="test",engine="1"} 2e1\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 999\n'
        'vllm:spec_decode_num_accepted_tokens_created 123\n'
    )
    assert probe.counters(text) == dict(zip(probe.COUNTERS, (120, 20, 10)))


def test_missing_metrics_fail_instead_of_becoming_zero():
    with pytest.raises(AssertionError, match="required metrics missing"):
        probe.counters('vllm:prefix_cache_hits_total 0')


def test_negative_metric_sample_is_rejected():
    with pytest.raises(AssertionError, match="invalid counter"):
        probe.counters(metrics(-1, 2, 1))


@pytest.mark.parametrize("mutation", ["details_missing", "cached_negative", "cached_full_prompt",
                                     "prompt_length", "decode_short", "token_ids_missing", "finish"])
def test_invalid_completion_cannot_pass(mutation):
    body = response()
    if mutation == "details_missing":body["usage"].pop("prompt_tokens_details")
    elif mutation == "cached_negative":body["usage"]["prompt_tokens_details"]["cached_tokens"] = -1
    elif mutation == "cached_full_prompt":body["usage"]["prompt_tokens_details"]["cached_tokens"] = 4103
    elif mutation == "prompt_length":body["usage"]["prompt_tokens"] = 1
    elif mutation == "decode_short":body["choices"][0]["token_ids"] = [17]
    elif mutation == "token_ids_missing":body["choices"][0].pop("token_ids")
    else:body["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(AssertionError):probe.check_completion(body, 4103, 128)


def test_replays_require_real_hits_and_token_parity():
    probe.check_replays(rows(), 1280)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_prefix_cache_enabled_but_zero_hits_is_not_success(index):
    data = rows();data[index]["cached_tokens"] = 0
    with pytest.raises(AssertionError, match="did not actually reuse"):
        probe.check_replays(data, 1280)


def test_cold_hit_is_not_accepted_as_a_cold_reference():
    data = rows();data[0]["cached_tokens"] = 640
    with pytest.raises(AssertionError, match="cold request unexpectedly hit"):
        probe.check_replays(data, 1280)


@pytest.mark.parametrize("index", [1, 3])
def test_replay_token_mismatch_requires_investigation(index):
    data = rows();data[index]["token_ids"] = [99] * 128
    with pytest.raises(AssertionError, match="token IDs differ"):
        probe.check_replays(data, 1280)


def test_counter_reset_is_not_treated_as_an_acceptance_rate():
    with pytest.raises(AssertionError, match="metrics reset"):
        probe.counter_delta(probe.counters(metrics(10, 20, 15)), probe.counters(metrics()))


@pytest.mark.parametrize("values", [(0, 20, 10), (1, 0, 0), (1, 20, 0), (1, 20, 21)])
def test_positive_mtp_and_prefix_activity_are_required(values):
    with pytest.raises(AssertionError):probe.check_activity(dict(zip(probe.COUNTERS, values)))


def test_fake_end_to_end_probe_is_sequential_and_writes_evidence(monkeypatch, tmp_path):
    requests = []
    metric_reads = 0
    def fake_http(base_url, path, payload, timeout):
        nonlocal metric_reads
        if path == "/tokenize":
            if payload["prompt"].startswith("Unique test prefix"):
                return {"tokens": list(range(5000))}
            return {"tokens": [6001 if "Task A" in payload["prompt"] else 6002] * 7}
        if path == "/metrics":
            metric_reads += 1
            return metrics() if metric_reads == 1 else metrics(11520, 400, 300)
        assert path == "/v1/completions"
        requests.append(payload)
        return response(prompt=len(payload["prompt"]), cached=0 if len(requests)==1 else 3840,
                        output=payload["max_tokens"], token=17 if len(requests)<=2 else 18)
    monkeypatch.setattr(probe, "http", fake_http)
    out = tmp_path / "probe"
    result = probe.run_probe("http://127.0.0.1:9999", "test", out)
    assert result["passed"] is True
    assert len(requests) == 4
    assert requests[0]["prompt"] == requests[1]["prompt"]
    assert requests[2]["prompt"] == requests[3]["prompt"]
    assert requests[0]["prompt"][:4352] == requests[2]["prompt"][:4352]
    assert requests[0]["prompt"] != requests[2]["prompt"]
    assert json.loads((out/"result.json").read_text())["passed"] is True


def test_probe_never_overwrites_existing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        probe.run_probe("http://unused", "test", tmp_path)
