# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit GLM5Next live prefix-cache probe.

These tests use synthetic HTTP responses, not a real model or HCU device.
"""
import json
import sys

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
    return [{"label": label, "cache_salt": "cold-b" if label == "B_cold" else "shared",
             **probe.check_completion(response(cached=c, token=t), 4103, 128)}
            for label, (c, t) in zip(probe.CASES,
                [(0, 17), (3840, 17), (3840, 18), (3840, 18), (0, 18)])]


def fake_server(monkeypatch, *, block_size=640, bad_cached_b=False):
    requests = []
    tokenizations = []
    metric_reads = 0

    def fake_http(base_url, path, payload, timeout):
        nonlocal metric_reads
        if path == "/tokenize":
            tokenizations.append(payload["prompt"])
            if payload["prompt"].startswith("Prefix-cache replay reference"):
                return {"tokens": list(range(20000))}
            return {"tokens": [60001 if "Task A" in payload["prompt"] else 60002] * 7}
        if path == "/metrics":
            metric_reads += 1
            return metrics() if metric_reads == 1 else metrics(11520, 400, 300)
        assert path == "/v1/completions"
        requests.append(payload)
        cold = len(requests) == 1 or payload["cache_salt"].endswith("-b-cold")
        cached = 0 if cold else ((len(payload["prompt"]) - 7) // block_size) * block_size
        token = 17 if len(requests) <= 2 else 18
        if bad_cached_b and len(requests) in (3, 4):
            token = 99
        return response(prompt=len(payload["prompt"]), cached=cached,
                        output=payload["max_tokens"], token=token)

    monkeypatch.setattr(probe, "http", fake_http)
    return requests, tokenizations


def test_payload_requires_decode_and_requests_exact_token_ids():
    payload = probe.completion_payload("test", [1, 2, 3], 128)
    assert payload["ignore_eos"] is True
    assert payload["temperature"] == 0
    assert payload["return_token_ids"] is True
    assert payload["add_special_tokens"] is False
    assert payload["max_tokens"] == 128
    with pytest.raises(ValueError):
        probe.completion_payload("test", [1], 1)


def test_cold_and_cached_b_payloads_only_differ_in_cache_salt():
    warm = probe.completion_payload("test", [1, 2, 3], 128, cache_salt="shared")
    cold = probe.completion_payload("test", [1, 2, 3], 128, cache_salt="cold-b")
    assert warm.pop("cache_salt") != cold.pop("cache_salt")
    assert warm == cold
    with pytest.raises(ValueError, match="non-empty string"):
        probe.completion_payload("test", [1], 128, cache_salt="")


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


@pytest.mark.parametrize("index", [0, 4])
def test_cold_hit_is_not_accepted_as_a_cold_reference(index):
    data = rows();data[index]["cached_tokens"] = 640
    with pytest.raises(AssertionError, match="cold request unexpectedly hit"):
        probe.check_replays(data, 1280)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_replay_token_mismatch_requires_investigation(index):
    data = rows();data[index]["token_ids"] = [99] * 128
    with pytest.raises(AssertionError, match="token IDs differ"):
        probe.check_replays(data, 1280)


def test_identical_b_replays_cannot_hide_a_stably_wrong_cached_state():
    data = rows()
    data[2]["token_ids"] = data[3]["token_ids"] = [99] * 128
    assert data[2]["token_ids"] == data[3]["token_ids"]
    with pytest.raises(AssertionError, match="cold/shared B token IDs differ"):
        probe.check_replays(data, 1280)


def test_four_requests_without_an_independent_b_cold_control_are_rejected():
    with pytest.raises(ValueError, match="five sequential"):
        probe.check_replays(rows()[:4], 1280)


@pytest.mark.parametrize("index,salt", [(4, "shared"), (4, None), (2, "other")])
def test_cache_namespaces_must_isolate_cold_b_and_share_the_cached_branch(index, salt):
    data = rows()
    data[index]["cache_salt"] = salt
    with pytest.raises(AssertionError, match="independent cache_salt"):
        probe.check_replays(data, 1280)


def test_counter_reset_is_not_treated_as_an_acceptance_rate():
    with pytest.raises(AssertionError, match="metrics reset"):
        probe.counter_delta(probe.counters(metrics(10, 20, 15)), probe.counters(metrics()))


@pytest.mark.parametrize("values", [(0, 20, 10), (1, 0, 0), (1, 20, 0), (1, 20, 21)])
def test_positive_mtp_and_prefix_activity_are_required(values):
    with pytest.raises(AssertionError):probe.check_activity(dict(zip(probe.COUNTERS, values)))


def test_fake_end_to_end_probe_is_sequential_and_writes_evidence(monkeypatch, tmp_path):
    requests, _ = fake_server(monkeypatch)
    out = tmp_path / "probe"
    result = probe.run_probe("http://127.0.0.1:9999", "test", out)
    assert result["passed"] is True
    assert len(requests) == 5
    assert requests[0]["prompt"] == requests[1]["prompt"]
    assert requests[2]["prompt"] == requests[3]["prompt"] == requests[4]["prompt"]
    assert requests[0]["prompt"][:10240] == requests[2]["prompt"][:10240]
    assert requests[0]["prompt"] != requests[2]["prompt"]
    assert len({request["cache_salt"] for request in requests[:4]}) == 1
    assert requests[4]["cache_salt"] != requests[2]["cache_salt"]
    assert {key:value for key,value in requests[2].items() if key != "cache_salt"} == {
        key:value for key,value in requests[4].items() if key != "cache_salt"}
    assert json.loads((out/"prompts.json").read_text())["prompt_b"] == requests[4]["prompt"]
    assert json.loads((out/"result.json").read_text())["passed"] is True


def test_failed_b_cold_control_preserves_all_five_responses_and_inputs(monkeypatch, tmp_path):
    fake_server(monkeypatch, bad_cached_b=True)
    out = tmp_path / "failed"
    with pytest.raises(AssertionError, match="cold/shared B token IDs differ"):
        probe.run_probe("http://unused", "test", out)
    result = json.loads((out/"result.json").read_text())
    assert result["passed"] is False
    assert len(result["requests"]) == 5
    assert (out/"2_B_shared.json").is_file()
    assert (out/"3_B_warm.json").is_file()
    assert (out/"4_B_cold.json").is_file()
    assert (out/"prompts.json").is_file()


def test_paired_head_modes_reuse_exact_saved_ids_without_tokenizing(monkeypatch, tmp_path):
    requests_on, _ = fake_server(monkeypatch, block_size=640)
    on = probe.run_probe("http://head-on", "test", tmp_path/"on")
    requests_off, tokenizations = fake_server(monkeypatch, block_size=4480)
    off = probe.run_probe("http://head-off", "test", tmp_path/"off",
                          prompts_file=tmp_path/"on"/"prompts.json")
    assert not tokenizations
    assert on["passed"] and off["passed"]
    assert on["prompts_sha256"] == off["prompts_sha256"]
    assert on["shared_cache_salt"] != off["shared_cache_salt"]
    assert all(off["requests"][index]["cached_tokens"] >= 4480 for index in (1,2,3))
    for left, right in zip(requests_on, requests_off):
        assert {key:value for key,value in left.items() if key != "cache_salt"} == {
            key:value for key,value in right.items() if key != "cache_salt"}


def test_a_short_prefix_still_fails_on_4480_token_blocks(monkeypatch, tmp_path):
    fake_server(monkeypatch, block_size=4480)
    with pytest.raises(AssertionError, match="did not actually reuse"):
        probe.run_probe("http://unused", "test", tmp_path/"short", prefix_tokens=4352)
    result = json.loads((tmp_path/"short"/"result.json").read_text())
    assert result["passed"] is False
    assert result["minimum_hit"] == 1280


@pytest.mark.parametrize("prefix_tokens", [8960, 10240])
def test_configurable_long_prefix_reaches_4480_token_cache_blocks(monkeypatch, tmp_path, prefix_tokens):
    fake_server(monkeypatch, block_size=4480)
    result = probe.run_probe("http://unused", "test", tmp_path/"long",
                             prefix_tokens=prefix_tokens, output_tokens=128)
    assert result["prefix_tokens"] == prefix_tokens
    assert result["output_tokens"] == 128
    assert result["passed"] is True


@pytest.mark.parametrize("name", ["prefix_tokens", "output_tokens"])
def test_saved_input_lengths_cannot_be_changed_in_a_paired_run(monkeypatch, tmp_path, name):
    fake_server(monkeypatch)
    source = probe.run_probe("http://unused", "test", tmp_path/"source")
    with pytest.raises(ValueError, match="differs from saved prompts"):
        probe.run_probe("http://unused", "test", tmp_path/"changed",
                         prompts_file=tmp_path/"source"/"prompts.json", **{name:source[name]+1})


@pytest.mark.parametrize("mutation", ["version", "bool_token", "negative_token", "short",
                                     "prefix", "same_branch", "output", "missing_field"])
def test_saved_prompt_manifest_is_validated(mutation):
    data = {"version":1, "prefix_tokens":2, "output_tokens":16,
            "prompt_a":[1,2,3], "prompt_b":[1,2,4]}
    if mutation == "version": data["version"] = True
    elif mutation == "bool_token": data["prompt_a"][0] = True
    elif mutation == "negative_token": data["prompt_a"][0] = -1
    elif mutation == "short": data["prompt_a"] = [1,2]
    elif mutation == "prefix": data["prompt_b"][0] = 5
    elif mutation == "same_branch": data["prompt_b"] = list(data["prompt_a"])
    elif mutation == "output": data["output_tokens"] = 1
    else: data.pop("prompt_b")
    with pytest.raises(ValueError):
        probe.validate_prompts(data)


def test_cli_exposes_reproducible_inputs_and_hit_threshold(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {"passed":True, "metrics_delta":{}}

    monkeypatch.setattr(probe, "run_probe", fake_run)
    monkeypatch.setattr(sys, "argv", ["probe", "--base-url", "http://unused", "--model", "test",
        "--output-dir", str(tmp_path/"out"), "--prefix-tokens", "10240", "--output-tokens", "128",
        "--minimum-hit", "4480", "--prompts-file", str(tmp_path/"prompts.json")])
    probe.main()
    assert captured == {"prefix_tokens":10240, "output_tokens":128, "minimum_hit":4480,
                        "prompts_file":tmp_path/"prompts.json"}


def test_probe_never_overwrites_existing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        probe.run_probe("http://unused", "test", tmp_path)
