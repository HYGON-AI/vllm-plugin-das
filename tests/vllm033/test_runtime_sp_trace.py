# SPDX-License-Identifier: Apache-2.0
"""CPU intermediate provenance/layout controls; no native or model claim."""
import json

import pytest
import torch
from torch import nn

from tools.runtime_sp_trace import RuntimeSPTrace, compare_trace_sets


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Identity()
        self.model.norm = nn.Identity()

    def forward(self, x):
        return self.model.norm(self.model.embed_tokens(x))


def test_capture_is_explicit_copies_outputs_and_removes_hooks(tmp_path):
    model = Toy()
    trace = RuntimeSPTrace(model, arm="reference", rank=0, size=1, num_tokens=3)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert model(x) is x and trace.data["samples"] == {}
    with trace.capture("eager", active_tokens=3):
        assert model(x) is x
    x.zero_()
    first = trace.data["samples"]["eager"]["events"][0]["leaves"]["output"]
    torch.testing.assert_close(first, torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    trace.save(tmp_path / "trace.pt")
    saved = torch.load(tmp_path / "trace.pt", weights_only=True)
    torch.testing.assert_close(saved["samples"]["eager"]["events"][0]["leaves"]["output"], first)
    trace.close()
    with trace.capture("after-close", active_tokens=3):
        model(x)
    assert trace.data["samples"]["after-close"]["events"] == []


def test_inactive_hooks_do_not_inspect_gpu_capture(monkeypatch):
    model = Toy()
    trace = RuntimeSPTrace(model, arm="reference", rank=0, size=1, num_tokens=1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: pytest.fail("inactive trace queried GPU"))
    model(torch.zeros(1, 4))
    trace.close()


def test_recording_refuses_inside_graph_and_preserves_incomplete_sample(monkeypatch):
    model = Toy()
    trace = RuntimeSPTrace(model, arm="reference", rank=0, size=1, num_tokens=1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="forbidden inside a GPU Graph"):
        with trace.capture("bad", active_tokens=1):
            model(torch.zeros(1, 4))
    assert not trace.data["samples"]["bad"]["complete"]
    trace.close()


def test_budget_does_not_publish_partial_capture_as_complete():
    model = Toy()
    trace = RuntimeSPTrace(model, arm="reference", rank=0, size=1, num_tokens=1, max_bytes=1)
    with pytest.raises(RuntimeError, match="budget"):
        with trace.capture("large", active_tokens=1):
            model(torch.zeros(1, 4))
    assert not trace.data["samples"]["large"]["complete"]
    trace.close()


def records():
    references, candidates = [], []
    full = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 32
    padded = torch.cat((full, torch.full((1, 4), float("nan"))))
    for rank in range(4):
        left = dict(schema=1, arm="reference", size=4, rank=rank, num_tokens=7, samples={})
        right = dict(schema=1, arm="candidate", size=4, rank=rank, num_tokens=7, samples={})
        for label, active in (("initial", 7), ("changed-padding", 5)):
            def event(name, layout, tensor):
                return dict(name=name, layout=layout, leaves={"output": tensor.clone()})
            head = full + rank
            left["samples"][label] = dict(complete=True, active_tokens=active, events=[
                event("model.layers.0.self_attn.qkv_proj", "head_sharded", head),
                event("model.layers.0.self_attn.o_proj", "replicated", full),
            ])
            right["samples"][label] = dict(complete=True, active_tokens=active, events=[
                event("model.layers.0.self_attn.qkv_proj", "head_sharded",
                      torch.cat((head, torch.full((1, 4), float("nan"))))),
                event("model.layers.0.self_attn.o_proj", "token_sharded", padded[rank * 2:(rank + 1) * 2]),
            ])
        references.append(left)
        candidates.append(right)
    return references, candidates


def test_global_token_join_head_rank_and_padding_are_distinct():
    left, right = records()
    result = compare_trace_sets(left[::-1], right[2:] + right[:2])
    assert result["first_difference"] is None and result["first_outside_tolerance"] is None
    assert len(result["comparisons"]) == 16
    right[0]["samples"]["initial"]["events"][1]["leaves"]["output"] += 2
    result = compare_trace_sets(left, right)
    assert result["first_outside_tolerance"]["event"].endswith("o_proj")


def test_first_exact_difference_does_not_claim_first_failed_tolerance():
    left, right = records()
    right[0]["samples"]["initial"]["events"][0]["leaves"]["output"] += .001
    right[0]["samples"]["initial"]["events"][1]["leaves"]["output"] += 2
    result = compare_trace_sets(left, right)
    assert result["first_difference"]["event"].endswith("qkv_proj")
    assert result["first_outside_tolerance"]["event"].endswith("o_proj")
    assert result["atol"] == result["rtol"] == .016


@pytest.mark.parametrize("damage", ["missing-rank", "duplicate-rank", "schema", "partial",
    "event-order", "layout", "rank-layout", "dtype", "extent", "active", "leaf", "empty-leaves"])
def test_incomplete_or_misaligned_data_cannot_be_reported_as_parity(damage):
    left, right = records()
    sample = right[0]["samples"]["initial"]
    event = sample["events"][1]
    if damage == "missing-rank": right.pop()
    elif damage == "duplicate-rank": right[-1]["rank"] = 0
    elif damage == "schema": right[0]["schema"] = 2
    elif damage == "partial": sample["complete"] = False
    elif damage == "event-order": sample["events"].reverse()
    elif damage == "layout":
        for r in right: r["samples"]["initial"]["events"][1]["layout"] = "unknown"
    elif damage == "rank-layout": event["layout"] = "replicated"
    elif damage == "dtype": event["leaves"]["output"] = event["leaves"]["output"].double()
    elif damage == "extent": event["leaves"]["output"] = event["leaves"]["output"][:1]
    elif damage == "active": sample["active_tokens"] = 6
    elif damage == "leaf": event["leaves"] = {}
    elif damage == "empty-leaves":
        for record in left + right:
            record["samples"]["initial"]["events"][1]["leaves"] = {}
    with pytest.raises(ValueError):
        compare_trace_sets(left, right)


def test_active_nonfinite_is_failure_with_strict_json_output():
    left, right = records()
    right[0]["samples"]["initial"]["events"][1]["leaves"]["output"][0, 0] = float("nan")
    result = compare_trace_sets(left, right)
    assert not result["first_outside_tolerance"]["finite"]
    assert result["first_outside_tolerance"]["max_abs_error"] is None
    json.dumps(result, allow_nan=False)


def test_equal_infinities_are_not_finite_intermediate_parity():
    left, right = records()
    for record in left + right:
        record["samples"]["initial"]["events"][0]["leaves"]["output"][0, 0] = float("inf")
    result = compare_trace_sets(left, right)
    assert result["first_outside_tolerance"]["exact"]
    assert not result["first_outside_tolerance"]["finite"]
    json.dumps(result, allow_nan=False)
