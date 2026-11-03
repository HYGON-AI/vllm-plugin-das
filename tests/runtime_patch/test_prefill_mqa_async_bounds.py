# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for async speculative upper bounds (native 752a3a50)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_mla_indexer as adapter


def _helper_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/v1/attention/ops/prefill_mqa.py"
    )
    spec = importlib.util.spec_from_file_location("upper_bound_repro", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.mark.parametrize(
    "async_setting,expected",
    [(False, True), (True, False), (None, False), ("missing", False)],
)
def test_builder_proves_exact_cpu_lengths_only_for_explicit_sync(
    monkeypatch, async_setting, expected,
):
    def split(seq_lens_cpu, query_lens_cpu, workspace_size, max_logits_bytes,
              request_offset=0):
        return []

    def batch(common_attn_metadata, decode_threshold=1, require_uniform=False,
              treat_short_extends_as_decodes=True):
        return ()

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return SimpleNamespace()

    upstream = ModuleType(adapter.TARGET_MODULE)
    upstream.DeepseekV32IndexerMetadataBuilder = Builder
    upstream.split_indexer_prefill_chunks = split
    upstream.split_decodes_and_prefills = batch
    adapter.apply_to_module(upstream)
    worker = Builder()
    scheduler = SimpleNamespace()
    if async_setting != "missing":
        scheduler.async_scheduling = async_setting
    worker.vllm_config = SimpleNamespace(
        scheduler_config=scheduler,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    pcp = ModuleType("vllm_hcu.model_executor.layers.attention.pcp")
    pcp.effective_pcp_world_size = lambda value: value
    monkeypatch.setitem(sys.modules, pcp.__name__, pcp)
    result = worker.build(0, SimpleNamespace(num_actual_tokens=1027))
    assert result.hcu_mqa_cpu_lengths_exact is expected


def test_rejected_draft_upper_bound_shifts_prompt_first_key():
    # Full batch q=[1, 2, 1024]; require_uniform=True leaves the second
    # 2-token decode in the prefill chunk with the 1024-token prompt.
    helper = _helper_module()
    chunks = helper.plan_prefill_mqa_chunks(
        [2, 1024], [8194, 1024], max_logits_bytes=256 * 1024 * 1024,
    )
    assert chunks == ((0, 2, 0, 8194), (2, 1026, 8194, 9218))
    # Device gather is [0, 8193, 9217], despite allocation capacity 9218.
    keys = torch.ones((9218, 1))
    q = torch.ones((1026, 1, 1))
    scales = torch.ones(9218)
    weights = torch.ones((1026, 1))
    starts = torch.tensor([0, 0] + [8193] * 1024)
    ends = torch.tensor([8192, 8193] + list(range(8194, 9218)))

    def mqa(q, kv, weights, ks, ke):
        return q[:, 0] @ kv[0].T

    def topk(logits, ks, ke, out, count):
        out.fill_(-1)
        for i in range(len(out)):
            start = max(0, min(int(ks[i]), logits.shape[1]))
            end = max(start, min(int(ke[i]), logits.shape[1]))
            if end > start:
                out[i, 0] = logits[i, start:end].argmax()

    original = torch.full((1026, 1), -1, dtype=torch.int32)
    topk(mqa(q, (keys, scales), weights, starts, ends),
         starts, ends, original, 1)
    unsafe = torch.full_like(original, -1)
    assert helper.run_grouped_prefill_mqa(
        q, keys, scales, weights, starts, ends, unsafe, chunks,
        mqa=mqa, topk=topk,
    )
    assert original[2, 0].item() == 0
    assert unsafe[2, 0].item() == -1
    exact = helper.plan_prefill_mqa_chunks(
        [2, 1024], [8193, 1024], max_logits_bytes=256 * 1024 * 1024,
    )
    corrected = torch.full_like(original, -1)
    assert helper.run_grouped_prefill_mqa(
        q, keys, scales, weights, starts, ends, corrected, exact,
        mqa=mqa, topk=topk,
    )
    torch.testing.assert_close(corrected, original)


def test_default_plan_can_cross_non_beta_lightop_dispatch_threshold():
    # This shape check does not establish GPU logits/TopK equivalence.
    assert _helper_module().plan_prefill_mqa_chunks(
        [128, 32], [131072, 4096], max_logits_bytes=256 * 1024 * 1024,
    ) == ((0, 128, 0, 131072), (128, 160, 131072, 135168))
