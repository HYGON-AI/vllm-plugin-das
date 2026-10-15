# SPDX-License-Identifier: Apache-2.0
"""DSpark-only rollout and query-row contracts; never enable generic speculation."""

from types import SimpleNamespace

import pytest
import torch

from tests.runtime_patch.test_deepseek_v41_flashmla_decode import (
    enabled,  # noqa: F401 -- pytest fixture
    forbidden,
    inputs,
    model,
)
from tests.runtime_patch.test_deepseek_v41_flashmla_metadata import builder
from vllm_hcu.patch.worker.op_opt import (
    patch_deepseek_v41_rocm_flashmla_decode as decode_patch,
    patch_deepseek_v41_rocm_flashmla_metadata as metadata_patch,
)
from vllm_hcu.platforms import envs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import (
    SCHEDULERS_ATTR,
    DecodeScheduler,
    decode_context_supported,
    decode_query_limit,
)


@pytest.mark.parametrize("method,count,parallel,limit", [
    ("dspark", 5, False, 6), ("dspark", 5, True, 11),
    ("mtp", 5, True, 0), ("eagle3", 5, True, 0),
    ("dspark", 0, False, 0), ("dspark", -1, False, 0),
    ("dspark", None, False, 0), ("dspark", True, False, 0),
])
def test_only_explicit_dspark_config_is_accepted(enabled, monkeypatch, method, count, parallel, limit):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "1")
    _, b, _, _ = builder()
    config = b.vllm_config
    config.speculative_config = SimpleNamespace(
        method=method, num_speculative_tokens=count, parallel_drafting=parallel,
    )
    assert decode_query_limit(config) == limit
    assert decode_context_supported(config) is bool(limit)
    config.parallel_config.prefill_context_parallel_size = 8
    assert not decode_context_supported(config)
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 2
    assert not decode_context_supported(config)


@pytest.mark.parametrize("child,decode,master,expected", [
    (None, "1", "1", True), ("1", "1", "1", True),
    ("0", "1", "1", False), ("1", "0", "1", False),
    ("1", "1", "0", False),
])
def test_dspark_rollout_switch(monkeypatch, child, decode, master, expected):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", decode)
    if child is None:
        monkeypatch.delenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", child)
    assert envs.dsv41_flashmla_dspark_enabled() is expected
    assert envs.hcu_vllm_environment_variables["VLLM_HCU_DSV41_FLASHMLA_DSPARK"]() is expected


def test_builder_prepares_multiquery_leases_and_preserves_ragged(enabled, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH", "1")
    module, b, metadata, _ = builder()
    b.vllm_config.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=5, parallel_drafting=True,
    )
    metadata.max_decode_query_len = 5
    ragged = metadata.decode_swa_ragged_indices
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (forbidden, forbidden, lambda: (object(), None)))
    metadata_patch.apply_to_module(module)
    b.build(0, metadata)
    schedulers = getattr(metadata, SCHEDULERS_ATTR)
    assert metadata.decode_swa_ragged_indices is ragged
    for lease in schedulers.values():
        assert lease.speculative and lease.query_limit == 11 and lease.allow_graph
    metadata.max_decode_query_len = 12
    b.build(0, metadata)
    assert getattr(metadata, SCHEDULERS_ATTR) is None
    metadata.max_decode_query_len = 1
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "0")
    b.build(0, metadata)
    assert getattr(metadata, SCHEDULERS_ATTR) is None


@pytest.mark.parametrize("query_len", [1, 3, 5, 6, 11])
def test_query_rows_not_requests_are_native_batch_rows(enabled, monkeypatch, query_len):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "1")
    args = inputs()
    rows = max(3, query_len)
    args["q"] = torch.zeros(rows, 64, 512, dtype=torch.bfloat16)
    args["output"] = torch.full_like(args["q"], 7)
    args["swa_indices"] = torch.arange(128, dtype=torch.int32).repeat(rows, 1)
    args["swa_lens"] = torch.full((rows,), 128, dtype=torch.int32)
    args["max_decode_query_len"] = query_len
    args["scheduler"] = DecodeScheduler(lambda: (object(), None), query_limit=11, speculative=True)
    # The future slot is explicit, even for the first query in the block.
    args["swa_indices"][0, :3] = torch.tensor([2, 1, 0], dtype=torch.int32)
    args["swa_lens"][0] = 3
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"].shape == (rows, 1, 64, 512)
        torch.testing.assert_close(kwargs["indices"][0, 0, :3], args["swa_indices"][0, :3])
        assert kwargs.get("causal", False) is False
        return torch.zeros_like(kwargs["q"]), None

    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (kernel, forbidden, forbidden))
    assert runtime.try_flashmla_decode(**args) and len(calls) == 1
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "0")
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    args["output"].fill_(7)
    assert not runtime.try_flashmla_decode(**args)
    assert torch.all(args["output"] == 7)


def test_no_global_config_and_runtime_optout_for_speculative_instance(enabled, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "1")
    module, template, _, _ = model()
    decode_patch.apply_to_module(module)
    _, b, _, _ = builder()
    config = b.vllm_config
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=5)
    attention = type(template)(config)
    assert decode_patch._context_supported(attention)
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DSPARK", "0")
    assert not decode_patch._context_supported(attention)
