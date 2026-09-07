# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""One targeted Qwen3.5 TP2/EP2 + MTP3 FULL decode graph nightly case."""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import (
    QWEN35_MTP3_SELECTOR_NAMES,
    require_gfx_arch,
    require_model_runtime,
    run_vllm_case,
)
from tests.integration.parallel.test_tp_ep_models import (
    QWEN35_35B_A3B,
    _qwen35_gpu_memory_utilization,
)


def _assert_mtp3_graph_parity(result: dict[str, Any], log: str) -> None:
    baseline = None
    for label in ("eager", "graph"):
        entry = result[label]
        assert len(entry["workers"]) == 2, (label, entry["workers"])
        for worker in entry["workers"]:
            assert worker["enforce_eager"] is (label == "eager"), worker
            assert all(
                worker["hcu_selectors"][name] is True
                for name in QWEN35_MTP3_SELECTOR_NAMES
            ), worker
            assert worker["parallel_config"] == {
                "tensor_parallel_size": 2,
                "data_parallel_size": 1,
                "enable_expert_parallel": True,
            }, worker
            assert worker["speculative_config"] == {
                "method": "mtp", "num_speculative_tokens": 3,
            }, worker
            assert worker["cache_config"] == {
                "mamba_cache_dtype": "float32",
            }, worker
            compilation = worker["compilation_config"]
            if label == "graph":
                # This is the target enum's decode_mode(), not a substring
                # match against the requested mode or the driver's config.
                assert compilation["decode_mode"] == "FULL", worker
                assert compilation["cudagraph_capture_sizes"], worker
                assert compilation["num_cudagraph_captured"] > 0, worker
            else:
                assert compilation["decode_mode"] == "NONE", worker

        assert len(entry["rounds"]) == 2, (label, entry["rounds"])
        for round_index, output in enumerate(entry["rounds"]):
            tokens = [record["token_ids"] for record in output]
            assert tokens and all(len(ids) >= 16 for ids in tokens), (label, tokens)
            for record in output:
                # Eager/graph reductions can differ in floating-point rounding.
                # Logprobs require presence, finiteness, and per-token coverage;
                # exact parity below applies only to generated token IDs.
                assert record["cumulative_logprob"] is not None, (label, record)
                assert math.isfinite(record["cumulative_logprob"]), (label, record)
                assert record["sample_logprob_count"] == 16, (label, record)
            if baseline is None:
                baseline = tokens
            assert tokens == baseline, (label, round_index, tokens, baseline)

            # Bound evidence to generation, excluding startup/capture logs.
            begin = f"VLLM_HCU_GENERATE_BEGIN={label}:{round_index}"
            end = f"VLLM_HCU_GENERATE_END={label}:{round_index}"
            assert begin in log and end in log, (label, round_index)
            segment = log.split(begin, 1)[1].split(end, 1)[0]
            drafts = re.findall(
                r"SpecDecoding metrics:[^\n]*Drafted: (\d+) tokens", segment
            )
            assert any(int(count) > 0 for count in drafts), (
                label, round_index, segment,
            )
            if label == "graph":
                full_counts = re.findall(
                    r"\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*FULL\s*\|\s*(\d+)\s*\|",
                    segment,
                )
                assert any(int(count) > 0 for count in full_counts), (
                    round_index, segment,
                )


@pytest.mark.hcu
@pytest.mark.model
@pytest.mark.multi_hcu
@pytest.mark.hcu_count(2)
@pytest.mark.slow
@pytest.mark.nightly
def test_qwen35_35b_a3b_tp2_ep2_mtp3_full_decode_graph_aiter_auto_shuffle(
    hcu_test_resources: HcuTestResources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise default-enabled selectors even if the invoking shell disables
    # them; pytest restores the caller environment after this targeted test.
    for name in QWEN35_MTP3_SELECTOR_NAMES:
        monkeypatch.delenv(name, raising=False)
    label = "Qwen3.5-35B-A3B TP2/EP2 MTP3 FULL decode graph"
    require_gfx_arch("gfx938", label)
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_QWEN35_35B_A3B_MODEL",
        relative_path=QWEN35_35B_A3B,
        label=label,
        hcu_count=2,
    )
    result = run_vllm_case(
        "qwen35-mtp3-graph-parity",
        model_path,
        timeout_s=5400,
        extra_args=[
            "--gpu-memory-utilization", str(_qwen35_gpu_memory_utilization(2)),
        ],
        extra_env={
            "VLLM_HCU_USE_AITER_MOE_SHUFFLE": "1",
            "VLLM_LOG_STATS_INTERVAL": "0.1",
            # The trusted local worker-config callback needs callable RPC.
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        },
        log_label="qwen35-tp2-ep2-mtp3-full-decode-graph-aiter-auto-shuffle",
    )
    log = Path(result["_log_path"]).read_text(encoding="utf-8", errors="replace")
    _assert_mtp3_graph_parity(result, log)
