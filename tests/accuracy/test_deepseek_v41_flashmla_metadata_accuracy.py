# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real ROCm builder -> model wrapper -> native attention (no model weights)."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.accuracy.test_deepseek_v41_flashmla_decode_accuracy import (
    make_cache,
    reference,
)
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import SCHEDULERS_ATTR

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_real_builder_and_model_wrapper(ratio):
    # Native binding discovery imports vLLM and can load MoE modules before
    # the dispatcher is armed. Use a cold child so this test also works after
    # the standalone attention tests, without late replacement or module resets.
    # Do not initialize a HIP context in the parent before launching the child.
    code = f"""
import pytest
from tests.accuracy.test_deepseek_v41_flashmla_metadata_accuracy import _run_real_builder_and_model_wrapper
with pytest.MonkeyPatch.context() as monkeypatch:
    _run_real_builder_and_model_wrapper(monkeypatch, {ratio})
"""
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["VLLM_HCU_USE_CUSTOM_OPS"] = "1"
    env["VLLM_HCU_DSV41_FLASHMLA_DECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode == 77:
        pytest.skip("requires gfx938 and native FlashMLA")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "real-builder-wrapper-native-ok" in result.stdout


@pytest.mark.parametrize("mode", ["graph", "verify", "draft", "verify_graph", "draft_graph"])
def test_real_builder_execution_modes(mode):
    code = f"""
import pytest
from tests.accuracy.test_deepseek_v41_flashmla_metadata_accuracy import _run_real_builder_and_model_wrapper
for ratio in {(0,) if mode.startswith('draft') else (0, 1, 2)}:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _run_real_builder_and_model_wrapper(monkeypatch, ratio, {mode!r})
"""
    env = dict(os.environ, VLLM_PLUGINS="__disabled__", VLLM_HCU_USE_CUSTOM_OPS="1",
               VLLM_HCU_DSV41_FLASHMLA_DECODE="1", VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH="1",
               VLLM_HCU_DSV41_FLASHMLA_DSPARK="1")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    if result.returncode == 77:
        pytest.skip("requires gfx938 and native FlashMLA")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "real-builder-wrapper-native-ok" in result.stdout


def _run_real_builder_and_model_wrapper(monkeypatch, ratio, mode="eager"):
    # Use the actual worker dispatcher, including sparse_swa module exchange;
    # no direct replacement of the original builder or GPU metadata kernels.
    from vllm_hcu.patch.worker import prepare_worker_patches

    prepare_worker_patches()
    if not torch.cuda.is_available():
        raise SystemExit(77)
    if not runtime._supported_device(torch.empty(1, device="cuda")):
        raise SystemExit(77)
    if runtime._decode_bindings() is None:
        raise SystemExit(77)
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm_hcu.patch.worker.op_opt import (
        patch_deepseek_v41_rocm_flashmla_decode as decode_patch,
        patch_deepseek_v41_rocm_flashmla_metadata as metadata_patch,
    )

    builder_cls = rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    attention_cls = rocm.DeepseekV41ROCMAiterMLAAttention
    assert getattr(builder_cls, metadata_patch._MARKER)
    assert getattr(attention_cls, decode_patch._MARKER)
    assert not builder_cls.supports_draft_decode_metadata_update
    graph_mode = "graph" in mode
    speculative = mode.startswith(("verify", "draft"))
    noncausal = mode.startswith("draft")
    spec_config = None if not speculative else SimpleNamespace(
        method="dspark", num_speculative_tokens=5, parallel_drafting=True,
        use_dspark=lambda: True,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=512,
            hf_config=SimpleNamespace(sliding_window=128, compress_ratios=[0, 1, 2]),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=3),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
        ),
        speculative_config=spec_config,
    )
    spec = SlidingWindowMLASpec(
        block_size=32, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=128, cache_dtype_str="fp8_ds_mla", state_content_bytes=584,
        alignment=576, model_version="deepseek_v4",
    )
    namespace = "draft" if noncausal else "target"
    names = [f"{namespace}.layer0.swa_cache", f"{namespace}.layer1.swa_cache"]
    builder = builder_cls(spec, names, config, torch.device("cuda"))
    query_lens = [5, 3, 1] if speculative else [1, 1, 1]
    rows = sum(query_lens)
    qstart_cpu = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0).int()
    lens_cpu = torch.tensor([131, 159, 193] if speculative else [32, 65, 97], dtype=torch.int32)
    table = torch.arange(8, device="cuda", dtype=torch.int32).repeat(3, 1)

    def make_common():
        # Real serving builds a NEW CommonAttentionMetadata every step.
        # token_to_req_indices() caches on the metadata object, so mutating a
        # previous instance's query_start_loc would keep the stale mapping.
        slots = torch.cat([
            torch.arange(int(n) - m, int(n), dtype=torch.int64)
            for n, m in zip(lens_cpu, query_lens)
        ]).cuda()
        if speculative:
            slots[-1] = -1  # padded row must not gather stale SWA or history
        return CommonAttentionMetadata(
            query_start_loc=qstart_cpu.cuda(), query_start_loc_cpu=qstart_cpu,
            seq_lens=lens_cpu.cuda(), seq_lens_cpu_upper_bound=lens_cpu,
            num_reqs=3, num_actual_tokens=rows, max_query_len=max(query_lens),
            max_seq_len=int(lens_cpu.max()), block_table_tensor=table,
            slot_mapping=slots, causal=not noncausal,
        )

    common = make_common()
    metadata = builder.build(0, common)
    leases = getattr(metadata, SCHEDULERS_ATTR)
    assert set(leases) == set(names)
    assert metadata.decode_swa_ragged_indices is not None
    assert metadata.decode_swa_ragged_indptr is not None
    assert metadata.max_decode_query_len == max(query_lens)
    if noncausal:
        assert metadata.decode_swa_width > 128
        assert metadata.decode_swa_lens[0].item() == 131
        # First draft query must see the final (future) token of its block.
        assert (metadata.decode_swa_indices[0] == 130).any()
    native, mapping, factory = runtime._decode_bindings()
    launched_plans = []

    def record(**kwargs):
        plan = kwargs["tile_scheduler_metadata"]
        assert not plan.have_initialized
        launched_plans.append(plan)
        return native(**kwargs)

    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (record, mapping, factory))
    torch.manual_seed(1729)
    swa_cache, swa_values = make_cache(32, blocks=8)
    cache, values = (None, None) if ratio == 0 else make_cache(64 // ratio, blocks=8)
    q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    sink = torch.linspace(-2, 2, 64, device="cuda")

    def forward(name, history_length):
        nonlocal common
        # Construct only the attention shell, not projections/model weights.
        attention = attention_cls.__new__(attention_cls)
        torch.nn.Module.__init__(attention)
        setattr(attention, decode_patch._CONTEXT_FLAG, True)
        setattr(attention, decode_patch._SPECULATIVE_FLAG, speculative)
        attention.compress_ratio = ratio
        attention.swa_cache_layer = SimpleNamespace(prefix=name, kv_cache=swa_cache)
        attention.scale = 512**-0.5
        attention.attn_sink = sink
        attention.head_dim, attention.nope_head_dim, attention.rope_head_dim = 512, 448, 64
        attention.kv_cache_dtype = "fp8_ds_mla"
        local = torch.arange(128, device="cuda", dtype=torch.int32).repeat(rows, 1)
        local[:, history_length:] = -1
        attention.topk_indices_buffer = local
        attn_metadata = None if ratio == 0 else SimpleNamespace(block_size=64, block_table=table)
        output = torch.empty_like(q)
        def run():
            assert attention._forward_decode(q, cache, metadata, attn_metadata, ratio == 0, output) is None

        run()  # warmup
        graph = None
        if graph_mode:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        pointers = tuple(getattr(metadata, key).data_ptr() for key in (
            "decode_swa_indices", "decode_swa_lens", "token_to_req_indices", "is_valid_token",
        ))
        for step in range(3 if graph_mode else 1):
            if step:
                # Change lengths AND row->request boundaries without changing
                # the captured row count. Real serving rebuilds the common
                # metadata each step (token_to_req_indices caches per object),
                # so build a fresh one and reuse the persistent buffers.
                if speculative:
                    query_lens[:] = [2, 5, 2] if step == 1 else [5, 3, 1]
                    qstart_cpu.copy_(torch.tensor([0] + query_lens).cumsum(0).int())
                lens_cpu.add_(3 if step == 1 else -4)
                common = make_common()
                updated = builder.build(0, common)
                assert pointers == tuple(getattr(updated, key).data_ptr() for key in (
                    "decode_swa_indices", "decode_swa_lens", "token_to_req_indices", "is_valid_token",
                ))
                q.copy_(torch.randn_like(q) * 0.2)
                local[:, 1::3] = -1
            if graph is not None:
                graph.replay()
            # Independent causal/non-causal visibility reference, not merely
            # reusing the GPU builder's indices as the expected result.
            expected_swa = torch.full((rows, metadata.decode_swa_width), -1, device="cuda", dtype=torch.int32)
            offset = 0
            for request, query_len in enumerate(query_lens):
                end = int(lens_cpu[request])
                prefix = end-query_len
                for i in range(query_len):
                    stop = end if noncausal else prefix+i+1
                    start = max(0, prefix-128) if noncausal else max(0, stop-128)
                    logical = torch.arange(start,stop,device="cuda",dtype=torch.int32)
                    physical = table[request,logical.long()//32]*32+logical%32
                    if common.slot_mapping[offset+i].item() >= 0:
                        expected_swa[offset+i,:physical.numel()] = physical
                offset += query_len
            columns = torch.arange(metadata.decode_swa_width,device="cuda")
            actual = torch.where(columns[None,:] < metadata.decode_swa_lens[:,None], metadata.decode_swa_indices.reshape(rows,-1), -1)
            try:
                torch.testing.assert_close(actual, expected_swa)
            except AssertionError:
                mismatch = actual != expected_swa
                row_ids = torch.nonzero(mismatch.any(dim=1)).flatten().tolist()
                details = []
                for row in row_ids:
                    details.append(
                        f"row={row} qstart={qstart_cpu.tolist()} req={int(torch.searchsorted(qstart_cpu[1:].cuda(), torch.tensor(row, device='cuda'), right=True))} "
                        f"lens={int(metadata.decode_swa_lens[row])} "
                        f"actual={actual[row, :16].tolist()} expected={expected_swa[row, :16].tolist()} "
                        f"actual_tail={actual[row, -8:].tolist()} expected_tail={expected_swa[row, -8:].tolist()}"
                    )
                raise AssertionError("DSpark/verify SWA index mismatch: " + "; ".join(details))
            indices = expected_swa
            decoded = swa_values
            if ratio:
                req = torch.repeat_interleave(torch.arange(3,device="cuda"),torch.tensor(query_lens,device="cuda"))
                page = 64//ratio
                extra = table[req].gather(1,local.clamp_min(0).long()//page)*page+local.clamp_min(0)%page
                extra = torch.where((local>=0)&(common.slot_mapping>=0)[:,None],extra+swa_values.shape[0],-1)
                indices = torch.cat((indices,extra),-1)
                decoded = torch.cat((swa_values,values))
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(),reference(q,decoded,indices,sink),atol=2e-3,rtol=2e-2)

    # Equal shapes/ratio, DIFFERENT history lengths: each layer gets its own plan.
    forward(names[0], 97)
    forward(names[1], 17)
    # Repeated execution of one metadata object also must get a new plan.
    forward(names[0], 33)
    expected_plans = 6 if graph_mode else 3
    assert len(launched_plans) == expected_plans
    assert len({id(plan) for plan in launched_plans}) == expected_plans
    next_metadata = builder.build(0, common)
    assert getattr(next_metadata, SCHEDULERS_ATTR) is not leases
    print("real-builder-wrapper-native-ok", ratio, mode)
