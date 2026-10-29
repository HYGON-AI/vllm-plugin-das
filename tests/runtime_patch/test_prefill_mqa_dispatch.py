# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of the production native indexer's grouping dispatch.

Extract the complete function so importing GPU decorators is unnecessary. Its
control flow and grouping helper are unmodified; kernel and collective leaves
use CPU implementations. No CUDA/HIP device or process group is initialized.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm_hcu/v1/attention/ops"


class _GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


def _helper_module():
    spec = importlib.util.spec_from_file_location(
        "prefill_mqa_dispatch_helper", OPS / "prefill_mqa.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_function(namespace):
    path = OPS / "rocm_aiter_mla_sparse.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    native = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "rocm_aiter_sparse_attn_indexer_native"
    )
    tree = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ), native],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[native.name]


@pytest.mark.parametrize(
    "enabled,mode,capturing,dcp,aiter,has_plan,expected_group_calls,expected_mqa_rows",
    [
        (True, _GraphMode.NONE, False, 1, False, True, 1, [1, 1]),
        (False, _GraphMode.NONE, False, 1, False, True, 0, [2]),
        (True, _GraphMode.PIECEWISE, False, 1, False, True, 0, [2]),
        (True, _GraphMode.FULL, False, 1, False, True, 0, [2]),
        (True, None, False, 1, False, True, 0, [2]),
        (True, _GraphMode.NONE, True, 1, False, True, 0, [2]),
        (True, _GraphMode.NONE, False, 2, False, True, 0, [2]),
        (True, _GraphMode.NONE, False, 1, True, True, 0, [2]),
        (True, _GraphMode.NONE, False, 1, False, False, 1, [2]),
    ],
    ids=["eligible", "flag-off", "piecewise", "full-graph", "unknown-mode",
         "stream-capture", "dcp-score-exchange", "aiter", "missing-plan"],
)
def test_native_prefill_grouping_dispatch_and_dcp_scores(
    monkeypatch, enabled, mode, capturing, dcp, aiter, has_plan,
    expected_group_calls, expected_mqa_rows,
):
    helper = _helper_module()
    group_calls, logits_calls, topk_calls, merge_calls = [], [], [], []
    chunk = SimpleNamespace(
        total_seq_lens=4,
        local_total_seq_lens=4,
        max_local_total_seq_lens=4,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([2, 4], dtype=torch.int32),
        token_start=0,
        token_end=2,
    )
    if has_plan:
        chunk.hcu_mqa_groups = helper.plan_prefill_mqa_chunks(
            [1, 1], [2, 2], min_saved_cells=0
        )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        num_kv_actual_tokens=2, num_decodes=0, num_decode_tokens=0,
        num_prefills=2, prefill=SimpleNamespace(chunks=[chunk]), decode=None,
    )
    context = SimpleNamespace(attn_metadata={"layer": metadata})
    if mode is not None:
        context.cudagraph_runtime_mode = mode

    def gather(cache, keys, scales, block_table, offsets):
        assert offsets is chunk.cu_seq_lens
        keys.copy_(torch.tensor([[1.0], [2.0], [4.0], [8.0]]))
        scales.view(torch.float32).fill_(1)

    def mqa(q, kv, weights, starts, ends):
        keys, scales = kv
        logits = (q[:, 0] @ keys.T) * scales.reshape(1, -1)
        logits_calls.append(logits)
        return logits

    def topk(logits, starts, ends, out, count):
        assert count == 1
        topk_calls.append(logits)
        for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            assert 0 <= start < end <= logits.shape[1]
            out[row, 0] = logits[row, start:end].argmax()

    actual_grouped = helper.run_grouped_prefill_mqa

    def grouped(*args, **kwargs):
        group_calls.append(args[7])
        return actual_grouped(*args, **kwargs)

    helper.run_grouped_prefill_mqa = grouped

    def merge(logits, indices, count, rank, world_size, interleave, *, row_starts):
        assert world_size == 2
        assert rank == 0 and count == 1 and interleave == 1
        assert row_starts is chunk.cu_seqlen_ks
        # DCP must receive the original full logits, not a grouped score matrix.
        assert logits is logits_calls[0]
        assert logits.shape == (2, 4)
        merge_calls.append((logits, indices.clone()))

    ops = _module("vllm._custom_ops", cp_gather_indexer_k_quant_cache=gather)
    torch_utils = _module(
        "vllm.utils.torch_utils", _resolve_layer_name=lambda value: value
    )
    modules = {
        "vllm": _module("vllm", _custom_ops=ops),
        "vllm._custom_ops": ops,
        "vllm.utils": _module("vllm.utils", torch_utils=torch_utils),
        "vllm.utils.torch_utils": torch_utils,
        "vllm._aiter_ops": _module(
            "vllm._aiter_ops",
            rocm_aiter_ops=SimpleNamespace(is_enabled=lambda: aiter),
        ),
        "vllm_hcu.v1.attention.ops.prefill_mqa": helper,
        "vllm_hcu.model_executor.layers.sparse_attn_indexer": _module(
            "vllm_hcu.model_executor.layers.sparse_attn_indexer",
            _merge_dcp_topk_global=merge,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    native = _native_function({
        "torch": torch,
        "get_forward_context": lambda: context,
        "current_platform": SimpleNamespace(
            is_rocm=lambda: True, fp8_dtype=lambda: torch.float32
        ),
        "on_gfx938": lambda: True,
        "DeepseekV32IndexerMetadata": SimpleNamespace,
        "henvs": SimpleNamespace(VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING=enabled),
        "CUDAGraphMode": _GraphMode,
        "_use_lightop_sparse_mla_topk": lambda: True,
        "rocm_fp8_mqa_logits": mqa,
        "_lightop_topk_indices_prefill": topk,
    })
    out = torch.full((2, 1), -77, dtype=torch.int32)
    result = native(
        hidden_states=torch.zeros((2, 1)), k_cache_prefix="layer",
        kv_cache=torch.zeros((1, 4, 5), dtype=torch.uint8),
        q_fp8=torch.tensor([[[1.0]], [[2.0]]]), k=torch.zeros((2, 1)),
        weights=torch.ones((2, 1)), quant_block_size=1, scale_fmt="e4m3",
        topk_tokens=1, head_dim=1, max_model_len=4, total_seq_lens=4,
        topk_indices_buffer=out, skip_k_cache_insert=True, dcp_world_size=dcp,
    )
    assert result is out
    assert out.tolist() == [[1], [1]]
    assert len(group_calls) == expected_group_calls
    assert [logits.shape[0] for logits in logits_calls] == expected_mqa_rows
    assert len(topk_calls) == len(logits_calls)
    assert all(selected is logits for selected, logits in zip(topk_calls, logits_calls))
    assert len(merge_calls) == (1 if dcp > 1 else 0)
    if dcp > 1:
        torch.testing.assert_close(
            merge_calls[0][0], torch.tensor([[1., 2., 4., 8.], [2., 4., 8., 16.]])
        )
        torch.testing.assert_close(merge_calls[0][1], out)


@pytest.mark.parametrize("mismatch", ["missing-parameter", "keyword-only", "default"])
def test_chunk_builder_signature_drift_fails_before_patching(mismatch):
    from vllm_hcu.patch.worker.op_opt import patch_mla_indexer as adapter

    def chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
               max_logits_bytes, request_offset=0):
        return []

    def split(common_attn_metadata, decode_threshold=1, require_uniform=False,
              treat_short_extends_as_decodes=True):
        return ()

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return None

    def chunk(start_idx, end_idx, query_start_loc, query_start_loc_cpu,
              uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
              block_table, compress_ratio, query_slice=None, skip_kv_gather=False,
              dcp_rank=0, dcp_world_size=1, cp_kv_cache_interleave_size=1):
        return None

    params = list(inspect.signature(chunk).parameters.values())
    if mismatch == "missing-parameter":
        params.pop()
    elif mismatch == "keyword-only":
        params[-1] = params[-1].replace(kind=inspect.Parameter.KEYWORD_ONLY)
    else:
        params[-1] = params[-1].replace(default=2)
    chunk.__signature__ = inspect.Signature(params)
    upstream = _module(
        adapter.TARGET_MODULE, DeepseekV32IndexerMetadataBuilder=Builder,
        split_indexer_prefill_chunks=chunks, split_decodes_and_prefills=split,
        build_prefill_chunk_metadata=chunk,
    )
    with pytest.raises(adapter.PatchCompatibilityError, match="build_prefill_chunk_metadata"):
        adapter.apply_to_module(upstream)
    assert upstream.build_prefill_chunk_metadata is chunk
    assert upstream.split_indexer_prefill_chunks is chunks
    assert upstream.split_decodes_and_prefills is split
    assert not getattr(upstream, adapter._MARKER, False)
