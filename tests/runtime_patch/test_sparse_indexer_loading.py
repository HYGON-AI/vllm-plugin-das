# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
MODEL_SOURCE = (REPO / "vllm_hcu/models/deepseek_v2.py").read_text(
    encoding="utf-8"
)


def _load_model_helpers():
    tree = ast.parse(MODEL_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_is_local_indexer_weight",
            "_try_load_quantized_indexer_wk",
            "_rewrite_stacked_param_name",
        }
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "GroupShape": lambda rows, columns: (rows, columns),
        "scaled_dequantize": lambda weight, scale, *, group_shape, out_dtype: (
            weight.to(out_dtype) * 0 + scale.flatten()[0].to(out_dtype)
        ),
    }
    exec(compile(module, "deepseek_v2_helpers", "exec"), namespace)
    return namespace


def _load_v32_sparse_indexer_contract(**dependencies):
    source = (
        REPO / "vllm_hcu/model_executor/layers/sparse_attn_indexer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "V32SparseAttnIndexer"
    )
    method = copy.deepcopy(
        next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_hip"
        )
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(dependencies)
    exec(compile(module, "v32_sparse_indexer_forward_hip", "exec"), namespace)
    return namespace["forward_hip"]


def _load_v32_sparse_indexer_class():
    source = (
        REPO / "vllm_hcu/model_executor/layers/sparse_attn_indexer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "V32SparseAttnIndexer"
        )
    )
    class_node.bases = [ast.Name(id="SparseAttnIndexer", ctx=ast.Load())]
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "SparseAttnIndexer": object}
    exec(compile(module, "v32_sparse_indexer_class", "exec"), namespace)
    return namespace["V32SparseAttnIndexer"]


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantized_indexer_wk_loader_source_contract_supports_int8_and_fp8(
    dtype: torch.dtype,
):
    helpers = _load_model_helpers()
    loader = helpers["_try_load_quantized_indexer_wk"]
    calls: list[tuple[torch.Tensor, int]] = []
    param = SimpleNamespace(
        weight_loader=lambda _param, tensor, shard: calls.append((tensor, shard))
    )
    params = {"model.layers.0.self_attn.indexer.wk_weights_proj.weight": param}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()
    weight = torch.ones((4, 8), dtype=dtype)
    scale = torch.full((2, 2), 2.0)

    assert loader(
        "model.layers.0.self_attn.indexer.wk.weight",
        weight,
        pending,
        params,
        loaded,
    )
    assert calls == []
    assert loader(
        "model.layers.0.self_attn.indexer.wk.weight_scale",
        scale,
        pending,
        params,
        loaded,
    )
    assert calls[0][0].dtype == torch.bfloat16
    assert calls[0][1] == 0
    assert loaded == set(params)
    assert pending == {}


def test_quantized_indexer_wk_loader_source_contract_rejects_bad_scale_shape():
    helpers = _load_model_helpers()
    loader = helpers["_try_load_quantized_indexer_wk"]
    prefix = "model.layers.0.self_attn.indexer"
    pending = {
        prefix: {
            "weight": torch.ones((4, 8), dtype=torch.int8),
        }
    }
    with pytest.raises(ValueError, match="not divisible"):
        loader(
            f"{prefix}.wk.weight_scale",
            torch.ones((3, 2)),
            pending,
            {},
            set(),
        )


def test_quantized_indexer_wk_loader_skips_pp_missing_layer_before_buffering():
    loader = _load_model_helpers()["_try_load_quantized_indexer_wk"]
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()

    assert loader(
        "model.layers.0.self_attn.indexer.wk.weight",
        torch.ones((4, 8), dtype=torch.int8),
        pending,
        {},
        loaded,
        pp_missing_layer_names=("model.layers.0",),
    )
    assert pending == {}
    assert loaded == set()


def test_local_indexer_weight_filter_is_layer_specific():
    is_local = _load_model_helpers()["_is_local_indexer_weight"]
    local_prefixes = {"model.layers.40.self_attn"}

    assert is_local(
        "model.layers.40.self_attn.indexer.wk.weight",
        local_prefixes,
    )
    assert not is_local(
        "model.layers.41.self_attn.indexer.wk.weight",
        local_prefixes,
    )
    assert is_local("model.layers.41.mlp.down_proj.weight", local_prefixes)


def test_stacked_name_rewrite_source_contract_is_component_bounded():
    rewrite = _load_model_helpers()["_rewrite_stacked_param_name"]
    name = "model.gate_proj_alias.gate_proj.weight"
    assert rewrite(name, "gate_proj", "gate_up_proj") == (
        "model.gate_proj_alias.gate_up_proj.weight"
    )


def test_v32_pcp_gathers_k_and_slots_before_hcu_cache_insertion():
    events: list[tuple[object, ...]] = []
    local_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gathered_k = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    expanded_slots = torch.tensor([10, 11, 20], dtype=torch.int64)
    gathered_slots = torch.tensor([10, 11, 20], dtype=torch.int64)
    metadata = SimpleNamespace(
        slot_mapping=expanded_slots,
        pcp_world_size=2,
        num_decode_tokens=0,
        num_prefills=2,
    )

    def gather(k, slots, actual_metadata):
        events.append(("gather", k, slots, actual_metadata))
        return gathered_k, gathered_slots

    def cache_insert(k, cache, slots, block_size, scale_fmt):
        events.append(
            ("cache", k, cache, slots, block_size, scale_fmt)
        )

    def hcu_op(*args):
        events.append(("hcu_op", *args))
        return "topk"

    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        ops=SimpleNamespace(vllm=SimpleNamespace(hcu_sparse_attn_indexer=hcu_op)),
    )
    forward_hip = _load_v32_sparse_indexer_contract(
        torch=fake_torch,
        effective_pcp_world_size=lambda value: value,
        get_forward_context=lambda: SimpleNamespace(
            attn_metadata={"indexer": metadata}
        ),
        maybe_gather_indexer_k=gather,
        ops=SimpleNamespace(indexer_k_quant_and_cache=cache_insert),
        on_gfx938=lambda: True,
        indexer_k_bf16_cache_triton=lambda *args: pytest.fail(
            "gfx938 PCP cache write used the BF16 fallback"
        ),
        _encode_layer_name=lambda value: value,
    )
    cache = object()
    q_quant = torch.tensor([[7.0, 8.0]])
    weights = torch.tensor([[9.0]])
    hidden_states = torch.tensor([[10.0]])
    indexer = SimpleNamespace(
        use_fp4_cache=False,
        use_pcp=True,
        pcp_world_size=2,
        skip_k_cache_insert=False,
        k_cache=SimpleNamespace(prefix="indexer", kv_cache=cache),
        quant_block_size=128,
        scale_fmt="e8m0",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=65536,
        max_total_seq_len=65536,
        topk_indices_buffer=object(),
    )

    assert forward_hip(indexer, hidden_states, q_quant, local_k, weights) == "topk"

    assert [event[0] for event in events] == ["gather", "cache", "hcu_op"]
    assert events[0][1] is local_k
    assert events[0][2] is expanded_slots
    assert events[0][3] is metadata
    assert events[1][1] is gathered_k
    assert events[1][2] is cache
    assert events[1][3] is gathered_slots
    hcu_args = events[2][1:]
    assert hcu_args[0] is hidden_states
    assert hcu_args[3] is q_quant
    assert hcu_args[4] is local_k
    assert hcu_args[5] is weights
    assert hcu_args[8] == 2048
    assert hcu_args[-1] is True


def test_v32_replicated_mtp_batch_bypasses_static_pcp_indexer_state():
    """The global MTP draft must use one-rank metadata and cache ownership."""

    calls: list[tuple[object, ...]] = []

    def hcu_op(*args):
        calls.append(args)
        return "topk"

    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        ops=SimpleNamespace(vllm=SimpleNamespace(hcu_sparse_attn_indexer=hcu_op)),
    )
    forward_hip = _load_v32_sparse_indexer_contract(
        torch=fake_torch,
        effective_pcp_world_size=lambda _value: 1,
        get_forward_context=lambda: pytest.fail(
            "replicated MTP draft re-entered PCP indexer cache gathering"
        ),
        maybe_gather_indexer_k=lambda *args: pytest.fail(
            "replicated MTP draft gathered PCP indexer inputs"
        ),
        ops=SimpleNamespace(
            indexer_k_quant_and_cache=lambda *args: pytest.fail(
                "replicated MTP draft used external PCP cache insertion"
            )
        ),
        on_gfx938=lambda: True,
        indexer_k_bf16_cache_triton=lambda *args: pytest.fail(
            "replicated MTP draft used PCP BF16 cache insertion"
        ),
        _encode_layer_name=lambda value: value,
    )
    local_k = torch.ones(2, 2)
    q_quant = torch.ones(2, 2)
    indexer = SimpleNamespace(
        use_fp4_cache=False,
        use_pcp=True,
        pcp_world_size=2,
        skip_k_cache_insert=False,
        k_cache=SimpleNamespace(prefix="indexer", kv_cache=object()),
        quant_block_size=128,
        scale_fmt="e8m0",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=65536,
        max_total_seq_len=65536,
        topk_indices_buffer=object(),
    )

    assert forward_hip(indexer, object(), q_quant, local_k, object()) == "topk"
    assert len(calls) == 1
    assert calls[0][4] is local_k
    assert calls[0][-1] is False


def test_v32_hcu_indexer_impl_advertises_pcp_capability():
    assert _load_v32_sparse_indexer_class().supports_pcp is True


def test_indexer_metadata_adapter_propagates_pcp_world_size():
    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_mla_indexer"
    )

    def split_chunks(
        seq_lens_cpu,
        query_lens_cpu,
        workspace_size,
        max_logits_bytes,
        request_offset=0,
    ):
        return [(slice(0, 1), slice(0, 1))]

    def split_batch(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        return (0, 1, 0, common_attn_metadata.num_actual_tokens)

    class Builder:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            return SimpleNamespace(decode=None)

    module = ModuleType(adapter.TARGET_MODULE)
    module.split_indexer_prefill_chunks = split_chunks
    module.split_decodes_and_prefills = split_batch
    module.DeepseekV32IndexerMetadataBuilder = Builder
    module.current_platform = SimpleNamespace(is_rocm=lambda: False)
    assert adapter.apply_to_module(module) is True
    builder = Builder()
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    common = SimpleNamespace(
        num_actual_tokens=2,
        num_kv_actual_tokens=3,
    )

    metadata = builder.build(0, common)

    assert metadata.pcp_world_size == 2


def test_rocm_indexer_metadata_adapter_skips_unused_lightop_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTP2 flattening must not call a schedule builder no HCU path consumes."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_mla_indexer"
    )

    def split_chunks(
        seq_lens_cpu,
        query_lens_cpu,
        workspace_size,
        max_logits_bytes,
        request_offset=0,
    ):
        del (
            seq_lens_cpu,
            query_lens_cpu,
            workspace_size,
            max_logits_bytes,
            request_offset,
        )
        return []

    def split_batch(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        del (
            common_attn_metadata,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes,
        )
        return (1, 0, 3, 0)

    upstream_schedule = object()

    class Builder:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            del common_prefix_len, common_attn_metadata, fast_build
            return SimpleNamespace(
                decode=SimpleNamespace(
                    seq_lens=torch.tensor([[5], [6], [7]], dtype=torch.int32),
                    schedule_metadata=upstream_schedule,
                )
            )

    module = ModuleType(adapter.TARGET_MODULE)
    module.split_indexer_prefill_chunks = split_chunks
    module.split_decodes_and_prefills = split_batch
    module.DeepseekV32IndexerMetadataBuilder = Builder
    module.current_platform = SimpleNamespace(is_rocm=lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "lightop",
        SimpleNamespace(
            gemmopt=SimpleNamespace(
                get_paged_mqa_logits_metadata=lambda *args: pytest.fail(
                    "ROCm built unused lightop schedule metadata"
                )
            )
        ),
    )
    assert adapter.apply_to_module(module) is True
    builder = Builder()
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    builder.kv_cache_spec = SimpleNamespace(storage_block_size=64)
    builder.num_sms = 64
    common = SimpleNamespace(num_actual_tokens=3)

    metadata = builder.build(0, common)

    assert metadata.decode.schedule_metadata is upstream_schedule


def test_rocm_lightop_paged_mqa_uses_categorized_kernel_and_builds_its_schedule_internally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime contract permits the metadata adapter to skip precompute."""

    runtime = importlib.import_module(
        "vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse"
    )
    from vllm._aiter_ops import rocm_aiter_ops

    calls: list[tuple[object, ...]] = []
    output = object()

    def paged_mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(paged_mqa_logits=paged_mqa_logits),
        raising=False,
    )
    monkeypatch.delattr(runtime, "gemmopt", raising=False)
    supplied_schedule = object()
    weights = torch.arange(16, dtype=torch.float16).reshape(8, 2).transpose(0, 1)

    result = runtime.rocm_fp8_paged_mqa_logits(
        torch.empty((2, 1, 8, 128)),
        torch.empty((1, 64, 1, 132), dtype=torch.uint8),
        weights,
        torch.ones((2, 1), dtype=torch.int32),
        torch.zeros((2, 1), dtype=torch.int32),
        supplied_schedule,
        64,
    )

    assert result is output
    assert len(calls) == 1
    assert calls[0][5] is None
    assert calls[0][-1] is True
    assert calls[0][2].dtype is torch.float32
    assert calls[0][2].is_contiguous()
    assert torch.equal(calls[0][2], weights.float().contiguous())


def test_v32_pcp_one_preserves_existing_hcu_custom_op_ownership():
    calls: list[tuple[object, ...]] = []

    def hcu_op(*args):
        calls.append(args)
        return "topk"

    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        ops=SimpleNamespace(vllm=SimpleNamespace(hcu_sparse_attn_indexer=hcu_op)),
    )
    forward_hip = _load_v32_sparse_indexer_contract(
        torch=fake_torch,
        effective_pcp_world_size=lambda value: value,
        get_forward_context=lambda: pytest.fail(
            "PCP=1 inspected forward metadata outside the custom op"
        ),
        maybe_gather_indexer_k=lambda *args: pytest.fail(
            "PCP=1 gathered sparse-indexer cache inputs"
        ),
        ops=SimpleNamespace(
            indexer_k_quant_and_cache=lambda *args: pytest.fail(
                "PCP=1 moved cache ownership outside the custom op"
            )
        ),
        on_gfx938=lambda: True,
        indexer_k_bf16_cache_triton=lambda *args: pytest.fail(
            "PCP=1 moved cache ownership outside the custom op"
        ),
        _encode_layer_name=lambda value: value,
    )
    local_k = torch.ones(1, 2)
    q_quant = torch.ones(1, 2)
    indexer = SimpleNamespace(
        use_fp4_cache=False,
        use_pcp=False,
        pcp_world_size=1,
        skip_k_cache_insert=False,
        k_cache=SimpleNamespace(prefix="indexer", kv_cache=object()),
        quant_block_size=128,
        scale_fmt="e8m0",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=65536,
        max_total_seq_len=65536,
        topk_indices_buffer=object(),
    )

    assert forward_hip(indexer, object(), q_quant, local_k, object()) == "topk"
    assert len(calls) == 1
    assert calls[0][4] is local_k
    assert calls[0][-1] is False
