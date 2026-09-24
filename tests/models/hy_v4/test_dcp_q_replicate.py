# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import torch

from vllm_hcu.model_executor.layers.mla_runtime import (
    _dcp_a2a_lse_reduce_with_overlap,
    mla_forward_impl,
)
from vllm_hcu.models.hy_v4.attention import (
    DCPGroupColumnParallelLinear,
    dcp_q_replication_enabled,
    resolve_dcp_q_replication_topology,
)
from vllm_hcu.models.hy_v4 import attention as attention_module


def test_dcp_a2a_overlap_launches_gate_after_local_attention(monkeypatch) -> None:
    events: list[str] = []

    class FakeEvent:
        def __init__(self, name: str):
            self.name = name

        def record(self, stream) -> None:
            events.append(f"record_{self.name}_{stream.name}")

    class FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_event(self, event) -> None:
            events.append(f"{self.name}_wait_{event.name}")

    class StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            events.append(f"enter_{self.stream.name}")

        def __exit__(self, *_args):
            events.append(f"exit_{self.stream.name}")

    main = FakeStream("main")
    aux = FakeStream("aux")
    start = FakeEvent("start")
    done = FakeEvent("done")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: StreamContext(stream))
    layer = SimpleNamespace(
        _hcu_dcp_a2a_overlap_fn=lambda: events.append("gate"),
        _hcu_dcp_a2a_overlap_stream=aux,
        _hcu_dcp_a2a_overlap_start_event=start,
        _hcu_dcp_a2a_overlap_done_event=done,
    )
    reduce_fn = lambda *args, **kwargs: events.append("a2a") or "out"

    result = _dcp_a2a_lse_reduce_with_overlap(
        reduce_fn,
        layer,
        torch.empty(0),
        torch.empty(0),
        object(),
        is_lse_base_on_e=True,
    )

    assert result == "out"
    assert events == [
        "record_start_main",
        "enter_aux",
        "aux_wait_start",
        "gate",
        "record_done_aux",
        "exit_aux",
        "a2a",
        "main_wait_done",
    ]


def test_dcp_q_replication_uses_official_environment_flag(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    assert dcp_q_replication_enabled() is False
    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "1")
    assert dcp_q_replication_enabled() is True


def test_dcp_q_replication_topology_shards_across_groups() -> None:
    topology = resolve_dcp_q_replication_topology(
        tp_rank=3,
        tp_world_size=8,
        dcp_world_size=2,
    )

    assert topology.group_size == 2
    assert topology.rank_in_group == 1
    assert topology.tp_rank == 1
    assert topology.tp_world_size == 4


def test_dcp_q_replication_local_view_selects_rank_head_shard() -> None:
    layer = object.__new__(DCPGroupColumnParallelLinear)
    layer.group_size = 2
    layer.rank_in_group = 1
    full_group_q = torch.arange(16).view(1, 4, 4)

    local_q = layer._local_view(full_group_q)

    torch.testing.assert_close(local_q, full_group_q[:, 2:4])
    assert local_q.is_contiguous()


def test_dcp_q_projection_loads_one_shared_shard_per_group(monkeypatch) -> None:
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(
        attention_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=2)
        ),
    )
    monkeypatch.setattr(attention_module, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        attention_module, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 8
    )
    layer = DCPGroupColumnParallelLinear(3, 8, bias=False)
    checkpoint_weight = torch.arange(24, dtype=torch.float32).view(8, 3)

    layer.weight_loader(layer.weight, checkpoint_weight)

    assert layer.weight.shape == (2, 3)
    assert layer.is_quantization is False
    torch.testing.assert_close(layer.weight, checkpoint_weight[2:4])


def test_hyv4_mla_prepares_group_wuk_for_replicated_q(monkeypatch) -> None:
    events: list[object] = []

    class Group:
        def all_gather(self, tensor, dim):
            events.append((tensor.clone(), dim))
            return torch.cat((tensor, tensor + 10), dim=dim)

    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.MLAAttention.process_weights_after_loading",
        lambda self, dtype: events.append(("parent", dtype)),
    )
    monkeypatch.setattr(attention_module, "get_dcp_group", lambda: Group())
    layer = object.__new__(attention_module.HYV4MLAAttentionLayer)
    layer._hcu_dcp_q_replicate = True
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.W_UK_T = torch.arange(2, dtype=torch.float32).view(1, 1, 2)
    layer.impl = SimpleNamespace(
        process_weights_after_loading=lambda dtype: events.append(("impl", dtype))
    )

    layer.process_weights_after_loading(torch.bfloat16)

    torch.testing.assert_close(
        layer.W_UK_T_dcp_qrep,
        torch.tensor([[[0.0, 1.0]], [[10.0, 11.0]]]),
    )
    assert events[0] == ("parent", torch.bfloat16)
    assert events[-1] == ("impl", torch.bfloat16)


def test_mla_q_replication_skips_query_all_gather(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    class SparseBase:
        pass

    class Impl(SparseBase):
        dcp_world_size = 2
        supports_quant_query_input = False

        def forward_mqa(self, q, kv_cache, metadata, layer):
            del kv_cache, metadata, layer
            captured["q"] = q.clone()
            return torch.ones((1, 2, 1)), torch.zeros((1, 2))

    class DcpGroup:
        world_size = 2

        def all_gather(self, tensor, dim):
            raise AssertionError("replicated Q must not run query all-gather")

    upstream = SimpleNamespace(
        _detect_output_quant_key=lambda *args: None,
        is_quantized_kv_cache=lambda value: value == "fp8_ds_mla",
        SparseMLAAttentionImpl=SparseBase,
        get_dcp_group=lambda: DcpGroup(),
        cp_lse_ag_out_rs=lambda out, lse, group, is_lse_base_on_e: out[:, :1],
    )
    layer = SimpleNamespace(
        impl=Impl(),
        kv_cache_dtype="fp8_ds_mla",
        num_heads=1,
        v_head_dim=1,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        chunked_prefill_workspace_size=1,
        q_pad_num_heads=None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        W_UK_T=torch.ones((1, 1, 1)),
        W_UK_T_dcp_qrep=torch.tensor([[[1.0]], [[2.0]]]),
        dcp_a2a=False,
        _use_fi_prefill=False,
    )

    def v_up_proj(x, out):
        out.copy_(x.reshape_as(out))

    layer._v_up_proj = v_up_proj
    metadata = SimpleNamespace(num_actual_tokens=1, num_kv_actual_tokens=1)
    local_q = torch.tensor([[[11.0, 13.0]]])
    replicated_q = torch.tensor([[[2.0, 5.0], [3.0, 7.0]]])
    output = torch.empty((1, 1))

    result = mla_forward_impl(
        upstream,
        layer,
        local_q,
        torch.ones((1, 1)),
        torch.ones((1, 1, 1)),
        torch.empty(0),
        metadata,
        output,
        q_dcp_replicated=replicated_q,
    )

    assert result is output
    torch.testing.assert_close(
        captured["q"],
        torch.tensor([[[2.0, 5.0], [6.0, 7.0]]]),
    )
