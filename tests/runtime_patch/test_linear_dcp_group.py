from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_hcu.model_executor.layers import linear


def test_dcp_group_column_parallel_linear_uses_group_shard(
    monkeypatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_column_init(self, *args, **kwargs) -> None:
        del self, args
        captured.update(tp_rank=kwargs["tp_rank"], tp_size=kwargs["tp_size"])

    monkeypatch.setattr(
        linear,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=2)
        ),
    )
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(linear.ColumnParallelLinear, "__init__", fake_column_init)

    layer = linear.DCPGroupColumnParallelLinear(16, 32)

    assert captured == {"tp_rank": 1, "tp_size": 4}
    assert layer.group_size == 2
    assert layer.qrep_active is True
    assert layer.rank_in_group == 1
    out = torch.arange(16).reshape(1, 8, 2)
    torch.testing.assert_close(layer._local_view(out), out[:, 4:8, :])
