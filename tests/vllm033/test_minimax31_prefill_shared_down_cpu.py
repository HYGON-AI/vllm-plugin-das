"""CPU regression for the policy-gated shared-down scheduling handoff."""

from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.parallel import minimax31_deepep_runner as runner


@pytest.mark.parametrize("optimized,rows,capturing,qualified,deferred", [
    (False, 6144, False, True, False),
    (True, 4096, False, True, False),
    (True, 6144, True, True, False),
    (True, 6144, False, False, False),
    (True, 6144, False, True, True),
])
def test_shared_down_handoff_preserves_route_padding_and_join(
    monkeypatch, optimized, rows, capturing, qualified, deferred
):
    monkeypatch.setattr(runner, "prefill_optimized", lambda: optimized)
    monkeypatch.setattr(runner, "_supports_shared_down", lambda device: qualified)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    events = []
    intermediate = object()
    packed = object()
    hidden = torch.zeros((rows, 1), dtype=torch.bfloat16)
    method = SimpleNamespace(
        is_monolithic=False, topk_indices_dtype=torch.int64,
        quant_config=SimpleNamespace(routed_backend="deepgemm_hipc_v1"),
    )
    experts = SimpleNamespace(
        quant_method=method, _ensure_moe_quant_config_init=lambda: None,
    )

    def launch(x, *, defer_down=False):
        assert x is hidden and defer_down == deferred
        events.append("gate_up" if defer_down else "shared_mlp")
        return intermediate if defer_down else True

    def down(x):
        assert x is intermediate
        events.append("shared_down")

    def execute(x, ids, weights, *, expert_weights, **options):
        assert ids[0, 0] == -1 and weights[0, 0] == 0
        assert expert_weights is packed
        events.append("routed")
        if deferred:
            options.pop("before_combine")()
        assert not options
        events.append("combine")
        return x

    monkeypatch.setattr(runner, "_get_aligned_moe", lambda *args: (execute, packed))
    idx = torch.zeros((rows, 1), dtype=torch.int64)
    idx[0, 0] = -1
    obj = SimpleNamespace(
        _pcp_ep_group=lambda: SimpleNamespace(rank_in_group=0),
        gate=None, routed_experts=experts, _quant_method=method,
        router=SimpleNamespace(select_experts=lambda **kw: (torch.ones(rows, 1), idx)),
        _shared_experts=SimpleNamespace(output=hidden, wait=lambda: events.append("join")),
        _launch_long_shared=launch, _launch_shared_down=down, _hcu_shared_pending=False,
    )
    if capturing:
        with pytest.raises(RuntimeError, match="requires eager execution"):
            runner.MiniMax31DeepEpRunner._forward_locked(obj, hidden, None, hidden)
        assert events == ["shared_mlp"]
        return
    result = runner.MiniMax31DeepEpRunner._forward_locked(obj, hidden, None, hidden)
    assert all(value is hidden for value in result)
    expected = ["gate_up", "routed", "shared_down"] if deferred else ["shared_mlp", "routed"]
    assert events == expected + ["combine", "join"]
