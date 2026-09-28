"""Production-size EPLB counts: exact CPU oracle and bounded scratch."""
from types import SimpleNamespace as NS
import pytest
import torch

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize('steps', [1, 35, 1000])
def test_kimi_load_window_exact_bounded_and_preserved(monkeypatch, steps):
    if not torch.cuda.is_available():
        pytest.skip('HCU required')
    from vllm_hcu.models.kimi_k3.amd.ops.eplb import compact_kimi_load_windows
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('VLLM_HCU_USE_KIMI_HT_EPLB', '1')
    torch.manual_seed(23)
    # 92 MoE layers, 896 logical + 16 redundant physical experts, plus four
    # deliberately inactive slots. Per-step logical sums cannot overflow int32.
    cpu = torch.randint(0, 8, (steps, 92, 916), dtype=torch.int32)
    cpu[:, :, 0] = 100_000_000
    mapping = (torch.arange(912) % 896).expand(92, -1).clone()
    expected = torch.zeros((92, 896), dtype=torch.int64)
    # Independent time-row accumulation, not the GPU reduction expression.
    for row in cpu:
        expected.scatter_add_(-1, mapping, row[:, :912].long())
    window = cpu.cuda()
    indices = mapping.cuda().unsqueeze(0)
    live = NS(expert_load_window=window, model=NS(_vllm_hcu_kimi_eplb_load_window=True))
    state = NS(model_states={'m': live}, is_async=False,
        expert_load_window_size=steps, expert_load_window_step=0,
        parallel_config=NS(enable_eplb=True, all2all_backend='deepep_high_throughput'))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with compact_kimi_load_windows(state, rank_mapping=None):
        logical = torch.zeros((1, 92, 896), dtype=torch.int64, device='cuda')
        logical.scatter_add_(-1, indices, live.expert_load_window[:, :, :912])
        result = logical.sum(0)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    print(dict(steps=steps, extra_peak_bytes=extra), flush=True)
    assert extra < 64 * 1024**2
    assert result.dtype == torch.int64
    assert torch.equal(result.cpu(), expected)
    assert live.expert_load_window is window
    assert torch.equal(window.cpu(), cpu)
    assert state.expert_load_window_size == steps
