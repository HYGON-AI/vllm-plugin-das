"""Narrow, opt-in synchronous Kimi HT EPLB configuration contract."""
import os
from contextlib import contextmanager


def sum_load_window_bounded(window):
    """Sum time in int64 without ROCm's large dim-0 reduction workspace.

    The live window stays int32. Reducing at most 16 time rows per call bounds
    scratch independently of the configured history length (normally 1000).
    """
    import torch
    if window.ndim != 3 or window.shape[0] < 1 or window.dtype not in (torch.int32, torch.int64):
        raise ValueError('Kimi EPLB requires a nonempty integer [time, layer, physical] window')
    total = torch.zeros(window.shape[1:], dtype=torch.int64, device=window.device)
    for chunk in window.split(16, dim=0):
        total.add_(chunk.sum(dim=0, dtype=torch.int64))
    return total.unsqueeze(0)


@contextmanager
def compact_kimi_load_windows(state, *, rank_mapping):
    """Supply an equivalent one-row snapshot only during synchronous rearrange.

    Time reduction commutes with the physical-to-logical scatter for valid
    non-overflowing per-step token counts. Upstream still owns all-reduce,
    policy, transfers and map commits. Neither window contents nor ring cursor
    are changed. Async/elastic and other models retain the exact upstream path.
    """
    models = list(state.model_states.values())
    parallel = state.parallel_config
    eligible = (kimi_ht_eplb_enabled() and not state.is_async
        and rank_mapping is None and getattr(parallel, 'enable_eplb', False)
        and parallel.all2all_backend == 'deepep_high_throughput'
        and models and all(getattr(m.model, '_vllm_hcu_kimi_eplb_load_window', False)
                               for m in models))
    if not eligible:
        yield
        return
    size = state.expert_load_window_size
    windows = [m.expert_load_window for m in models]
    if any(w.shape[0] != size for w in windows):
        raise ValueError('Kimi EPLB window length disagrees with state')
    # Complete all allocations before modifying any state; restoration also
    # covers failures from the original policy, transfer, or commit functions.
    compact = [sum_load_window_bounded(w) for w in windows]
    if not getattr(state, '_vllm_hcu_kimi_compact_window_logged', False):
        from vllm.logger import init_logger
        init_logger(__name__).warning('Kimi HT EPLB bounded load-window reduction active (chunk=16)')
        state._vllm_hcu_kimi_compact_window_logged = True
    try:
        state.expert_load_window_size = 1
        for model, snapshot in zip(models, compact):
            model.expert_load_window = snapshot
        yield
    finally:
        for model, window in zip(models, windows):
            model.expert_load_window = window
        state.expert_load_window_size = size


def kimi_ht_eplb_enabled():
    import vllm_hcu.platforms.envs as henvs
    return henvs.VLLM_HCU_USE_CUSTOM_OPS and os.environ.get(
        'VLLM_HCU_USE_KIMI_HT_EPLB', '0').strip().lower() in ('1', 'true')


def kimi_eplb_options(vllm_config):
    parallel = vllm_config.parallel_config
    if not parallel.enable_eplb:
        return False, 0
    if not kimi_ht_eplb_enabled():
        raise ValueError('Kimi EPLB requires VLLM_HCU_USE_CUSTOM_OPS and VLLM_HCU_USE_KIMI_HT_EPLB')
    if (not parallel.enable_expert_parallel or
            parallel.all2all_backend != 'deepep_high_throughput'):
        raise ValueError('Kimi EPLB currently requires DeepEP HT expert parallelism')
    if parallel.eplb_config.use_async or not vllm_config.model_config.enforce_eager:
        raise ValueError('Kimi EPLB currently requires synchronous rearrangement and eager execution')
    return True, parallel.eplb_config.num_redundant_experts
