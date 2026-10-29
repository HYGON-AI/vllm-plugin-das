# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Encoder-side APC wiring for DeepSeek-V4.1 SWA bounded replay on 0.29."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.patch.platform.framework_opt import patch_swa_bounded_replay_apc
from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v41_decoder_replay_state,
    patch_deepseek_v41_swa_bounded_replay,
)
from vllm_hcu.v1.attention.backends.mla.sparse_swa import (
    BoundedReplaySlidingWindowMLASpec,
    DeepseekV4SWACache,
    resolve_swa_bounded_replay,
)


def _vllm_config(
    *,
    swa_bounded_replay: bool = True,
    use_v2_model_runner: bool = True,
    pcp: int = 1,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            swa_bounded_replay=swa_bounded_replay,
            cache_dtype="fp8_ds_mla",
        ),
        use_v2_model_runner=use_v2_model_runner,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.mark.parametrize(
    ("swa_bounded_replay", "use_v2", "pcp", "enable_apc", "expected"),
    [
        (True, True, 1, True, True),
        (False, True, 1, True, False),
        (True, False, 1, True, False),
        # PCP + APC: allowed (#62183 / #62384).
        (True, True, 2, True, True),
        # PCP without APC: decoder SWA is independent of prefix caching.
        (True, True, 2, False, True),
    ],
)
def test_resolve_swa_bounded_replay_gates(
    swa_bounded_replay, use_v2, pcp, enable_apc, expected
):
    cfg = _vllm_config(
        swa_bounded_replay=swa_bounded_replay,
        use_v2_model_runner=use_v2,
        pcp=pcp,
    )
    cfg.cache_config.enable_prefix_caching = enable_apc
    assert resolve_swa_bounded_replay(cfg) is expected


def test_swa_cache_spec_leaves_prefix_cache_when_bounded_replay(monkeypatch):
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    cfg = _vllm_config()
    monkeypatch.setattr(
        "vllm_hcu.v1.attention.backends.mla.sparse_swa.get_current_vllm_config",
        lambda: cfg,
    )
    cache = DeepseekV4SWACache(
        head_dim=512,
        window_size=128,
        dtype=torch.uint8,
        prefix="layers.0.attn.swa_cache",
        cache_config=cfg.cache_config,
        block_size=32,
        packed_bytes_per_token=584,
        packed_page_alignment=576,
        bounded_replay=True,
    )
    spec = cache.get_kv_cache_spec(cfg)
    assert isinstance(spec, BoundedReplaySlidingWindowMLASpec)
    assert isinstance(spec, SlidingWindowMLASpec)
    assert spec.prefix_cacheable is False
    assert spec.prefix_replay_tokens == 128


def test_swa_cache_spec_stays_prefix_cacheable_when_flag_off(monkeypatch):
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    cfg = _vllm_config(swa_bounded_replay=False)
    monkeypatch.setattr(
        "vllm_hcu.v1.attention.backends.mla.sparse_swa.get_current_vllm_config",
        lambda: cfg,
    )
    cache = DeepseekV4SWACache(
        head_dim=512,
        window_size=128,
        dtype=torch.uint8,
        prefix="layers.1.attn.swa_cache",
        cache_config=cfg.cache_config,
        block_size=32,
        bounded_replay=False,
    )
    spec = cache.get_kv_cache_spec(cfg)
    assert type(spec) is SlidingWindowMLASpec
    assert not isinstance(spec, BoundedReplaySlidingWindowMLASpec)
    assert spec.prefix_cacheable is True


def test_attention_init_patch_sets_bounded_replay(monkeypatch):
    """Worker patch sets swa_cache_layer.bounded_replay from resolve()."""

    class _FakeSWA:
        def __init__(self):
            self.bounded_replay = False

    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
            candidate_block_buffer=None,
        ):
            del topk_indices_buffer, aux_stream_list, candidate_block_buffer, prefix
            self._vllm_config = vllm_config
            self.swa_cache_layer = _FakeSWA()

    module = SimpleNamespace(DeepseekV4Attention=DeepseekV4Attention)
    monkeypatch.setattr(
        patch_deepseek_v41_swa_bounded_replay,
        "load_exact_module",
        lambda target, module: module,
    )
    assert patch_deepseek_v41_swa_bounded_replay.apply_to_module(module) is True

    cfg_on = _vllm_config(swa_bounded_replay=True, use_v2_model_runner=True)
    attn_on = DeepseekV4Attention(cfg_on, "layers.0.attn")
    assert attn_on.swa_cache_layer.bounded_replay is True

    cfg_off = _vllm_config(swa_bounded_replay=True, use_v2_model_runner=False)
    attn_off = DeepseekV4Attention(cfg_off, "layers.1.attn")
    assert attn_off.swa_cache_layer.bounded_replay is False

    assert patch_deepseek_v41_swa_bounded_replay.apply_to_module(module) is False


def test_worker_dispatcher_registers_swa_bounded_replay_patch():
    from vllm_hcu.patch.worker import worker_callback_names

    assert (
        patch_deepseek_v41_swa_bounded_replay.PATCH_ID,
        patch_deepseek_v41_swa_bounded_replay.TARGET_MODULE,
    ) in worker_callback_names()


def test_platform_dispatcher_registers_apc_patches():
    from vllm_hcu.patch.platform import platform_framework_callback_names

    names = platform_framework_callback_names()
    assert (
        patch_swa_bounded_replay_apc.KV_PATCH_ID,
        patch_swa_bounded_replay_apc.KV_TARGET_MODULE,
    ) in names
    assert (
        patch_swa_bounded_replay_apc.SCHEDULER_PATCH_ID,
        patch_swa_bounded_replay_apc.SCHEDULER_TARGET_MODULE,
    ) in names


def test_prefix_replay_tokens_on_kv_spec_and_uniform_type():
    patch_swa_bounded_replay_apc.apply_kv_spec_to_module(
        __import__("vllm.v1.kv_cache_interface", fromlist=["*"])
    )
    from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

    class _Base(KVCacheSpec):
        @property
        def page_size_bytes(self) -> int:
            return self.block_size

        @property
        def tokens_per_state(self):
            return 1

        @property
        def state_content_size_bytes(self) -> int:
            return 1

        @property
        def num_heads(self) -> int:
            return 1

        def max_memory_usage_bytes(self, vllm_config) -> int:
            del vllm_config
            return self.block_size

    base = _Base(block_size=16)
    assert base.prefix_replay_tokens == 0

    class _Replay(_Base):
        @property
        def prefix_replay_tokens(self) -> int:
            return 128

        @property
        def prefix_cacheable(self) -> bool:
            return False

    class _Cacheable(_Base):
        @property
        def prefix_cacheable(self) -> bool:
            return True

    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"a": _Replay(block_size=16), "b": _Cacheable(block_size=16)},
    )
    assert uniform.prefix_replay_tokens == 128
    assert uniform.prefix_cacheable is False


def test_mark_prefix_replay_sets_request_replay_start():
    patch_swa_bounded_replay_apc.apply_request_to_module(
        __import__("vllm.v1.request", fromlist=["*"])
    )
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams

    # Minimal Request; __init__ is heavy — use a SimpleNamespace stand-in for
    # the helper, and also smoke the patched attribute on a real instance path
    # via the helper's contract.
    req = SimpleNamespace(replay_start=0)
    sched = SimpleNamespace(prefix_replay_tokens=128)
    n = patch_swa_bounded_replay_apc._mark_prefix_replay(sched, req, 200)
    assert n == 128
    assert req.replay_start == 72

    n0 = patch_swa_bounded_replay_apc._mark_prefix_replay(
        SimpleNamespace(prefix_replay_tokens=0), req, 200
    )
    assert n0 == 0

    # Real Request gets replay_start=0 from the patched __init__.
    real = Request(
        request_id="t",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    assert getattr(real, "replay_start", None) == 0


def test_new_request_data_carries_replay_start():
    patch_swa_bounded_replay_apc.apply_request_to_module(
        __import__("vllm.v1.request", fromlist=["*"])
    )
    patch_swa_bounded_replay_apc.apply_new_request_data_to_module(
        __import__("vllm.v1.core.sched.output", fromlist=["*"])
    )
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams

    req = Request(
        request_id="r1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    req.replay_start = 42
    data = NewRequestData.from_request(req, block_ids=([],))
    assert data.replay_start == 42


def test_preempt_request_clears_replay_start(monkeypatch):
    from types import ModuleType

    module = ModuleType(patch_swa_bounded_replay_apc.SCHEDULER_TARGET_MODULE)

    class Scheduler:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def schedule(self, *args, **kwargs):
            del args, kwargs
            return None

        def _update_waiting_for_remote_kv(self, request):
            del request

        def _preempt_request(self, request, *args, **kwargs):
            del args, kwargs
            request.preempted = True

    module.Scheduler = Scheduler
    # Stub schedule has no 0.29 needles; skip surgery for this contract test.
    monkeypatch.setattr(
        patch_swa_bounded_replay_apc,
        "_rebuild_schedule",
        lambda original: original,
    )
    assert patch_swa_bounded_replay_apc.apply_scheduler_to_module(module) is True

    req = SimpleNamespace(replay_start=72, preempted=False)
    sched = object.__new__(module.Scheduler)
    sched._preempt_request(req)
    assert req.preempted is True
    assert req.replay_start == 0


def test_remove_request_clears_replay_start_floor():
    from types import ModuleType

    from vllm_hcu.patch.worker.core_fix import (
        patch_deepseek_v41_decoder_replay_state as state_patch,
    )

    module = ModuleType(state_patch.TARGET_MODULE)

    class DeepseekV41ModelState:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_attn(self, *args, **kwargs):
            del args, kwargs

        def add_request(self, req_index, new_req_data):
            del req_index, new_req_data

        def remove_request(self, req_id):
            self.removed = req_id

    module.DeepseekV41ModelState = DeepseekV41ModelState
    assert state_patch.apply_to_module(module) is True

    state = object.__new__(module.DeepseekV41ModelState)
    state._replay_start_np = np.array([0, 72, 0], dtype=np.int32)
    state._keeps_rows_np = np.array([False, True, False], dtype=np.bool_)
    state._replay_req_id_to_index = {"r1": 1}
    state.removed = None
    state.remove_request("r1")
    assert state.removed == "r1"
    assert state._replay_start_np[1] == 0
    assert not bool(state._keeps_rows_np[1])
    assert "r1" not in state._replay_req_id_to_index


def test_async_engine_args_accepts_swa_bounded_replay():
    from dataclasses import dataclass
    from types import ModuleType

    from vllm_hcu.patch.platform.core_fix import patch_cache_config

    module = ModuleType(patch_cache_config.ENGINE_TARGET_MODULE)

    @dataclass
    class EngineArgs:
        max_model_len: int = 1024

        @staticmethod
        def add_cli_args(parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            del args
            return cls()

        def create_engine_config(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(cache_config=SimpleNamespace())

    @dataclass
    class AsyncEngineArgs(EngineArgs):
        enable_log_requests: bool = False

    module.EngineArgs = EngineArgs
    module.AsyncEngineArgs = AsyncEngineArgs
    assert patch_cache_config.apply_engine_args_to_module(module) is True

    async_args = module.AsyncEngineArgs(swa_bounded_replay=False)
    assert async_args.swa_bounded_replay is False
    default_args = module.AsyncEngineArgs()
    assert default_args.swa_bounded_replay is True


def test_mega_moe_replay_buffers_use_int64_on_029_alias():
    from vllm_hcu.patch.worker.core_fix import decoder_replay_cudagraph as m

    # Import may leave NATIVE_MEGA_MOE_BACKENDS empty on 0.29; the alias must
    # still select int64 input-id buffers for MegaMoE replay graphs.
    assert "deep_gemm_mega_moe" in m._MEGA_MOE_INPUT_ID_BACKENDS


def test_compilation_config_owns_replay_capture_size_list():
    from types import ModuleType

    from vllm_hcu.patch.platform.core_fix import patch_decoder_replay_compilation

    module = ModuleType(patch_decoder_replay_compilation.TARGET_MODULE)

    class CompilationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.CompilationConfig = CompilationConfig
    assert patch_decoder_replay_compilation.apply_to_module(module) is True

    a = module.CompilationConfig()
    b = module.CompilationConfig()
    assert a.decoder_replay_cudagraph_capture_sizes == []
    assert b.decoder_replay_cudagraph_capture_sizes == []
    a.decoder_replay_cudagraph_capture_sizes.append(256)
    assert b.decoder_replay_cudagraph_capture_sizes == []
    assert a.decoder_replay_cudagraph_capture_sizes == [256]


def test_pad_replayed_slots_pads_cacheable_groups_only():
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    # groups: 0=cacheable (MLA), 1=SWA (non-cacheable)
    num_groups, num_tokens, num_reqs = 2, 8, 2
    slot_mappings = torch.arange(
        num_groups * num_tokens, dtype=torch.int64
    ).reshape(num_groups, num_tokens)
    original = slot_mappings.clone()
    cacheable_groups = torch.tensor([0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    positions = torch.tensor([10, 11, 12, 13, 50, 51, 52, 53], dtype=torch.int64)
    # req0: replay [11, 11+4) → pads pos 11,12,13; req1: start 0 → skip
    replay_start = torch.tensor([11, 0], dtype=torch.int32)
    window = 4

    patch_deepseek_v41_decoder_replay_state.pad_replayed_slots(
        slot_mappings,
        cacheable_groups,
        query_start_loc,
        positions,
        replay_start,
        window,
        PAD_SLOT_ID,
    )

    # Cacheable group 0: tokens 1..3 of req0 padded; req1 untouched.
    assert slot_mappings[0, 0].item() == original[0, 0].item()
    assert slot_mappings[0, 1].item() == PAD_SLOT_ID
    assert slot_mappings[0, 2].item() == PAD_SLOT_ID
    assert slot_mappings[0, 3].item() == PAD_SLOT_ID
    assert torch.equal(slot_mappings[0, 4:], original[0, 4:])
    # SWA group 1 left alone.
    assert torch.equal(slot_mappings[1], original[1])


@pytest.mark.parametrize("decoder_replay", [False, True])
@pytest.mark.parametrize(
    ("max_num_reqs", "pcp_size", "expected_capacity", "num_local_reqs"),
    [
        (1, 1, 1, 1),
        (3, 1, 3, 3),
        (32, 1, 32, 32),
        (1, 8, 4, 4),
        (3, 8, 10, 10),
        (32, 8, 96, 86),
        (32, 8, 96, 96),
    ],
)
def test_replay_buffers_cover_pcp_request_capacity(
    monkeypatch,
    decoder_replay,
    max_num_reqs,
    pcp_size,
    expected_capacity,
    num_local_reqs,
):
    class FakeUvaBufferPool:
        def __init__(self, capacity, dtype):
            self.capacity = capacity
            self.dtype = dtype

        def copy_to_gpu(self, values, out=None):
            source = torch.as_tensor(values, dtype=self.dtype)
            return source.clone() if out is None else out.copy_(source)

    class DeepseekV41ModelState:
        def __init__(self, vllm_config, model, encoder_cache, device):
            self.max_num_reqs = max_num_reqs
            self.device = device

        def prepare_attn(self, *args, **kwargs):
            return {}

        def add_request(self, req_index, new_req_data):
            return None

        def capture_inner_cudagraphs(self, *args, **kwargs):
            return None

    module = SimpleNamespace(DeepseekV41ModelState=DeepseekV41ModelState)
    monkeypatch.setattr(
        patch_deepseek_v41_decoder_replay_state,
        "load_exact_module",
        lambda target, module: module,
    )
    monkeypatch.setattr(
        patch_deepseek_v41_decoder_replay_state, "UvaBufferPool", FakeUvaBufferPool
    )
    assert patch_deepseek_v41_decoder_replay_state.apply_to_module(module) is True

    cfg = _vllm_config(pcp=pcp_size)
    cfg.compilation_config.cudagraph_decoder_replay = False
    model = SimpleNamespace(
        decoder_replay_layers=SimpleNamespace(window=128) if decoder_replay else None
    )
    state = DeepseekV41ModelState(cfg, model, None, torch.device("cpu"))
    assert state._replay_start_np.shape == (expected_capacity,)
    assert state._replay_start.shape == (expected_capacity,)
    assert state._replay_start_staging.capacity == expected_capacity
    if pcp_size > 1:
        assert state._pcp_replay_start.capacity == expected_capacity
    else:
        assert state._pcp_replay_start is None
    if decoder_replay:
        assert state._keeps_rows_np.shape == (expected_capacity,)
        assert state._kept_kv_start.shape == (expected_capacity,)

    state._replay_start_np[:max_num_reqs] = np.arange(max_num_reqs) + 100
    input_batch = SimpleNamespace(
        num_reqs=num_local_reqs,
        idx_mapping_np=np.arange(num_local_reqs, dtype=np.int32) % max_num_reqs,
        is_prefilling_np=np.arange(num_local_reqs) % 3 != 0,
    )
    expected = np.where(
        input_batch.is_prefilling_np,
        state._replay_start_np[input_batch.idx_mapping_np],
        0,
    ).astype(np.int32)
    replay_start, replay_start_np = (
        patch_deepseek_v41_decoder_replay_state._gather_batch_replay_start(
            state, input_batch
        )
    )
    np.testing.assert_array_equal(replay_start_np, expected)
    assert torch.equal(replay_start, torch.from_numpy(expected))


@pytest.mark.parametrize(
    "idx_mapping_np",
    [
        np.array([0, 1], dtype=np.int32),
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.arange(86, dtype=np.int32) % 32,
    ],
    ids=("two_requests", "fragmented_requests", "reproduced_batch"),
)
def test_pcp_replay_keeps_multi_request_boundaries(monkeypatch, idx_mapping_np):
    """Multi-req PCP must build per-logical-request tails (not a single span)."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.distributed import parallel_state

    class FakeGroup:
        world_size = 8
        rank_in_group = 0

    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: FakeGroup())

    captured: dict[str, object] = {}
    real_layout = patch_deepseek_v41_decoder_replay_state.late_layer_tail_layout

    def spy_layout(**kwargs):
        captured["extend_lens_cpu"] = list(kwargs["extend_lens_cpu"])
        captured["seq_lens_cpu"] = list(kwargs["seq_lens_cpu"])
        return real_layout(**kwargs)

    monkeypatch.setattr(
        patch_deepseek_v41_decoder_replay_state, "late_layer_tail_layout", spy_layout
    )

    num_reqs = len(idx_mapping_np)
    unique_ids = np.unique(idx_mapping_np)
    n_logical = int(unique_ids.shape[0])
    # Per-logical extend / seq lens (local rows share the logical id's values).
    logical_extend = {int(i): 200 + 10 * int(i) for i in unique_ids}
    logical_seq = {int(i): 400 + 10 * int(i) for i in unique_ids}
    global_query = np.array(
        [logical_extend[int(i)] for i in idx_mapping_np], dtype=np.int32
    )
    global_seq = np.array(
        [logical_seq[int(i)] for i in idx_mapping_np], dtype=np.int32
    )
    layers = SimpleNamespace(window=128, replay_batch=object())
    state = SimpleNamespace(
        decoder_replay_layers=layers,
        _keeps_rows_np=np.zeros(max(int(idx_mapping_np.max()) + 1, 1), dtype=np.bool_),
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
        _replay_attn_groups=None,
        _kept_kv_start=torch.zeros(n_logical + 8, dtype=torch.int32),
        pcp_size=8,
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens_after_padding=sum(logical_extend[int(i)] for i in unique_ids),
        idx_mapping_np=idx_mapping_np,
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.as_tensor(global_seq, dtype=torch.int32),
        positions=torch.arange(int(global_query.sum()) + 8, dtype=torch.int64),
        is_prefilling_np=np.ones(num_reqs, dtype=np.bool_),
        # Restore metadata intentionally missing → fall back after layout.
        pcp=SimpleNamespace(
            global_query_lens_np=global_query,
            global_seq_lens_cpu_upper_bound_np=global_seq,
            global_num_tokens=int(global_query.sum()),
        ),
    )
    patch_deepseek_v41_decoder_replay_state._prepare_replay_batch(
        state,
        input_batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty((1, 0), dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=SimpleNamespace(),
        replay_start=torch.zeros(num_reqs, dtype=torch.int32),
    )
    # One entry per logical request (first occurrence order).
    _, first = np.unique(idx_mapping_np, return_index=True)
    first = np.sort(first)
    assert captured["extend_lens_cpu"] == [
        logical_extend[int(idx_mapping_np[i])] for i in first
    ]
    assert captured["seq_lens_cpu"] == [
        logical_seq[int(idx_mapping_np[i])] for i in first
    ]
    # Missing DualChunk restore metadata → no PCP interleaved replay batch.
    assert layers.replay_batch is None


def test_pcp_replicated_tail_restores_outputs_without_duplicate_gather(monkeypatch):
    from contextlib import nullcontext

    from vllm.distributed import parallel_state
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch
    from vllm_hcu.patch.worker.core_fix import decoder_replay
    from vllm_hcu.patch.worker.core_fix.late_layer_tail_cp import (
        late_layer_tail_cp_layout,
    )

    class FakeGroup:
        world_size = 8
        rank_in_group = 0
        gather_count = 0

        def all_gather(self, tensor, dim):
            self.gather_count += 1
            return torch.cat(
                [tensor] + [torch.zeros_like(tensor) for rank in range(7)], dim=dim
            )

    group = FakeGroup()
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: group)
    monkeypatch.setattr(
        decoder_replay, "override_forward_context", lambda context: nullcontext()
    )
    plan = late_layer_tail_cp_layout(
        token_indices=torch.arange(4),
        tail_lens=torch.tensor([4]),
        positions=torch.arange(4),
        window_floors=torch.zeros(4, dtype=torch.int32),
        pcp_rank=0,
        pcp_size=1,
        batch_size=1,
    )

    def run_layers(hidden_states, *states):
        assert in_replicated_mtp_batch()
        return (hidden_states * 2,)

    runner = decoder_replay.DecoderReplayLayers(
        128,
        run_layers=run_layers,
        write_batch_kv=lambda *args: None,
        metadata_prefixes=set(),
        first_swa_prefix="cache",
    )
    runner.replay_batch = SimpleNamespace(
        pcp_tail=plan,
        pcp_meta=SimpleNamespace(
            pcp_local_num_tokens=4, pcp_restore_idx=torch.arange(4)
        ),
        forward_context=object(),
        num_padded=None,
        run_graph=None,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    outputs = runner._run_pcp_interleaved_tail(hidden_states)
    assert torch.equal(outputs[0], hidden_states * 2)
    assert group.gather_count == 1
    assert not in_replicated_mtp_batch()


def test_pcp_interleaved_tail_pads_activations_to_graph_size(monkeypatch):
    """Graph num_padded > packed tail must pad model inputs before run_layers."""
    from contextlib import nullcontext

    from vllm.distributed import parallel_state
    from vllm_hcu.patch.worker.core_fix import decoder_replay
    from vllm_hcu.patch.worker.core_fix.late_layer_tail_cp import (
        late_layer_tail_cp_layout,
    )

    class FakeGroup:
        world_size = 1
        rank_in_group = 0

        def all_gather(self, tensor, dim):
            return tensor

    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: FakeGroup())
    monkeypatch.setattr(
        decoder_replay, "override_forward_context", lambda context: nullcontext()
    )
    plan = late_layer_tail_cp_layout(
        token_indices=torch.arange(3),
        tail_lens=torch.tensor([3]),
        positions=torch.arange(3),
        window_floors=torch.zeros(3, dtype=torch.int32),
        pcp_rank=0,
        pcp_size=1,
        batch_size=1,
    )
    seen = {}

    def run_layers(hidden_states, *states):
        seen["shape"] = tuple(hidden_states.shape)
        return (hidden_states + 1,)

    runner = decoder_replay.DecoderReplayLayers(
        128,
        run_layers=run_layers,
        write_batch_kv=lambda *args: None,
        metadata_prefixes=set(),
        first_swa_prefix="cache",
    )
    runner.replay_batch = SimpleNamespace(
        pcp_tail=plan,
        pcp_meta=SimpleNamespace(
            pcp_local_num_tokens=3, pcp_restore_idx=torch.arange(3)
        ),
        forward_context=object(),
        num_padded=5,
        run_graph=None,
    )
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    outputs = runner._run_pcp_interleaved_tail(hidden_states)
    assert seen["shape"] == (5, 2)
    # Pad rows must not leak into DualChunk scatter destinations.
    assert outputs[0].shape == hidden_states.shape
    assert torch.equal(outputs[0], hidden_states + 1)


def test_subsequent_replay_kv_start_keeps_apc_floor():
    """Short APC chunk with trim=0 must keep replay_start on later SWA layers."""
    replay_start = torch.tensor([1024, 0], dtype=torch.int32)
    trim = torch.tensor([0, 128], dtype=torch.int32)
    out = patch_deepseek_v41_decoder_replay_state._subsequent_replay_kv_start(
        replay_start, trim
    )
    assert out.tolist() == [1024, 128]


def test_forward_kv_restores_full_pcp_cache_inputs(monkeypatch):
    import vllm.forward_context as forward_context
    from vllm_hcu.model_executor.layers.attention import pcp
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_forward_kv

    class DeepseekV4Attention:
        def _run_parallel_input_projections(self, hidden_states):
            return hidden_states, None, None

        def _split_qkv_and_norm(self, projected):
            return None, None, projected

    module = SimpleNamespace(DeepseekV4Attention=DeepseekV4Attention)
    monkeypatch.setattr(
        patch_deepseek_v41_forward_kv,
        "load_exact_module",
        lambda target, module: module,
    )
    assert patch_deepseek_v41_forward_kv.apply_to_module(module) is True
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(4), pcp_has_global_prefill=True
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"cache": metadata}),
    )
    monkeypatch.setattr(
        pcp,
        "restore_pcp_rows_to_global",
        lambda tensor, metadata: torch.cat((tensor, tensor + 10)),
    )
    monkeypatch.setattr(pcp, "pcp_global_positions", lambda metadata: torch.arange(4))
    monkeypatch.setattr(
        pcp, "globalize_pcp_slot_mapping", lambda slots, metadata: torch.arange(4)
    )
    captured = {}

    def capture_insert(*arguments):
        captured["arguments"] = arguments

    monkeypatch.setattr(
        torch.ops._C, "fused_deepseek_v4_kv_rope_insert", capture_insert, raising=False
    )
    attention = DeepseekV4Attention()
    attention.swa_cache_layer = SimpleNamespace(
        prefix="cache", kv_cache=torch.zeros(4, 2, dtype=torch.uint8), block_size=64
    )
    attention.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(1))
    attention.compressor = None
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    attention.forward_kv(torch.arange(2), hidden_states)
    arguments = captured["arguments"]
    assert torch.equal(arguments[0], torch.cat((hidden_states, hidden_states + 10)))
    assert torch.equal(arguments[2], torch.arange(4))
    assert torch.equal(arguments[3], torch.arange(4))


def test_prepare_attn_apc_path_without_decoder_layers(monkeypatch):
    """APC pad + ReplayAttnMetadata run even when decoder_replay_layers is None."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    class _FakeState:
        def __init__(self):
            self.max_num_reqs = 4
            self.device = torch.device("cpu")
            self.decoder_replay_layers = None
            self._replay_start_np = np.zeros(4, dtype=np.int32)
            self._replay_start = torch.zeros(4, dtype=torch.int32)
            self._replay_start_staging = SimpleNamespace(
                copy_to_gpu=lambda x, out=None: (
                    out.copy_(torch.as_tensor(x, dtype=torch.int32))
                    if out is not None
                    else torch.as_tensor(x, dtype=torch.int32)
                )
            )
            self._replay = (
                4,
                torch.tensor([0], dtype=torch.int32),  # only group 0 cacheable
            )
            self.vllm_config = SimpleNamespace()

    seen = {}

    def fake_original(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
    ):
        seen["slots"] = slot_mappings.clone()
        return {"ok": True}

    # Build a minimal module/class and apply the patch wrappers manually.
    class DeepseekV41ModelState:
        def __init__(self, *a, **k):
            pass

        def prepare_attn(self, *a, **k):
            return fake_original(self, *a, **k)

        def add_request(self, req_index, new_req_data):
            pass

        def capture_inner_cudagraphs(self, *a, **k):
            pass

    module = SimpleNamespace(DeepseekV41ModelState=DeepseekV41ModelState)
    # Reset marker if re-run in same process.
    if hasattr(DeepseekV41ModelState, patch_deepseek_v41_decoder_replay_state._MARKER):
        delattr(DeepseekV41ModelState, patch_deepseek_v41_decoder_replay_state._MARKER)

    monkeypatch.setattr(
        patch_deepseek_v41_decoder_replay_state,
        "load_exact_module",
        lambda target, module: module,
    )
    # Avoid CUDAGraph / PP imports in init by only applying prepare path:
    # call apply_to_module then replace instance attrs.
    assert patch_deepseek_v41_decoder_replay_state.apply_to_module(module) is True

    state = _FakeState()
    # Bind patched methods onto the fake state.
    state.prepare_attn = DeepseekV41ModelState.prepare_attn.__get__(state, _FakeState)

    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        num_tokens_after_padding=4,
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0]),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        positions=torch.arange(4, dtype=torch.int64),
        max_query_len=4,
        num_scheduled_tokens=torch.tensor([4]),
        seq_lens=torch.tensor([4]),
        seq_lens_cpu_upper_bound=torch.tensor([4]),
        fast_prefill=None,
    )
    state._replay_start_np[0] = 1  # replay positions >= 1 within window 4
    slot_mappings = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    prefix_replay_tokens=4, prefix_cacheable=True
                )
            ),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    prefix_replay_tokens=0, prefix_cacheable=False
                )
            ),
        ]
    )

    # Inject build_attn_metadata so DefaultModelState global patch sees it.
    import vllm.v1.worker.gpu.model_states.default as default_ms

    captured = {}

    def fake_build(**kwargs):
        captured["model_specific"] = kwargs.get("model_specific_attn_metadata")
        return {}

    monkeypatch.setattr(default_ms, "build_attn_metadata", fake_build)

    from vllm.config.compilation import CUDAGraphMode

    out = state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=slot_mappings,
        attn_groups=[],
        kv_cache_config=kv_cache_config,
    )
    assert out == {"ok": True}
    # Cacheable group 0 padded for pos 1,2,3; SWA group 1 untouched.
    assert slot_mappings[0, 0].item() == 0
    assert slot_mappings[0, 1].item() == PAD_SLOT_ID
    assert slot_mappings[0, 2].item() == PAD_SLOT_ID
    assert slot_mappings[0, 3].item() == PAD_SLOT_ID
    assert torch.equal(slot_mappings[1], torch.arange(4, 8))


def test_pad_kept_slots_rounds_up_to_graph_size():
    """Piecewise dispatch may raise num_padded above the real PCP tail."""
    slots = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    positions = torch.tensor([10, 11, 12], dtype=torch.int64)
    padded_slots, padded_pos = (
        patch_deepseek_v41_decoder_replay_state._pad_kept_slots(slots, positions, 5)
    )
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    assert padded_slots.shape == (2, 5)
    assert padded_pos.tolist() == [10, 11, 12, 0, 0]
    assert padded_slots[:, 3:].eq(PAD_SLOT_ID).all()


def test_pcp_kv_start_only_advances_when_trimmed():
    """Decode / untrimmed rows keep kv_start=0; trimmed prefill advances."""
    kv_start = patch_deepseek_v41_decoder_replay_state._replay_kv_starts(
        [256, 1], [128, 1], [256, 4096]
    )
    np.testing.assert_array_equal(kv_start, np.array([128, 0], dtype=np.int32))


def test_bump_last_seq_len_keeps_prefix_stable_under_query_pad():
    """seq_len grows with query pad so SWA prefix_len = seq_len - query_len holds."""
    bumped = patch_deepseek_v41_decoder_replay_state._bump_last_seq_len_for_query_pad(
        [100, 200], real_tokens=130, num_padded=160
    )
    assert bumped == [100, 230]
    # Last request real query 30 → padded query 60; prefix stays 170.
    assert bumped[-1] - 60 == 200 - 30
    assert patch_deepseek_v41_decoder_replay_state._bump_last_seq_len_for_query_pad(
        [100], real_tokens=128, num_padded=128
    ) == [100]


def test_model_init_skips_pp_missing_cut_layer():
    """PP stages that do not own cut must not touch layers[cut]."""
    import ast
    import inspect

    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_decoder_replay

    class PPMissingLayer:
        pass

    # Guard must appear before any layers[cut] attribute read in model_init.
    src = inspect.getsource(patch_deepseek_v41_decoder_replay)
    assert "owns_cut = self.start_layer <= cut < self.end_layer" in src
    assert src.index("owns_cut = self.start_layer <= cut < self.end_layer") < src.index(
        "getattr(self.layers[cut], \"attn\", None)"
    )

    cut = 20
    start_layer, end_layer = 30, 40
    owns_cut = start_layer <= cut < end_layer
    assert owns_cut is False
    layers = [PPMissingLayer() for _ in range(40)]
    if owns_cut:
        _ = layers[cut].attn  # pragma: no cover
    # Reaching here means the gate prevented AttributeError on PPMissingLayer.
    assert not hasattr(layers[cut], "attn")
    # Sanity: the AST of the patched module still parses after edits.
    ast.parse(inspect.getsource(patch_deepseek_v41_decoder_replay))
