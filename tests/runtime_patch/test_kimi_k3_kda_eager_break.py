# SPDX-License-Identifier: Apache-2.0
"""KDA checkpoint binding must run inside breakable CUDA-graph replay."""
from types import ModuleType, SimpleNamespace

import pytest

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm_hcu.model_executor.layers import kimi_k3_kda as kda
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_kda_reference_paths as patch


class _Capture:
    """Breakable capture double; GPU segments are not created."""

    def __init__(self):
        self.active = True
        self._capturing = True
        self.segments = []

    def add_eager(self, fn):
        self._capturing = False
        try:
            result = fn()
        finally:
            self._capturing = True
        self.segments.append(fn)
        return result

    def replay(self):
        self.active = False
        for segment in self.segments:
            segment()


def test_eager_break_replay_binds_layer_and_finalizes(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_KIMI_KDA_PREFILL_BACKEND", "boltops")
    monkeypatch.setenv("VLLM_KIMI_KDA_CACHE_STRATEGY", "extra_buffer")
    events = []
    finalized = []

    class Layer:
        def __init__(self, prefix="layer"):
            self.prefix = prefix

        @eager_break_during_capture
        def _forward(self, value):
            events.append((value, kda.ACTIVE_LAYER.get()))

    module = ModuleType("vllm.models.kimi_k3.amd.kda")
    module.causal_conv1d_fn = lambda *args, **kwargs: None
    module.causal_conv1d_update = lambda *args, **kwargs: None
    module.chunk_kda_with_fused_gate = lambda **kwargs: None
    module.KimiK3DeltaAttention = Layer
    current = {}
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={"layer": current.get("metadata")}
    )

    monkeypatch.setattr(kda, "install_checkpoint_spec", lambda cls: None)
    monkeypatch.setattr(
        kda, "finalize_checkpoints",
        lambda metadata, layer=None: finalized.append((metadata, layer)),
    )
    capture = _Capture()
    monkeypatch.setattr(
        BreakableCUDAGraphCapture, "current",
        classmethod(lambda cls: capture if capture.active else None),
    )

    assert patch.apply_to_module(module)
    layer = Layer()

    capture_metadata = SimpleNamespace(nums_dict={}, num_decodes=0)
    current["metadata"] = capture_metadata
    layer._forward("capture")
    assert [event[0] for event in events] == ["capture"]
    active = events[0][1]
    assert active is not None and active[0] is layer and active[1] is capture_metadata
    assert finalized == [(capture_metadata, layer)]
    assert len(capture.segments) == 1

    # Replay uses a different request's metadata. The recorded eager segment
    # must bind that metadata and publish its checkpoint state.
    events.clear()
    finalized.clear()
    replay_metadata = SimpleNamespace(nums_dict={}, num_decodes=0)
    current["metadata"] = replay_metadata
    capture.replay()
    assert [event[0] for event in events] == ["capture"]
    active = events[0][1]
    assert active is not None and active[0] is layer and active[1] is replay_metadata
    assert finalized == [(replay_metadata, layer)]
