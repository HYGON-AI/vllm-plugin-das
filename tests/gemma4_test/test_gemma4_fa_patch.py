# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model-local runtime injection and worker registration contracts."""

from types import ModuleType

from vllm_hcu.models.gemma4 import config
from vllm_hcu.patch.worker.op_opt import patch_gemma4_attention as patch


def test_model_local_injection_is_idempotent_and_preserves_explicit_backend(
    monkeypatch,
):
    class Attention:
        def __init__(
            self,
            num_heads,
            head_size,
            scale,
            *,
            per_layer_sliding_window=None,
            attn_backend=None,
        ):
            self.backend = attn_backend
            self.head_size = head_size

    module = ModuleType(patch.TARGET_MODULE)
    module.Attention = Attention
    selected = object()
    calls = []

    def select(**kwargs):
        calls.append(kwargs)
        return selected

    monkeypatch.setattr(config, "select_gemma4_hcu_attention_backend", select)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    assert module.Attention is not Attention
    assert module.Attention(16, 512, 1.0).backend is selected
    assert calls == [{"head_dim": 512, "is_sliding": False}]
    explicit = object()
    assert module.Attention(16, 512, 1.0, attn_backend=explicit).backend is explicit
    assert len(calls) == 1
    module.Attention(8, 256, 1.0, per_layer_sliding_window=1024)
    assert calls[-1] == {"head_dim": 256, "is_sliding": True}
    assert Attention(16, 512, 1.0).backend is None


def test_worker_dispatcher_registers_gemma4_callback():
    from vllm_hcu.patch.worker import worker_callback_names

    assert (patch.PATCH_ID, patch.TARGET_MODULE) in worker_callback_names()
