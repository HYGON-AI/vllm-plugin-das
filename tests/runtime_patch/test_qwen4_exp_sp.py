# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Qwen4Exp sequence-parallel MoE / HC SP patch contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.config import HcuFeatureConfig, set_hcu_config
from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_sp as sp


def _vllm_config(*, hc_sp: bool, sp_moe: bool) -> SimpleNamespace:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sp_moe),
        additional_config={},
    )
    set_hcu_config(config, HcuFeatureConfig(qwen4_exp_hc_sp=hc_sp))
    return config


@pytest.mark.parametrize(
    ("hc_sp", "sp_moe", "expected"),
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_hc_sp_requires_flag_and_sequence_parallel_moe(hc_sp, sp_moe, expected):
    config = _vllm_config(hc_sp=hc_sp, sp_moe=sp_moe)
    assert sp.hc_sequence_parallel_enabled(config) is expected


def test_sp_moe_guard_matcher_is_exact():
    import ast

    guard = ast.parse(
        "if c.use_sequence_parallel_moe:\n"
        "    raise NotImplementedError('x')\n"
    ).body[0]
    other = ast.parse(
        "if c.use_sequence_parallel_moe:\n"
        "    raise ValueError('x')\n"
    ).body[0]
    with_else = ast.parse(
        "if c.use_sequence_parallel_moe:\n"
        "    raise NotImplementedError('x')\nelse:\n    pass\n"
    ).body[0]
    assert sp._is_sp_moe_guard(guard)
    assert not sp._is_sp_moe_guard(other)
    assert not sp._is_sp_moe_guard(with_else)


@pytest.mark.parametrize("num_tokens", [1, 2, 7, 8])
@pytest.mark.parametrize("rank", [0, 1])
def test_shard_and_padding_mask_cover_all_tokens(monkeypatch, num_tokens, rank):
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda: rank, raising=False)
    x = torch.arange(num_tokens * 3, dtype=torch.float32).view(num_tokens, 3) + 1
    chunk = (num_tokens + 1) // 2

    shard = sp.sp_shard(x)
    assert shard.shape == (chunk, 3)
    expected = torch.zeros(chunk, 3)
    rows = x[rank * chunk : (rank + 1) * chunk]
    expected[: rows.shape[0]] = rows
    assert torch.equal(shard, expected)

    mask = sp.sp_padding_mask(None, x)
    assert mask.dtype == torch.bool and mask.shape == (chunk,)
    assert mask.tolist() == [rank * chunk + i >= num_tokens for i in range(chunk)]


def test_gather_packed_splits_back(monkeypatch):
    monkeypatch.setattr(sp, "sp_all_gather", lambda t: torch.cat([t, t + 100]), raising=False)
    first = torch.zeros(2, 2)
    second = torch.ones(2, 3)
    a, b = sp.gather_packed(first, second, 3)
    assert a.shape == (3, 2) and b.shape == (3, 3)
    assert a.is_contiguous() and b.is_contiguous()
    assert torch.equal(b[2], torch.full((3,), 101.0))


def test_patches_apply_in_registered_order_and_are_idempotent():
    # Real model modules register custom ops; keep them out of this process.
    script = textwrap.dedent(
        """
        import vllm.models.qwen4_exp.amd.model as model
        import vllm.models.qwen4_exp.amd.mtp as mtp
        from vllm_hcu.patch.worker.core_fix import (
            patch_qwen4_exp_mtp_pp,
            patch_qwen4_exp_ple_prefetch,
            patch_qwen4_exp_sp as sp,
            patch_qwen4_exp_sp_mtp,
        )

        assert sp.apply_to_module(model) is True
        patch_qwen4_exp_ple_prefetch.apply_to_module(model)
        patch_qwen4_exp_mtp_pp.apply_to_module(mtp)
        assert patch_qwen4_exp_sp_mtp.apply_to_module(mtp) is True

        assert sp.apply_to_module(model) is False
        assert patch_qwen4_exp_sp_mtp.apply_to_module(mtp) is False
        assert patch_qwen4_exp_mtp_pp.apply_to_module(mtp) is False
        for cls in (model.Qwen4ExpSparseMoeBlock, model.Qwen4ExpDecoderLayer):
            assert "use_sequence_parallel_moe" not in cls.__init__.__code__.co_names
        """
    )
    env = {**os.environ, "VLLM_HCU_PLE_PREFETCH_STREAM": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-4000:]


_COMPOSE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from types import SimpleNamespace

    import torch
    import vllm
    from vllm_hcu.patch.module_exchange import register_qwen4_exp_ple_exchange
    from vllm_hcu.patch.worker import prepare_worker_patches
    from vllm_hcu.patch import import_coordinator as ic

    register_qwen4_exp_ple_exchange()
    prepare_worker_patches()
    import vllm.models.qwen4_exp.amd.model as model
    ic.IMPORT_COORDINATOR.drain_ready_callbacks()
    from vllm_hcu.patch.worker.core_fix import (
        patch_qwen4_exp_fused_ple as fused,
        patch_qwen4_exp_sp as sp,
    )

    forward = model.Qwen4ExpDecoderLayer.forward
    hcu_ple = model.Qwen4ExpPLELayer.__module__.startswith("vllm_hcu.")
    report = {
        "installed_is_sp": forward is sp.decoder_forward,
        "fused_applied": bool(getattr(model, fused._MARKER, False)),
        "hcu_ple": hcu_ple,
        "sp_fused_mode": sp._fused_ple,
        "reapply_sp": sp.apply_to_module(model),
        "reapply_fused": fused.apply_to_module(model),
    }

    # Run the installed forward on stubs: PLE returns the residual-included
    # state for the HCU (fused) PLE and only the delta for the original one.
    class Ple:
        def __call__(self, h, input_ids, qsl, ctx):
            delta = torch.full_like(h, 10.0)
            return h + delta if hcu_ple else delta

    hc = SimpleNamespace(
        combine=lambda h, b, i: h,
        mix=lambda h: (h, h, None),
        combine_and_mix=lambda h, a, i: (h, a, i),
    )
    layer = SimpleNamespace(
        ple=Ple(), attn_hyper_connection=hc, mlp_hyper_connection=hc,
        layer_type="linear_attention", linear_attn=lambda hidden_states: hidden_states,
        mlp=lambda x: x,
    )
    T = 3
    ids = torch.zeros(T, dtype=torch.long)
    args = dict(input_ids=ids, query_start_loc=ids, ngram_context=ids)
    h = torch.arange(T, dtype=torch.float32).view(T, 1)

    setattr(layer, sp.HC_SP_ATTR, False)
    out, _, _ = forward(layer, h, None, None, torch.zeros(T), **args)
    report["off"] = out.flatten().tolist()

    # HC SP on, emulated TP=2 rank 0: this rank holds rows [0, 2) (padded).
    sp.get_tensor_model_parallel_world_size = lambda: 2
    sp.get_tensor_model_parallel_rank = lambda: 0
    sp.sp_all_gather = lambda t: torch.cat([t, t + 100.0])
    sp.sp_reduce_scatter = lambda t: t
    setattr(layer, sp.HC_SP_ATTR, True)
    local = h[:2]
    out, _, _ = forward(layer, local, None, None, torch.zeros(T), **args)
    report["on"] = out.flatten().tolist()
    print("REPORT" + json.dumps(report))
    """
)


@pytest.mark.parametrize("custom_ops", ["0", "1"])
def test_sp_composes_with_fused_ple_without_double_residual(custom_ops):
    env = {**os.environ, "VLLM_HCU_USE_CUSTOM_OPS": custom_ops}
    result = subprocess.run(
        [sys.executable, "-c", _COMPOSE_SCRIPT], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("REPORT"))
    report = __import__("json").loads(line[len("REPORT"):])

    fused = custom_ops == "1"
    assert report["installed_is_sp"] is True
    assert report["fused_applied"] is fused
    assert report["hcu_ple"] is fused
    assert report["sp_fused_mode"] is fused
    assert report["reapply_sp"] is False
    assert report["reapply_fused"] is False
    # Residual added exactly once: h + 10 for both PLE flavors.
    assert report["off"] == [10.0, 11.0, 12.0]
    # HC SP: PLE on the gathered full state, then this rank's rows.
    assert report["on"] == [10.0, 11.0]


def test_sp_refuses_foreign_decoder_forward_wrapper():
    import functools

    def original(self):
        return None

    original.__module__ = sp.TARGET_MODULE

    @functools.wraps(original)
    def foreign(self):
        return None

    sp._require_unwrapped(original, "target")
    with pytest.raises(sp.PatchCompatibilityError):
        sp._require_unwrapped(foreign, "target")
