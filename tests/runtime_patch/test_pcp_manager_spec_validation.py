# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for selecting the plugin manager through official MRV2 hooks."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
from vllm_hcu.v1 import pcp_manager


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            max_num_batched_tokens=128,
        ),
    )


def test_hcu_manager_owns_validation_without_patching_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    calls: list[object] = []
    monkeypatch.setattr(
        pcp_manager,
        "_validate_hcu_pcp_scope",
        lambda value: calls.append(value) or True,
    )

    pcp_manager.HcuPCPManager.validate_config(config, False)

    assert calls == [config]
    with pytest.raises(ValueError, match="multimodal"):
        pcp_manager.HcuPCPManager.validate_config(config, True)


def test_bound_manager_matches_official_constructor_contract() -> None:
    manager_cls = pcp_manager.make_hcu_pcp_manager_cls(_config())
    signature = inspect.signature(manager_cls.__init__)

    assert list(signature.parameters) == [
        "self",
        "pcp_world_size",
        "pcp_rank",
        "device",
        "req_states",
        "max_num_reqs",
        "max_num_tokens",
        "block_tables",
        "dcp_world_size",
        "dcp_rank",
        "cp_interleave",
    ]
    assert issubclass(manager_cls, pcp_manager.HcuPCPManager)


def test_bound_manager_fails_closed_before_allocating_on_argument_drift() -> None:
    manager_cls = pcp_manager.make_hcu_pcp_manager_cls(_config())

    with pytest.raises(PatchCompatibilityError, match="constructor arguments"):
        manager_cls(
            pcp_world_size=4,
            pcp_rank=0,
            device="cpu",
            req_states=object(),
            max_num_reqs=16,
            max_num_tokens=128,
            block_tables=object(),
            dcp_world_size=1,
            dcp_rank=0,
            cp_interleave=1,
        )
