# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dispatch-size compatibility for the HCU PCP manager."""

from __future__ import annotations

import inspect

import numpy as np

from vllm_hcu.v1.pcp_manager import HcuPCPManager


def _manager(pcp_size: int) -> HcuPCPManager:
    manager = object.__new__(HcuPCPManager)
    manager.pcp_size = pcp_size
    return manager


def test_pcp_dispatch_size_uses_largest_rank_local_batch() -> None:
    manager = _manager(2)

    result = manager.get_num_tokens_for_dispatch(
        np.array([10, 3], dtype=np.int32),
        np.array([True, False], dtype=np.bool_),
    )

    # The 10-token prefill splits 4/6 across PCP ranks; the 3 decode tokens
    # are replicated, so the runner must dispatch the larger 6+3 batch.
    assert result == 9


def test_pcp_dispatch_size_preserves_decode_and_empty_batches() -> None:
    manager = _manager(4)

    assert manager.get_num_tokens_for_dispatch(
        np.array([2, 3], dtype=np.int32),
        np.array([False, False], dtype=np.bool_),
    ) == 5
    assert manager.get_num_tokens_for_dispatch(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.bool_),
    ) == 0


def test_pcp_partition_accepts_official_runner_padding_contract() -> None:
    signature = inspect.signature(HcuPCPManager.partition_batch)

    assert signature.parameters["padded_num_tokens"].default is None
