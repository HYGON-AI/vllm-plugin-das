# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType

from vllm_hcu.patch.worker.core_fix import patch_minimax_m3_sparse_decode


class _Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return "launched"

        return launch


def test_sparse_decode_launch_forces_one_stage():
    module = ModuleType(patch_minimax_m3_sparse_decode.TARGET_MODULE)
    kernel = _Kernel()
    module._gqa_sparse_decode_kernel = kernel

    assert patch_minimax_m3_sparse_decode.apply_to_module(module) is True
    assert module._gqa_sparse_decode_kernel[(2, 1)]("q", BLOCK_SIZE_K=128) == "launched"
    assert kernel.calls == [
        ((2, 1), ("q",), {"BLOCK_SIZE_K": 128, "num_stages": 1})
    ]
    assert patch_minimax_m3_sparse_decode.apply_to_module(module) is False
