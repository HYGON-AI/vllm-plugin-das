# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Narrow HCU Worker adapter around the official vLLM v0.28 Worker."""

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker


class HcuGPUWorker(Worker):
    """Arm HCU callbacks while retaining the target Worker lifecycle."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        # Worker callbacks must be armed against the deserialized config before
        # the parent constructor imports model runners or custom operators.
        from vllm_hcu.patch.runtime_state import set_process_role
        from vllm_hcu.patch.worker import apply_worker_patches

        set_process_role("Worker")
        apply_worker_patches(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load via target Worker, then prove every enabled patch is live."""

        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm_hcu.patch.worker import validate_worker_patches

        validate_worker_patches(require_applied=True)

    def init_device(self):
        """Install the HCU split-group shim, then delegate to target v0.28."""

        from vllm import envs

        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
            from vllm_hcu.v1.worker_framework_runtime import (
                install_split_group_backend_compat,
            )

            install_split_group_backend_compat(torch.distributed)
        return super().init_device()


__all__ = ["HcuGPUWorker"]
