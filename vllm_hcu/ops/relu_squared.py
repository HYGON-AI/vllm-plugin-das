# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native relu^2 replacement for HCU.

``torch.ops._C.relu_squared`` returns NaNs and values up to ~1e36 on
DTK2610.  ReLUSquaredActivation is used by Mamba2 and NemotronH MLPs, so the
corrupted output poisons every downstream layer.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.activation import ReLUSquaredActivation

from vllm_hcu.platforms import envs as henvs


@ReLUSquaredActivation.register_oot
class HcuReLUSquaredActivation(ReLUSquaredActivation):
    """HCU replacement that avoids the DTK2610 relu_squared kernel."""

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        if henvs.VLLM_HCU_USE_NATIVE_RELU_SQUARED:
            return self.forward_native(x)
        return super().forward_hip(x)