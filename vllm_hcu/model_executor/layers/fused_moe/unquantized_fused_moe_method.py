# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""
HCU version of UnquantizedFusedMoEMethod.
Inherits from vllm's version and overrides process_weights_after_loading
to preserve canonical AITER weights while initializing the MoE kernel.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    make_unquantized_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as _Original,
)


def _is_hcu_aiter_moe_requested(method: object | None = None) -> bool:
    from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
        is_aiter_moe_requested,
    )

    return is_aiter_moe_requested(getattr(method, "moe", None))


def _expert_routing_tables(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    routing_tables = getattr(layer, "_expert_routing_tables", None)
    if callable(routing_tables):
        return routing_tables()

    legacy_routing_tables = getattr(layer, "_maybe_init_expert_routing_tables", None)
    if callable(legacy_routing_tables):
        return legacy_routing_tables()

    return None

class HcuUnquantizedFusedMoEMethod(_Original):
    """HCU version of UnquantizedFusedMoEMethod.

    Initializes the AITER runner without replacing canonical model weights.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            not _is_hcu_aiter_moe_requested(self)
            or getattr(self, "unquantized_backend", None)
            != UnquantizedMoeBackend.AITER
        ):
            return super().process_weights_after_loading(layer)

        if getattr(layer, "_hcu_aiter_moe_initialized", False):
            return

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=_expert_routing_tables(layer),
        )
        layer._hcu_aiter_moe_initialized = True
