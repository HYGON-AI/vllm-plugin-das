# SPDX-License-Identifier: Apache-2.0
"""Official AutoAWQ config with an explicitly enabled dense provider adapter."""

from __future__ import annotations

from vllm import envs
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQConfig,
    AutoAWQLinearMethod,
    AutoAWQMarlinLinearMethod,
)

from .autoawq_linear import LightOpAutoAWQLinearMethod
from .environment import lightop_awq_enabled


class HcuAutoAWQConfig(AutoAWQConfig):
    def get_quant_method(self, layer, prefix):
        selected = super().get_quant_method(layer, prefix)
        if (
            lightop_awq_enabled()
            and not envs.VLLM_BATCH_INVARIANT
            and self.weight_bits == 4
            and self.group_size == 128
            and self.zero_point is True
            and isinstance(selected, (AutoAWQLinearMethod, AutoAWQMarlinLinearMethod))
        ):
            return LightOpAutoAWQLinearMethod(selected, self)
        return selected
