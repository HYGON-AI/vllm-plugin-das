# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 admission over the existing packed HCU method and Core weight loader."""
from vllm.config import get_current_vllm_config

from .ct_w4a8 import HcuDSV41CTW4A8Config
from .dsv41_channel_fp8 import _is_hcu
from .hy4_ct_policy import NAME, accepts_checkpoint, is_hy4


class HcuHY4CTW4A8Config(HcuDSV41CTW4A8Config):
    @classmethod
    def get_name(cls):
        return NAME

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if _is_hcu() and accepts_checkpoint(hf_quant_cfg, user_quant, hf_config):
            from hcu_vllm_das.hy4_quant_activation import check_health
            check_health()
            return NAME
        return None

    def get_quant_method(self, layer, prefix):
        if not is_hy4(get_current_vllm_config().model_config.hf_config):
            raise ValueError('hcu_hy4_ct_w4a8 requires a HY4 model configuration')
        # Keep parsed scheme matching, native loader, INT8 dense layers, packed
        # HIPC weights, modular HT/LL and the model's SwiGLU clamp unchanged.
        return super().get_quant_method(layer, prefix)
