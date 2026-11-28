# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CT checkpoint admission around native HY4 constructors and loaders."""
from vllm.models.hy_v4.amd import model as native
from vllm.models.hy_v4.nvidia import mtp

from ..hy4_constructor import bind_constructor
from ..hy4_indexer_loader import load_indexer_wk
from .. import hy4_ihc_launch as ihc_launch
from .. import hy4_dense_fusion as dense_fusion
from .. import hy4_dp_metadata
from .. import hy4_fused_routing as fused_routing


class HcuHY4CTBackbone(native.HYV4Model):
    load_weights = bind_constructor(native.HYV4Model.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})


class HcuHY4CTLaunchBackbone(HcuHY4CTBackbone):
    __init__ = bind_constructor(native.HYV4Model.__init__,
                               {'HYV4DecoderLayer': ihc_launch.TunedDecoder})


def make_ct_backbone(*, vllm_config, prefix=''):
    if getattr(vllm_config.speculative_config, "method", None) == "dfly":
        from .hy_v4_flash.dfly_compat import validate_dfly_runtime
        validate_dfly_runtime(vllm_config)
    hy4_dp_metadata.prepare(vllm_config)
    from .hy4_standard.target_policy import uses_standard_target
    standard = uses_standard_target(
        vllm_config.model_config.hf_config,
        method=getattr(vllm_config.speculative_config, "method", None))
    if standard:
        from .hy4_standard.model import HYV4StandardModel
        cls = HYV4StandardModel
    else:
        cls = HcuHY4CTLaunchBackbone if ihc_launch.selected(vllm_config) else HcuHY4CTBackbone
    return fused_routing.attach(
        dense_fusion.attach(ihc_launch.attach_vendor(
            cls(vllm_config=vllm_config, prefix=prefix), vllm_config), vllm_config), vllm_config)


def make_ct_predictor(*, vllm_config, prefix=''):
    return fused_routing.attach(dense_fusion.attach(
        mtp.HYV4MultiTokenPredictor(vllm_config=vllm_config, prefix=prefix), vllm_config), vllm_config)


class HcuHY4CTForCausalLM(native.HYV4ForCausalLM):
    __init__ = bind_constructor(native.HYV4ForCausalLM.__init__,
                               {'HYV4Model': make_ct_backbone})

    def load_weights(self, weights):
        from .hy4_standard.weight_utils import filter_target_weights
        return super().load_weights(filter_target_weights(weights))


class HcuHY4CTMTP(mtp.HYV4MTP):
    __init__ = bind_constructor(mtp.HYV4MTP.__init__,
                               {'HYV4MultiTokenPredictor': make_ct_predictor})
    load_weights = bind_constructor(mtp.HYV4MTP.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})
