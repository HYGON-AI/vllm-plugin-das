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


class HcuHY4CTBackbone(native.HYV4Model):
    load_weights = bind_constructor(native.HYV4Model.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})


class HcuHY4CTLaunchBackbone(HcuHY4CTBackbone):
    __init__ = bind_constructor(native.HYV4Model.__init__,
                               {'HYV4DecoderLayer': ihc_launch.TunedDecoder})


def make_ct_backbone(*, vllm_config, prefix=''):
    hy4_dp_metadata.prepare(vllm_config)
    cls = HcuHY4CTLaunchBackbone if ihc_launch.selected(vllm_config) else HcuHY4CTBackbone
    return dense_fusion.attach(cls(vllm_config=vllm_config, prefix=prefix), vllm_config)


def make_ct_predictor(*, vllm_config, prefix=''):
    return dense_fusion.attach(
        mtp.HYV4MultiTokenPredictor(vllm_config=vllm_config, prefix=prefix), vllm_config)


class HcuHY4CTForCausalLM(native.HYV4ForCausalLM):
    __init__ = bind_constructor(native.HYV4ForCausalLM.__init__,
                               {'HYV4Model': make_ct_backbone})


class HcuHY4CTMTP(mtp.HYV4MTP):
    __init__ = bind_constructor(mtp.HYV4MTP.__init__,
                               {'HYV4MultiTokenPredictor': make_ct_predictor})
    load_weights = bind_constructor(mtp.HYV4MTP.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})
