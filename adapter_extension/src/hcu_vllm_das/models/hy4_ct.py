# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CT checkpoint admission around native HY4 constructors and loaders."""
from vllm.models.hy_v4.amd import model as native
from vllm.models.hy_v4.nvidia import mtp

from ..hy4_constructor import bind_constructor
from ..hy4_indexer_loader import load_indexer_wk
from ..hy4_ihc_launch import make_ct_decoder


class HcuHY4CTBackbone(native.HYV4Model):
    __init__ = bind_constructor(native.HYV4Model.__init__,
                               {'HYV4DecoderLayer': make_ct_decoder})
    load_weights = bind_constructor(native.HYV4Model.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})


class HcuHY4CTForCausalLM(native.HYV4ForCausalLM):
    __init__ = bind_constructor(native.HYV4ForCausalLM.__init__,
                               {'HYV4Model': HcuHY4CTBackbone})


class HcuHY4CTMTP(mtp.HYV4MTP):
    load_weights = bind_constructor(mtp.HYV4MTP.load_weights,
        {'_try_load_fp8_indexer_wk': load_indexer_wk})
