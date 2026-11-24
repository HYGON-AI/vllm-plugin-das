# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional native iHC launch tuning on owned HY4 backbone layers.

Only launch constants differ. Core owns the kernels, rounding boundaries,
decoder loop, checkpoint parameters and graph lifecycle. Restart to change
this policy; the master rollback or an unset child retains native layers.
"""
import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.models.hy_v4.amd import hc, model as native, triton_ihc as kernels

from .environment import _flag, custom_ops_enabled
from .hy4_constructor import bind_constructor

logger = init_logger(f'vllm.{__name__}')


def _device_arch():
    from vllm.platforms import current_platform
    if current_platform.device_name != 'hcu':
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0]


def _supported_config(cfg):
    model = cfg.model_config
    hf = model.hf_config
    return (getattr(hf, 'enable_ihc', False)
            and hf.hidden_size == 6144 and hf.hc_mult == 4
            and model.dtype == torch.bfloat16
            and cfg.use_v2_model_runner
            and cfg.compilation_config.mode == 0
            and _device_arch() == 'gfx938')


def enabled(cfg):
    master = custom_ops_enabled()
    child = _flag('HCU_VLLM_HY4_IHC_LAUNCH', default=False)
    return master and child and _supported_config(cfg)


_LAUNCH = {'_PRE_STAGE1_WARPS': 4, '_PRE_STAGE2_WARPS': 4}
tuned_pre = bind_constructor(kernels.triton_ihc_pre, _LAUNCH)
tuned_chain = bind_constructor(kernels.triton_ihc_post_pre_rms_norm, _LAUNCH)


def _vendor_topology(cfg=None):
    cfg = get_current_vllm_config() if cfg is None else cfg
    model, parallel = cfg.model_config, cfg.parallel_config
    return (getattr(model.hf_config, 'model_type', None) == 'hy_v4'
            and not model.is_multimodal_model and model.enforce_eager
            and (parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
                 parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size,
                 parallel.data_parallel_size) == (1, 1, 16, 1, 1))


def _use_vendor(x):
    return (custom_ops_enabled() and _flag('HCU_VLLM_HY4_IHC_VENDOR', default=False)
            and _vendor_topology() and x.device.type == 'cuda'
            and x.dtype == torch.bfloat16 and x.ndim == 3
            and x.shape[0] >= 378 and tuple(x.shape[1:]) == (4, 6144)
            and x.is_contiguous())


def _vendor():
    from hcu_adapter.bridges import vllm_ihc
    return vllm_ihc


def _pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    function = _vendor().pre if _use_vendor(x) else tuned_pre
    return function(x, weight, scale, base, magnitude, hc_eps, norm_eps)


def _chain(x, residual, post, weight, scale, base, norm_weight,
           magnitude, hc_eps, norm_eps):
    function = _vendor().post_pre_rms_norm if _use_vendor(residual) else tuned_chain
    return function(x, residual, post, weight, scale, base, norm_weight,
                    magnitude, hc_eps, norm_eps)


class TunedHCPre(hc.HYV4HCPreLayer):
    forward = bind_constructor(hc.HYV4HCPreLayer.forward,
                               {'triton_ihc_pre': _pre})


class TunedHCLayer(hc.HYV4HCLayer):
    __init__ = bind_constructor(hc.HYV4HCLayer.__init__,
                               {'HYV4HCPreLayer': TunedHCPre})


class TunedDecoder(native.HYV4DecoderLayer):
    __init__ = bind_constructor(native.HYV4DecoderLayer.__init__,
                               {'HYV4HCLayer': TunedHCLayer})
    _forward_ihc = bind_constructor(native.HYV4DecoderLayer._forward_ihc,
        {'triton_ihc_post_pre_rms_norm': _chain})


def selected(cfg):
    use_tuned = enabled(cfg)
    if use_tuned:
        logger.info_once('HY4 native iHC launch 4/4 enabled for gfx938 backbone')
        if _flag('HCU_VLLM_HY4_IHC_VENDOR', default=False) and _vendor_topology(cfg):
            _vendor().validate_abi()
            logger.info_once('HY4 PCP16 BoltOps iHC selected for rows >= 378')
    return use_tuned
