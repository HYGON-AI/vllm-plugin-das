# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned decoder attention RMS and native CT input-quantization handoff."""
import logging
import math

import torch
from hcu_adapter import rms_int8_scope as scope
from .adapter_api import adapter_module

HcuDSV41Int8LinearMethod = adapter_module('dsv41_int8_linear').HcuDSV41Int8LinearMethod
from vllm import envs, ir
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8

from .environment import rms_norm_enabled
from .rms_norm import HcuRMSNorm

_LOG = logging.getLogger(__name__)


class PairedAttentionRMSNorm(HcuRMSNorm):
    """Keep the registered weight name and all unsupported native behavior."""

    def __init__(self, original, fused):
        super().__init__(original.hidden_size, original.variance_epsilon,
                         var_hidden_size=original.variance_size_override,
                         has_weight=original.has_weight, dtype=original.weight.dtype)
        self.weight = original.weight
        self.pass_weight = original.pass_weight
        self.pass_weight_add = original.pass_weight_add
        self.train(original.training)
        self._fused = fused

    def forward(self, x, residual=None):
        if (residual is None and scope.active() and self.pass_weight
                and self.variance_size_override is None
                and not rms_norm_enabled() and not envs.VLLM_BATCH_INVARIANT
                and ir.ops.rms_norm.get_priority()[:1] == ['vllm_c']
                and scope.supported(x, self.weight, self.variance_epsilon)):
            y, codes, scales = self._fused(x, self.weight, self.variance_epsilon)
            scope.publish(y, codes, scales)
            return y
        return super().forward(x, residual)


def install_owned_decoder(decoder):
    """Only this FP4 decoder's contiguous, separate attention norm is paired."""
    original = decoder.attn_norm
    projection = decoder.attn.fused_wqa_wkv
    scheme = getattr(projection, 'scheme', None)
    if (torch.version.hip is None or type(original) is not HcuRMSNorm
            or original.weight.device.type != 'cuda'
            or original.hidden_size != 5120
            or not math.isfinite(original.variance_epsilon) or original.variance_epsilon < 0
            or not original.has_weight or not original.pass_weight
            or original.variance_size_override is not None
            or decoder.use_sequence_parallel or decoder.fuse_seam_norm
            or decoder.mono is not None or rms_norm_enabled()
            or envs.VLLM_BATCH_INVARIANT
            or ir.ops.rms_norm.get_priority()[:1] != ['vllm_c']
            or getattr(decoder.attn, '_wqa_wkv_scale', None) is not None
            or type(getattr(projection, 'quant_method', None)) is not HcuDSV41Int8LinearMethod
            or type(scheme) is not CompressedTensorsW8A8Int8
            or scheme.strategy != 'channel' or scheme.is_static_input_scheme
            or not scheme.input_symmetric
            or type(scheme.kernel) is not TritonInt8ScaledMMLinearKernel):
        return False
    # The scoped handoff is Python orchestration. It is captured normally by
    # whole-model CUDA graphs, but is not an Inductor graph rewrite.
    if get_current_vllm_config().compilation_config.mode != CompilationMode.NONE:
        return False
    props = torch.cuda.get_device_properties(original.weight.device)
    if props.gcnArchName.split(':')[0] != 'gfx938':
        return False
    # Compile/load once while constructing the model, outside CUDA graph capture.
    from hcu_adapter.kernels.rms_int8_cpp import module
    fused = module().apply
    decoder.attn_norm = PairedAttentionRMSNorm(original, fused)
    _LOG.info('HCU W4A8 paired native RMS/dynamic INT8 enabled: %s', decoder.attn.prefix)
    return True
