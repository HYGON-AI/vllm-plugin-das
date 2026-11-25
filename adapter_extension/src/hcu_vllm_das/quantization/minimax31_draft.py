# SPDX-License-Identifier: Apache-2.0
"""MiniMax DSpark's CT channel-INT8 scheme with explicit IPC restoration."""
import torch
from vllm.model_executor.layers.linear import (
    LinearBase,UnquantizedLinearMethod,register_weight_loader_v2_supported_method)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,CompressedTensorsLinearMethod)
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.model_executor.layers.attention import Attention


class MiniMax31DraftInt8Config(CompressedTensorsConfig):
    @classmethod
    def get_name(cls):return "minimax31_dspark_int8"

    @classmethod
    def override_quantization_method(cls,hf_quant_cfg,user_quant,*,hf_config=None):
        if user_quant != cls.get_name():return None
        if (hf_quant_cfg.get('quant_method')!='compressed-tensors' or
                getattr(hf_config,'architectures',None)!=['DSparkMiniMaxDraftModel']):
            raise ValueError('MiniMax draft INT8 requires its explicit CT checkpoint architecture')
        return cls.get_name()

    def get_quant_method(self,layer,prefix):
        # This checkpoint has weight/activation INT8, but no KV quantization.
        # Keep Core's ordinary unit-scale attention. CT's generic KV method
        # deletes its placeholder Parameters on the producer, then tries to
        # read newly-created meta placeholders during IPC restoration.
        if isinstance(layer,Attention) and self.kv_cache_scheme is None:
            return None
        # These checkpoint modules are deliberately BF16. The scope is this
        # quantization name only; generic CT policies remain untouched.
        if isinstance(layer,LinearBase) and any(prefix.endswith(s) for s in (
                '.fc','.markov_head.markov_w1','.markov_head.markov_w2','.confidence_head.proj')):
            return UnquantizedLinearMethod()
        method=super().get_quant_method(layer,prefix)
        if isinstance(method,CompressedTensorsLinearMethod):
            return MiniMax31DraftLinearMethod(self)
        return method


@register_weight_loader_v2_supported_method
class MiniMax31DraftLinearMethod(CompressedTensorsLinearMethod):
    supports_pre_processed_weights=True

    def process_weights_after_loading(self,layer):
        from .minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel
        kernel=layer.scheme.kernel
        c=kernel.config
        if c.is_static_input_scheme or not c.is_channelwise or not c.input_symmetric:
            raise ValueError("MiniMax draft IPC only supports dynamic symmetric token/channel INT8")
        if not is_weights_pre_processed():
            super().process_weights_after_loading(layer)
        k,n=layer.weight.shape
        if (layer.weight.dtype!=torch.int8 or layer.weight.stride()!=(1,k)
                or layer.weight_scale.dtype!=torch.float32 or layer.weight_scale.shape!=(n,1)):
            raise ValueError("MiniMax draft processed KN weight/scale layout differs")
        layer.scheme.kernel=MiniMax31HipBlasLtInt8Kernel(c,kernel.layer_param_names)
