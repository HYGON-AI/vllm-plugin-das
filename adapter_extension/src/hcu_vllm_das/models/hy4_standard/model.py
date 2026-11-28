# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 VL/DFly bindings around unchanged stock c440 backbones."""

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.hy_v4.amd import model as native

from ...hy4_constructor import bind_constructor
from ...hy4_indexer_loader import load_indexer_wk
from .attention import (
    HYV4StandardAttention,
    bind_stock_constructor,
    raw_constructor,
)
from .weight_utils import filter_target_weights

_STOCK_DECODER_INIT = native.HYV4DecoderLayer.__init__
_STOCK_MODEL_INIT = native.HYV4Model.__init__
_STOCK_MODEL_FORWARD = native.HYV4Model.forward
_STOCK_LM_LOAD = native.HYV4ForCausalLM.load_weights
_BOUND_DECODER_INIT = bind_stock_constructor(
    _STOCK_DECODER_INIT,
    "23f13c4ac55911256dd824d311527ce8fbeb1d4d2d90ccad55f9c7d0affc189e",
    {"HYV4MLAAttention": HYV4StandardAttention},
)


def require_eager(vllm_config):
    compilation = vllm_config.compilation_config
    if (
        not vllm_config.model_config.enforce_eager
        or compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        raise ValueError(
            "Standard HY4 VL/DFly stock-Core binding requires explicit eager execution"
        )


def check_stock_model_anchors():
    if (
        native.HYV4DecoderLayer.__init__ is not _STOCK_DECODER_INIT
        or native.HYV4Model.__init__ is not _STOCK_MODEL_INIT
        or native.HYV4Model.forward is not _STOCK_MODEL_FORWARD
        or native.HYV4ForCausalLM.load_weights is not _STOCK_LM_LOAD
        or raw_constructor(_STOCK_MODEL_INIT).__globals__["HYV4DecoderLayer"]
        is not native.HYV4DecoderLayer
        or _STOCK_DECODER_INIT.__globals__["HYV4MLAAttention"]
        is not HYV4StandardAttention.__bases__[0]
    ):
        raise RuntimeError("HY4 stock model code/global owner changed")


class HYV4StandardDecoderLayer(native.HYV4DecoderLayer):
    __init__ = _BOUND_DECODER_INIT


_BOUND_MODEL_INIT = bind_stock_constructor(
    _STOCK_MODEL_INIT,
    "c629c45638077b0c3f7634e1bf52979b5522c7acec8d113bc1da361125f537c0",
    {"HYV4DecoderLayer": HYV4StandardDecoderLayer},
)


class HYV4StandardModel(native.HYV4Model):
    load_weights = bind_constructor(
        native.HYV4Model.load_weights, {"_try_load_fp8_indexer_wk": load_indexer_wk}
    )

    def __init__(self, *, vllm_config, prefix=""):
        check_stock_model_anchors()
        require_eager(vllm_config)
        from .pd_runtime import runtime_role
        runtime_role(vllm_config)
        if vllm_config.lora_config is not None:
            raise ValueError("Standard HY4 VL/DFly does not admit LoRA")
        if vllm_config.attention_config.hisparse_config is not None:
            raise ValueError("Standard HY4 VL/DFly does not admit HiSparse")
        _BOUND_MODEL_INIT(self, vllm_config=vllm_config, prefix=prefix)
        self.dfly_enabled = (
            getattr(vllm_config.speculative_config, "method", None) == "dfly"
        )

    def embed_input_ids(self, input_ids):
        if self.dfly_enabled:
            input_ids = input_ids.clamp_min(0)
        return super().embed_input_ids(input_ids)


class HYV4ForCausalLM(native.HYV4ForCausalLM):
    __init__ = bind_constructor(
        native.HYV4ForCausalLM.__init__, {"HYV4Model": HYV4StandardModel}
    )

    def load_weights(self, weights):
        check_stock_model_anchors()
        return super().load_weights(filter_target_weights(weights))
