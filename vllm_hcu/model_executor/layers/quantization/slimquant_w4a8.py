from typing import Any, Dict, List, Optional

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.utils import set_weight_attrs

from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import (LinearBase,LinearMethodBase)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

from vllm import envs

from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm2
    from aiter.moe import (
    get_aiter_moe_config,
    aiter_moe,
    MoeQuantType,
    )
    from aiter import dtypes
except ImportError as e:
    print("Import error msg: import aiter")

class SlimQuantW4A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlimQuantW4A8Int8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            #return SlimQuantW4A8Int8LinearMethod(self)
            layer.scheme = CompressedTensorsW8A8Int8(
                QuantizationStrategy.CHANNEL, False, True
            )
            return SlimQuantW4A8Int8LinearMethod(self)
            # return CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL,False,True)
        elif isinstance(layer, FusedMoE):
            # if envs.VLLM_ROCM_USE_AITER_MOE:
            return SlimQuantW4A8Int8AiterMoEMethod(self, layer.moe_config)
            # else:
            #     return SlimQuantW4A8Int8MoEMethod(self, layer.moe_config)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class SlimQuantW4A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: SlimQuantW4A8Int8Config):
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)

class SlimQuantW4A8Int8AiterMoEMethod:
    """MoE method for W4A8INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config, moe):
        self.moe = moe
        self.quant_config = quant_config
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.moe_kernel: mk.FusedMoEKernel | None = None
        
    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module)-> Optional[FusedMoEQuantConfig]:
        self.moe_quant_config = FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
            weight_dtype='int4'
        )
        return self.moe_quant_config
    
    @property
    def is_monolithic(self) -> bool:
        return False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size=intermediate_size_per_partition
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size//2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size//2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def repack_and_shuffle_w4a8(self, weight_data, E):
        """
        逐 expert 处理 [n, k_half]
        处理完直接写回 weight_data[i]
        """
        # 原始 shape: [E, n, k_half]
        for i in range(E):
            # 1. 取当前 expert [n, k_half]
            expert = weight_data[i]
            n, k_half = expert.shape

            # 2. repack 逻辑（连续 → blocked）
            w_u8 = expert.to(torch.uint8)
            
            # 解包 1byte → 2个4bit
            w_unpacked = torch.stack([
                (w_u8 >> 4) & 0x0F,
                w_u8 & 0x0F
            ], dim=-1).view(n, -1)

            # 8个4bit分块重排
            blocks = w_unpacked.view(n, -1, 8)
            w_low = blocks[..., :4]
            w_high = blocks[..., 4:]
            packed = (w_low << 4) | w_high
            packed = packed.view(n, k_half)

            # 3. shuffle
            w_marlin_in = w4a8_moe_layout_shuffle_gemm2(packed)
            w_marlin_in = w_marlin_in.reshape(n, k_half)
            # 4. 直接写回
            weight_data[i] = w_marlin_in

        return weight_data
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        E=layer.w13_weight.shape[0]  
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.w13_weight = Parameter(self.repack_and_shuffle_w4a8(layer.w13_weight.data, E), requires_grad=False)
        layer.w2_weight = Parameter(self.repack_and_shuffle_w4a8(layer.w2_weight.data, E), requires_grad=False)
  
    def apply(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
            **_,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        E = layer.w13_weight.size(0)
        K = x.size(-1)
        N1 = layer.w13_weight.size(1)

        if x.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            M = x.size(0)
        else:
            assert x.dim() == 3
            assert x.size(0) == E, f"{x.size(0)} == {E}"
            M = x.size(1)
        topk = topk_ids.size(1)
        status, moe_cfg = get_aiter_moe_config(
            M=M,
            E=E,
            N1=N1,
            N2=N1//2,
            K=K,
            top_k=topk,
            block_size=None,
            dtype=x.dtype,
            quant_type=MoeQuantType.W4A8,
        )
        if not status:
            assert moe_cfg.solution_type is None
            assert moe_cfg.config is None
            logger.info(f"[get_config_w4a8] {M=}, no solution found")

        return aiter_moe(
            x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_cfg,
            activation="silu",
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            global_num_experts=E,
            expert_map=None,
        )