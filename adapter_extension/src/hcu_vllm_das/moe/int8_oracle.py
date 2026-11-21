"""Unregistered INT8 expert prototype; official v0.33 owns model routing.

The historical installer rebound official oracle globals. It is deliberately
unavailable: v0.33 has no public INT8 expert registry. Direct-provider/operator
experiments do not certify production model integration.
"""

from __future__ import annotations

from .fused_moe_aiter import apply_aiter_int8_moe

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
EXPERTS_MODULE = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
QUANT_UTILS_MODULE = (
    "vllm.model_executor.layers.quantization.utils.quant_utils"
)
INT8_CONSUMER_MODULES = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
    "vllm.model_executor.layers.quantization.online.int8",
    "vllm.model_executor.layers.quantization.quark.quark_moe",
)
INSTALL_MARKER = "_hcu_vllm_das_aiter_int8_installed"


class Int8OracleInstallError(RuntimeError):
    """Raised before a changed or late vLLM INT8 oracle can be modified."""


HcuAiterInt8Experts: type | None = None


def _make_experts_class(
    parent: type,
    int8_weight_key: object,
    int8_activation_key: object,
) -> type:
    class _HcuAiterInt8Experts(parent):
        consumes_expert_mask = False

        @staticmethod
        def _supports_current_device() -> bool:
            return True

        @staticmethod
        def _supports_activation(activation: object) -> bool:
            # Other parent activations need parameters this INT8 ABI does not
            # forward (for example SITU beta or SwiGLU-OAI clamping).
            return getattr(activation, "value", activation) in ("silu", "gelu")

        @staticmethod
        def is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        ):
            if getattr(moe_config, "has_bias", False):
                return False, "HCU AITER INT8 MoE does not support expert bias"
            return super().is_supported_config(
                cls, moe_config, weight_key, activation_key, activation_format
            )

        @staticmethod
        def _supports_quant_scheme(
            weight_key: object,
            activation_key: object,
        ) -> bool:
            return (
                weight_key == int8_weight_key
                and activation_key == int8_activation_key
            )

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            if not bool(getattr(self.quant_config, "use_int8_w8a8", False)):
                return super().apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            if expert_tokens_meta is not None:
                raise Int8OracleInstallError(
                    "HCU AITER INT8 MoE does not support expert token metadata"
                )
            if apply_router_weight_on_input:
                raise Int8OracleInstallError(
                    "HCU AITER INT8 MoE does not support router weight on input"
                )
            result = apply_aiter_int8_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=self.moe_config,
                quant_config=self.quant_config,
                activation=activation,
                expert_map=expert_map,
                output_dtype=output.dtype,
            )
            output.copy_(result)
            return None

    _HcuAiterInt8Experts.__name__ = "HcuAiterInt8Experts"
    _HcuAiterInt8Experts.__qualname__ = "HcuAiterInt8Experts"
    _HcuAiterInt8Experts.__module__ = __name__
    return _HcuAiterInt8Experts



def install() -> bool:
    """Reject the retired private-oracle integration without importing Core."""
    raise Int8OracleInstallError(
        "HCU AITER INT8 model integration requires a public expert registration "
        "API unavailable in vLLM 0.33; use official INT8 auto/triton. "
        "No official oracle was modified."
    )
