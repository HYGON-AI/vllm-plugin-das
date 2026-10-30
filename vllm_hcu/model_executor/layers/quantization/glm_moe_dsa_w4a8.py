# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mixed quantization for GLM MoE DSA: channel INT4 experts and channel INT8 linears.

Routed experts whose prefix matches the INT4 targets are compressed-tensors
pack-quantized INT4 with BF16 per-channel scales and dynamic per-token INT8
activations. Other linears stay on compressed-tensors channel INT8. MTP experts
retry only when ``.mtp_block.`` hides the published name, or when their layer
index is past the backbone and the backbone expert scheme is channel INT4.
Ignored prefixes and non-expert INT4 groups stay on the parent method.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
    convert_compressed_tensors_int4_to_slimquant,
    is_glm_channel_int4_w4a8,
    is_glm_moe_dsa_model,
    repack_and_shuffle_w4a8_gemm2,
    routed_expert_uses_glm_channel_int4,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
)


_GLM_W4A8_GEMM2_GENERATION = "_glm_w4a8_gemm2_generation"
_GLM_EXPERT_WEIGHTS = ("w13_weight", "w2_weight")


def _expert_weight_generation(layer: torch.nn.Module) -> tuple[object, ...]:
    """Identity of the two expert weight tensors for one load generation."""
    generation = []
    for name in _GLM_EXPERT_WEIGHTS:
        tensor = getattr(layer, name, None)
        if not isinstance(tensor, torch.Tensor):
            return ()
        generation.append(
            (
                id(tensor),
                tensor.data_ptr(),
                tensor._version,
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
        )
    return tuple(generation)


def glm_expert_weights_use_gemm2_layout(layer: torch.nn.Module) -> bool:
    """Return whether the current expert weights are the gemm2 conversion."""
    recorded = getattr(layer, _GLM_W4A8_GEMM2_GENERATION, None)
    if not recorded:
        return False
    return recorded == _expert_weight_generation(layer)


def commit_installed_glm_w4a8_gemm2_generation(layer: torch.nn.Module) -> None:
    """Refresh installed GLM layout metadata and runtime scale owners.

    Checkpoint reload records the marker on the temporary materialized
    Parameter, then copies those bytes back onto the original runtime
    Parameter. Kernel-format reload copies into that same Parameter and
    bumps its version. Rebind the existing quant config to the installed
    scale owners as well, preserving references held by the execution path.
    The native fallback also needs its scale owners rebound after reload,
    although its unpacked weights have no gemm2 marker. Neither path may
    run gemm2 repack again.
    """
    if getattr(layer, _GLM_W4A8_GEMM2_GENERATION, None):
        generation = _expert_weight_generation(layer)
        if not generation:
            return
        setattr(layer, _GLM_W4A8_GEMM2_GENERATION, generation)
        layer._glm_w4a8_gemm2_shuffled = True
    elif not all(
        getattr(getattr(layer, name, None), "_hcu_vllm_w4a8_unpacked", False)
        for name in _GLM_EXPERT_WEIGHTS
    ):
        return
    method = getattr(layer, "quant_method", None)
    if isinstance(method, GlmMoeDsaW4A8MoEMethod):
        quant_config = getattr(method, "moe_quant_config", None)
        if quant_config is not None:
            from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
                _replace_tensor_reference,
            )

            _replace_tensor_reference(quant_config, "w1_scale", layer.w13_weight_scale)
            _replace_tensor_reference(quant_config, "w2_scale", layer.w2_weight_scale)


def commit_installed_glm_w4a8_gemm2_generations(model: torch.nn.Module) -> None:
    """Refresh GLM gemm2 markers and scale owners on final runtime parameters."""
    for module in model.modules():
        commit_installed_glm_w4a8_gemm2_generation(module)


def _disable_generic_w4a8_shuffle(config: object) -> None:
    """gemm2 is already the final layout, so skip the generic weight/scale shuffle."""
    if getattr(config, "need_shuffle", False):
        config.need_shuffle = False
    if getattr(config, "need_shuffle_scale", False):
        config.need_shuffle_scale = False


def _wrap_checkpoint_loader(param: torch.nn.Parameter, *, scale: bool) -> None:
    """Convert packing and rescale before the original expert loader runs.

    compressed-tensors scales match the real signed INT4 values. HIPC/AITER
    read the high nibble as 16 times that value, so the SlimQuant canonical
    scale is the checkpoint scale divided by 16. The Triton fallback
    multiplies by 16 again at the call site.
    """
    original = param.weight_loader

    # The first parameter must be named ``param``. Layerwise reload binds this
    # signature and then replaces ``arguments["param"]`` with the materialized
    # Parameter. A different name leaves that replacement unused, so reload
    # keeps writing the meta Parameter.
    def loader(param, loaded_weight, *args, **kwargs):
        if scale:
            loaded_weight = loaded_weight.to(dtype=param.dtype) / 16.0
        elif loaded_weight.dtype == torch.int32:
            loaded_weight = convert_compressed_tensors_int4_to_slimquant(
                loaded_weight
            )
        return original(param, loaded_weight, *args, **kwargs)

    param.weight_loader = loader


class GlmMoeDsaW4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    """Load packed INT4 experts and keep computation on SlimQuant W4A8."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        for name in ("w13_weight", "w2_weight"):
            _wrap_checkpoint_loader(getattr(layer, name), scale=False)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            _wrap_checkpoint_loader(getattr(layer, name), scale=True)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Fold experts into the AITER W4A8 gemm2 layout and skip the generic shuffle.

        An 8-nibble regroup alone is still SlimQuant's pairwise byte order.
        AITER moe_c expects that regroup followed by
        ``w4a8_moe_layout_shuffle_gemm2``. When ``need_shuffle`` is set, the
        parent also calls ``aiter_moe_shfl_weight``, which is a different
        layout; stacking the two still yields garbage on a single run.
        Triton (BoltOps) and DeepEP keep the unshuffled SlimQuant packing.
        """
        backend = getattr(self.moe, "moe_backend", "auto")
        from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import (
            slimquant_w4a8_uses_deepep_auto,
        )

        if backend == "triton" or slimquant_w4a8_uses_deepep_auto(self.moe):
            super().process_weights_after_loading(layer)
            return

        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
            prewarm_aiter_w4a8_moe,
        )

        aiter_config = prewarm_aiter_w4a8_moe(self, layer)
        if aiter_config is None:
            super().process_weights_after_loading(layer)
            return

        if not glm_expert_weights_use_gemm2_layout(layer):
            for name in _GLM_EXPERT_WEIGHTS:
                parameter = getattr(layer, name)
                parameter.data.copy_(repack_and_shuffle_w4a8_gemm2(parameter.data))
            # Record the generation after copy_ so the version matches the
            # converted storage. Replacing either Parameter makes this miss
            # and the conversion runs again.
            setattr(
                layer,
                _GLM_W4A8_GEMM2_GENERATION,
                _expert_weight_generation(layer),
            )
        layer._glm_w4a8_gemm2_shuffled = glm_expert_weights_use_gemm2_layout(layer)

        # Prewarm returns the M=1 config. Clear the generic shuffle so the
        # parent registers the gemm2 layout and does not call
        # aiter_moe_shfl_weight.
        _disable_generic_w4a8_shuffle(aiter_config)
        super().process_weights_after_loading(layer)


class GlmMoeDsaChannelW4A8Config(CompressedTensorsConfig):
    """GLM MoE DSA mixed quant: INT4 experts, original INT8 for other linears."""

    @classmethod
    def get_name(cls) -> str:
        return "glm_moe_dsa_w4a8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlmMoeDsaChannelW4A8Config":
        parsed = super().from_config(config)
        if type(parsed) is cls:
            return parsed
        # If the parent hardcodes the base class, copy parsed fields onto this subclass.
        instance = cls.__new__(cls)
        instance.__dict__.update(parsed.__dict__)
        return instance

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        if user_quant not in (None, cls.get_name()):
            return None
        if not is_glm_moe_dsa_model(hf_config):
            return None
        if not is_glm_channel_int4_w4a8(hf_quant_cfg, hf_config):
            return None
        return cls.get_name()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts) and routed_expert_uses_glm_channel_int4(
            self, layer, prefix
        ):
            return GlmMoeDsaW4A8MoEMethod(self, layer.moe_config)
        return super().get_quant_method(layer, prefix)
