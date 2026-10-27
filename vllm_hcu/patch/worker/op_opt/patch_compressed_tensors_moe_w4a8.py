# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route channel compressed-tensors W4A8 to fixed-HT HCU DeepGEMM."""
from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w4a8_deepgemm"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method",)
_MARKER = "_vllm_hcu_ct_w4a8_deepgemm"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "CompressedTensorsMoEMethod", TARGETS[0])
    original = require_callable(cls, "get_moe_method", TARGETS[0])
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("stale HCU CT W4A8 routing marker; restart worker")
        return False
    require_exact_signature(
        original, TARGETS[0], positional=("quant_config", "layer", "layer_name")
    )

    @functools.wraps(original)
    def get_moe_method(quant_config, layer, layer_name):
        # Every other quantization/backend retains upstream selection.
        if not target.current_platform.is_rocm() or layer.moe_config.moe_backend != "deep_gemm":
            return original(quant_config, layer, layer_name)
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, layer_name + suffix)
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        scheme = schemes[0]
        if scheme is None:
            return original(quant_config, layer, layer_name)
        weight = scheme.get("weights")
        activation = scheme.get("input_activations")
        if not quant_config._is_dynamic_token_w4a8_int(weight, activation):
            return original(quant_config, layer, layer_name)
        if not all(other == scheme for other in schemes):
            raise ValueError("All HCU W4A8 MoE projections require the same quantization scheme")
        if scheme.get("format") != "pack-quantized":
            raise ValueError("HCU W4A8 requires the pack-quantized checkpoint format")
        from vllm.config import get_current_vllm_config
        from vllm_hcu.deepseek_v4_runtime import model_architectures
        config = get_current_vllm_config()
        if "DeepseekV41ForCausalLM" not in model_architectures(config):
            raise ValueError("HCU CT W4A8 DeepGEMM bring-up currently supports DeepSeek-V4.1 only")
        # Reject unsupported topology before allocating/repacking weights.
        #
        # Speculative draft models are a legitimate exception to the PCP8 topology
        # rule. vLLM forces prefill_context_parallel_size=1 on draft models by
        # design (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:39-45,
        # inherited by DSparkSpeculator at dspark/speculator.py:40), because PCP
        # splits the PREFILL context and a draft model only serves decode-time
        # speculation. The draft is still W4A8: the checkpoint carries a dedicated
        # group_int4_mtp_experts scheme (same INT4 channel weights / INT8 dynamic
        # token activations as group_int4_routed_experts) and SGLang likewise runs
        # the draft on deep_gemm (--speculative-moe-runner-backend deep_gemm).
        # So the draft keeps the same DeepGEMM path, only the PCP expectation differs.
        from vllm.compilation import backends as _compile_backends
        is_draft_model = _compile_backends.model_tag != "backbone"
        parallel = config.parallel_config
        reasons = []
        if parallel.tensor_parallel_size != 1: reasons.append(f"tp={parallel.tensor_parallel_size}")
        if parallel.pipeline_parallel_size != 1: reasons.append(f"pp={parallel.pipeline_parallel_size}")
        if parallel.decode_context_parallel_size != 1: reasons.append(f"dcp={parallel.decode_context_parallel_size}")
        if is_draft_model:
            # vLLM forces pcp=1 for draft models; anything else would mean the
            # assumption above no longer holds.
            if parallel.prefill_context_parallel_size != 1:
                reasons.append(f"draft_pcp={parallel.prefill_context_parallel_size}")
        elif parallel.prefill_context_parallel_size != 8:
            reasons.append(f"pcp={parallel.prefill_context_parallel_size}")
        if parallel.data_parallel_size != 1: reasons.append(f"dp={parallel.data_parallel_size}")
        if parallel.enable_eplb: reasons.append("eplb=True")
        if not config.model_config.enforce_eager: reasons.append("enforce_eager=False")
        if config.lora_config is not None: reasons.append("lora!=None")
        if reasons:
            scope = "draft" if is_draft_model else "target"
            raise ValueError(
                "HCU CT W4A8 bring-up requires eager TP1/DP1/PP1/DCP1 without LoRA/EPLB "
                f"(target PCP8, draft PCP1); scope={scope}; mismatched: {', '.join(reasons)}"
            )
        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_deepgemm import (
            HCUCompressedTensorsW4A8Int8MoEMethod,
        )
        return HCUCompressedTensorsW4A8Int8MoEMethod(weight, activation, layer.moe_config)

    setattr(get_moe_method, _MARKER, True)
    cls._vllm_hcu_original_get_moe_method = staticmethod(original)
    cls.get_moe_method = staticmethod(get_moe_method)
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
