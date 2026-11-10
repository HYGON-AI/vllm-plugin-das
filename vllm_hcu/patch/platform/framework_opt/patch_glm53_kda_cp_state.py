# SPDX-License-Identifier: Apache-2.0
"""Size GLM5Next head-sharded states before any distributed groups exist."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.v1.glm53_kda_cp_config import (
    glm53_text_config,
    resolve_kda_head_parallel,
    validate_glm53_pcp_topology,
)

from ..core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_positional_signature,
)

TARGET_MODULE = "vllm.models.glm5next.nvidia.model"
PATCH_ID = "platform.framework_opt.glm53.kda_cp_state"
TARGETS = (f"{TARGET_MODULE}.Glm5NextForCausalLM.get_mamba_state_shape_from_config",)
_MARKER = "_hcu_glm53_kda_cp_state"


def apply_to_module(module: ModuleType) -> bool:
    model = load_exact_module(TARGET_MODULE, module)
    cls = vars(model).get("Glm5NextForCausalLM")
    if not isinstance(cls, type):
        raise PatchCompatibilityError(f"required class {TARGETS[0]} is missing")
    descriptor = vars(cls).get("get_mamba_state_shape_from_config")
    if not isinstance(descriptor, classmethod):
        raise PatchCompatibilityError(f"{TARGETS[0]} must be a classmethod")
    original = descriptor.__func__
    if getattr(original, _MARKER, False):
        return False
    require_positional_signature(original, TARGETS[0], ("cls", "vllm_config"))

    @functools.wraps(original)
    def get_shape(cls, vllm_config):
        plan = resolve_kda_head_parallel(vllm_config)
        if plan.group == "tp":
            return original(cls, vllm_config)
        validate_glm53_pcp_topology(vllm_config)
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        config = glm53_text_config(vllm_config)
        speculative = vllm_config.speculative_config
        num_spec = speculative.num_speculative_tokens if speculative else 0
        return MambaStateShapeCalculator.kda_state_shape(
            plan.size, config.linear_num_heads, config.linear_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim, num_spec=num_spec,
        )

    setattr(get_shape, _MARKER, True)
    cls.get_mamba_state_shape_from_config = classmethod(get_shape)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
