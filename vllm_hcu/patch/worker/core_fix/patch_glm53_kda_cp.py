# SPDX-License-Identifier: Apache-2.0
"""Opt-in KDA CP-head construction, without mutating vLLM source or global TP."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.v1.glm53_kda_cp_config import (
    resolve_kda_head_parallel,
    validate_glm53_pcp_topology,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.glm5next.nvidia.kda"
PATCH_ID = "worker.core_fix.glm53.kda_cp_heads"
TARGETS = (
    f"{TARGET_MODULE}.Glm5NextLinearAttention.__init__",
    f"{TARGET_MODULE}.Glm5NextLinearAttention.get_state_shape",
)
_MARKER = "_hcu_glm53_kda_cp_heads"


def apply_to_module(module: ModuleType) -> bool:
    kda = load_exact_module(TARGET_MODULE, module)
    cls = require_class(kda, "Glm5NextLinearAttention", TARGETS[0])
    original_init = require_callable(cls, "__init__", TARGETS[0])
    original_shape = require_callable(cls, "get_state_shape", TARGETS[1])
    if vars(cls).get(_MARKER, False):
        if not getattr(original_init, _MARKER, False):
            raise PatchCompatibilityError("KDA CP-head initializer marker is stale")
        return False
    require_exact_signature(
        original_init, TARGETS[0],
        positional=("self", "config", "vllm_config", "prefix"),
        defaults={"prefix": ""},
    )
    require_exact_signature(original_shape, TARGETS[1], positional=("self",))

    @functools.wraps(original_init)
    def initialize(self, config, vllm_config, prefix=""):
        plan = resolve_kda_head_parallel(vllm_config)
        if plan.group == "tp":
            return original_init(self, config, vllm_config, prefix)
        # Head ownership and full-model capability are different decisions.
        # Keep this boundary check even if construction bypasses the platform.
        validate_glm53_pcp_topology(vllm_config)
        from vllm_hcu.model_executor.layers.glm53_kda_cp import initialize_cp_kda

        initialize_cp_kda(self, config, vllm_config, prefix, kda, plan.size)

    @functools.wraps(original_shape)
    def get_state_shape(self):
        if not getattr(self, "_hcu_kda_cp_head_sharded", False):
            return original_shape(self)
        return kda.MambaStateShapeCalculator.kda_state_shape(
            self.head_parallel_size, self.num_heads, self.head_dim,
            conv_kernel_size=self.conv_size, num_spec=self.num_spec,
        )

    setattr(initialize, _MARKER, True)
    setattr(get_state_shape, _MARKER, True)
    cls.__init__ = initialize
    cls.get_state_shape = get_state_shape
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
