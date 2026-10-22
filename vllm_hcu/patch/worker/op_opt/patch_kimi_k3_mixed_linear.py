# SPDX-License-Identifier: Apache-2.0
"""Connect SlimQuant K3 mixed precision through plugin-owned callbacks."""
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from types import ModuleType

from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.mixed_linear"
TARGETS = (f"{TARGET_MODULE}.KimiLinearForCausalLM.__init__",
           f"{TARGET_MODULE}.KimiMoE.__init__",
           f"{TARGET_MODULE}.KimiLinearModel.load_weights")
_MARKER = "_hcu_kimi_mixed_linear"
_LATENT_SCOPE = ContextVar("hcu_kimi_mixed_latent", default=None)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.kimi_k3_mixed_precision import (
        KimiK3MixedKDAProjection, is_mixed_slimquant,
    )
    cls = target.KimiLinearForCausalLM
    original_init = require_callable(cls, "__init__", TARGETS[0])

    @wraps(original_init)
    def init(self, *, vllm_config, **kwargs):
        config = vllm_config.quant_config
        if is_mixed_slimquant(config):
            # K3's model class does not advertise these fused names upstream.
            config.packed_modules_mapping = {
                **config.packed_modules_mapping,
                "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }
        original_init(self, vllm_config=vllm_config, **kwargs)

    moe_cls = target.KimiMoE
    original_moe = require_callable(moe_cls, "__init__", TARGETS[1])
    moe_signature = signature(original_moe)

    @wraps(original_moe)
    def moe_init(self, *args, **kwargs):
        bound = moe_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments["quant_config"]
        scope = (config, bound.arguments["prefix"]) if is_mixed_slimquant(config) else None
        token = _LATENT_SCOPE.set(scope)
        try:
            return original_moe(self, *args, **kwargs)
        finally:
            _LATENT_SCOPE.reset(token)

    class MixedLatentConfig:
        def __init__(self, *args, **kwargs):
            scope = _LATENT_SCOPE.get()
            if scope is not None:
                config, parent = scope
                if kwargs.get("prefix") in (
                    f"{parent}.routed_expert_down_proj",
                    f"{parent}.routed_expert_up_proj",
                ):
                    kwargs["quant_config"] = config
            super().__init__(*args, **kwargs)

    class MixedLatentLinear(MixedLatentConfig, target.ReplicatedLinear):
        pass

    class MixedLatentColumnLinear(MixedLatentConfig, target.ColumnParallelLinear):
        pass

    model_cls = target.KimiLinearModel
    original_load = require_callable(model_cls, "load_weights", TARGETS[2])

    @wraps(original_load)
    def load(self, weights):
        owners = {name.removesuffix(".in_proj_qkvgfab"): (name, sub)
                  for name, sub in self.named_modules()
                  if isinstance(sub, KimiK3MixedKDAProjection)}
        if not owners:
            return original_load(self, weights)
        loaded = set()

        def remaining():
            for entry in weights:
                name, tensor = entry[:2]
                parent, _, suffix = name.rpartition(".")
                attention, _, branch = parent.rpartition(".")
                owner = owners.get(attention)
                if owner is None or branch not in (
                    "q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj",
                ):
                    yield entry
                    continue
                if len(entry) > 2 and entry[2]:
                    raise ValueError(f"Unexpected loader metadata for mixed KDA {name}")
                module_name, projection = owner
                param_name = projection.load_branch(branch, suffix, tensor)
                loaded.add(f"{module_name}.{param_name}")

        return original_load(self, remaining()) | loaded

    cls.__init__ = init
    moe_cls.__init__ = moe_init
    target.ReplicatedLinear = MixedLatentLinear
    target.ColumnParallelLinear = MixedLatentColumnLinear
    model_cls.load_weights = load
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
