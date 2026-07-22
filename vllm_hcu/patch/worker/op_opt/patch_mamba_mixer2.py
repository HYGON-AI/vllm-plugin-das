# SPDX-License-Identifier: Apache-2.0
"""vLLM v0.25 target MambaMixer2 behavior plus HCU NN-layout deltas."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_class, require_exact_signature

TARGET_MODULE = "vllm.model_executor.layers.mamba.mamba_mixer2"
PATCH_ID = "worker.op_opt.mamba.mixer2_nn_layout"
TARGETS = (
    f"{TARGET_MODULE}.mamba_v2_sharded_weight_loader",
    f"{TARGET_MODULE}.MambaMixer2.__init__",
)
_MARKER = "_vllm_hcu_mamba_mixer2_applied"
_WRAPPER = "_vllm_hcu_mamba_mixer2_wrapper"


def _nn_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_USE_NN)


def apply_to_module(module: ModuleType) -> bool:
    mixer = load_exact_module(TARGET_MODULE, module)
    cls = require_class(mixer, "MambaMixer2", f"{TARGET_MODULE}.MambaMixer2")
    wrapped = (
        (mixer, "mamba_v2_sharded_weight_loader", TARGETS[0], _WRAPPER),
        (cls, "__init__", TARGETS[1], _WRAPPER),
    )
    if already_applied(mixer, _MARKER, wrapped):
        return False
    loader_factory = require_callable(mixer, "mamba_v2_sharded_weight_loader", TARGETS[0])
    require_exact_signature(loader_factory, TARGETS[0], positional=("shard_spec", "tp_size", "tp_rank"))
    original_init = require_callable(cls, "__init__", TARGETS[1])
    require_exact_signature(
        original_init, TARGETS[1],
        positional=("self", "hidden_size", "ssm_state_size", "conv_kernel_size",
                    "intermediate_size", "use_conv_bias", "use_bias", "n_groups",
                    "num_heads", "head_dim", "rms_norm_eps", "activation",
                    "use_rms_norm", "model_config", "cache_config", "quant_config", "prefix"),
        defaults={"n_groups": 1, "num_heads": 128, "head_dim": 64,
                  "rms_norm_eps": 1e-5, "activation": "silu", "use_rms_norm": True,
                  "model_config": None, "cache_config": None, "quant_config": None,
                  "prefix": ""},
    )

    @functools.wraps(loader_factory)
    def hcu_loader_factory(shard_spec, tp_size, tp_rank):
        if not _nn_enabled():
            return loader_factory(shard_spec, tp_size, tp_rank)
        from vllm_hcu.model_executor.layers.mamba_runtime import (
            mamba_v2_nn_sharded_weight_loader,
        )
        return mamba_v2_nn_sharded_weight_loader(shard_spec, tp_size, tp_rank)

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _nn_enabled():
            conv_weights = self.conv1d.weight.squeeze(1).transpose(0, 1).contiguous()
            self.conv_weights = conv_weights

    setattr(hcu_loader_factory, _WRAPPER, True)
    setattr(hcu_init, _WRAPPER, True)
    setattr(mixer, "_vllm_hcu_original_mamba_v2_loader", loader_factory)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(mixer, "mamba_v2_sharded_weight_loader", hcu_loader_factory)
    setattr(cls, "__init__", hcu_init)
    setattr(mixer, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
