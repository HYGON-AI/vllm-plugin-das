# SPDX-License-Identifier: Apache-2.0
# vllm-hcu: route ModelOpt W8A16 (int8 weight-only) layers to the vllm-hcu
# deepgemm/aiter W8A16 path. Uses upstream 0.28's LINEAR_METHOD_BUILDERS
# escape hatch for linear layers and intercepts RoutedExperts for the MoE
# method; no module swap, so upstream ModelOpt formats are untouched.

from __future__ import annotations

from types import ModuleType

from ._common import (
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.model_executor.layers.quantization.modelopt"
PATCH_ID = "worker.op_opt.modelopt.w8a16_hcu"
TARGETS = (
    f"{TARGET_MODULE}.LINEAR_METHOD_BUILDERS",
    f"{TARGET_MODULE}.LINEAR_ALGOS",
    f"{TARGET_MODULE}.ModelOptQuantConfigBase.get_quant_method",
    f"{TARGET_MODULE}.ModelOptQuantConfigBase.__init__",
)
_MARKER = "_vllm_hcu_modelopt_w8a16_patched"


def apply_to_module(module: ModuleType) -> bool:
    mo = load_exact_module(TARGET_MODULE, module)
    if getattr(mo, _MARKER, False):
        return False

    builders = getattr(mo, "LINEAR_METHOD_BUILDERS", None)
    linear_algos = getattr(mo, "LINEAR_ALGOS", None)
    quant_algos = getattr(mo, "QUANT_ALGOS", None)
    # 0.28 splits the config into ModelOptQuantConfigBase + per-format
    # subclasses (Fp8/NvFp4/MxFp4/Mixed); the shared dispatch and __init__
    # live on the base class, which the flat W8A16 format uses via the
    # modelopt-registered Fp8 config (algos_owned_by("modelopt") already
    # includes W8A16 through the LINEAR_ALGOS entry above).
    config_cls = require_class(
        mo, "ModelOptQuantConfigBase",
        f"{TARGET_MODULE}.ModelOptQuantConfigBase")
    require_callable(
        config_cls, "get_quant_method",
        f"{TARGET_MODULE}.ModelOptQuantConfigBase.get_quant_method")

    from vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu import (
        ModelOptW8A16LinearMethod,
        ModelOptW8A16MoEMethod,
    )

    # Linear: bespoke-method escape hatch (see upstream modelopt.py notes).
    if builders is not None and "W8A16" not in builders:
        builders["W8A16"] = lambda cfg, prefix: ModelOptW8A16LinearMethod(cfg)

    # get_scheme routes algo -> (owner, sub-config attr); the flat
    # (non-mixed) W8A16 format has no separate sub-config, so the owning
    # ModelOptConfig doubles as it via the w8a16_config attribute attached
    # by the __init__ wrapper below.
    if linear_algos is not None and "W8A16" not in linear_algos:
        linear_algos["W8A16"] = ("modelopt", "w8a16_config")
    if quant_algos is not None and "W8A16" not in quant_algos:
        quant_algos.append("W8A16")

    if not hasattr(config_cls, "w8a16_config"):
        orig_init = config_cls.__init__

        def _hcu_w8a16_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.w8a16_config = self

        config_cls.__init__ = _hcu_w8a16_init

    # MoE: the base class dispatches RoutedExperts straight to
    # FusedMoEMethodCls (FP8 assert for W8A16); intercept before it does.
    # Flat configs carry the algo in self.quant_method; mixed-precision
    # configs resolve it per prefix via _resolve_quant_algo.
    orig_gqm = config_cls.get_quant_method

    def _hcu_w8a16_get_quant_method(self, layer, prefix=""):
        if type(layer).__name__ == "RoutedExperts":
            algo = getattr(self, "quant_method", None)
            if algo is None and hasattr(self, "_resolve_quant_algo"):
                algo = self._resolve_quant_algo(prefix)
            if str(algo).upper() == "W8A16":
                return ModelOptW8A16MoEMethod(
                    quant_config=self, moe_config=layer.moe_config)
        return orig_gqm(self, layer, prefix)

    config_cls.get_quant_method = _hcu_w8a16_get_quant_method
    setattr(mo, _MARKER, True)
    return True
