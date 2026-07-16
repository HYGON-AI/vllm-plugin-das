# SPDX-License-Identifier: Apache-2.0
"""Lightweight registry facades for HCU SlimQuant implementations.

Registry discovery imports this module, but the concrete SlimQuant modules are
loaded only when vLLM validates or instantiates the selected quantization.  A
registered facade therefore proves configuration recognition, not successful
kernel loading; concrete dependency/kernel failures remain explicit at the
feature boundary.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class _SlimQuantFacade(QuantizationConfig):
    _registry_name: ClassVar[str]
    _implementation_module: ClassVar[str]
    _implementation_class: ClassVar[str]

    @classmethod
    def _implementation(cls) -> type[QuantizationConfig]:
        module = importlib.import_module(cls._implementation_module)
        implementation = getattr(module, cls._implementation_class, None)
        if not isinstance(implementation, type) or not issubclass(
            implementation, QuantizationConfig
        ):
            raise RuntimeError(
                f"SlimQuant implementation {cls._implementation_module}."
                f"{cls._implementation_class} is unavailable or incompatible"
            )
        return implementation

    @classmethod
    def get_name(cls) -> str:
        return cls._registry_name

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return cls._implementation().get_supported_act_dtypes()

    @classmethod
    def get_min_capability(cls) -> int:
        return cls._implementation().get_min_capability()

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return cls._implementation().get_config_filenames()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig:
        return cls._implementation().from_config(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        raise RuntimeError(
            "SlimQuant registry facade must be materialized through from_config()"
        )


class SlimQuantMarlinFacade(_SlimQuantFacade):
    _registry_name = "slimquant_marlin"
    _implementation_module = (
        "vllm_hcu.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_marlin"
    )
    _implementation_class = "SlimQuantCompressedTensorsMarlinConfig"

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        del hf_config
        if (
            hf_quant_cfg.get("quant_method") == "compressed-tensors"
            and user_quant == cls._registry_name
        ):
            return "slimquant_compressed_tensors_marlin"
        return None


class SlimQuantCompressedTensorsMarlinFacade(SlimQuantMarlinFacade):
    _registry_name = "slimquant_compressed_tensors_marlin"

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        del hf_config
        if (
            hf_quant_cfg.get("quant_method") == "compressed-tensors"
            and user_quant == cls._registry_name
        ):
            return cls._registry_name
        return None


class SlimQuantW4A8Facade(_SlimQuantFacade):
    _registry_name = "slimquant_w4a8"
    _implementation_module = (
        "vllm_hcu.model_executor.layers.quantization.slimquant_w4a8"
    )
    _implementation_class = "SlimQuantW4A8Int8Config"

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        del hf_config
        if hf_quant_cfg.get("quant_method") != cls._registry_name:
            return None
        if user_quant in (None, cls._registry_name):
            return cls._registry_name
        return None


SLIMQUANT_FACADES: dict[str, type[QuantizationConfig]] = {
    "slimquant_marlin": SlimQuantMarlinFacade,
    "slimquant_compressed_tensors_marlin": (
        SlimQuantCompressedTensorsMarlinFacade
    ),
    "slimquant_w4a8": SlimQuantW4A8Facade,
}


__all__ = [
    "SLIMQUANT_FACADES",
    "SlimQuantCompressedTensorsMarlinFacade",
    "SlimQuantMarlinFacade",
    "SlimQuantW4A8Facade",
]
