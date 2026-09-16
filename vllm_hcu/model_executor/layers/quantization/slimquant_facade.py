# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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


class KimiK3W4A8Facade(_SlimQuantFacade):
    """Kimi-K3-specific W4A8 registry entry with strict metadata checks."""

    _registry_name = "kimi_k3_w4a8"
    _implementation_module = (
        "vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8"
    )
    _implementation_class = "KimiK3W4A8Config"

    def __init__(self) -> None:
        super().__init__()
        self._materialized = None

    def _materialized_config(self):
        if self._materialized is None:
            self._materialized = self._implementation()()
        return self._materialized

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        del model_name, revision
        implementation = self._implementation()
        metadata = None
        text_config = getattr(hf_config, "text_config", hf_config)
        if text_config is not None:
            from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
                KIMI_K3_W4A8_DEFAULT_METADATA,
                KimiK3W4A8Metadata,
            )

            values = dict(KIMI_K3_W4A8_DEFAULT_METADATA)
            values.update(
                {
                    "num_experts": getattr(
                        text_config, "num_experts", values["num_experts"]
                    ),
                    "top_k": getattr(
                        text_config,
                        "num_experts_per_token",
                        values["top_k"],
                    ),
                    "hidden_size": getattr(
                        text_config,
                        "routed_expert_hidden_size",
                        values["hidden_size"],
                    ),
                    "intermediate_size": getattr(
                        text_config,
                        "moe_intermediate_size",
                        values["intermediate_size"],
                    ),
                }
            )
            metadata = KimiK3W4A8Metadata(**values)
        self._materialized = implementation(metadata)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._materialized_config().get_quant_method(layer, prefix)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        if getattr(hf_config, "model_type", None) != "kimi_k3":
            return None
        if hf_quant_cfg.get("quant_method") not in {
            "slimquant_w4a8",
            "kimi_k3_w4a8",
        }:
            return None
        if user_quant not in (None, cls._registry_name):
            return None
        return cls._registry_name


SLIMQUANT_FACADES: dict[str, type[QuantizationConfig]] = {
    "slimquant_marlin": SlimQuantMarlinFacade,
    "slimquant_compressed_tensors_marlin": (
        SlimQuantCompressedTensorsMarlinFacade
    ),
    "slimquant_w4a8": SlimQuantW4A8Facade,
    "kimi_k3_w4a8": KimiK3W4A8Facade,
}


__all__ = [
    "SLIMQUANT_FACADES",
    "SlimQuantCompressedTensorsMarlinFacade",
    "SlimQuantMarlinFacade",
    "SlimQuantW4A8Facade",
    "KimiK3W4A8Facade",
]
