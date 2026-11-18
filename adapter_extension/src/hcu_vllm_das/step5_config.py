# SPDX-License-Identifier: Apache-2.0
"""Standalone draft metadata; target configuration remains owned by vLLM."""

from typing import Any

from vllm.transformers_utils.configs import Step5Config


class Step5MTPConfig(Step5Config):
    """Parse a separately saved draft without changing the target model type."""

    model_type = "step5_mtp"
    architectures = ["Step5MTP"]

    def __init__(self, num_nextn_predict_layers: int = 1, **kwargs: Any) -> None:
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(type(self).architectures)
        super().__init__(num_nextn_predict_layers=num_nextn_predict_layers, **kwargs)
