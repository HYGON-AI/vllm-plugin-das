# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


# Register CustomRotaryEmbedding to CustomOP.
@RotaryEmbedding.register_oot
class HcuRotaryEmbedding(RotaryEmbedding):
    """Original rotary positional embedding."""

    def forward_hip(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(*args, **kwargs)
