# SPDX-License-Identifier: Apache-2.0
"""Token layouts around the unchanged official GLM4/HY3 model loops."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context, is_forward_context_available

_BATCH_KEY = "hcu_runtime_sp_batch"


@dataclass
class RuntimeSPBatch:
    size: int
    rank: int
    num_tokens: int
    global_padding: torch.Tensor
    local_padding: torch.Tensor

    @classmethod
    def create(cls, size, rank, num_tokens, padding, device):
        if type(size) is not int or size <= 0 or not 0 <= rank < size:
            raise ValueError("runtime SP requires valid TP geometry")
        aligned = ((num_tokens + size - 1) // size) * size
        if padding is None:
            padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if (padding.dtype != torch.bool or padding.ndim != 1
                or padding.shape[0] < num_tokens or padding.device != device):
            raise ValueError("runtime SP requires a matching GPU/CPU token padding mask")
        # Keep mask values on device: graph replay may change the number of real
        # tokens without changing any addresses or the selected padded shape.
        mask = F.pad(padding[:num_tokens], (0, aligned - num_tokens), value=True)
        rows = aligned // size
        return cls(size, rank, num_tokens, mask, mask[rank * rows:(rank + 1) * rows])

    @property
    def padded_tokens(self):
        return self.global_padding.shape[0]

    def shard(self, tensor):
        if tensor.ndim != 2 or tensor.shape[0] != self.num_tokens:
            raise ValueError("runtime SP boundary requires the full token tensor")
        padded = F.pad(tensor, (0, 0, 0, self.padded_tokens - self.num_tokens))
        rows = self.padded_tokens // self.size
        local = padded[self.rank * rows:(self.rank + 1) * rows]
        return local.masked_fill(self.local_padding[:, None], 0)

    def positions(self, positions):
        if positions.ndim != 1 or positions.shape[0] != self.num_tokens:
            raise ValueError("runtime SP GLM4/HY3 requires full one-dimensional positions")
        return F.pad(positions, (0, self.padded_tokens - self.num_tokens))

    def gather(self, tensor, group):
        if tensor.ndim != 2 or tensor.shape[0] * self.size != self.padded_tokens:
            raise ValueError("runtime SP model returned an invalid local token layout")
        tensor = tensor.masked_fill(self.local_padding[:, None], 0)
        full = group.all_gather(tensor, dim=0) if self.size > 1 else tensor
        return full[:self.num_tokens]


def current_sp_batch():
    if not is_forward_context_available():
        return None
    return get_forward_context().additional_kwargs.get(_BATCH_KEY)


def mask_local_padding(tensor):
    batch = current_sp_batch()
    if batch is None:
        return tensor
    if tensor.shape[0] != batch.local_padding.shape[0]:
        raise ValueError("runtime SP projection received a nonlocal token layout")
    return tensor.masked_fill(batch.local_padding[:, None], 0)
