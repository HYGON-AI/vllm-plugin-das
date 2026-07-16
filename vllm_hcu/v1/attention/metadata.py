# SPDX-License-Identifier: Apache-2.0
"""HCU-owned metadata types for Lightly-CP."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CpCommonAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    _seq_lens_cpu: torch.Tensor
    num_actual_tokens: int
    num_kv_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    num_reqs: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    _num_computed_tokens_cpu: torch.Tensor
    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None

    def batch_size(self) -> int:
        return self.seq_lens.shape[0]

    @property
    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu


__all__ = ["CpCommonAttentionMetadata"]
