# SPDX-License-Identifier: Apache-2.0
"""HY4 sink and valid-length semantics for the flat BF16 FlashMLA API."""

import torch


def resident_d576_with_sink(q, kv, indices, scale, sink, lengths):
    from flash_mla import flash_mla_sparse_fwd

    if lengths is not None:
        columns = torch.arange(indices.shape[-1], device=q.device, dtype=torch.int32)
        indices = indices.masked_fill(columns >= lengths.view(-1, 1, 1), -1)
    output, maximum, lse = flash_mla_sparse_fwd(q, kv, indices, scale, 512)
    # MLS marks an empty row with max=-inf and lse=+inf. Its real log mass
    # is -inf before the virtual sink is added.
    empty = torch.isneginf(maximum)
    lse = lse.masked_fill(empty, -torch.inf)
    if sink is not None:
        weight = torch.sigmoid(lse - sink).masked_fill(empty, 0.0)
        output.mul_(weight.unsqueeze(-1))
        lse = torch.logaddexp(lse, sink)
    return output, lse
