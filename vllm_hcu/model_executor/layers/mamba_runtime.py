# SPDX-License-Identifier: Apache-2.0
"""HCU-owned Mamba weight-layout helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch


def mamba_v2_nn_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Create the v0.21 sharded loader for HCU's output-last NN layout."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        boundary = loaded_boundary = 0
        loaded_total_dim = sum(full_dim - extra for full_dim, extra, _ in shard_spec)
        param_out_axis = 0 if param.dim() == 1 else param.dim() - 1
        loaded_out_axis = 0
        if (
            loaded_weight.dim() > 1
            and loaded_weight.shape[-1] == loaded_total_dim
            and loaded_weight.shape[0] != loaded_total_dim
        ):
            loaded_out_axis = loaded_weight.dim() - 1

        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            loaded_start_idx = loaded_boundary + rank * shard_size
            take = min(shard_size, full_dim - extra - rank * shard_size)
            if take > 0:
                param_slice = param.data.narrow(param_out_axis, boundary, take)
                loaded_slice = loaded_weight.narrow(
                    loaded_out_axis, loaded_start_idx, take
                )
                if (
                    param_slice.dim() == loaded_slice.dim() + 1
                    and param_slice.shape[1] == 1
                ):
                    loaded_slice = loaded_slice.unsqueeze(1)
                elif (
                    loaded_slice.dim() == param_slice.dim() + 1
                    and loaded_slice.shape[1] == 1
                ):
                    loaded_slice = loaded_slice.squeeze(1)
                if param_slice.shape != loaded_slice.shape:
                    loaded_slice = loaded_slice.permute(
                        *reversed(range(loaded_slice.dim()))
                    )
                if param_slice.shape != loaded_slice.shape:
                    raise RuntimeError(
                        "mamba_v2_sharded_weight_loader shape mismatch: "
                        f"param_slice={tuple(param_slice.shape)} "
                        f"loaded_slice={tuple(loaded_slice.shape)} "
                        f"(param_out_axis={param_out_axis}, "
                        f"loaded_out_axis={loaded_out_axis})"
                    )
                param_slice.copy_(loaded_slice)
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


__all__ = ["mamba_v2_nn_sharded_weight_loader"]
