# SPDX-License-Identifier: Apache-2.0
"""Fix GDN fused-conv1d TP weight loading for the packed (k,1,out) param.

Root cause (nmz single-node TP8 Kimi-K3 smoke, turn C): vLLM core
``kimi_gdn_linear_attn._make_fused_conv1d_weight_loader`` copies checkpoint
shards as ``param.data[target:target+shard].copy_(loaded_shard)`` assuming
``param.data`` rows are (shard, 1, kernel).  Under plain TP8 the
``ColumnParallelLinear`` conv weight arrives transposed as ``(kernel, 1,
shard)`` (probe evidence: ``param=(4,1,4608) loaded=(12288,1,4)``), so the
copy broadcasts and dies with ``a (4608) vs b (4) at dim 2``.  The
DP2xTP8+EP16 baseline never hits this because its packing keeps the
(row,1,kernel) orientation.

Adapter policy: plugin-side only, never touching core code.  When the
orientation differs, transpose the *loaded shard* into the param layout and
write it directly; revert silently to the stock copy when shapes already
match.  Logs one line per fixed call so the behaviour stays visible.
"""
from __future__ import annotations

import functools
from types import ModuleType

PATCH_ID = "platform.core.gdn_conv_tp_packing"
TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"
TARGETS = (f"{TARGET_MODULE}._make_fused_conv1d_weight_loader",)

_MARKER = "_k3_gdn_conv_tp_packing_armed"


def apply_to_module(module: ModuleType) -> None:
    import logging

    log = logging.getLogger("vllm_hcu.patch.gdn_conv_tp_packing")
    factory = getattr(module, "_make_fused_conv1d_weight_loader")
    if getattr(module, _MARKER, False):
        return

    @functools.wraps(factory)
    def patched(dims, tp_size, tp_rank):
        loader = factory(dims, tp_size, tp_rank)

        @functools.wraps(loader)
        def wrapper(param, loaded_weight, loaded_shard_id=None, **kw):
            # Probe evidence (single-node TP8): param=(4,1,4608)
            # [kernel, 1, 3*shard] vs loaded=(12288,1,4) [full_out, 1, kernel].
            # Stock loader assumes param rows are (shard,1,kernel); bail out
            # to it whenever shapes already fit that layout.
            transposed = (
                loaded_shard_id is not None
                and param.data.dim() == 3
                and loaded_weight.dim() == 3
                and param.data.shape[0] == loaded_weight.shape[2]
                and param.data.shape[0] != param.data.shape[2]
                and int(param.data.shape[2]) % int(tp_size) == 0
            )
            if not transposed:
                return loader(param, loaded_weight, loaded_shard_id, **kw)

            num_shards = len(dims)
            packed_out = int(param.data.shape[2])
            if packed_out != num_shards * (packed_out // num_shards):
                return loader(param, loaded_weight, loaded_shard_id, **kw)
            shard_size = packed_out // num_shards
            source_start = int(tp_rank) * shard_size
            shard = loaded_weight.detach()[
                source_start : source_start + shard_size
            ]  # (shard, 1, kernel)
            target_start = int(loaded_shard_id) * shard_size
            view = param.data[
                :, :, target_start : target_start + shard_size
            ]  # (kernel, 1, shard)
            view.copy_(shard.permute(2, 1, 0))
            log.info(
                "[gdn-conv-fix] transposed load shard_id=%s shard=%d kernel=%d packed_out=%d",
                loaded_shard_id,
                shard_size,
                int(param.data.shape[0]),
                packed_out,
            )
            return None

        return wrapper

    module._make_fused_conv1d_weight_loader = patched
    setattr(module, _MARKER, True)
    log.info("[gdn-conv-fix] armed on %s", TARGET_MODULE)


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply_to_module"]
