# SPDX-License-Identifier: Apache-2.0
"""Use channel INT8 projections in the owned DSV4.1 CT W4A8 models."""

from types import MethodType

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

logger = init_logger(f"vllm.{__name__}")


def _is_dynamic_channel_int8(scheme):
    return (
        isinstance(scheme, CompressedTensorsW8A8Int8)
        and scheme.strategy == "channel"
        and not scheme.is_static_input_scheme
        and scheme.input_symmetric
    )


def bind_vocab_head(head, quant_config, prefix):
    # Core constructs these two DSV4.1 vocabulary projections without CT.
    # Ask CT to select and allocate the quantized method for the actual prefix.
    if quant_config.get_scheme_dict(head, prefix) is None:
        return False
    method = quant_config.get_quant_method(head, prefix)
    if not _is_dynamic_channel_int8(getattr(head, "scheme", None)):
        raise ValueError(f"Unsupported DSV4.1 CT vocabulary projection: {prefix}")
    if head.tp_size != 1 or head.num_embeddings_per_partition != head.org_vocab_size:
        raise ValueError("DSV4.1 INT8 vocabulary projection requires an unpadded TP1 head")
    with torch.device(head.weight.device):
        method.create_weights(
            head,
            head.input_size_per_partition,
            head.output_partition_sizes,
            head.input_size,
            head.output_size,
            params_dtype=head.params_dtype,
            weight_loader=head.weight_loader,
        )
    head.quant_method = method
    head.update_param_tp_status()
    logger.info_once("DSV4.1 channel INT8 vocabulary projection: %s", prefix)
    return True


def _project_wo_a(self, attn_out, positions):
    # On HCU the attention backend has already run inverse RoPE. Each WO-A
    # group has its own K inputs, so quantize each (token, group) independently.
    groups, rank = self.n_local_groups, self.o_lora_rank
    rows = attn_out.shape[0]
    if rows == 0:
        return self._wo_b_after_wo_a(attn_out.new_empty((0, groups * rank)))
    width = self.n_local_heads * self.head_dim // groups
    x = attn_out[:, :self.n_local_heads].reshape(rows, groups, width)
    q, scale, _ = ops.scaled_int8_quant(
        x.transpose(0, 1).reshape(groups * rows, width).contiguous(),
        symmetric=True,
    )
    q = q.view(groups, rows, width)
    scale = scale.view(groups, rows, 1)
    # CT finalizes WO-A as [K, G*R], with one scale per output channel.
    weight = self.wo_a.weight.split(rank, dim=1)
    weight_scale = self.wo_a.weight_scale.reshape(-1, 1).split(rank, dim=0)
    out = [
        torch.ops.vllm.int8_scaled_mm_func(
            q[g], weight[g], scale[g], weight_scale[g], attn_out.dtype, None,
        )
        for g in range(groups)
    ]
    return self._wo_b_after_wo_a(torch.stack(out, dim=1).flatten(1))


def bind_wo_a(layers):
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

    bound = 0
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if (isinstance(attn, DeepseekV41ROCMAiterMLAAttention)
                and _is_dynamic_channel_int8(getattr(attn.wo_a, "scheme", None))):
            attn._o_proj = MethodType(_project_wo_a, attn)
            bound += 1
    if bound:
        logger.info_once("DSV4.1 grouped channel INT8 WO-A: %d layers", bound)
