# SPDX-License-Identifier: Apache-2.0
"""Packed CT INT4 weights and token-INT8 activations for DSV4.1 MegaMoE."""
import torch

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_int8 import (
    CompressedTensorsW4A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import _per_token_quant_int8
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static, kInt8DynamicTokenSym
from vllm.model_executor.utils import set_weight_attrs

from ..dsv41.profile_worker import local_token_bound
from ..quantization.ct_w4a8_weights import PackedCTParameters
from .dsv41_megamoe import (
    DSV41MegaMoEMethodMixin, MegaExperts, MegaKernel, MegaPrepare, _layer_buffer,
)


def w4a8_provider():
    import inspect
    import deepgemm.mega as mega

    native = getattr(mega._C, "w4a8_mega_moe", None)
    if not callable(native):
        raise RuntimeError("DeepGEMM must provide the native W4A8 MegaMoE entry")
    # Persistent, pre-divided scales avoid the vendor's legacy temporary-scale
    # path, which synchronizes the host before releasing those temporaries.
    if any(name not in (native.__doc__ or "") for name in
           ("l1_weights_sf_div16", "l2_weights_sf_div16")):
        raise RuntimeError("DeepGEMM W4A8 requires persistent pre-divided scales")
    if "per_token_intermediate" not in (native.__doc__ or ""):
        raise RuntimeError("DeepGEMM must support CT per-token intermediate quantization")
    for name, args in (
        ("transform_weights_for_mega_moe", (None, None)),
        ("mega_moe_preprocess", (None, None, None, None, 0)),
    ):
        inspect.signature(getattr(mega, name)).bind(*args, mode="w4a8")
    return mega


def unpack_ct_words(words):
    """CT packs the earliest unsigned offset-binary nibble in the low bits."""
    if words.dtype != torch.int32:
        raise ValueError("Expected INT32 compressed-tensors weight_packed")
    pairs = words.contiguous().view(torch.uint8)
    return (torch.stack((pairs & 15, pairs >> 4), dim=-1).flatten(-2) ^ 8)


@triton.jit
def _stage_ct_w4a8(x, q, sf, ids, weights, out_ids, out_weights,
                   x_stride, id_row, id_col, weight_row, weight_col,
                   hidden: tl.constexpr, topk: tl.constexpr,
                   block: tl.constexpr, route_block: tl.constexpr):
    # Inline Core's exact token quantizer and write directly into the IPC view.
    _per_token_quant_int8(x, q, sf, x_stride, hidden, hidden, BLOCK=block)
    row = tl.program_id(0)
    k = tl.arange(0, route_block)
    expert = tl.load(ids + row * id_row + k * id_col, mask=k < topk, other=-1)
    route = tl.load(weights + row * weight_row + k * weight_col, mask=k < topk, other=0)
    tl.store(out_ids + row * topk + k, expert, mask=k < topk)
    tl.store(out_weights + row * topk + k, route, mask=k < topk)


def stage_ct_w4a8(buf, x, ids, weights):
    if x.stride(-1) != 1:
        x = x.contiguous()
    rows, hidden = x.shape
    topk = ids.shape[1]
    block = triton.next_power_of_2(hidden)
    _stage_ct_w4a8[(rows,)](
        x, buf.x, buf.x_sf, ids, weights, buf.topk_idx, buf.topk_weights,
        x.stride(0), *ids.stride(), *weights.stride(), hidden, topk,
        block, triton.next_power_of_2(topk),
        num_warps=min(max(block // 256, 1), 8), num_stages=1)


class W4A8MegaExperts(MegaExperts):
    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (kInt4Static, kInt8DynamicTokenSym)

    def run(self, output, x, ids, weights):
        # Core pads PCP inputs to the largest rank. Profile/replay may carry a
        # full scheduler batch, while ordinary prefills fit in one local tile.
        # Use the common descriptor even on ranks with fewer (or zero) rows.
        extent = x.shape[0]
        if is_forward_context_available():
            desc = get_forward_context().batch_descriptor
            if desc is not None:
                extent = desc.num_tokens
        if extent < x.shape[0]:
            raise ValueError("PCP batch descriptor is smaller than the W4A8 input")
        cap = self.buffer.num_max_tokens_per_rank
        if extent <= cap:
            return super().run(output, x, ids, weights)
        if not is_forward_context_available() or desc is None:
            raise ValueError("Oversize PCP batches require a common batch descriptor")
        for start in range(0, max(1, extent), cap):
            tile = slice(start, start + cap)
            super().run(output[tile], x[tile], ids[tile], weights[tile])

    def stage(self, x, ids, weights):
        stage_ct_w4a8(self.buffer, x, ids, weights)

    def launch(self, output):
        # Use the vendor's existing native ABI; no private Python wrapper
        # extensions or per-forward scale-cache allocations are required.
        buf = self.buffer
        native = self.mega._C.w4a8_mega_moe
        args = (
            output, self.l1[0], self.l2[0], self.l1[1], self.l2[1],
            buf.buffer, buf.local_buffer, buf.handle.buffer_ptrs,
            buf.handle.signal_ptrs, buf.group.rank(),
            buf.num_max_tokens_per_rank, buf.num_experts, buf.num_topk,
            self.quant_config.gemm1_clamp_limit, True, self.l1[1], self.l2[1],
        )
        native(*args, per_token_intermediate=True)


class HcuDSV41W4A8MegaMoEMethod(DSV41MegaMoEMethodMixin, CompressedTensorsW4A8Int8MoEMethod):
    supports_static_eplb = True
    supports_swa_bounded_replay = True
    supports_dspark = True

    @staticmethod
    def _buffer_capacity(config, moe):
        sched = config.scheduler_config
        return min(moe.max_num_tokens, local_token_bound(
            sched.max_num_batched_tokens, sched.max_num_seqs, moe.pcp_size))

    def __init__(self, weight_quant, input_quant, moe, prefix):
        super().__init__(weight_quant, input_quant, moe, prefix)
        self.experts_cls = W4A8MegaExperts
        self.mega = w4a8_provider()

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        self._packed = PackedCTParameters(layer, num_experts, hidden_size,
                                          intermediate_size_per_partition,
                                          extra_weight_attrs.get("weight_loader"))
        extra = {k: v for k, v in extra_weight_attrs.items()
                 if k not in ("weight_loader", "is_transposed", "quant_method")}
        for param in layer.parameters(recurse=False):
            set_weight_attrs(param, extra)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8, weight_dtype="int4", per_act_token_quant=True, per_out_ch_quant=True,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=layer.swiglu_limit)


    @torch.inference_mode()
    def _process_weights(self, layer):
        if self._processed:
            raise RuntimeError("MegaMoE packed weights cannot be postprocessed twice")
        if (layer.apply_router_weight_on_input or layer.expert_map_manager.num_fused_shared_experts
                or layer.swiglu_limit != 10.0):
            raise ValueError("DSV4.1 W4A8 requires separate shared experts, output routing and clamp=10")
        ep = get_ep_group()
        if ep.world_size != self.moe.ep_size:
            raise ValueError("MegaMoE EP group differs from expert placement")
        h, i = self.moe.hidden_dim, self.moe.intermediate_size_per_partition
        for meta, shape in ((layer.w13_weight_shape, (i, h)), (layer.w2_weight_shape, (h, i))):
            if not torch.equal(meta, meta.new_tensor(shape).expand_as(meta)):
                raise ValueError("Unexpected DSV4.1 checkpoint expert weight_shape")
        self._packed.state = "packing"
        w1, w2 = layer.w13_weight_packed, layer.w2_weight_packed
        # Reuse checkpoint storage, expanding just one local expert at a time.
        p1, p2 = w1.data.view(torch.int8), w2.data.view(torch.int8)
        s1 = layer.w13_weight_scale.data.squeeze(-1)
        s2 = layer.w2_weight_scale.data.squeeze(-1)
        for idx in range(w1.shape[0]):
            row = slice(idx, idx + 1)
            l1, l2 = self.mega.transform_weights_for_mega_moe(
                (unpack_ct_words(w1[row]), s1[row]),
                (unpack_ct_words(w2[row]), s2[row]), mode="w4a8")
            p1[row].copy_(l1[0])
            p2[row].copy_(l2[0])
            s1[row].copy_(l1[1])
            s2[row].copy_(l2[1])
        s1.div_(16)
        s2.div_(16)
        # Keep the registered parameters used by Core's native CT W4A8 apply.
        w1.data, w2.data = p1, p2
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        with _layer_buffer(ep, self.mega, w1.device, self.moe, self.buffer_capacity, "w4a8") as buf:
            experts = W4A8MegaExperts(self.moe, quant, buf, (p1, s1), (p2, s2), self.mega)
            self.moe_kernel = MegaKernel(MegaPrepare(), experts)
            self._packed.state = "ready"
            self._processed = True
        if ep.rank_in_group == 0:
            init_logger(f"vllm.{__name__}").info_once(
                "HCU DSV4.1 W4A8 MegaMoE ready: provider=%s local_experts=%d capacity=%d",
                self.mega.__file__, w1.shape[0], buf.num_max_tokens_per_rank)
