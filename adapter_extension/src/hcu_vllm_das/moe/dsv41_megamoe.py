# SPDX-License-Identifier: Apache-2.0
"""V4.1 Channel-FP8 MegaMoE via the installed deepgemm.mega API.

The official runner owns routing, shared experts and routed scaling. MegaMoE
owns dispatch, the two GEMMs and combine. Each PCP rank submits only its rows.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import inspect
import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import PassThroughPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
from vllm.platforms import current_platform

from ..dsv41.profile_worker import WORKER, local_token_bound


def validate_config(config, moe):
    p = config.parallel_config
    if (p.tensor_parallel_size != 1 or p.data_parallel_size != 1
            or p.pipeline_parallel_size != 1 or moe.pcp_size != 8
            or moe.ep_size != 8 or not p.enable_expert_parallel):
        raise ValueError("V41 MegaMoE requires TP1/DP1/PP1/PCP8/EP8")
    if not config.model_config.enforce_eager or p.enable_dbo:
        raise ValueError("V41 MegaMoE currently requires eager execution without DBO")
    if config.speculative_config is not None:
        raise ValueError("V41 PCP MegaMoE does not support speculative decoding")
    if p.enable_eplb or moe.num_experts != moe.num_logical_experts:
        raise ValueError("V41 PCP MegaMoE does not support EPLB or redundant experts")
    if not moe.moe_parallel_config.use_deepep_ht_kernels:
        raise ValueError("V41 MegaMoE requires deepep_high_throughput metadata")
    if (moe.in_dtype != torch.bfloat16 or moe.is_sequence_parallel
            or moe.has_bias or moe.is_lora_enabled or not moe.is_act_and_mul
            or moe.activation.value != "silu"):
        raise ValueError("V41 MegaMoE requires BF16 gated SiLU without SP/bias/LoRA")
    if (moe.num_experts, moe.experts_per_token, moe.hidden_dim, moe.intermediate_size_per_partition) != (384, 6, 5120, 2304):
        raise ValueError("V41 MegaMoE requires the vendor-compiled target shape (6, 5120, 2304)")
    if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        raise ValueError("V41 MegaMoE does not support parameterized SwiGLU")
    offload = getattr(config, "offload_config", None)
    if offload is not None and (
            offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size):
        raise ValueError("V41 MegaMoE does not support model weight offloading")
    if getattr(config.model_config, "enable_sleep_mode", False):
        raise ValueError("V41 MegaMoE IPC buffers do not support sleep mode")
    if moe.skip_final_all_reduce:
        raise ValueError("V41 MegaMoE owns its final combine")
    if str(p.expert_placement_strategy).split(".")[-1].lower() != "linear":
        raise ValueError("V41 MegaMoE requires linear physical expert placement")


def provider():
    import deepgemm.mega as mega
    for name, args, kwargs in (
        ("transform_weights_for_mega_moe", (None, None), {}),
        ("get_symm_buffer_for_mega_moe", (None, 256, 64, 6, 4096, 2048),
         dict(use_fp8_dispatch=True, activation="swiglu")),
        ("mega_moe_preprocess", (None, None, None, None, 0), {}),
        ("fp8_mega_moe", (None, None, None, None),
         dict(activation_clamp=float("inf"), fast_math=True)),
    ):
        fn = getattr(mega, name, None)
        if not callable(fn):
            raise RuntimeError(f"Missing deepgemm.mega.{name}")
        inspect.signature(fn).bind(*args, **kwargs)
    return mega



def make_buffer(mega, group, num_experts, capacity, topk, hidden, intermediate):
    # With 384 experts and EP8, rounding local capacity aligns both HT tiles.
    capacity = (capacity + 127) // 128 * 128
    buf = mega.get_symm_buffer_for_mega_moe(
        group, num_experts, capacity, topk, hidden, intermediate,
        use_fp8_dispatch=True, activation="swiglu")
    if buf.l1_acts.shape[0] % 128 or buf.l2_acts.shape[0] % 128:
        buf.destroy()
        raise RuntimeError("MegaMoE native pool capacity or ring is not HT aligned")
    return buf


@contextmanager
def _layer_buffer(ep, mega, device, moe, capacity):
    """Publish a group's shared buffer only after its first layer loads."""
    cache = getattr(ep, "_hcu_dsv41_mega_buffers", None)
    if cache is None:
        cache = {}
        destroy_group = ep.destroy

        def destroy():
            # Core owns the process groups used by the vendor IPC handles.
            # ExitStack also attempts the other releases if one raises.
            with ExitStack() as cleanup:
                cleanup.callback(destroy_group)
                for buf in cache.values():
                    cleanup.callback(buf.destroy)
                cache.clear()

        ep.destroy = destroy
        ep._hcu_dsv41_mega_buffers = cache

    shape = (moe.num_experts, capacity, moe.experts_per_token,
             moe.hidden_dim, moe.intermediate_size_per_partition)
    key = (device, *shape)
    if key in cache:
        yield cache[key]
        return

    buf = make_buffer(mega, ep.device_group, *shape)
    try:
        yield buf
        cache[key] = buf
    except BaseException:
        buf.destroy()
        raise


_PREFLIGHT_KEYS = set()


def preflight_staging(mega, device):
    """Reject non-executing vendor binaries before packing model weights.

    Host checks are confined to this startup probe; the forward path has no
    device-to-host reads. A compatible Python signature is not execution proof.
    """
    import os
    from types import SimpleNamespace
    key = (os.getpid(), str(device), mega.mega_moe_preprocess)
    if key in _PREFLIGHT_KEYS:
        return
    rows, hidden, topk = 2, 5120, 6
    x = torch.ones((rows, hidden), dtype=torch.bfloat16, device=device)
    # BF16 values on FP8 midpoints distinguish rounded division from the
    # provider's FP32 multiply-by-reciprocal scale (3.0625 -> 0x3be00001).
    x[:, 0] = 3.0625
    x[:, 1] = -0.2119140625
    x[:, 2] = -0.314453125
    x[:, 3] = -0.62890625
    ids = torch.arange(rows * topk, device=device, dtype=torch.int64).view(rows, topk)
    weights = torch.full((rows, topk), 0.25, dtype=torch.float32, device=device)
    buf = SimpleNamespace(
        x=torch.zeros_like(x, dtype=torch.float8_e4m3fn),
        x_sf=torch.zeros(rows, dtype=torch.float32, device=device),
        topk_idx=torch.full_like(ids, -1),
        topk_weights=torch.zeros_like(weights),
    )
    from contextlib import nullcontext
    guard = torch.cuda.device(device) if torch.device(device).type == "cuda" else nullcontext()
    with guard:
        mega.mega_moe_preprocess(x, ids, weights, buf, rows)
    expected_scale = x.float().abs().amax(-1) * torch.tensor(
        1.0 / 448.0, dtype=torch.float32, device=device)
    expected_q = (x.float() / expected_scale[:, None]).to(torch.float8_e4m3fn)
    if not (torch.equal(buf.topk_idx, ids)
            and torch.equal(buf.topk_weights, weights)
            and torch.equal(buf.x.view(torch.uint8), expected_q.view(torch.uint8))
            and torch.equal(buf.x_sf, expected_scale)):
        raise RuntimeError(
            "Installed deepgemm.mega preprocess failed its startup execution probe "
            "(routing or FP8 rounding/scale contract mismatch). A valid Python ABI "
            "does not prove a usable gfx938 kernel image. Check the vendor binary "
            "and HIP runtime; restart after installing a compatible provider."
        )
    _PREFLIGHT_KEYS.add(key)


class MegaPrepare(PassThroughPrepareAndFinalize):
    def topk_indices_dtype(self):
        return torch.int64


class MegaExperts(mk.FusedMoEExpertsModular):
    def __init__(self, moe, quant, buffer, l1, l2, mega):
        mk.FusedMoEExpertsModular.__init__(self, moe, quant)
        self.buffer, self.l1, self.l2, self.mega = buffer, l1, l2, mega

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        return current_platform.is_rocm() and current_platform.device_name == "hcu"

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (kFp8StaticChannelSym, kFp8DynamicTokenSym)

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(parallel):
        return parallel.use_deepep_ht_kernels and not parallel.use_deepep_ll_kernels

    def supports_expert_map(self):
        return True

    @property
    def expects_unquantized_inputs(self):
        return True

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return (self.moe_config.num_local_experts, a1.shape[0],
                2 * self.moe_config.intermediate_size_per_partition,
                self.moe_config.hidden_dim, topk_ids.shape[1])

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    def run(self, output, x, ids, weights):
        if (x.ndim != 2 or x.dtype != torch.bfloat16
                or x.shape[1] != self.moe_config.hidden_dim
                or ids.shape != weights.shape
                or ids.shape != (x.shape[0], self.moe_config.experts_per_token)):
            raise ValueError("MegaMoE activation/routing shape or dtype mismatch")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V41 MegaMoE eager path cannot be captured")
        if x.shape[0] > self.buffer.num_max_tokens_per_rank:
            raise ValueError("MegaMoE per-rank token capacity exceeded")
        # No early return on an empty rank: peers still need its IPC barriers.
        if x.shape[0]:
            self.mega.mega_moe_preprocess(
                x.contiguous(), ids.to(torch.int64).contiguous(),
                weights.float().contiguous(), self.buffer, x.shape[0])
        self.mega.fp8_mega_moe(
            output, self.l1, self.l2, self.buffer,
            activation_clamp=(self.quant_config.gemm1_clamp_limit
                              if self.quant_config.gemm1_clamp_limit is not None
                              else float("inf")), fast_math=True)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        self.run(output, hidden_states, topk_ids, topk_weights)


class MegaKernel(mk.FusedMoEKernel):
    def apply(self, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, apply_router_weight_on_input,
              shared_experts=None, shared_experts_input=None):
        # The stock modular executor skips experts when M=0, which would
        # deadlock an IPC collective. All row counts use this same launch path.
        if apply_router_weight_on_input:
            raise ValueError("MegaMoE applies routing weights after GEMM2")
        x, *_ = self.prepare_finalize.prepare(
            hidden_states, topk_weights, topk_ids, global_num_experts,
            expert_map, apply_router_weight_on_input,
            self.fused_experts.quant_config, True)
        output = torch.empty_like(x)
        self.fused_experts.run(output, x, topk_ids, topk_weights)
        return output


class HcuDSV41MegaMoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    def __init__(self, weight_quant, input_quant, moe, prefix):
        FusedMoEMethodBase.__init__(self, moe)
        config = get_current_vllm_config()
        validate_config(config, moe)
        # Stock and custom workers can profile with a full-size dummy batch.
        self.buffer_capacity = moe.max_num_tokens
        if config.parallel_config.worker_cls == WORKER:
            self.buffer_capacity = local_token_bound(
                config.scheduler_config.max_num_batched_tokens,
                config.scheduler_config.max_num_seqs,
                moe.pcp_size,
            )
        self.weight_quant, self.input_quant = weight_quant, input_quant
        self.weight_block_size = None
        self.block_quant = self.static_input_scales = False
        self._prefix = prefix
        self._processed = False
        self._failure = None
        self.experts_cls = MegaExperts

    @property
    def supports_eplb(self):
        return False

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                            act_dtype, moe_parallel_config):
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        return FusedMoEQuantConfig.make(
            torch.float8_e4m3fn, w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale, per_act_token_quant=True,
            per_out_ch_quant=True, gemm1_clamp_limit=layer.swiglu_limit)

    def process_weights_after_loading(self, layer):
        if self._failure is not None:
            raise RuntimeError(self._failure)
        try:
            return self._process_weights(layer)
        except BaseException:
            if self._failure is None:
                self._failure = "MegaMoE post-load failed; restart process"
            raise

    @torch.inference_mode()
    def _process_weights(self, layer):
        if self._processed:
            self._failure = "MegaMoE packed weights cannot be postprocessed twice; restart for reload"
            raise RuntimeError(self._failure)
        mega = provider()
        preflight_staging(mega, layer.w13_weight.device)
        if layer.apply_router_weight_on_input or layer.expert_map_manager.num_fused_shared_experts:
            raise ValueError("MegaMoE requires separate shared experts and output routing weights")
        w1, w2 = layer.w13_weight, layer.w2_weight
        s1, s2 = layer.w13_weight_scale, layer.w2_weight_scale
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise ValueError("MegaMoE requires E4M3FN checkpoint weights")
        l1, l2 = mega.transform_weights_for_mega_moe(
            (w1, s1.reshape(w1.shape[:2]).float().contiguous()),
            (w2, s2.reshape(w2.shape[:2]).float().contiguous()))
        ep = get_ep_group()
        if ep.world_size != self.moe.ep_size:
            raise ValueError("MegaMoE EP group differs from expert placement")
        # Keep exactly one copy of expert weights. The native loader's shapes
        # and Parameter objects remain visible for model memory accounting.
        pairs = ((w1, l1[0]), (w2, l2[0]), (s1, l1[1]), (s2, l2[1]))
        for param, packed in pairs:
            if param.numel() != packed.numel() or param.dtype != packed.dtype:
                raise ValueError("MegaMoE packing changed weight element count/dtype")
        for param, packed in pairs:
            param.data = packed.reshape(param.shape)
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        with _layer_buffer(ep, mega, w1.device, self.moe, self.buffer_capacity) as buffer:
            self.moe_kernel = MegaKernel(
                MegaPrepare(), MegaExperts(self.moe, quant, buffer, l1, l2, mega))
            self._processed = True
            # Once per process: every layer shares these packed shapes and buffer.
            if ep.rank_in_group == 0:
                from vllm.logger import init_logger
                init_logger(f"vllm.{__name__}").info_once(
                    "HCU V41 MegaMoE ready: provider=%s physical=%d "
                    "local=%d topk=%d capacity=%d",
                    mega.__file__, self.moe.num_experts,
                    self.moe.num_local_experts, self.moe.experts_per_token,
                    buffer.num_max_tokens_per_rank)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if self._failure is not None:
            raise RuntimeError(self._failure)
        return super().apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
