# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step-5 model.

Step-5 shares Step-3.5's dense backbone (zero-centered RMSNorm, clamped SwiGLU,
sigmoid router with a selection-only bias, per-head attention gate, per-layer
rope_theta, sliding/full layer mix), so this file follows ``step3p5.py`` closely.
The three deltas that matter:

* **The residual stream is fp32.** At 92 layers the difference is visible, so the
  decoder accumulates in fp32 and narrows to the compute dtype only on the way
  into each GEMM. The final narrow happens before ``model.norm``, matching the
  reference implementation.
* **``partial_rotary_factors`` is 1/3 on full-attention layers**, so the rotary
  dim comes from ``get_rope`` rather than a ``head_dim // 2`` special case.
* **Only the MoE expert weights are fp8.** Every other weight in the checkpoint
  is bf16, so ``quant_config`` is handed to ``FusedMoE`` alone. (``FP8_DENSE``
  and ``FP8_BF16_MOE`` quantize more of them at load time.)

Two fp8 checkpoint layouts load through this file unchanged:

===============  ===========================  ==============================
                 block (the released weights) channel (converted for DCU)
===============  ===========================  ==============================
quant_method     ``fp8``                      ``compressed-tensors``
weight scale     ``weight_scale_inv`` (M/128, ``weight_scale`` (M, 1)
                 K/128) per expert            per expert
activations      dynamic per 128-block        dynamic per token
bf16 MoE layers  ``modules_to_not_convert``   ``ignore`` (regex entries)
===============  ===========================  ==============================

Nothing here branches on which one is in use: ``vllm_config.quant_config`` is
passed straight through, and ``expert_params_mapping`` routes both scale names
because it matches by substring and rewrites by *prefix* replacement -- the
``.moe.gate_proj.weight`` entry carries ``...weight_scale_inv`` and
``...weight_scale`` alike onto ``w13_weight_scale{_inv}``. The three explicit
``_scale_inv`` entries in that list are therefore unreachable; they are kept
only because removing them is a behaviour-neutral edit to a load path that is
expensive to retest.

DSA (top-k sparse attention on the 23 ``full_attention`` layers) is wired up:
those layers construct a ``Step4Indexer`` and route their ``Attention`` to
``Step4SparseAttentionBackend``; the 69 sliding layers are untouched and keep
whatever backend the platform picks. Setting ``VLLM_STEP5_DISABLE_DSA=1`` falls
back to the earlier dense-equivalent path, which skips the indexer weights at
load time and counts them. That fallback is exact only up to ``topk *
region_block_size == 4096`` tokens, where the selected region set still covers
the whole causal history; past that it is a different (denser) model.
"""

import os
import re
import functools
import typing
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm_hcu.platforms.step5_envs import is_dsa_disabled
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.model_executor.layers.step4_dsa import (
    Step4DSAGeometry,
    Step4DSAState,
    Step4Indexer,
    Step4SparseAttentionBackend,
)
from vllm_hcu.models.step5.kernels import router_bias_func
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)

logger = init_logger("vllm.step5.step5_text")

# Checkpoint tensors that belong to the DSA indexer. Only consulted when DSA is
# switched off: the dense-equivalent path does not allocate them, and they are
# counted at load time so a skip stays distinguishable from a successful load.
_DSA_INDEXER_SUBSTRINGS = (
    "sparse_indexer_q",
    "sparse_indexer_k",
    "sparse_indexer_w",
    "sparse_indexer_z",
    "sparse_indexer_q_norm",
    "sparse_indexer_k_norm",
    "ssmax_s",
)

# Escape hatch back to the pre-DSA dense path, for A/B against this backend.
DISABLE_DSA = is_dsa_disabled()

# Run qkv_proj and g_proj as one GEMM. Both weights become column views of one
# [hidden, qkv + heads] buffer, so the loaders fill it in place and no
# post-load step is needed (AutoWeightsLoader may call load_weights twice).
# Only valid for the HCU [in, out] weight layout (VLLM_USE_NN).
FUSE_QKVG = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FUSE_QKVG", "1") == "1"
)

# Keep the router weight in its checkpoint dtype (bf16) and compute fp32 logits
# with a bf16 MFMA GEMM instead of upcasting both operands to fp32. Exact up to
# accumulation order, since bf16 x bf16 products are exact in fp32.
BF16_ROUTER = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_BF16_ROUTER", "1") == "1"
)

# With BF16_ROUTER and the shared expert folded into FusedMoE: at decode sizes run
# the router GEMM, the shared-expert gate_up GEMM and its SwigluStep as one Triton
# kernel (the router input is also the shared-expert input).
FUSED_ROUTER_SHARED = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FUSED_ROUTER_SHARED", "1") == "1"
)

# After loading, store the bf16 attention (qkv/g/o) and dense-MLP weights as fp8
# with one scale per output channel and run them as fp8 GEMMs with per-token
# activation scales (ops/fp8_dense.py), the scheme of the fp8 MoE experts. The
# shared expert (fused into the router kernel), router, indexer, eh_proj and
# lm_head stay bf16.
FP8_DENSE = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FP8_DENSE", "0") == "1"
)

# Quantize the bf16 routed experts (layers 88-90) to fp8 per output channel at
# load time, the scheme of the other layers' checkpoint experts, so they run the
# same fp8 MoE path instead of the bf16 Triton experts. Channel (compressed-
# tensors) checkpoint only.
FP8_BF16_MOE = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FP8_BF16_MOE", "0") == "1"
)

SWA_FUSED_QK = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_SWA_FUSED_QK", "0") == "1"
)
FUSED_GATE = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_STEP5_FUSED_GATE", "0") == "1"
)

step5_swa_fused = None
if SWA_FUSED_QK or FUSED_GATE:
    # Enabled custom ops must register during model-module initialization,
    # before torch.compile tracing or CUDA/HCU graph capture begins.
    from vllm_hcu.models import step5_swa_fused


def _get_step5_swa_fused():
    """Return the provider selected during startup feature resolution."""

    assert step5_swa_fused is not None
    return step5_swa_fused

_PACKED_EXPERT_WEIGHT = re.compile(r"\.moe\.(gate|up|down)_proj\.weight$")


def _unignore_bf16_experts(quant_config: QuantizationConfig | None) -> None:
    """FP8_BF16_MOE: drop the routed-expert entries from the compressed-tensors
    ignore list, so layers 88-90 build the fp8 MoE method like the others."""
    ignore = getattr(quant_config, "ignore", None)
    if quant_config is None or quant_config.get_name() != "compressed-tensors":
        raise ValueError(
            "VLLM_HCU_STEP5_FP8_BF16_MOE needs the compressed-tensors (channel) "
            "checkpoint"
        )
    kept = [t for t in ignore if "moe\\.experts" not in t]
    if len(kept) != len(ignore):
        logger.info_once("step5: quantizing the bf16 routed experts to fp8")
    quant_config.ignore = kept


def _quantize_bf16_experts(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """FP8_BF16_MOE: packed bf16 expert weights ``[E, out, in]`` -> fp8 plus a
    ``weight_scale`` ``[E, out, 1]`` (absmax per output row), as in the
    checkpoint's fp8 layers. Quantized on the GPU in expert chunks."""
    dev = torch.device("cuda", torch.cuda.current_device())
    for name, w in weights:
        if not (
            w.dtype == torch.bfloat16
            and w.dim() == 3
            and _PACKED_EXPERT_WEIGHT.search(name)
        ):
            yield name, w
            continue
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=dev)
        scale = torch.empty(*w.shape[:2], 1, dtype=torch.float32, device=dev)
        for e0 in range(0, w.shape[0], 16):
            x = w[e0 : e0 + 16].to(dev).float()
            sc = (x.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-12)
            q[e0 : e0 + 16] = (x / sc).clamp(-448.0, 448.0).to(q.dtype)
            scale[e0 : e0 + 16] = sc
            del x
        yield name, q
        yield name + "_scale", scale


def dsa_layer_indices(config: Any) -> list[int]:
    """Loaded layer indices that run DSA, in order.

    Read from ``sparse_config.apply_to_layer_types`` rather than assumed, and
    bounded by ``num_hidden_layers`` -- ``layer_types`` is length 93 because it
    includes the MTP layer at index 92, which is never loaded.
    """
    if DISABLE_DSA:
        return []
    sparse_config = getattr(config, "sparse_config", None)
    if not sparse_config:
        return []
    if isinstance(sparse_config, dict):

        def get(name, default=None):
            return sparse_config.get(name, default)
    else:

        def get(name, default=None):
            return getattr(sparse_config, name, default)

    if not get("enabled", False):
        return []
    wanted = set(get("apply_to_layer_types", ["full_attention"]) or [])
    layer_types = getattr(config, "layer_types", None) or []
    return [
        idx
        for idx in range(min(config.num_hidden_layers, len(layer_types)))
        if layer_types[idx] in wanted
    ]


class FP32ReplicatedLinear(ReplicatedLinear):
    """
    Use FP32 for higher precision.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        assert self.params_dtype == torch.float32
        return super().forward(x.to(torch.float32))


class BF16FP32RouterLinear(ReplicatedLinear):
    """Router with a bf16 weight and fp32 logits (see ``BF16_ROUTER``)."""

    def __init__(self, input_size: int, output_size: int, prefix: str = ""):
        super().__init__(
            input_size,
            output_size,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=prefix,
        )
        from vllm_hcu.ops.step5.router_gemm import (
            init_router_gemm_workspace,
            router_gemm_bf16_fp32,
        )

        self._gemm = router_gemm_bf16_fp32
        if self.weight.is_cuda:
            init_router_gemm_workspace(self.weight.device, output_size)
        self._fused_shared: list[Step5MLP] | None = None
        self._rsg_workspace: tuple[torch.Tensor, torch.Tensor] | None = None

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.dtype != torch.bfloat16:
            raise ValueError(
                f"{self.prefix}: router weight is {loaded_weight.dtype}, not "
                "bf16; set VLLM_HCU_STEP5_BF16_ROUTER=0"
            )
        super().weight_loader(param, loaded_weight)

    def fuse_shared_expert(self, mlp: "Step5MLP") -> None:
        """Compute ``mlp``'s gate_up + SwigluStep together with the logits for
        decode-sized inputs; ``mlp.forward`` on the same input picks it up."""
        from vllm_hcu.ops.step5.router_gemm import (
            make_router_shared_workspace,
            router_shared_swiglustep,
        )

        assert mlp.limit is not None
        # A list, not an attribute: keep the MLP out of this module's children.
        self._fused_shared = [mlp]
        self._rsg = router_shared_swiglustep
        if self.weight.is_cuda:
            self._rsg_workspace = make_router_shared_workspace(
                self.weight.device,
                self.weight.shape[1],
                mlp.gate_up_proj.output_size_per_partition // 2,
            )

    def _forward_fused_shared(
        self, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        from vllm_hcu.ops.step5.router_gemm import RSG_MAX_M

        mlp = self._fused_shared[0]
        w_gu = mlp.gate_up_proj.weight
        if (
            x2.shape[0] > RSG_MAX_M
            or w_gu.dtype != torch.bfloat16
            or w_gu.shape[0] != x2.shape[1]
            or x2.stride(1) != 1
            or w_gu.stride(1) != 1
            or self.weight.stride(1) != 1
        ):
            return None
        logits, act = self._rsg(
            x2,
            self.weight,
            w_gu,
            mlp.limit,
            workspace=self._rsg_workspace,
        )
        return logits, act

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        w = self.weight
        if self._fused_shared is not None:
            # A prior eligible call may have left a cache that must never be
            # consumed by an ineligible dtype or weight layout.
            self._fused_shared[0].fused_act = None
        if x.dtype == torch.bfloat16 and w.shape[0] == x.shape[-1]:
            x2 = x.reshape(-1, x.shape[-1])
            if self._fused_shared is not None:
                # Return the activation explicitly so Dynamo records the
                # cross-module dataflow.
                fused = self._forward_fused_shared(x2)
                if fused is not None:
                    out, shared_act = fused
                    return out.view(*x.shape[:-1], w.shape[1]), shared_act
            out = self._gemm(x2, w)
            return out.view(*x.shape[:-1], w.shape[1]), None
        # Weight in [out, in] layout (VLLM_USE_NN off) or non-bf16 activations.
        w = w.float() if w.shape[0] == x.shape[-1] else w.float().t()
        return x.float() @ w, None


def _fp32_gemma_rms_norm(hidden_size: int, eps: float) -> GemmaRMSNorm:
    """GemmaRMSNorm holding its weight in fp32, on the native forward path.

    Every norm weight in the Step-5 checkpoint is serialized as fp32, and the
    reference implementation keeps it that way. GemmaRMSNorm allocates with the
    default dtype instead, which rounds the weight to bf16; because the norm
    computes ``x * (1 + w)`` and these weights are centred on zero, rounding
    perturbs ``1 + w`` by up to ~8e-3 -- far above bf16 compute noise, on all
    four norms of all 92 layers.

    Device-specific GemmaRMSNorm kernels require the activation and the weight to
    share a dtype, so an fp32 weight also has to opt out of them. forward_native
    reduces in fp32 and returns the input dtype, which is the wanted semantics.
    """
    norm = GemmaRMSNorm(hidden_size, eps)
    norm.weight = nn.Parameter(
        torch.zeros(hidden_size, dtype=torch.float32), requires_grad=False
    )
    norm._forward_method = norm.forward_native
    return norm


class Step5FP8LinearMethod(LinearMethodBase):
    """``FP8_DENSE``: a bf16 linear run as an fp8 GEMM (ops/fp8_dense.py).

    Swapped in after the layer is built, so the layer is created and loaded as an
    unquantized bf16 linear; process_weights_after_loading then replaces the
    weight with fp8 + a per-output-channel scale and frees the bf16 copy. With
    ``post`` the owner quantizes instead (the fused qkvg weight).
    """

    def __init__(self, post: Callable[[], None] | None = None):
        self.post = post

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError("swapped in after the layer is created")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self.post is not None:
            self.post()
            return
        from vllm_hcu.ops.step5.fp8_dense import quantize_weight_fp8

        w = layer.weight
        k, n = layer.input_size_per_partition, layer.output_size_per_partition
        if not _weight_is_kn(w, k, n):
            w = w.t()
        layer.fp8_wq, layer.fp8_ws = quantize_weight_fp8(w)
        layer.fp8_n = n
        _free_weight(layer.weight)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm_hcu.ops.step5.fp8_dense import fp8_linear

        if isinstance(x, tuple):
            # Pre-quantized (xq, xs) from step5_swa_fused.
            xq, xs = x
            out = torch.ops.vllm.step5_fp8_gemm(xq, xs, layer.fp8_wq, layer.fp8_ws)
            if layer.fp8_n != out.shape[-1]:
                out = out[:, : layer.fp8_n]
            return out if bias is None else out + bias
        out = fp8_linear(x, layer.fp8_wq, layer.fp8_ws, layer.fp8_n)
        return out if bias is None else out + bias


def _weight_is_kn(w: torch.Tensor, k: int, n: int) -> bool:
    """Whether a bf16 linear weight is stored ``[in, out]`` (HCU VLLM_USE_NN)."""
    if k != n:
        return tuple(w.shape) == (k, n)
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_USE_NN)


def _free_weight(p: Parameter) -> None:
    p.data = torch.empty(0, dtype=p.dtype, device=p.device)


def _use_fp8(*layers: nn.Module) -> None:
    for layer in layers:
        layer.quant_method = Step5FP8LinearMethod()


class Step5MLP(nn.Module):
    """Dense FFN, used both for the four non-MoE layers and the shared expert."""

    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        # (stream, input shape, SwigluStep output) left by the router
        # (BF16FP32RouterLinear.fuse_shared_expert) for the next forward. Not
        # keyed by data_ptr: the MoE op mutates its hidden_states, so after
        # functionalization the router and the shared expert see different
        # buffers holding the same values. The stream check rejects a stale
        # entry when the shared expert runs on the aux stream before the router.
        self.fused_act: tuple[int, torch.Size, torch.Tensor] | None = None
        self.limit = None
        layer_idx = extract_layer_index(prefix)
        swiglu_limits_shared = getattr(config, "swiglu_limits_shared", None)
        if (
            swiglu_limits_shared
            and layer_idx < len(swiglu_limits_shared)
            and swiglu_limits_shared[layer_idx] is not None
            and swiglu_limits_shared[layer_idx] != 0
        ):
            self.limit = swiglu_limits_shared[layer_idx]
            self.act_fn = SwigluStepAndMul(limit=self.limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fused, self.fused_act = self.fused_act, None
        if (
            fused is not None
            and fused[0] == torch.cuda.current_stream().cuda_stream
            and fused[1] == hidden_states.shape
        ):
            intermediate_act = fused[2]
        else:
            gate_up, _ = self.gate_up_proj(hidden_states)
            intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step5Attention(nn.Module):
    """GQA with qk-norm, a per-head sigmoid output gate, and partial rotary.

    Every weight here is bf16 in the checkpoint, so ``quant_config`` is
    deliberately not threaded into the projections -- see the module docstring.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: CacheConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        sparse_config: Any = None,
        dsa_state: Step4DSAState | None = None,
        dsa_slot: int | None = None,
        total_num_attention_heads: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        if layer_types:
            enable_sliding_window = layer_types[self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0
        if yarn_only_types and layer_types[self.layer_idx] not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            rope_theta = rope_theta[self.layer_idx]

        self.rank = get_tensor_model_parallel_rank()
        self.partial_rotary_factor = partial_rotary_factor
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Fewer KV heads than ranks: replicate. TP8 with 4 groups gives one
            # KV head per rank and two replicas per group.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            raise ValueError("rope_scaling must be a dict for Step5Attention.")

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.q_norm = _fp32_gemma_rms_norm(self.head_dim, rms_norm_eps)
        self.k_norm = _fp32_gemma_rms_norm(self.head_dim, rms_norm_eps)
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.register_buffer("qkvg_weight", None, persistent=False)
        self.register_buffer("qkvg_wq", None, persistent=False)
        self.register_buffer("qkvg_ws", None, persistent=False)
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.g_proj",
            )
            self._maybe_fuse_qkvg()
        if FP8_DENSE:
            _use_fp8(self.o_proj)
            if self.qkvg_weight is not None:
                self.qkv_proj.quant_method = Step5FP8LinearMethod(
                    post=self._quantize_qkvg
                )
            else:
                _use_fp8(self.qkv_proj)
                if use_head_wise_attn_gate:
                    _use_fp8(self.g_proj)

        self.use_rope = True
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]

        # 23 full_attention layers run DSA; the other 69 keep the platform's
        # default backend. attn_backend is a per-layer override that skips
        # get_attn_backend's automatic selection entirely.
        self.is_sparse_layer = dsa_state is not None and dsa_slot is not None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            attn_backend=Step4SparseAttentionBackend if self.is_sparse_layer else None,
        )

        self.indexer: Step4Indexer | None = None
        if self.is_sparse_layer:
            if sliding_window is not None:
                raise ValueError(
                    f"layer {self.layer_idx} is both sliding and sparse; DSA "
                    f"applies to full_attention layers only"
                )
            # The indexer's parameters are attached to *this* module rather than
            # to a submodule, because that is where the checkpoint puts them.
            self.indexer = Step4Indexer(
                self,
                geometry=Step4DSAGeometry.from_config(sparse_config),
                state=dsa_state,
                dsa_slot=dsa_slot,
                num_kv_groups=self.num_kv_heads,
                hidden_size=hidden_size,
                tp_rank=self.rank,
                tp_size=tp_size,
                rms_norm_eps=rms_norm_eps,
                rope_theta=self.rope_theta,
                # The indexer rotates with the same theta and the same llama3
                # scaling as the layer it feeds, so it has to see the scaling
                # dict after the yarn_only_types filter above, not before.
                rope_scaling=rope_scaling,
                max_position=max_position,
                params_dtype=torch.get_default_dtype(),
                num_attention_heads=total_num_attention_heads or self.total_num_heads,
            )

        self.max_position_embeddings = max_position
        # Sliding layers: q/k norm + RoPE + q quant + KV cache write in one op.
        self.swa_fused_qk = SWA_FUSED_QK and _get_step5_swa_fused().can_use(self)
        # Gate + fp8 quant of the o_proj input as one op (folded into the SWA
        # decode combine when swa_fused_qk); needs the fused qkvg fp8 GEMM.
        self.fused_gate = (
            FUSED_GATE
            and FP8_DENSE
            and use_head_wise_attn_gate
            and self.qkvg_weight is not None
            and isinstance(self.o_proj.quant_method, Step5FP8LinearMethod)
            and self.num_heads & (self.num_heads - 1) == 0
        )
        # Step-5 uses 1/3 on full-attention layers, so take the rotary width from
        # get_rope (int(head_size * factor)) instead of assuming 1 or 1/2.
        self.rotary_dim = getattr(
            self.rotary_emb,
            "rotary_dim",
            int(self.head_dim * self.partial_rotary_factor),
        )

    def _maybe_fuse_qkvg(self) -> None:
        if not FUSE_QKVG:
            return
        try:
            from vllm_hcu.platforms import envs as henvs
        except ImportError:
            return
        # The fused GEMM bypasses both linear methods and their bias handling.
        # Keep those methods for biased or incompatible projections.
        if self.qkv_proj.bias is not None or self.g_proj.bias is not None:
            return
        wq, wg = self.qkv_proj.weight, self.g_proj.weight
        if (
            not henvs.VLLM_USE_NN
            or wq.dim() != 2
            or wg.dim() != 2
            or wq.shape[0] != wg.shape[0]
            or wq.dtype != wg.dtype
            or wq.device != wg.device
            or wq.dtype not in (torch.bfloat16, torch.float16)
        ):
            return
        nq = wq.shape[1]
        fused = torch.empty(
            wq.shape[0], nq + wg.shape[1], dtype=wq.dtype, device=wq.device
        )
        wq.data = fused[:, :nq]
        wg.data = fused[:, nq:]
        self.register_buffer("qkvg_weight", fused, persistent=False)

    def _quantize_qkvg(self) -> None:
        from vllm_hcu.ops.step5.fp8_dense import quantize_weight_fp8

        self.qkvg_wq, self.qkvg_ws = quantize_weight_fp8(self.qkvg_weight)
        self.qkvg_weight = None
        _free_weight(self.qkv_proj.weight)
        _free_weight(self.g_proj.weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.qkvg_wq is not None:
            from vllm_hcu.ops.step5.fp8_dense import fp8_linear

            # N is padded; slice instead of split.
            qkvg = fp8_linear(hidden_states, self.qkvg_wq, self.qkvg_ws)
            nqkv = self.q_size + 2 * self.kv_size
            qkv = qkvg[..., :nqkv]
            extra_dims = qkvg[..., nqkv : nqkv + self.num_heads]
        elif self.qkvg_weight is not None:
            qkvg = torch.matmul(hidden_states, self.qkvg_weight)
            qkv, extra_dims = qkvg.split(
                [self.q_size + 2 * self.kv_size, self.num_heads], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            extra_dims = None
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.swa_fused_qk:
            attn_output = _get_step5_swa_fused().forward(
                self,
                positions,
                qkv,
                v,
                k,
                gate=extra_dims if self.fused_gate else None,
            )
            return self._gate_o_proj(attn_output, extra_dims, hidden_states)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head.contiguous())
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head.contiguous())
        k = k_by_head.view(k.shape)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        if self.indexer is not None:
            # Publishes layer.attn.step4_sparse_plan, which the DSA backend
            # reads. It needs hidden_states, which Attention.forward never sees.
            self.indexer.forward(hidden_states, positions)
        attn_output = self.attn(q, k, v)
        return self._gate_o_proj(attn_output, extra_dims, hidden_states)

    def _gate_o_proj(
        self,
        attn_output: torch.Tensor,
        extra_dims: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(attn_output, tuple):
            # Already gated and quantized (step5_swa_fused, fused_gate).
            output, _ = self.o_proj(attn_output)
            return output
        if self.fused_gate and extra_dims is not None:
            output, _ = self.o_proj(
                _get_step5_swa_fused().gate_quant(
                    attn_output.view(-1, attn_output.shape[-1]), extra_dims
                )
            )
            return output
        if self.use_head_wise_attn_gate:
            if extra_dims is None:
                extra_dims, _ = self.g_proj(hidden_states)
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


def _step5_moe_reduce_policy(tp_size: int, dp_size: int) -> tuple[bool, bool]:
    """Return combined-reduce and per-path-reduce settings for Step5 MoE."""
    fuse_all_reduce = tp_size > 1 and dp_size == 1
    return fuse_all_reduce, not fuse_all_reduce


class FusedMoEBlock(nn.Module):
    """Router + shared expert + 352 routed experts.

    ``quant_config`` reaches ``FusedMoE`` and nothing else: the expert weights are
    the only fp8 tensors in the checkpoint. Layers 88-90 keep bf16 experts; the
    block checkpoint lists them in ``modules_to_not_convert`` and the channel one
    in the compressed-tensors ``ignore`` list. Either way the layer resolves to an
    unquantized method for those three ``FusedMoE`` instances.

    Note the ``ignore`` list is matched against the *unfused* expert names vLLM
    synthesises (``...moe.experts.0.gate_proj``), not against this module's
    ``...moe.experts`` prefix.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        if BF16_ROUTER:
            self.gate = BF16FP32RouterLinear(
                config.hidden_size, config.moe_num_experts, prefix=f"{prefix}.gate"
            )
        else:
            self.gate = FP32ReplicatedLinear(
                config.hidden_size,
                config.moe_num_experts,
                bias=False,
                quant_config=None,
                params_dtype=torch.float32,  # Use FP32 for higher precision.
                prefix=f"{prefix}.gate",
            )
        self.use_moe_router_bias = config.use_moe_router_bias
        assert self.use_moe_router_bias, "Only support use_moe_router_bias is true."
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        assert self.need_fp32_gate, (
            "Router logits must use FP32 precision for numerical stability."
        )

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            swiglu_limit = float(swiglu_limit)
            assert swiglu_limit == 7.0, (
                "Swiglu limit in fused moe block only support 7.0 now."
            )
            activation = "swiglustep"

        self.fuse_all_reduce, reduce_results = _step5_moe_reduce_policy(
            self.tp_size, get_dp_group().world_size
        )
        self.effective_sequence_parallel = (
            vllm_config.compilation_config.pass_config.enable_sp and self.tp_size > 1
        )
        # Outside SP, FusedMoE owns the shared expert (as 0.21's step4.py did): the
        # runner adds shared + routed before its single TP all-reduce -- under DP>1
        # one all-reduce per layer instead of two -- and can run the shared expert
        # on the aux stream alongside the gate, router and routed experts.
        self.fold_shared_expert = not self.effective_sequence_parallel
        if self.fold_shared_expert:
            self.fuse_all_reduce = False
        self.share_expert = Step5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=False if self.fold_shared_expert else reduce_results,
            # The shared expert is bf16 in both checkpoints. Do not rely on the
            # ignore list: hf_to_vllm_mapper renames it to ".moe.share_expert.",
            # which the channel checkpoint's regex ("layers.N.share_expert.")
            # does not match, so it would be built as fp8 and the bf16 weights
            # cast into it with no scale.
            quant_config=None,
            prefix=f"{prefix}.share_expert",
        )
        if (
            FUSED_ROUTER_SHARED
            and isinstance(self.gate, BF16FP32RouterLinear)
            and self.fold_shared_expert
            and self.share_expert.limit is not None
        ):
            self.gate.fuse_shared_expert(self.share_expert)
        custom_routing_function = functools.partial(
            router_bias_func,
            router_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
        )
        self.experts = FusedMoEFactory(
            gate=self.gate,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True if self.fold_shared_expert else reduce_results,
            shared_experts=self.share_expert if self.fold_shared_expert else None,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.effective_sequence_parallel,
            router_logits_dtype=torch.float32,
            custom_routing_function=custom_routing_function,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return (shared, routed); shared is None when FusedMoE already added it."""
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        if (
            self.experts.moe_config.is_sequence_parallel
            and not input_is_sequence_parallel
        ):
            hidden_states = sequence_parallel_chunk(hidden_states)
        shared_output = (
            None if self.fold_shared_expert else self.share_expert(hidden_states)
        )
        if self.experts.is_internal_router:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, fused_shared_act = self.gate(hidden_states)
            if fused_shared_act is not None:
                self.share_expert.fused_act = (
                    torch.cuda.current_stream().cuda_stream,
                    hidden_states.shape,
                    fused_shared_act,
                )
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        return shared_output, routed_output


class Step5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        dsa_state: Step4DSAState | None = None,
        dsa_slots: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )
        self.compute_dtype = vllm_config.model_config.dtype
        if cache_config is not None:
            cache_config.sliding_window = None
        if config.att_impl_type != "GQA":
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )

        # HF exposes main-model layer_types separately; MTP uses the tail.
        layer_types = (
            getattr(config, "layer_types_with_mtp", None)
            or getattr(config, "layer_types", [])
        )
        num_attention_heads = None
        num_attention_groups = None
        head_dim = None
        if (
            getattr(config, "attention_other_setting", None)
            and layer_types
            and layer_types[layer_idx]
            == config.attention_other_setting["attention_type"]
        ):
            num_attention_heads = config.attention_other_setting["num_attention_heads"]
            num_attention_groups = config.attention_other_setting[
                "num_attention_groups"
            ]
            head_dim = config.attention_other_setting["head_dim"]
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        self.self_attn = Step5Attention(
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads
            if num_attention_heads
            else config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=num_attention_groups
            if num_attention_groups
            else config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
            cache_config=cache_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=getattr(config, "sliding_window", None),
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=layer_types,
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            partial_rotary_factor=partial_rotary_factors[layer_idx]
            if partial_rotary_factors
            else 1.0,
            prefix=f"{prefix}.self_attn",
            sparse_config=getattr(config, "sparse_config", None),
            dsa_state=dsa_state if dsa_slots and layer_idx in dsa_slots else None,
            dsa_slot=(dsa_slots or {}).get(layer_idx),
            total_num_attention_heads=config.num_attention_heads,
        )

        self.use_moe = False
        self.tp_group = get_tp_group()
        self.use_fused_all_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and get_dp_group().world_size == 1
        )

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(
                vllm_config,
                prefix=f"{prefix}.moe",
            )
            self.use_moe = True
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=None,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
            if FP8_DENSE:
                _use_fp8(self.mlp.gate_up_proj, self.mlp.down_proj)
        self.input_layernorm = _fp32_gemma_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = _fp32_gemma_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.prefix = prefix

    def _cast_for_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual_connection:
            return hidden_states
        return hidden_states.to(torch.float32)

    def _forward_ffn(
        self,
        hidden_states: torch.Tensor,
        input_is_sequence_parallel: bool = False,
        residual: torch.Tensor | None = None,
        orig_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if not self.use_moe:
            return self.mlp(hidden_states)
        shared_output, routed_output = self.moe(
            hidden_states, input_is_sequence_parallel=input_is_sequence_parallel
        )
        if self.moe.experts.moe_config.is_sequence_parallel:
            combined = self._cast_for_residual(shared_output) + self._cast_for_residual(
                routed_output
            )
            if input_is_sequence_parallel:
                assert residual is not None and orig_num_tokens is not None
                return tensor_model_parallel_all_gather(
                    combined + self._cast_for_residual(residual), dim=0
                )[:orig_num_tokens]
            return tensor_model_parallel_all_gather(combined, dim=0)[
                : hidden_states.shape[0]
            ]
        if shared_output is None:
            return routed_output
        combined = self._cast_for_residual(shared_output) + self._cast_for_residual(
            routed_output
        )
        if self.moe.fuse_all_reduce:
            combined = tensor_model_parallel_all_reduce(combined)
        return combined

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        orig_num_tokens = hidden_states.shape[0]
        residual = hidden_states
        if not self.fp32_residual_connection:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            return self._forward_ffn(hidden_states) + residual

        hidden_states = self.input_layernorm(residual).to(self.compute_dtype)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self._cast_for_residual(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(
            self.compute_dtype
        )
        ffn_output = self._forward_ffn(
            hidden_states,
            input_is_sequence_parallel=False,
            residual=None,
            orig_num_tokens=orig_num_tokens,
        )
        return self._cast_for_residual(ffn_output) + residual


@support_torch_compile
class Step5Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.compute_dtype = vllm_config.model_config.dtype
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )

        self.moe_num_experts = config.moe_num_experts
        self.skipped_dsa_weights = 0
        if FP8_BF16_MOE:
            _unignore_bf16_experts(vllm_config.quant_config)

        # One Step4DSAState is shared by every DSA layer on this rank: the
        # region bookkeeping it does is layer-independent, so the first layer of
        # a step pays for it and the other 22 reuse the result.
        sparse_layers = dsa_layer_indices(config)
        self.dsa_slots = {idx: slot for slot, idx in enumerate(sparse_layers)}
        self.dsa_state: Step4DSAState | None = None
        if sparse_layers:
            self.dsa_state = Step4DSAState(
                num_layers=len(sparse_layers),
                geometry=Step4DSAGeometry.from_config(config.sparse_config),
                # A running request holds at most one half-filled region, so
                # max_num_seqs would do; the slack absorbs rows left behind by
                # requests that finished mid-region, which are only reclaimed
                # least-recently-touched.
                max_pending=2 * vllm_config.scheduler_config.max_num_seqs + 16,
                dtype=self.compute_dtype,
                # Capture bounds: the decode schedule lives in [max_num_reqs]
                # buffers at a score width covering max_model_len, so that both
                # its addresses and its shapes survive a CUDA graph replay.
                max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
                max_model_len=vllm_config.model_config.max_model_len,
                # MTP verify steps can be rolled back by rejection; DSA keeps
                # the regions they close recoverable (see Step4PendingTable.retire).
                spec_query_len=1
                + (
                    vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config is not None
                    else 0
                ),
            )
            logger.info(
                "step5: DSA enabled on %d of %d layers (%d...%d)",
                len(sparse_layers),
                config.num_hidden_layers,
                sparse_layers[0],
                sparse_layers[-1],
            )

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step5DecoderLayer(
                vllm_config,
                prefix=prefix,
                dsa_state=self.dsa_state,
                dsa_slots=self.dsa_slots,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = _fp32_gemma_rms_norm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.fp32_residual_connection:
            hidden_states = hidden_states.float()

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        # Narrow before the final RMSNorm, not after: normalising in fp32 and
        # casting only at lm_head is measurably different.
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(self.compute_dtype)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        assert config.num_attention_groups > 1, "Only support GQA"
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        skipped_dsa = 0

        # Packed 3D expert format: .moe.gate_proj.weight [num_experts, out, in].
        # vLLM 0.28's FusedMoEFactory nests the expert parameters under
        # ``.moe.experts.routed_experts.``; the 0.21-era names this file was
        # written against had no such level, so each prefix below carries it.
        _routed = ".moe.experts.routed_experts."
        expert_params_mapping = [
            (f"{_routed}{base_layer}w13_weight", ".moe.gate_proj.weight", "w1"),
            (f"{_routed}{base_layer}w13_weight", ".moe.up_proj.weight", "w3"),
            (f"{_routed}{base_layer}w2_weight", ".moe.down_proj.weight", "w2"),
            (
                f"{_routed}{base_layer}w13_weight_scale_inv",
                ".moe.gate_proj.weight_scale_inv",
                "w1",
            ),
            (
                f"{_routed}{base_layer}w13_weight_scale_inv",
                ".moe.up_proj.weight_scale_inv",
                "w3",
            ),
            (
                f"{_routed}{base_layer}w2_weight_scale_inv",
                ".moe.down_proj.weight_scale_inv",
                "w2",
            ),
        ]

        # Per-expert format: .moe.experts.E.{gate,up,down}_proj.*
        per_expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.moe_num_experts,
        )

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]
        if FP8_BF16_MOE:
            weights = _quantize_bf16_experts(weights)

        for name, loaded_weight in weights:
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            spec_layer = get_spec_layer_idx_from_weight_name(config, full_name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # Skip any layers beyond the main model's depth (e.g., MTP layers)
            if full_name.startswith("model.layers."):
                parts = full_name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= config.num_hidden_layers:
                        continue

            # With DSA switched off there is nothing to load the indexer
            # tensors into. Count instead of silently dropping so that a load
            # problem stays visible.
            if DISABLE_DSA and any(
                sub in local_name for sub in _DSA_INDEXER_SUBSTRINGS
            ):
                skipped_dsa += 1
                continue

            if ".moe.experts." in local_name:
                is_expert_weight = False
                for mapping in per_expert_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in local_name:
                        continue
                    is_expert_weight = True
                    name_mapped = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if (
                        not is_expert_weight
                        and not is_pp_missing_parameter(local_name, self)
                        and local_name in params_dict
                    ):
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            default_weight_loader,
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(local_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in local_name:
                    continue
                if any(
                    disable_moe_stacked_param in local_name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                replaced_name = local_name.replace(weight_name, param_name)
                if is_pp_missing_parameter(replaced_name, self):
                    continue
                if replaced_name not in params_dict:
                    continue
                param = params_dict[replaced_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(replaced_name)
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    if weight_name not in local_name:
                        continue
                    replaced_name = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(replaced_name, self):
                        continue
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    param = params_dict[replaced_name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.moe_num_experts
                    assert loaded_weight.shape[0] == moe_expert_num
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            replaced_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(replaced_name)
                    break
                else:
                    if is_pp_missing_parameter(local_name, self):
                        continue
                    if "expert_bias" in local_name:
                        logger.warning_once("ignore expert_bias")
                        continue
                    if local_name not in params_dict:
                        logger.warning_once(
                            "step5: no parameter for checkpoint tensor %s", local_name
                        )
                        continue
                    param = params_dict[local_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(local_name)

        self.skipped_dsa_weights = skipped_dsa
        if skipped_dsa:
            logger.info(
                "step5: VLLM_STEP5_DISABLE_DSA is set, so %d DSA indexer "
                "tensor(s) were skipped; dense attention is exact only for "
                "sequences up to topk * region_block_size tokens.",
                skipped_dsa,
            )
        return loaded_params


class Step5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".share_expert.": ".moe.share_expert."}
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Step5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,  # lm_head is bf16 in the checkpoint.
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.logits_processor._step5_skinny_lm_head = True
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[FusedMoEBlock] = []
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step5DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                self.moe_layers.append(layer.moe)

        self.expert_weights = []
        assert len(self.moe_layers) > 0, "No MoE layers found in the model."
        example_layer = self.moe_layers[0]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            experts = layer.experts
            self.expert_weights.append(experts.get_expert_weights())
            experts.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.moe_layers:
            assert isinstance(layer, FusedMoEBlock)
            layer.n_local_physical_experts = num_local_physical_experts
            layer.n_physical_experts = num_physical_experts
            layer.n_redundant_experts = self.num_redundant_experts
            layer.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def get_spec_layer_idx_from_weight_name(
    config: ModelConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"layers.{layer_idx + i}."
            ) or weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
