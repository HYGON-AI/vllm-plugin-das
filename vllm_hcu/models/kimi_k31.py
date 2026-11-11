# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3.1 Pro DSA (DeepSeek Sparse Attention) inference path on HCU.

Adapted from the vLLM PR #65685 ROCm reference (``kimi_k3.amd.dsa``) and the
HY-V4/V3.2 sparse stack already shipped in this plugin:

* every DSA full-attention layer builds its own indexer (``index_topk_freq``
  must be 1; the K3.1 checkpoint stores per-layer indexer weights);
* the indexer mirrors the V3.2 ROCm contract: FP8 per-token-group query
  quantization with UE8M0 scales folded into the per-head weights, an FP8
  index-key cache (gfx938: 132 B/token uint8) and
  ``SparseAttnIndexer.forward_hip``;
* submodule names follow the harvested K3.1 checkpoint final layout
  (``self_attn.indexer.{wq_b,wk,weights_proj,k_norm}``) so the stock
  ``AutoWeightsLoader`` routes the BF16 indexer tensors without renames;
* the sparse MLA attention is wired through the plain
  ``MultiHeadLatentAttentionWrapper`` (``MLAModules(is_sparse=True)``).  The
  AMD ``KimiK3MultiHeadLatentAttentionWrapper`` fused decode Q-prep writes
  dense 576-wide rows, which the 656-byte ``fp8_ds_mla`` records cannot take.
"""

from __future__ import annotations

from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
from vllm.v1.attention.selector import get_attn_backend

from vllm_hcu.platforms.hcu import on_gfx938

logger = init_logger(__name__)

# Model-instance scope: the shared top-k buffer lives on ``KimiLinearModel``
# and must be visible to every DSA layer constructed inside ``__init__``.
K31_DSA_CTX: ContextVar[dict[str, Any] | None] = ContextVar(
    "kimi_k31_dsa_ctx", default=None
)


def _dsa_config(text_config: Any) -> SimpleNamespace | None:
    """Return the K3.1 ``dsa_config`` as an attribute accessor, else None."""
    raw = getattr(text_config, "dsa_config", None)
    if raw is None and isinstance(text_config, dict):
        # The built-in KimiK3Config path can hand us raw dicts (AutoConfig
        # before vLLM normalizes sub-configs).
        raw = text_config.get("dsa_config")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return SimpleNamespace(**raw)
    return raw


def kimi_k31_is_k31(model_config: Any, hf_config: Any) -> bool:
    hf_config = hf_config if hf_config is not None else getattr(
        model_config, "hf_config", None
    )
    return getattr(hf_config, "model_type", None) == "kimi_k31"


def kimi_k31_dsa_layers(text_config: Any) -> frozenset[int] | None:
    """0-based decoder layer indices owning a DSA indexer, or None."""
    dsa = _dsa_config(text_config)
    if dsa is None:
        return None
    layers = getattr(dsa, "dsa_layers", None)
    if layers is None:
        return None
    # Checkpoint/config dsa_layers are 1-based.
    return frozenset(int(idx) - 1 for idx in layers)


def kimi_k31_is_dsa_layer(text_config: Any, layer_idx: int) -> bool:
    layers = kimi_k31_dsa_layers(text_config)
    return layers is not None and layer_idx in layers


class KimiK31DsaContext:
    """Short-lived model-construction scope binding the shared top-k buffer."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None,
    ) -> None:
        text_config = vllm_config.model_config.hf_text_config
        dsa = _dsa_config(text_config)
        if dsa is None:
            raise ValueError("KimiK31DsaContext requires text dsa_config")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.topk_indices_buffer = torch.empty(
            (
                vllm_config.scheduler_config.max_num_batched_tokens,
                int(dsa.index_topk),
            ),
            dtype=torch.int32,
            device=device,
        )
        self.dsa_layers = kimi_k31_dsa_layers(text_config)
        self.cache_config = cache_config
        self._vllm_config = vllm_config
        self._token = None

    def __enter__(self) -> "KimiK31DsaContext":
        self._token = K31_DSA_CTX.set(
            {
                "buffer": self.topk_indices_buffer,
                "dsa_layers": self.dsa_layers,
                "vllm_config": self._vllm_config,
                "cache_config": self.cache_config,
            }
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        K31_DSA_CTX.reset(self._token)
        self._token = None


class KimiK31IndexerKNorm(nn.Module):
    """Bias-full fp32 LayerNorm for the DSA indexer shared key.

    PR #65685 reference: ``weight``/``bias`` load and stay in fp32 with an
    activation-dtype copy cached for the forward, so the checkpoint's BF16
    tensors are upcast by the stock weight loader and never round-trip per
    call in the captured graph.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.weight_act: torch.Tensor | None
        self.bias_act: torch.Tensor | None
        self.register_buffer("weight_act", None, persistent=False)
        self.register_buffer("bias_act", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.weight_act, self.bias_act
        if weight is None or weight.dtype != x.dtype:
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype)
            if not torch.compiler.is_compiling():
                self.weight_act, self.bias_act = weight, bias
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, weight, bias, self.eps
        )


class KimiK31Indexer(nn.Module):
    """Lightning indexer for one K3.1 DSA layer, V3.2-ROCm contract.

    Checkpoint mapping (BF16, excluded from quantization):
    ``wq_b[4096,1536]``, ``wk[128,7168]``, ``weights_proj[32,7168]``,
    ``k_norm.{weight,bias}`` -- the submodule names below match exactly so
    loader routing stays direct.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        hidden_size: int,
        index_q_input_size: int,
        index_head_dim: int,
        index_num_heads: int,
        index_topk: int,
        rms_norm_eps: float,
        index_k_norm_type: str = "rmsnorm",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_num_heads = index_num_heads
        self.indexer_weight_scale = self.index_num_heads**-0.5
        self.topk_tokens = index_topk
        self.quant_config = quant_config
        self.prefix = prefix

        # Checkpoint stores BF16 under self_attn.indexer (modules_to_not_
        # convert); SlimQuant W4A8 Config.from_config() only honors `ignore`,
        # so passing the global quant_config would create packed W4A8
        # parameters that the BF16 safetensors can never fill.
        self.wq_b = ReplicatedLinear(
            index_q_input_size,
            self.index_num_heads * self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.index_num_heads,
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.weights_proj",
        )
        if index_k_norm_type == "layernorm":
            self.k_norm: nn.Module = KimiK31IndexerKNorm(
                self.index_head_dim, eps=rms_norm_eps
            )
        elif index_k_norm_type == "rmsnorm":
            self.k_norm = RMSNorm(self.index_head_dim, eps=rms_norm_eps)
        else:
            raise ValueError(
                "dsa_config.index_k_norm_type must be 'rmsnorm' or "
                f"'layernorm', got {index_k_norm_type!r}"
            )

        # Index cache scales use UE8M0 (power-of-2 rounding at write), the
        # DeepSeek V3.2 convention; the scoring kernels read them as fp32.
        self.scale_fmt = "ue8m0"
        self.quant_block_size = self.index_head_dim
        self.index_softmax_scale = self.index_head_dim**-0.5
        self.topk_indices_buffer = topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError(
                f"{prefix}: the indexer requires the model-level "
                "topk_indices_buffer shared across DSA layers"
            )
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=(
                self.index_head_dim + self.index_head_dim // 128 * 4
                if on_gfx938()
                else self.index_head_dim
            ),
            dtype=torch.uint8 if on_gfx938() else torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.index_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            topk_indices_buffer,
        )

    def _quantize_index_q_and_weights(
        self,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8-quantize the query, fold scales into fp32 per-head weights.

        Mirrors the PR #65685 ``_index_weights_kernel`` rounding: the
        head-count scale applies in the projection dtype first, and only then
        do the fp32 query scales and softmax scale multiply.
        """
        q_flat = index_q.reshape(-1, self.index_head_dim).contiguous()
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_flat,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(index_q.shape)
        q_scale = q_scale.reshape(index_q.shape[:-1])
        # Preserve the projection dtype's rounding before the fp32 products.
        scaled = (index_weights * self.indexer_weight_scale).to(
            index_weights.dtype
        )
        weights = (
            scaled.to(torch.float32)
            * q_scale.to(torch.float32)
            * self.index_softmax_scale
        )
        return q_fp8, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_q_input: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        rotary_emb: nn.Module | None = None,
    ) -> torch.Tensor:
        if rotary_emb is not None:
            # K3.1 sets index_rope_dim=0 (NoPE); a rotary module here belongs
            # to a different model generation and is not supported.
            raise NotImplementedError(
                f"{self.prefix}: K3.1 indexer does not support index RoPE"
            )
        if index_q_input is None:
            # q_lora_rank is None: MultiHeadLatentAttention feeds hidden.
            index_q_input = hidden_states
        index_q = self.wq_b(index_q_input)[0].view(
            -1, self.index_num_heads, self.index_head_dim
        )
        index_k = self.k_norm(self.wk(hidden_states)[0])
        index_weights = self.weights_proj(hidden_states)[0]
        if not on_gfx938():
            # BF16 index-cache path: no fp8 scoring, weights stay raw
            # (mirrors the HCU V3.2 indexer branch).
            return self.indexer_op(
                hidden_states, index_q, index_k, index_weights
            )
        q_fp8, weights = self._quantize_index_q_and_weights(
            index_q, index_weights
        )
        return self.indexer_op(hidden_states, q_fp8, index_k, weights)


def _normalize_k31_kv_cache_dtype(
    cache_config: CacheConfig | None, *, use_sparse: bool
) -> None:
    """Adopt the sparse FlashMLA cache layout in place (HY V4 pattern)."""
    if cache_config is None or not use_sparse:
        return
    if cache_config.cache_dtype in ("auto", "bfloat16"):
        raise RuntimeError(
            "Kimi-K3.1 DSA layers require fp8_e4m3 / fp8_ds_mla KV cache; "
            f"got {cache_config.cache_dtype!r}"
        )
    if cache_config.cache_dtype == "fp8_e4m3":
        cache_config.cache_dtype = "fp8_ds_mla"  # type: ignore[assignment]
        logger.info_once(
            "Kimi-K3.1 Pro: using fp8_ds_mla as the internal sparse MLA "
            "KV-cache layout (sparse records 656 B/token)."
        )


class KimiK31DSAAttention:
    """Kimi-K3.1 DSA full-attention state (sparse MLA + inline indexer).

    Same projections as the dense Kimi-K3 MLA path, plus the indexer wired
    through ``MLAModules(is_sparse=True)`` so the sparse MLA backend consumes
    the shared top-k indices buffer.

    Deliberately NOT an ``nn.Module`` subclass: the worker patch invokes this
    ``__init__`` ON a stock ``KimiMLAAttention`` instance to transplant the
    sparse state in place.  Subclassing would break the zero-argument
    ``super()`` (``obj must be an instance or subtype of type``); instead the
    body initializes the ``nn.Module`` machinery of the foreign instance
    explicitly and every submodule assignment registers onto it through the
    ordinary ``nn.Module.__setattr__`` path.
    """

    def __init__(
        self,
        config: Any,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        model_config: Any = None,
        prefix: str = "",
        use_nope: bool = True,
        vllm_config: VllmConfig | None = None,
    ) -> None:
        # The foreign (stock KimiMLAAttention) instance's nn.Module machinery
        # was never initialized (its stock __init__ is preempted), so bring
        # it up explicitly; subsequent submodule assignments register through
        # nn.Module.__setattr__ exactly as in a normal constructor.
        nn.Module.__init__(self)
        # Prefer the explicitly supplied engine config (the construction-
        # argument path used by the worker patch); fall back to the ambient
        # config, and fail precisely when neither exists.
        if vllm_config is None:
            vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            raise RuntimeError(
                "KimiK31DSAAttention requires an explicit vllm_config "
                "(constructed outside the engine configuration scope)"
            )
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "Kimi-K3.1 DSA does not support prefill context parallelism"
            )
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "Kimi-K3.1 DSA does not support decode context parallelism"
            )
        dsa = _dsa_config(config)
        if dsa is None:
            raise ValueError("KimiK31DSAAttention requires text dsa_config")

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size > 0:
            padded = (num_heads + tp_size - 1) // tp_size * tp_size
            logger.warning_once(
                "K3.1 DSA num_heads %d is not divisible by tp_size %d; "
                "padding to %d heads",
                num_heads,
                tp_size,
                padded,
            )
            self.num_heads = padded
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        # NoPE-only main path, same as dense K3 MLA (rotary_emb disabled).
        if not getattr(config, "mla_use_nope", True) or not use_nope:
            raise NotImplementedError(
                "Kimi-K3.1 DSA requires mla_use_nope=True (NoPE-only MLA)"
            )
        self.use_nope = True
        self.rotary_emb = None

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))
        if self.use_output_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.v_head_dim * self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.index_topk = int(dsa.index_topk)
        self.index_head_dim = int(dsa.index_head_dim)
        self.index_num_heads = int(dsa.index_num_heads)
        if int(getattr(dsa, "index_rope_dim", 0) or 0) != 0:
            raise NotImplementedError(
                "Kimi-K3.1 DSA requires index_rope_dim == 0 "
                f"(got {getattr(dsa, 'index_rope_dim', None)})"
            )
        if int(getattr(dsa, "index_topk_freq", 1) or 1) != 1:
            raise NotImplementedError(
                "Kimi-K3.1 DSA has no index-share path; index_topk_freq must "
                f"be 1 (got {getattr(dsa, 'index_topk_freq', None)})"
            )

        ctx = K31_DSA_CTX.get()
        if ctx is None or ctx.get("buffer") is None:
            raise ValueError(
                f"{prefix}: KimiK31DSAAttention requires the model-level "
                "topk_indices_buffer provided by the KimiLinearModel wrapper"
            )
        self.topk_indices_buffer = ctx["buffer"]
        _normalize_k31_kv_cache_dtype(
            cache_config, use_sparse=True
        )
        # Fail fast instead of silently degrading sparse layers to dense.
        try:
            get_attn_backend(
                head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.get_default_dtype(),
                kv_cache_dtype=cache_config.cache_dtype
                if cache_config is not None
                else "auto",
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:  # pragma: no cover - surface the real error
            raise RuntimeError(
                "Kimi-K3.1 sparse MLA backend is unavailable for the current "
                "runtime/config; refusing to fall back to dense attention."
            ) from exc

        if self.q_lora_rank is not None:
            index_q_input_size = self.q_lora_rank
        else:
            index_q_input_size = self.hidden_size
        # The layer prefix below `.self_attn` MUST stay `indexer`: the K3.1
        # checkpoint stores final names under self_attn.indexer.*.
        self.indexer = KimiK31Indexer(
            vllm_config=vllm_config,
            hidden_size=self.hidden_size,
            index_q_input_size=index_q_input_size,
            index_head_dim=self.index_head_dim,
            index_num_heads=self.index_num_heads,
            index_topk=self.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            index_k_norm_type=str(
                getattr(dsa, "index_k_norm_type", "rmsnorm") or "rmsnorm"
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=self.topk_indices_buffer,
            prefix=f"{prefix}.indexer",
        )
        # The DSA __init__ runs ON the stock KimiMLAAttention instance, so
        # mirror the sparse attribute surface other patches / stock methods
        # may still touch on this object.
        self.is_sparse = True
        self.skip_topk = False
        self.indexer_rope_emb = None

        mla_modules = MLAModules(  # type: ignore[call-arg]
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm
            if self.q_lora_rank is not None
            else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=None,
            is_sparse=True,
            topk_indices_buffer=self.topk_indices_buffer,
            g_proj=getattr(self, "g_proj", None),
        )
        # The plain wrapper, not the AMD Kimi wrapper: see module docstring.
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def get_input_quant_key(self) -> None:
        # The generic MLA wrapper also feeds these activations to the gate.
        return None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output[:] = self.mla_attn(positions, hidden_states)

# === K31 MUTE DSA (env-gated experiment) ===
import os as _os_mute
if _os_mute.environ.get('VLLM_K31_MUTE_DSA') == '1':
    def _mute_fwd9(self, positions, hidden_states, output):
        output.zero_()
    KimiK31DSAAttention.forward = _mute_fwd9
