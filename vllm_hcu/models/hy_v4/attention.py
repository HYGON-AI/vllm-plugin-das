# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA attention and lightning indexer for HY V4 on HCU.

The eager MLA path uses the existing HCU FP8 indexer cache. A model-private
sparse backend forwards the per-head learnable sink through both prefill and
decode without changing backend behavior for other models.
"""

from dataclasses import dataclass, replace
import os
from typing import cast

import regex as re
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import (
    divide,
    get_dcp_group,
    get_dp_group,
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    init_model_parallel_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, get_kv_quant_mode

from .fp8_kv_dequant import LightOpKVReuseState

logger = init_logger(__name__)

_SPARSE_LAYER_TYPES = ("sparse_attention", "sparse", "deepseek_sparse_attention")
_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _require_accuracy_safe_kv_cache_dtype(kv_cache_dtype: str) -> None:
    """Reject HY V4 KV-cache formats without verified output parity."""
    if kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "fp8_e4m3",
        "fp8_ds_mla",
    ):
        raise RuntimeError(
            "HY V4 accuracy-first inference supports auto, bfloat16, "
            "fp8_e4m3, or fp8_ds_mla KV cache; "
            f"got {kv_cache_dtype!r}. "
            "use --kv-cache-dtype fp8_e4m3 or fp8_ds_mla for quantized "
            "KV cache."
        )


def _normalize_hy_v4_kv_cache_dtype(
    kv_cache_dtype: str,
    *,
    use_sparse: bool,
) -> str:
    """Normalize HY V4's E4M3 alias to the sparse FlashMLA cache layout."""
    _require_accuracy_safe_kv_cache_dtype(kv_cache_dtype)
    if use_sparse and kv_cache_dtype == "fp8_e4m3":
        return "fp8_ds_mla"
    return kv_cache_dtype


def require_hyv4_sink_backend(
    backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    """Return a sparse sink-capable backend or fail closed."""
    if not backend.is_sparse() or not backend.supports_sink():
        raise ValueError(
            "HY V4 attention sink requires a sink-capable sparse MLA backend "
            f"for both prefill and decode; got {backend.get_name()}."
        )
    return backend


def _require_sparse_mqa_backend(backend: type[AttentionBackend]) -> None:
    """Verify that prefill and decode both use sparse MQA dispatch."""
    impl_cls = backend.get_impl_cls()
    if not isinstance(impl_cls, type) or not issubclass(
        impl_cls, SparseMLAAttentionImpl
    ):
        raise RuntimeError(
            "HY V4 learnable sink requires a sparse MQA implementation for "
            f"prefill and decode; got {backend.get_name()}."
        )


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.
    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        seen_full = False
        for layer_idx, indexer_type in enumerate(indexer_types):
            if indexer_type == "full":
                seen_full = True
            elif not seen_full:
                raise ValueError(
                    "A 'shared' indexer requires a preceding 'full' producer; "
                    f"layer {layer_idx} has none."
                )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


def require_local_indexer_producer(
    config: PretrainedConfig,
    *,
    start_layer: int,
    end_layer: int,
) -> None:
    """Reject PP stages that cannot obtain shared top-k indexer results."""
    if not hasattr(config, "index_topk") or start_layer == end_layer:
        return
    if not 0 <= start_layer < end_layer <= config.num_hidden_layers:
        raise ValueError(
            "Invalid HY V4 pipeline layer range: "
            f"[{start_layer}, {end_layer}) for {config.num_hidden_layers} layers."
        )
    skip_topk_layers = compute_skip_topk_layers(config)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        has_local_producer = False
        for layer_idx in range(start_layer, end_layer):
            # Match HYV4MLAAttention construction: a declared "full"
            # indexer on a dense attention layer does not create an indexer.
            if (
                layer_idx >= len(layer_types)
                or layer_types[layer_idx] not in _SPARSE_LAYER_TYPES
            ):
                continue
            if layer_idx not in skip_topk_layers:
                has_local_producer = True
            elif not has_local_producer:
                raise ValueError(
                    f"HY V4 shared sparse indexer layer {layer_idx} requires "
                    "a preceding local 'full' sparse indexer producer in "
                    f"pipeline layer range [{start_layer}, {end_layer})."
                )
        return

    # Retain the conservative declared-pattern check for callers that do not
    # supply attention layer types.
    if start_layer in skip_topk_layers:
        raise ValueError(
            "HY V4 pipeline stage starts at shared indexer layer "
            f"{start_layer}, but top-k indices are not transferred between "
            "pipeline stages; align the partition to a 'full' indexer layer."
        )


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.
    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # Quantized checkpoint adaptation for these BF16 projections is
        # provided by the separate HYV4 quantization integration.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # FP8 naive cache: values in fp8 plus one fp32 scale per
        # ``quant_block_size`` elements.
        assert cache_config is not None, "HYV4 indexer requires cache_config"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        indexer_cls = SparseAttnIndexer
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            # PCP metadata carries rank-ordered global cache slots. The V32
            # owner gathers matching K before computing top-k for local Q.
            from vllm.model_executor.layers.sparse_attn_indexer import (
                V32SparseAttnIndexer,
            )

            indexer_cls = V32SparseAttnIndexer
        self.indexer_op = indexer_cls(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states, q_quant, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the quantized query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims, so flatten back to the
        # token-major shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # Reassemble with the original physical layout: no_pe first, pe last.
        q = torch.cat([q_nope, q_pe], dim=-1)
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_nope, k_pe.squeeze(-2)], dim=-1)

        # Only q is quantized here; k quantization is fused with cache insertion.
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return hidden_states, q_fp8, k, weights


class HYV4MLAAttentionLayer(MLAAttention):
    """Attach the quantization mode omitted by vLLM 0.25.1 MLA specs."""

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Prepare MLA projections, then run the selected backend's hook."""
        super().process_weights_after_loading(act_dtype)
        if getattr(self, "_hcu_dcp_q_replicate", False):
            if (
                self.is_aiter_triton_fp4_bmm_enabled
                or self.is_aiter_triton_fp8_bmm_enabled
            ):
                raise RuntimeError(
                    "VLLM_DCP_Q_REPLICATE does not support Aiter MLA BMM "
                    "weight formats on vLLM 0.25.1"
                )
            self.W_UK_T_dcp_qrep = get_dcp_group().all_gather(
                self.W_UK_T.contiguous(), dim=0
            )
        self.impl.process_weights_after_loading(act_dtype)

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward local Q while exposing group-replicated Q to decode."""
        self._hcu_q_dcp_replicated = q_dcp_replicated
        try:
            return super().forward(q, kv_c_normed, k_pe, output_shape)
        finally:
            self._hcu_q_dcp_replicated = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        return replace(
            spec,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


_LINEAR_GATE_PCP_SHARD_ENV = "VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD"
_LINEAR_GATE_PCP_GROUP_SIZE_ENV = "VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE"
_LINEAR_GATE_PCP_CHUNK_ENV = "VLLM_HCU_LINEAR_GATE_PCP_CHUNKING"
_LINEAR_GATE_PCP_BLOCK_TOKENS_ENV = "VLLM_HCU_LINEAR_GATE_PCP_BLOCK_TOKENS"
_LINEAR_GATE_PCP_DEFAULT_BLOCK_TOKENS = 4096
_linear_gate_pcp_shard_logged = False
_linear_gate_pcp_groups: dict[tuple[int, ...], object] = {}
_LINEAR_GATE_DP_SHARD_ENV = "VLLM_HCU_ENABLE_LINEAR_GATE_DP_SHARD"
_LINEAR_GATE_DP_GROUP_SIZE_ENV = "VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE"
_LINEAR_GATE_DP_CHUNK_ENV = "VLLM_HCU_LINEAR_GATE_DP_CHUNKING"
_LINEAR_GATE_DP_BLOCK_TOKENS_ENV = "VLLM_HCU_LINEAR_GATE_DP_BLOCK_TOKENS"
_LINEAR_GATE_DP_DEFAULT_GROUP_SIZE = 8
_LINEAR_GATE_DP_DEFAULT_BLOCK_TOKENS = 4096
_linear_gate_dp_shard_logged = False
_linear_gate_dp_groups: dict[tuple[int, ...], object] = {}
_LINEAR_GATE_ALLOWED_GROUP_SIZES = (2, 4, 8)

_DCP_Q_REPLICATE_ENV = "VLLM_DCP_Q_REPLICATE"


@dataclass(frozen=True)
class DCPQReplicationTopology:
    """Effective TP topology used by an official-style DCP Q projection."""

    group_size: int
    rank_in_group: int
    tp_rank: int
    tp_world_size: int


def resolve_dcp_q_replication_topology(
    *,
    tp_rank: int,
    tp_world_size: int,
    dcp_world_size: int,
) -> DCPQReplicationTopology:
    """Map a TP rank to the coarser shard shared by its DCP group."""
    group_size = max(int(dcp_world_size), 1)
    if tp_world_size % group_size != 0:
        raise ValueError(
            "DCP Q replication requires tensor_parallel_size to be divisible "
            f"by decode_context_parallel_size; got TP={tp_world_size}, "
            f"DCP={group_size}"
        )
    return DCPQReplicationTopology(
        group_size=group_size,
        rank_in_group=tp_rank % group_size,
        tp_rank=tp_rank // group_size,
        tp_world_size=tp_world_size // group_size,
    )


def dcp_q_replication_enabled() -> bool:
    """Return the official vLLM DCP query-replication opt-in."""
    return _env_flag(_DCP_Q_REPLICATE_ENV, False)


class DCPGroupColumnParallelLinear(ColumnParallelLinear):
    """Shard a Q projection across DCP groups and replicate within a group.

    vLLM 0.25.1 does not yet let ``ColumnParallelLinear`` override its TP
    rank and world size. This is the equivalent of upstream's newer
    ``DCPGroupColumnParallelLinear`` adapted to the 0.25.1 constructor.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ) -> None:
        parallel_config = get_current_vllm_config().parallel_config
        topology = resolve_dcp_q_replication_topology(
            tp_rank=get_tensor_model_parallel_rank(),
            tp_world_size=get_tensor_model_parallel_world_size(),
            dcp_world_size=parallel_config.decode_context_parallel_size,
        )
        self.group_size = topology.group_size
        self.rank_in_group = topology.rank_in_group
        self.qrep_active = self.group_size > 1

        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(
            output_size, topology.tp_world_size
        )
        self.output_partition_sizes = [self.output_size_per_partition]

        # LinearBase resolves the quant method before weights are created.
        # Restore the coarse TP coordinates immediately afterward so both
        # legacy and v2 parameter loaders select the DCP-group shard.
        LinearBase.__init__(
            self,
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=True,
        )
        self.disable_tp = False
        self.tp_rank = topology.tp_rank
        self.tp_size = topology.tp_world_size
        self._maybe_allow_fp8_block_shape_mismatch()
        self.gather_output = gather_output
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=self.params_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()
        # The HCU replacement for ColumnParallelLinear uses this stable
        # class-name contract in both legacy and v2 weight loaders.
        self.is_quantization = (
            self.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        )

    def _local_view(self, out: torch.Tensor) -> torch.Tensor:
        """Select this rank's original TP head shard from group-wide Q."""
        if self.group_size == 1:
            return out
        local_heads = divide(out.shape[-2], self.group_size)
        start = self.rank_in_group * local_heads
        return out[..., start : start + local_heads, :].contiguous()


def _env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    return os.environ.get(name, default_value).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def linear_gate_pcp_shard_enabled() -> bool:
    """Return whether gated MLA linear_gate PCP sharding is enabled."""
    return _env_flag(_LINEAR_GATE_PCP_SHARD_ENV, False)


def linear_gate_pcp_chunking_enabled() -> bool:
    """Return whether large linear_gate PCP inputs should be chunked.

    Chunking defaults to disabled. Set
    ``VLLM_HCU_LINEAR_GATE_PCP_CHUNKING=1`` to enable chunked collectives.
    """
    return _env_flag(_LINEAR_GATE_PCP_CHUNK_ENV, False)


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive integer; got {raw_value!r}."
        ) from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value}.")
    return value


def linear_gate_pcp_block_tokens() -> int:
    """Return the positive local-token block size for linear_gate PCP."""
    return _env_positive_int(
        _LINEAR_GATE_PCP_BLOCK_TOKENS_ENV,
        _LINEAR_GATE_PCP_DEFAULT_BLOCK_TOKENS,
    )


def _linear_gate_pcp_group_size(pcp_size: int) -> int:
    """Return the PCP subgroup size that shares one K-sharded linear_gate.

    Unset ``VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE`` keeps the historical behavior
    of sharding across the full PCP group. When set, the value must be 2, 4, or
    8 and must divide ``pcp_size``.
    """
    raw = os.environ.get(_LINEAR_GATE_PCP_GROUP_SIZE_ENV)
    if raw is None or raw.strip() == "":
        return pcp_size
    try:
        size = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{_LINEAR_GATE_PCP_GROUP_SIZE_ENV} must be 2, 4, or 8; got {raw!r}."
        ) from exc
    if size not in _LINEAR_GATE_ALLOWED_GROUP_SIZES:
        raise ValueError(
            f"{_LINEAR_GATE_PCP_GROUP_SIZE_ENV} must be 2, 4, or 8; got {size}."
        )
    return size


def _linear_gate_pcp_group_ranks(
    world: int,
    dp_size: int,
    pp_size: int,
    pcp_size: int,
    tp_size: int,
    group_size: int,
) -> list[list[int]]:
    """Split each PCP group into K-shard subgroups of ``group_size`` ranks."""
    coordinates = dp_size * pp_size * pcp_size * tp_size
    if world % coordinates != 0:
        raise RuntimeError(
            "unable to derive PCP linear_gate subgroup topology: "
            f"world={world}, dp={dp_size}, pp={pp_size}, pcp={pcp_size}, "
            f"tp={tp_size}."
        )
    if pcp_size % group_size != 0:
        raise ValueError(
            f"PCP size {pcp_size} is not divisible by group size {group_size}."
        )
    # Layout matches vLLM: ExternalDP x DP x PP x PCP x TP.
    ranks = torch.arange(world).reshape(-1, dp_size, pp_size, pcp_size, tp_size)
    groups = ranks.transpose(3, 4).reshape(-1, pcp_size)
    return [
        chunk.tolist() for row in groups for chunk in row.reshape(-1, group_size)
    ]


def _get_linear_gate_pcp_group():
    """Return the PCP subgroup that K-shards one replica of linear_gate.

    ``VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE`` splits the PCP group into independent
    replicas. For PCP=32 and group size 8, ranks ``[0,8)``, ``[8,16)``,
    ``[16,24)``, and ``[24,32)`` each shard the same full weight and run
    all-to-all plus reduce-scatter inside the subgroup. When the env is unset,
    the full PCP group is used.
    """
    pcp = get_pcp_group()
    size = _linear_gate_pcp_group_size(pcp.world_size)
    if pcp.world_size <= 1:
        raise ValueError(
            "linear_gate PCP sharding requires prefill_context_parallel_size > 1; "
            f"got pcp_size={pcp.world_size}. Unset {_LINEAR_GATE_PCP_SHARD_ENV}."
        )
    if pcp.world_size % size != 0:
        raise ValueError(
            f"PCP size {pcp.world_size} is not divisible by "
            f"{_LINEAR_GATE_PCP_GROUP_SIZE_ENV}={size}."
        )
    if size == pcp.world_size:
        return pcp
    world = dist.get_world_size()
    dp = get_dp_group().world_size
    pp = get_pp_group().world_size
    tp = get_tensor_model_parallel_world_size()
    group_ranks = _linear_gate_pcp_group_ranks(
        world, dp, pp, pcp.world_size, tp, size
    )
    key = tuple(rank for group in group_ranks for rank in group)
    if key not in _linear_gate_pcp_groups:
        _linear_gate_pcp_groups[key] = init_model_parallel_group(
            group_ranks,
            pcp.local_rank,
            pcp.torch_distributed_backend,
            group_name="hy_v4_linear_gate_pcp",
        )
    return _linear_gate_pcp_groups[key]


def linear_gate_dp_shard_enabled() -> bool:
    """Return whether gated MLA linear_gate DP sharding is enabled."""
    return _env_flag(_LINEAR_GATE_DP_SHARD_ENV, False)


def linear_gate_dp_chunking_enabled() -> bool:
    """Return whether large linear_gate DP inputs should be chunked.

    Chunking defaults to disabled. Set
    ``VLLM_HCU_LINEAR_GATE_DP_CHUNKING=1`` to enable chunked collectives.
    """
    return _env_flag(_LINEAR_GATE_DP_CHUNK_ENV, False)


def linear_gate_dp_block_tokens() -> int:
    """Return the positive local-token block size for linear_gate DP."""
    return _env_positive_int(
        _LINEAR_GATE_DP_BLOCK_TOKENS_ENV,
        _LINEAR_GATE_DP_DEFAULT_BLOCK_TOKENS,
    )


def _linear_gate_dp_group_size() -> int:
    """Return the DP subgroup size that shares one K-sharded linear_gate."""
    size = _env_positive_int(
        _LINEAR_GATE_DP_GROUP_SIZE_ENV,
        _LINEAR_GATE_DP_DEFAULT_GROUP_SIZE,
    )
    if size not in _LINEAR_GATE_ALLOWED_GROUP_SIZES:
        raise ValueError(
            f"{_LINEAR_GATE_DP_GROUP_SIZE_ENV} must be 2, 4, or 8; got {size}."
        )
    return size


def _linear_gate_dp_group_ranks(
    world: int,
    dp_size: int,
    pp_size: int,
    pcp_size: int,
    tp_size: int,
    group_size: int,
) -> list[list[int]]:
    """Split each DP group into K-shard subgroups of ``group_size`` ranks."""
    coordinates = dp_size * pp_size * pcp_size * tp_size
    if world % coordinates != 0:
        raise RuntimeError(
            "unable to derive DP linear_gate subgroup topology: "
            f"world={world}, dp={dp_size}, pp={pp_size}, pcp={pcp_size}, "
            f"tp={tp_size}."
        )
    if dp_size % group_size != 0:
        raise ValueError(
            f"DP size {dp_size} is not divisible by group size {group_size}."
        )
    ranks = torch.arange(world).reshape(-1, dp_size, pp_size, pcp_size, tp_size)
    groups = ranks.transpose(1, 4).reshape(-1, dp_size)
    return [chunk.tolist() for row in groups for chunk in row.reshape(-1, group_size)]


def _get_linear_gate_dp_group():
    """Return the DP subgroup that K-shards one replica of linear_gate.

    ``VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE`` splits the DP group into independent
    replicas. For DP=32 and group size 8, ranks ``[0,8)``, ``[8,16)``,
    ``[16,24)``, and ``[24,32)`` each shard the same full weight and run
    all-to-all plus reduce-scatter inside the subgroup.
    """
    dp = get_dp_group()
    size = _linear_gate_dp_group_size()
    if dp.world_size <= 1:
        raise ValueError(
            "linear_gate DP sharding requires data_parallel_size > 1; got "
            f"dp_size={dp.world_size}. Unset {_LINEAR_GATE_DP_SHARD_ENV}."
        )
    if dp.world_size % size != 0:
        raise ValueError(
            f"DP size {dp.world_size} is not divisible by "
            f"{_LINEAR_GATE_DP_GROUP_SIZE_ENV}={size}."
        )
    if size == dp.world_size:
        return dp
    world = dist.get_world_size()
    pp = get_pp_group().world_size
    pcp = get_pcp_group().world_size
    tp = get_tensor_model_parallel_world_size()
    group_ranks = _linear_gate_dp_group_ranks(
        world, dp.world_size, pp, pcp, tp, size
    )
    key = tuple(rank for group in group_ranks for rank in group)
    if key not in _linear_gate_dp_groups:
        _linear_gate_dp_groups[key] = init_model_parallel_group(
            group_ranks,
            dp.local_rank,
            dp.torch_distributed_backend,
            group_name="hy_v4_linear_gate_dp",
        )
    return _linear_gate_dp_groups[key]


class _KShardedGateLinear(ColumnParallelLinear):
    """K-shard ``linear_gate`` over an arbitrary process group.

    ``k_shard`` slices the weight input dimension. An all-to-all sends each
    destination rank its K slice for every token, producing
    ``[group_tokens, K/group]`` before a partial GEMM and token reduce-scatter.
    reduce-scatter is the fused equivalent of all-reduce followed by
    redistribution to each token-owning rank.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group,
        parallel_name: str,
        enable_env: str,
        align_uneven_tokens: bool,
        chunking_enabled_fn,
        block_tokens_fn,
        log_once_flag: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.shard_group = group
        self.shard_rank = group.rank_in_group
        self.shard_size = group.world_size
        self.parallel_name = parallel_name
        self.align_uneven_tokens = align_uneven_tokens
        self._chunking_enabled_fn = chunking_enabled_fn
        self._block_tokens_fn = block_tokens_fn
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise ValueError(
                f"linear_gate {parallel_name} sharding requires "
                "tensor_parallel_size == 1 (TP already shards linear_gate); "
                f"got tp_size={tp_size}. Unset {enable_env}."
            )
        if self.shard_size <= 1:
            raise ValueError(
                f"linear_gate {parallel_name} sharding requires "
                f"{parallel_name} group size > 1; got "
                f"{parallel_name}_size={self.shard_size}. Unset {enable_env}."
            )
        if input_size % self.shard_size != 0:
            raise ValueError(
                f"linear_gate k_shard dimension {input_size} is not divisible "
                f"by {parallel_name} size {self.shard_size}."
            )
        self.gate_full_input_size = input_size
        self.gate_output_size = output_size
        self.gate_shard_input_size = input_size // self.shard_size
        local_input_size = self.gate_shard_input_size
        local_output_size = output_size
        super().__init__(
            local_input_size,
            local_output_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )
        if self.is_quantization:
            raise ValueError(
                f"linear_gate {parallel_name} sharding only supports "
                f"unquantized gate weights; layer {prefix} resolved quantized "
                f"method {self.quant_method.__class__.__name__}."
            )
        expected_weight_numel = local_input_size * local_output_size
        if self.weight.numel() != expected_weight_numel:
            raise RuntimeError(
                f"linear_gate k_shard allocated {self.weight.numel()} weight "
                f"elements; expected {expected_weight_numel} for local shape "
                f"[{local_output_size}, {local_input_size}]"
            )
        logged = globals()[log_once_flag]
        if not logged:
            globals()[log_once_flag] = True
            logger.info(
                "HY V4 linear_gate %s K sharding enabled: %s=%d, "
                "full weight=[%d, %d], local weight=[%d, %d].",
                parallel_name,
                parallel_name,
                self.shard_size,
                self.gate_output_size,
                self.gate_full_input_size,
                self.gate_output_size,
                self.gate_shard_input_size,
            )

    def _narrow_to_k_shard(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        """Slice this rank's input columns from the full checkpoint weight."""
        if loaded_weight.dim() == 2 and loaded_weight.shape == (
            self.gate_output_size,
            self.gate_full_input_size,
        ):
            return loaded_weight.narrow(
                1,
                self.shard_rank * self.gate_shard_input_size,
                self.gate_shard_input_size,
            )
        raise ValueError(
            f"linear_gate {self.parallel_name} sharding expected the full "
            f"checkpoint weight of shape [{self.gate_output_size}, "
            f"{self.gate_full_input_size}], got {tuple(loaded_weight.shape)}."
        )

    def weight_loader(self, param, loaded_weight: torch.Tensor):
        super().weight_loader(param, self._narrow_to_k_shard(loaded_weight))

    def weight_loader_v2(self, param, loaded_weight: torch.Tensor):
        super().weight_loader_v2(param, self._narrow_to_k_shard(loaded_weight))

    def _linear(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dim() != 2:
            raise RuntimeError(f"unexpected linear_gate weight rank: {weight.dim()}")
        if weight.shape[0] == hidden.shape[-1]:
            return torch.matmul(hidden, weight)
        if weight.shape[1] == hidden.shape[-1]:
            return torch.nn.functional.linear(hidden, weight)
        raise RuntimeError(
            f"linear_gate {self.parallel_name} sharding found an unexpected "
            f"weight layout {tuple(weight.shape)} for input width "
            f"{hidden.shape[-1]}"
        )

    def _align_tokens(
        self, input_: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        local_tokens = input_.shape[0]
        if not self.align_uneven_tokens:
            return input_, local_tokens, local_tokens
        count = torch.tensor(
            [local_tokens], device=input_.device, dtype=torch.int32
        )
        dist.all_reduce(
            count,
            op=dist.ReduceOp.MAX,
            group=self.shard_group.device_group,
        )
        max_tokens = int(count.item())
        if max_tokens > local_tokens:
            input_ = torch.nn.functional.pad(
                input_, (0, 0, 0, max_tokens - local_tokens)
            )
        return input_, local_tokens, max_tokens

    def _k_shard_block(self, block: torch.Tensor) -> torch.Tensor:
        current_tokens = block.shape[0]
        send = (
            block.reshape(
                current_tokens, self.shard_size, self.gate_shard_input_size
            )
            .transpose(0, 1)
            .contiguous()
        )
        received = torch.empty_like(send)
        dist.all_to_all_single(
            received.view(-1),
            send.view(-1),
            group=self.shard_group.device_group,
        )
        partial = self._linear(
            received.view(
                self.shard_size * current_tokens,
                self.gate_shard_input_size,
            )
        )
        return self.shard_group.reduce_scatter(partial.contiguous(), dim=0)

    def _forward_k_shard(self, input_: torch.Tensor) -> torch.Tensor:
        aligned, local_tokens, aligned_tokens = self._align_tokens(input_)
        if aligned_tokens == 0:
            return input_.new_empty((0, self.gate_output_size))
        chunking_enabled = self._chunking_enabled_fn()
        block_tokens = (
            self._block_tokens_fn() if chunking_enabled else aligned_tokens
        )
        if chunking_enabled and aligned_tokens > block_tokens:
            output = aligned.new_empty(
                (aligned_tokens, self.gate_output_size)
            )
            for start in range(0, aligned_tokens, block_tokens):
                end = min(start + block_tokens, aligned_tokens)
                output[start:end].copy_(self._k_shard_block(aligned[start:end]))
        else:
            output = self._k_shard_block(aligned)
        if local_tokens != aligned_tokens:
            return output[:local_tokens]
        return output

    def forward(self, input_):
        output = self._forward_k_shard(input_)
        if not self.return_bias:
            return output
        return output, None


class PCPShardedGateLinear(_KShardedGateLinear):
    """PCP-shard ``linear_gate`` using K sharding inside a PCP subgroup."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        pcp_group = _get_linear_gate_pcp_group()
        super().__init__(
            input_size,
            output_size,
            group=pcp_group,
            parallel_name="PCP",
            enable_env=_LINEAR_GATE_PCP_SHARD_ENV,
            align_uneven_tokens=False,
            chunking_enabled_fn=linear_gate_pcp_chunking_enabled,
            block_tokens_fn=linear_gate_pcp_block_tokens,
            log_once_flag="_linear_gate_pcp_shard_logged",
            quant_config=quant_config,
            prefix=prefix,
        )
        self.pcp_rank = self.shard_rank
        self.pcp_size = self.shard_size


class DPShardedGateLinear(_KShardedGateLinear):
    """DP-shard ``linear_gate`` using K sharding inside a DP subgroup."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        dp_group = _get_linear_gate_dp_group()
        super().__init__(
            input_size,
            output_size,
            group=dp_group,
            parallel_name="DP",
            enable_env=_LINEAR_GATE_DP_SHARD_ENV,
            align_uneven_tokens=True,
            chunking_enabled_fn=linear_gate_dp_chunking_enabled,
            block_tokens_fn=linear_gate_dp_block_tokens,
            log_once_flag="_linear_gate_dp_shard_logged",
            quant_config=quant_config,
            prefix=prefix,
        )
        self.dp_rank = self.shard_rank
        self.dp_size = self.shard_size


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    `.hcu_sparse`; unsupported backend configurations fail closed.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
        lightop_kv_reuse_state: LightOpKVReuseState | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        requested_kv_cache_dtype = (
            cache_config.cache_dtype if cache_config else "auto"
        )
        kv_cache_dtype = _normalize_hy_v4_kv_cache_dtype(
            requested_kv_cache_dtype,
            use_sparse=self.is_sparse,
        )
        if (
            cache_config is not None
            and kv_cache_dtype != requested_kv_cache_dtype
        ):
            cache_config.cache_dtype = cast(CacheDType, kv_cache_dtype)
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.q_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            self.q_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
                disable_tp=True,
            )
            self.kv_a_proj_with_mqa = MergedColumnParallelLinear(
                self.hidden_size,
                [self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        q_proj_cls = (
            DCPGroupColumnParallelLinear
            if dcp_q_replication_enabled()
            else ColumnParallelLinear
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = q_proj_cls(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = q_proj_cls(
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
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            if linear_gate_pcp_shard_enabled() and linear_gate_dp_shard_enabled():
                raise ValueError(
                    "linear_gate PCP sharding and DP sharding cannot be "
                    f"enabled together. Unset {_LINEAR_GATE_PCP_SHARD_ENV} or "
                    f"{_LINEAR_GATE_DP_SHARD_ENV}."
                )
            if linear_gate_pcp_shard_enabled():
                self.linear_gate = PCPShardedGateLinear(
                    self.hidden_size,
                    self.num_heads * self.gate_projection_size_per_head,
                    quant_config=quant_config,
                    prefix=f"{prefix}.linear_gate",
                )
            elif linear_gate_dp_shard_enabled():
                self.linear_gate = DPShardedGateLinear(
                    self.hidden_size,
                    self.num_heads * self.gate_projection_size_per_head,
                    quant_config=quant_config,
                    prefix=f"{prefix}.linear_gate",
                )
            else:
                self.linear_gate = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_heads * self.gate_projection_size_per_head,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.linear_gate",
                )
        else:
            self.linear_gate = None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            enable_sink = sink_backend is not None
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    # The kernels require fp32 sinks; the disabled path keeps
                    # the checkpoint dtype since the value is never consumed.
                    dtype=torch.float32 if enable_sink else torch.bfloat16,
                )
            )
            if enable_sink:
                sinks = self.learnable_sink_param
                _require_sparse_mqa_backend(sink_backend)

        extra_impl_args = {} if sinks is None else {"sinks": sinks}
        if self.is_sparse and lightop_kv_reuse_state is not None:
            extra_impl_args["lightop_kv_reuse_state"] = lightop_kv_reuse_state
        self.mla_attn = HYV4MLAAttentionLayer(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )
        q_proj_layer = self.q_b_proj if self.q_b_proj is not None else self.q_proj
        self.dcp_q_replicate = bool(
            getattr(q_proj_layer, "qrep_active", False)
        )
        self.mla_attn._hcu_dcp_q_replicate = self.dcp_q_replicate

    def _resolve_sink_backend(
        self, kv_cache_dtype: str
    ) -> type[AttentionBackend]:
        """Resolve a sink-capable sparse MLA backend and fail closed."""
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            raise RuntimeError(
                "HY V4 could not select a sparse MLA backend required by "
                "the learnable sink."
            ) from exc

        if selected_cls.is_sparse() and selected_cls.supports_sink():
            return require_hyv4_sink_backend(selected_cls)

        from .hcu_sparse import HYV4FlashMLASparseBackend

        capability = current_platform.get_device_capability()
        if capability is None:
            raise RuntimeError(
                "HY V4 learnable sink requires a known HCU device capability."
            )
        cache_config = get_current_vllm_config().cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None
            and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
        )
        if invalid_reasons:
            raise RuntimeError(
                "HY V4 learnable sink has no valid HCU sparse MLA backend: "
                + ", ".join(invalid_reasons)
            )
        return require_hyv4_sink_backend(HYV4FlashMLASparseBackend)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q_c = self.q_a_proj(hidden_states)[0]
            q_c = self.q_a_layernorm(q_c)
            q_proj_layer = self.q_b_proj
            q = q_proj_layer(q_c)[0]
        else:
            assert self.q_proj is not None
            q_proj_layer = self.q_proj
            q = q_proj_layer(hidden_states)[0]

        assert self.kv_a_proj_with_mqa is not None
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q_heads = self.num_local_heads
        if self.dcp_q_replicate:
            q_heads *= q_proj_layer.group_size
        q = q.view(-1, q_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # Single coarse eager break covering the indexer and MLA attention, as
        # the breakable cudagraph contract requires: everything that reads
        # per-batch metadata runs in one eager segment, so no tensor has to stay
        # alive across a capture-segment boundary.
        attn_out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        self._indexer_and_attn(
            hidden_states,
            q_c,
            positions,
            q,
            kv_c_normed,
            k_pe,
            attn_out,
            q_dcp_replicated,
        )

        if self.gated_mla and self.linear_gate is not None:
            gate_score = self.linear_gate(hidden_states)[0]
            if self.config.gating_type == "headwise":
                gate_score = gate_score.unsqueeze(-1)
                attn_out = attn_out.reshape(*attn_out.shape[:-1], -1, self.v_head_dim)
                attn_out = attn_out * torch.sigmoid(gate_score)
                attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
            else:
                attn_out = attn_out * torch.sigmoid(gate_score)

        out, _ = self.o_proj(attn_out)
        return out

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary. The nested ``sparse_attn_indexer`` and
        ``unified_mla_attention_with_output`` breaks short-circuit here, since
        the capture is no longer active inside an eager segment.
        """
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
                q_dcp_replicated=q_dcp_replicated,
            )
        )


__all__ = [
    "HYV4MLAAttention",
    "DCPGroupColumnParallelLinear",
    "dcp_q_replication_enabled",
    "resolve_dcp_q_replication_topology",
    "Indexer",
    "compute_skip_topk_layers",
    "is_skip_topk_indexer_weight",
    "require_local_indexer_producer",
    "require_hyv4_sink_backend",
]
