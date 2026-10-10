# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU Qwen4Exp PLE with opt-in INT8 UVA prefetch support.

Upstream source: vLLM commit 4574da606553cad5c22448d498f144630a23641e
Upstream SHA256: 8bb2e441dad2e9f3c8686bc48aa95001e5adc81b91199cc68b2e4bbe9eb69292
"""

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import nn

import vllm


_UPSTREAM_COMMIT = "4574da606553cad5c22448d498f144630a23641e"
_UPSTREAM_SHA256 = "8bb2e441dad2e9f3c8686bc48aa95001e5adc81b91199cc68b2e4bbe9eb69292"


def _verify_upstream_source() -> None:
    source = (
        Path(vllm.__file__).resolve().parent
        / "models/qwen4_exp/amd/ple_layer.py"
    )
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(
            f"cannot verify Qwen4Exp PLE source for HCU mirror at {source}"
        ) from exc
    if digest != _UPSTREAM_SHA256:
        raise RuntimeError(
            "Qwen4Exp PLE source changed since the HCU mirror was audited: "
            f"expected {_UPSTREAM_SHA256} from {_UPSTREAM_COMMIT}, got {digest}"
        )


_verify_upstream_source()

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding
from vllm_hcu.models.qwen4_exp.common.ops.ple import ple_conv, ple_gate, ple_ngram_ids
from vllm_hcu.models.qwen4_exp.engram import cpu_offload_enabled
from vllm_hcu.platforms import envs as hcu_envs

_PREFETCH_METHODS = (
    "prefetch_output_dtype",
    "prefetch_lookup_into",
    "finalize_prefetched",
)

logger = init_logger(__name__)


def _prefetch_method_enabled(method: object) -> bool:
    return bool(
        hcu_envs.VLLM_HCU_PLE_PREFETCH_STREAM
        and cpu_offload_enabled()
        and getattr(method, "supports_prefetch", False)
    )


def _prefetch_output_dtype(method: object, layer: nn.Module) -> torch.dtype:
    missing = [
        name for name in _PREFETCH_METHODS if not callable(getattr(method, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"{type(method).__name__} declares supports_prefetch=True but "
            f"does not implement {', '.join(missing)}"
        )
    output_dtype = method.prefetch_output_dtype(layer)
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError(
            f"{type(method).__name__}.prefetch_output_dtype must return "
            f"torch.dtype, got {type(output_dtype).__name__}"
        )
    return output_dtype


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        layer_name: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_name = layer_name
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
        self.max_total_tokens = max_total_tokens
        self._hcu_prefetch_successor: Qwen4ExpNGramEmbedding | None = None
        self._hcu_prefetch_pending = False
        self._hcu_prefetch_prepared = False
        method = self.ngram_embedding.quant_method
        self._hcu_prefetch_enabled = _prefetch_method_enabled(method)
        if self._hcu_prefetch_enabled:
            output_dtype = _prefetch_output_dtype(method, self.ngram_embedding)
            workspace_tokens = (
                max_total_tokens * self.ngram_embedding.etp_data_parallel_size
            )
            ids_buffer = torch.empty(
                (workspace_tokens, self.ngram_heads), dtype=torch.int64
            )
            rows_buffer = torch.empty(
                (workspace_tokens, embedding_dim), dtype=output_dtype
            )
            self._hcu_prefetch_stream = torch.cuda.Stream()
            self._hcu_prefetch_event = torch.cuda.Event()
            self._hcu_prefetch_fork_event = torch.cuda.Event()
        else:
            ids_buffer = torch.empty((0, self.ngram_heads), dtype=torch.int64)
            rows_buffer = torch.empty(
                (0, embedding_dim), dtype=self.ngram_embedding.params_dtype
            )
            self._hcu_prefetch_stream = None
            self._hcu_prefetch_event = None
            self._hcu_prefetch_fork_event = None
        self.register_buffer(
            "_hcu_prefetch_ids_buffer", ids_buffer, persistent=False
        )
        self.register_buffer(
            "_hcu_prefetch_rows_buffer", rows_buffer, persistent=False
        )
        self.register_buffer(
            "_hcu_prefetch_successor_ids_sentinel",
            torch.empty((0, self.ngram_heads), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_hcu_prefetch_successor_rows_sentinel",
            torch.empty(
                (0, embedding_dim), dtype=self.ngram_embedding.params_dtype
            ),
            persistent=False,
        )
        if self._hcu_prefetch_enabled:
            workspace_bytes = (
                ids_buffer.numel() * ids_buffer.element_size()
                + rows_buffer.numel() * rows_buffer.element_size()
            )
            logger.info_once(
                "Qwen4Exp PLE UVA prefetch active: %.3f MiB workspace "
                "for %s (ids=%s, rows=%s)",
                workspace_bytes / (1024**2),
                layer_name,
                tuple(ids_buffer.shape),
                tuple(rows_buffer.shape),
            )
        # Kept in the public constructor for vLLM 0.28.1 call compatibility;
        # fused ID generation no longer allocates a [requests, tokens] workspace.
        del max_num_reqs

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.max_total_tokens:
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.max_total_tokens}"
            )
        if ngram_context.shape[0] < num_reqs:
            raise ValueError(
                f"PLE received {num_reqs} requests, but ngram_context has "
                f"only {ngram_context.shape[0]} rows"
            )
        return ple_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context[:num_reqs],
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.eos_token_id,
            self.heads_per_ngram,
            output=output,
        )

    def prepare_prefetch(self) -> None:
        if not self._hcu_prefetch_enabled:
            return
        prepare = getattr(self.ngram_embedding.quant_method, "prepare_prefetch", None)
        if callable(prepare):
            prepare(self.ngram_embedding)
        self._hcu_prefetch_prepared = True

    def _start_prefetch_impl(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> bool:
        if not self._hcu_prefetch_enabled:
            return False
        if self._hcu_prefetch_pending:
            raise RuntimeError(f"duplicate PLE prefetch fire for {self.layer_name}")
        method = self.ngram_embedding.quant_method
        is_prepared = getattr(method, "is_prefetch_prepared", None)
        ready = self._hcu_prefetch_prepared and (
            not callable(is_prepared) or is_prepared(self.ngram_embedding)
        )
        if not ready:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"PLE prefetch for {self.layer_name} was not prepared "
                    "before graph capture"
                )
            self.prepare_prefetch()

        num_tokens = input_ids.reshape(-1).shape[0]
        if num_tokens > self.max_total_tokens:
            raise ValueError(
                f"PLE prefetch received {num_tokens} tokens, but its workspace "
                f"supports at most {self.max_total_tokens}"
            )
        side = self._hcu_prefetch_stream
        event = self._hcu_prefetch_event
        fork_event = self._hcu_prefetch_fork_event
        assert side is not None and event is not None and fork_event is not None
        main = torch.cuda.current_stream()
        # Fork the side stream with an explicit event.  On ROCm this is more
        # reliable during CUDA-graph capture than wait_stream(main): the
        # captured main stream owns the event node and the side stream waits
        # on that node before reading the input/workspace.
        main.record_event(fork_event)
        side.wait_event(fork_event)
        for tensor in (input_ids, query_start_loc, ngram_context):
            if tensor.device.type != "cpu":
                tensor.record_stream(side)

        self._hcu_prefetch_pending = True
        try:
            with torch.cuda.stream(side):
                generated_ids = self._hcu_prefetch_ids_buffer[:num_tokens]
                ngram_ids = self.compute_ngram_ids(
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    output=generated_ids,
                )
                embedding = self.ngram_embedding
                slot_size, _ = embedding._get_dp_gather_slot(num_tokens)
                gathered_ids = embedding._gather_dp_ids(ngram_ids, slot_size)
                gathered_num_tokens = gathered_ids.shape[0]
                if gathered_num_tokens > self._hcu_prefetch_ids_buffer.shape[0]:
                    raise ValueError(
                        "PLE prefetch received "
                        f"{gathered_num_tokens} gathered tokens, but its "
                        "workspace supports "
                        f"at most {self._hcu_prefetch_ids_buffer.shape[0]}"
                    )
                input_mask = None
                if embedding.tp_size > 1:
                    from vllm.model_executor.layers.vocab_parallel_embedding import (
                        get_masked_input_and_mask,
                    )

                    local_ids, input_mask = get_masked_input_and_mask(
                        gathered_ids,
                        embedding.shard_indices.org_vocab_start_index,
                        embedding.shard_indices.org_vocab_end_index,
                        embedding.shard_indices.num_org_vocab_padding,
                        embedding.shard_indices.added_vocab_start_index,
                        embedding.shard_indices.added_vocab_end_index,
                    )
                else:
                    local_ids = gathered_ids
                ids = self._hcu_prefetch_ids_buffer[:gathered_num_tokens]
                ids.copy_(local_ids)
                rows = self._hcu_prefetch_rows_buffer[:gathered_num_tokens].view(
                    gathered_num_tokens, self.ngram_heads, self.head_dim
                )
                embedding.quant_method.prefetch_lookup_into(embedding, ids, rows)
                if input_mask is not None:
                    rows.masked_fill_(input_mask.unsqueeze(-1), 0)
                event.record(side)
        except BaseException:
            self._hcu_prefetch_pending = False
            raise
        return True

    def _consume_prefetched_impl(self, num_tokens: int) -> torch.Tensor:
        if not self._hcu_prefetch_enabled:
            raise RuntimeError(f"PLE prefetch is disabled for {self.layer_name}")
        if not self._hcu_prefetch_pending:
            raise RuntimeError(f"PLE prefetch miss for {self.layer_name}")
        event = self._hcu_prefetch_event
        assert event is not None
        torch.cuda.current_stream().wait_event(event)
        embedding = self.ngram_embedding
        slot_size, slot_offset = embedding._get_dp_gather_slot(num_tokens)
        gathered_num_tokens = slot_size * embedding.etp_data_parallel_size
        rows = self._hcu_prefetch_rows_buffer[:gathered_num_tokens]
        try:
            rows = embedding.quant_method.finalize_prefetched(embedding, rows)
            return embedding._select_embeddings(
                rows,
                num_tokens,
                slot_offset,
            )
        finally:
            self._hcu_prefetch_pending = False

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        successor = self._hcu_prefetch_successor
        if successor is not None:
            successor._start_prefetch_impl(
                input_ids, query_start_loc, ngram_context
            )

        num_tokens = input_ids.reshape(-1).shape[0]
        if self._hcu_prefetch_enabled:
            return self._consume_prefetched_impl(num_tokens)

        ngram_ids = self.compute_ngram_ids(
            input_ids, query_start_loc, ngram_context
        )
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=self.ngram_embedding.params_dtype,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
            prefix,
        )
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]
        outer_residual = outer_residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                outer_residual=outer_residual,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                outer_residual_d, outer_residual_p = torch.split(
                    outer_residual,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                outer_residual_d = outer_residual_p = outer_residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    outer_residual=outer_residual_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.query_start_loc_p
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                outer_residual=outer_residual_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                outer_residual=outer_residual,
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. Short convolution
        # is a no-op there, but preserve the outer residual addition.
        if attn_metadata is None:
            residual.add_(outer_residual)
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            residual.add_(outer_residual)
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            outer_residual=outer_residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        dequantize = getattr(self.ple_embedding.ngram_embedding, "dequantize", None)
        if callable(dequantize):
            embeddings = dequantize(embeddings, hidden_states.dtype)
        elif embeddings.dtype != hidden_states.dtype:
            embeddings = embeddings.to(hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        # The short-conv op is a piecewise-graph splitting op: state routing
        # reads runtime request metadata. It accumulates the convolution and
        # the outer residual into gated_output.
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            conv_input,
            gated_output,
            hidden_states,
            self.prefix,
        )
        return gated_output


def qwen4_exp_amd_ple_ngram_embedding(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run the large PLE embedding lookup outside Inductor's FX graph.

    Keeping the embedding weight in ``static_forward_context`` prevents AOT
    compile-time autotuning from materializing a synthetic copy of the weight.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    owner = layer.ple_embedding
    result = owner.ngram_embedding(ngram_ids).flatten(-2)
    output.copy_(result)


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._short_conv(inputs, residual, outer_residual)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding",
    op_func=qwen4_exp_amd_ple_ngram_embedding,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["residual"],
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
