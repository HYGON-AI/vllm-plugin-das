# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import inspect
from collections.abc import Callable

import torch
from einops import rearrange
from torch import nn
from torch.nn.parameter import Parameter

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.models.kimi_k3.amd.ops.kda_gated_norm import (
    KimiFusedRMSNormGated as FusedRMSNormGated,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm_hcu.models.kimi_k3.amd.ops.gather_initial_states import (
    gather_initial_states,
)

# Empirical lower bound for the KDA gate to avoid numerical underflow.
_KDA_GATE_LOGBOUND_MIN = -5.0

logger = init_logger(__name__)

_CAUSAL_CONV_UPDATE_HAS_OUT = "out" in inspect.signature(
    causal_conv1d_update
).parameters


def _causal_conv1d_update_compat(*args, out=None, **kwargs):
    """Bridge the vLLM 0.25 update API, which lacks ``out``."""

    if _CAUSAL_CONV_UPDATE_HAS_OUT:
        return causal_conv1d_update(*args, out=out, **kwargs)
    result = causal_conv1d_update(*args, **kwargs)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _use_ext_causal_conv1d() -> bool:
    """Whether to dispatch Kimi KDA prefill/decode Conv1D to the external
    causal_conv1d package.

    Enabled by default; set VLLM_KIMI_CONV1D_BACKEND=0 to use vLLM Triton.
    spec-decode always stays on Triton because the external package lacks its
    state-update parameters.
    """
    return os.environ.get("VLLM_KIMI_CONV1D_BACKEND", "1") == "1"


_EXT_CAUSAL_CONV1D_CACHE = None


def _load_ext_causal_conv1d():
    """Lazily import the external causal_conv1d package (v1.5.4+das).

    Returns (causal_conv1d_fn_hcu, causal_conv1d_update_ext), or (None, None)
    when the package is unavailable (callers fall back to Triton).
    """
    global _EXT_CAUSAL_CONV1D_CACHE
    if _EXT_CAUSAL_CONV1D_CACHE is None:
        try:
            from causal_conv1d import causal_conv1d_fn_hcu
            from causal_conv1d import causal_conv1d_update as causal_conv1d_update_ext

            _EXT_CAUSAL_CONV1D_CACHE = (
                causal_conv1d_fn_hcu,
                causal_conv1d_update_ext,
            )
        except ImportError:
            logger.warning(
                "causal_conv1d package is unavailable; "
                "Kimi KDA Conv1D falls back to Triton"
            )
            _EXT_CAUSAL_CONV1D_CACHE = (None, None)
    return _EXT_CAUSAL_CONV1D_CACHE


def _use_flash_kda() -> bool:
    """Whether to dispatch Kimi KDA prefill to the external flash_kda package.

    Enabled by default; set VLLM_KIMI_FLASHKDA_BACKEND=0 to use the
    vendored Triton chunk kernels.
    """
    return os.environ.get("VLLM_KIMI_FLASHKDA_BACKEND", "1") == "1"


_FLASH_KDA_CACHE = None


def _load_flash_kda():
    """Lazily import the external flash_kda package (prebuilt HIP .so).

    Returns (flash_kda_fwd, get_workspace_size), or (None, None) when the
    package is unavailable (callers fall back to Triton).
    """
    global _FLASH_KDA_CACHE
    if _FLASH_KDA_CACHE is None:
        try:
            from flash_kda import fwd as flash_kda_fwd
            from flash_kda import get_workspace_size

            _FLASH_KDA_CACHE = (flash_kda_fwd, get_workspace_size)
        except ImportError:
            logger.warning(
                "flash_kda package is unavailable; "
                "Kimi KDA prefill falls back to Triton"
            )
            _FLASH_KDA_CACHE = (None, None)
    return _FLASH_KDA_CACHE


def a_log_weight_loader(
    shard_axis: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load KDA A_log stored as either old 4D or current 1D weights."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if loaded_weight.dim() == 4:
            assert loaded_weight.shape[:2] == (1, 1), (
                f"Expected old A_log shape (1, 1, H, 1), got {loaded_weight.shape}"
            )
            assert loaded_weight.shape[-1] == 1, (
                f"Expected old A_log last dim to be 1, got {loaded_weight.shape}"
            )
            loaded_weight = loaded_weight.view(loaded_weight.shape[2])

        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)

    return loader


def _make_fused_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
) -> Callable[..., None]:
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if (
            loaded_weight.dim() == 3
            and param.dim() == 3
            and param.shape[0] == loaded_weight.shape[-1]
            and param.shape[-1] != loaded_weight.shape[-1]
        ):
            # Legacy vLLM stores ColumnParallelLinear weights as
            # [kernel, 1, output], while K3 checkpoints use [output, 1, kernel].
            loaded_weight = loaded_weight.transpose(0, 2)
            shard_size = sharded_dims[loaded_shard_id]
            source_start = tp_rank * shard_size
            target_start = sum(sharded_dims[:loaded_shard_id])
            loaded_shard = loaded_weight[..., source_start : source_start + shard_size]
            target = param.data[..., target_start : target_start + shard_size]
            if target.shape != loaded_shard.shape:
                raise RuntimeError(
                    "Kimi conv1d loader shape mismatch: "
                    f"param={tuple(param.shape)} target={tuple(target.shape)} "
                    f"loaded={tuple(loaded_weight.shape)} shard={tuple(loaded_shard.shape)} "
                    f"shard_id={loaded_shard_id} tp={tp_size}"
                )
            target.copy_(loaded_shard)
            return
        if loaded_weight.dim() == 3 and param.dim() == 2:
            loaded_weight = loaded_weight.squeeze(1)
        elif loaded_weight.dim() == 2 and param.dim() == 3:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        if param.data[target_start : target_start + shard_size].shape != loaded_shard.shape:
            raise RuntimeError(
                "Kimi conv1d loader shape mismatch: "
                f"param={tuple(param.shape)} target="
                f"{tuple(param.data[target_start : target_start + shard_size].shape)} "
                f"loaded={tuple(loaded_weight.shape)} shard={tuple(loaded_shard.shape)} "
                f"shard_id={loaded_shard_id} tp={tp_size}"
            )
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)

    return weight_loader


class _KimiGDNMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with one output replicated across TP ranks.

    The replicated shard is represented as ``size * tp_size`` so the merged
    parameter reserves ``size`` local rows on every rank. Loading that shard
    from rank zero then gives every rank the complete checkpoint weight.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_id: int,
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_id = replicated_shard_id
        output_sizes = output_sizes.copy()
        output_sizes[replicated_shard_id] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)

        if self.use_full_rank_gate:
            # Keep f_a before the narrow beta shard, then pad each TP-local row
            # to select the aligned BF16 GEMM path. The padding also avoids an
            # Inductor correctness issue seen with the row-strided G view.
            qkvg_output_sizes = [self.projection_size] * 4
            in_proj_output_sizes = qkvg_output_sizes + [
                self.head_dim,
                self.num_heads,
            ]
            local_output_size = (
                4 * self.local_projection_size + self.head_dim + self.local_num_heads
            )
            self.in_proj_padding = -local_output_size % 16
            if self.in_proj_padding:
                in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        else:
            in_proj_output_sizes = [self.projection_size] * 3 + [
                self.num_heads,
                self.head_dim,
            ]
            self.in_proj_padding = 0
        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache let decode run a single conv update.
        # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        # HCU's NN-layout linear factory allocates [kernel, 1, output], while
        # the Kimi KDA kernels and checkpoint use [output, 1, kernel].
        if henvs.VLLM_USE_NN:
            self.conv1d.weight.data = (
                self.conv1d.weight.data.permute(1, 0).contiguous()
            )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_fused_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )
        self.use_safe_gate = self.gate_lower_bound is not None
        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        backend = "triton" if backend == "auto" else backend
        assert backend == "triton", (
            "The shared Kimi GDN layer only supports the Triton KDA "
            f"prefill backend, got {backend!r}."
        )
        if not self.use_full_rank_gate:
            self.g_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = mixed_qkv.shape[0]
        qkv = mixed_qkv.view(seq_len, 3, self.local_num_heads, self.head_dim)
        # Materialize all three row-strided inputs with one token-major to
        # QKV-major permutation. Each unbound tensor is then contiguous.
        qkv = qkv.permute(1, 0, 2, 3).contiguous().unsqueeze(1)
        return qkv.unbind(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        else:
            mixed_qkv, beta, f_a = projected_qkvgfab.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        # Vendor-specific KDA kernels: AMD/ROCm and NVIDIA keep their own copies
        # under kimi_k3/{amd,nvidia}/ops so each can diverge independently.
        # On ROCm the decode/recurrent kernels dispatch to boltops; the prefill
        # orchestration (chunk_kda_with_fused_gate) stays vendored in vLLM.
        if current_platform.is_rocm():
            from boltops.fla.kda import (
                fused_recurrent_kda,
                fused_recurrent_kda_packed_decode,
            )
            from vllm_hcu.models.kimi_k3.amd.ops.third_party.kda import (
                chunk_kda_with_fused_gate,
            )
        else:
            from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
                chunk_kda_with_fused_gate,
                fused_recurrent_kda,
                fused_recurrent_kda_packed_decode,
            )

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        m = attn_metadata_narrowed
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_sequence_masks = m.spec_sequence_masks
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        constant_caches = self.kv_cache

        conv_state, recurrent_state = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Split tokens into the multi-query spec-decode part and the remaining
        # (prefill / plain decode) part.
        if spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

        # ---------- spec-decode multi-query path ----------
        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)

            # Sibling beta and, for full-rank gates, output-gate views remain
            # live, so write the convolution output separately.
            spec_conv_out = torch.empty(
                mixed_qkv_spec.shape,
                dtype=mixed_qkv_spec.dtype,
                device=mixed_qkv_spec.device,
            )
            mixed_qkv_spec = _causal_conv1d_update_compat(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            # Spec-only batches write directly into core_attn_out.
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                raw_g=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                cu_seqlens=spec_cu_seqlens,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )

        # ---------- non-spec path (prefill or plain decode) ----------
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                # Packed prefill conv would require copying V solely to make
                # it dense for KDA. Separate calls accept the strided inputs
                # and produce dense Q/K/V without that extra traffic.
                # TODO: Use packed conv once every KDA prefill backend accepts
                # row-strided Q/K/V directly.
                def _prefill_conv(
                    x: torch.Tensor,
                    state: torch.Tensor,
                    weight: torch.Tensor,
                ) -> torch.Tensor:
                    if _use_ext_causal_conv1d():
                        hcu_fn, _ = _load_ext_causal_conv1d()
                        if hcu_fn is not None:
                            return hcu_fn(
                                x.transpose(0, 1),
                                weight,
                                None,
                                activation="silu",
                                initial_states=state,
                                has_initial_state=has_initial_state,
                                cache_indices=non_spec_state_indices_tensor,
                                query_start_loc=non_spec_query_start_loc,
                                seq_lens_cpu=non_spec_query_start_loc.diff()
                                .tolist(),
                            ).transpose(0, 1)
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )

                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                initial_state = gather_initial_states(
                    recurrent_state,
                    non_spec_state_indices_tensor,
                    has_initial_state,
                )
                # flash_kda (prebuilt HIP .so): a single fused prefill entry
                # point, enabled by default. It applies
                # qk-l2norm internally and requires K=V=128 plus a bounded gate.
                if (
                    _use_flash_kda()
                    and self.gate_lower_bound is not None
                    and self.head_dim == 128
                ):
                    flash_fwd, _ = _load_flash_kda()
                else:
                    flash_fwd = None

                if flash_fwd is not None:
                    logger.info_once(
                        "Using flash_kda KDA prefill backend "
                        "(VLLM_KIMI_FLASHKDA_BACKEND=1)."
                    )
                    core_attn_out_non_spec = torch.empty_like(v_ns)
                    last_recurrent_state = torch.empty_like(initial_state)
                    flash_fwd(
                        q_ns.contiguous(),
                        k_ns.contiguous(),
                        v_ns.contiguous(),
                        g1_ns.contiguous(),
                        beta_ns.contiguous(),
                        float(q_ns.shape[-1] ** -0.5),
                        core_attn_out_non_spec,
                        self.A_log.contiguous(),
                        self.dt_bias.view(
                            self.local_num_heads, self.head_dim
                        ).contiguous(),
                        float(self.gate_lower_bound),
                        initial_state=initial_state.contiguous(),
                        final_state=last_recurrent_state,
                        # flash_kda requires int64 cu_seqlens; vLLM metadata
                        # query_start_loc is int32.
                        cu_seqlens=non_spec_query_start_loc.to(
                            torch.int64
                        ).contiguous(),
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                # Init cache
                recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state

            else:
                # pure-decode non-spec batch
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                # Sibling beta and, for full-rank gates, output-gate views
                # remain live, so write the conv output separately.
                _, ext_update_fn = _load_ext_causal_conv1d()
                if _use_ext_causal_conv1d() and ext_update_fn is not None:
                    # The external kernel allocates its own output and does not
                    # overwrite the input, so no separate out buffer is needed.
                    mixed_qkv_ns = ext_update_fn(
                        mixed_qkv_ns,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=decode_conv_indices,
                    )
                else:
                    packed_conv_out = torch.empty(
                        mixed_qkv_ns.shape,
                        dtype=mixed_qkv_ns.dtype,
                        device=mixed_qkv_ns.device,
                    )
                    mixed_qkv_ns = _causal_conv1d_update_compat(
                        mixed_qkv_ns,
                        conv_state,
                        conv_weights,
                        self.conv1d.bias,
                        activation="silu",
                        conv_state_indices=decode_conv_indices,
                        validate_data=True,
                        out=packed_conv_out,
                    )
                core_attn_out_non_spec, _ = fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                )

        # ---------- merge spec and non-spec outputs ----------
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            # Mixed batches require indexed placement in the original order.
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0, :num_actual_tokens]
        elif core_attn_out_non_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_spec is not None
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))
