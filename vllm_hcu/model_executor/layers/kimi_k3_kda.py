# SPDX-License-Identifier: Apache-2.0
"""BoltOps KDA prefill and vLLM-owned, separately allocated checkpoints.

The deployed 0.28 MambaManager reserves the penultimate block for an
internal prefill checkpoint and keeps the final block private for execution.
It owns publication, copy-on-write, eviction and request cleanup. This module
only exports states to those slots; it never invents block IDs or cache keys.
"""
from __future__ import annotations

import importlib
import inspect
import os
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache, wraps

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.hcu.kimi_k3_kda")
CHECKPOINT_KEY = "_vllm_hcu_kimi_kda_checkpoints"
ACTIVE_LAYER = ContextVar("hcu_kimi_kda_layer", default=None)
CHUNK_SIZE = 64


def prefill_backend():
    value = os.environ.get("VLLM_KIMI_KDA_PREFILL_BACKEND", "legacy").lower()
    if value not in ("legacy", "boltops"):
        raise ValueError("VLLM_KIMI_KDA_PREFILL_BACKEND must be legacy or boltops")
    return value


def cache_strategy():
    value = os.environ.get("VLLM_KIMI_KDA_CACHE_STRATEGY", "align").lower()
    if value not in ("align", "extra_buffer"):
        raise ValueError("VLLM_KIMI_KDA_CACHE_STRATEGY must be align or extra_buffer")
    if value == "extra_buffer" and prefill_backend() != "boltops":
        raise ValueError("KDA extra_buffer requires the boltops prefill backend")
    if prefill_backend() == "boltops" and os.environ.get(
        "VLLM_HCU_USE_CUSTOM_OPS", "1"
    ).lower() not in ("1", "true"):
        raise ValueError("BoltOps KDA requires VLLM_HCU_USE_CUSTOM_OPS=1")
    return value


def state_backend():
    value = os.environ.get("VLLM_KIMI_KDA_STATE_BACKEND", "boltops").lower()
    if value not in ("boltops", "aiter"):
        raise ValueError("VLLM_KIMI_KDA_STATE_BACKEND must be boltops or aiter")
    return value


@lru_cache(maxsize=1)
def aiter_state_kernel():
    module = importlib.import_module("aiter.ops.fla")
    name = "chunk_gated_delta_rule_fwd_sglang_hip_blockdim64"
    fn = getattr(module, name, None)
    required = {"k", "w", "u", "gk", "initial_state", "initial_state_indices",
                "cu_seqlens", "chunk_indices", "chunk_size", "use_exp2",
                "output_final_state", "save_new_value", "transpose_state_layout"}
    if not callable(fn) or not required <= inspect.signature(fn).parameters.keys():
        raise RuntimeError(f"Incompatible AITER KDA state interface: {name}")
    return fn


def _chunk_gated_delta_rule_fwd_h_aiter(
    *, k, w, u, gk, initial_state, cu_seqlens, chunk_indices
):
    """SGLang HIP path, with a private in-place FP32 final-state pool.

    The returned chunk history is BF16. Never pass the caller's state pool to
    AITER: the SGLang entry point always updates valid slots in place.
    """
    if k.shape[-1] != 128 or u.shape[-1] != 128:
        raise ValueError("AITER KDA state kernel requires K=V=128")
    logger.info_once("Kimi KDA executing BoltOps prefill with AITER state")
    final = initial_state.to(dtype=torch.float32).contiguous().clone()
    indices = torch.arange(final.shape[0], device=final.device, dtype=torch.int32)
    h, vn = aiter_state_kernel()(
        k=k, w=w, u=u, gk=gk, initial_state=final,
        initial_state_indices=indices, output_final_state=True,
        chunk_size=CHUNK_SIZE, save_new_value=True, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, use_exp2=True, transpose_state_layout=True,
    )
    if h.dtype != torch.bfloat16:
        raise RuntimeError(f"AITER KDA expected BF16 chunk history, got {h.dtype}")
    return h, vn, final


@lru_cache(maxsize=1)
def boltops_kernels():
    """Check named parameters; **kwargs on the public API discards exports."""
    chunk = importlib.import_module("boltops.fla.kda.triton.chunk")
    state = importlib.import_module("boltops.fla.gdn.triton.chunk_delta_h")
    requirements = {
        "fused_kda_gate_chunk_cumsum": (chunk, ("chunk_indices", "lower_bound")),
        "chunk_kda_fwd_intra": (chunk, ("chunk_indices", "safe_gate")),
        "recompute_w_u_fwd": (chunk, ("chunk_indices", "gk")),
        "chunk_gated_delta_rule_fwd_h": (state, (
            "chunk_indices", "use_exp2", "transpose_state_layout")),
        "chunk_gla_fwd_o_gk": (chunk, ("chunk_indices", "h", "o")),
    }
    for name, (owner, required) in requirements.items():
        fn = getattr(owner, name, None)
        if not callable(fn) or not set(required) <= inspect.signature(fn).parameters.keys():
            raise RuntimeError(f"Incompatible BoltOps KDA interface: {name}")
    fp32_h_launcher = getattr(
        state, "launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None
    )
    fp32_h_requirements = {
        "k", "u", "w", "v_new", "g", "gk", "h", "initial_state",
        "initial_state_indices", "final_state", "cu_seqlens", "chunk_offsets",
        "N", "T", "H", "Hg", "K", "V", "BT", "use_exp2",
        "transpose_state_layout", "inplace_final_state", "null_state_index",
        "kernel_cfg",
    }
    if (
        not callable(fp32_h_launcher)
        or not fp32_h_requirements <= inspect.signature(fp32_h_launcher).parameters.keys()
        or not callable(getattr(state, "prepare_chunk_offsets", None))
    ):
        raise RuntimeError("Incompatible BoltOps low-level KDA state launcher for FP32 checkpoints")
    logger.info("Kimi KDA prefill: BoltOps Triton gate/intra/GDN-state/output; chunk=64")
    return chunk, state


def install_checkpoint_spec(layer):
    original = getattr(layer, "get_kv_cache_spec", None)
    if original is None:
        return  # Lightweight adapter test doubles have no cache allocator.

    @wraps(original)
    def spec(self, config):
        result = original(self, config)
        strategy = cache_strategy()
        if prefill_backend() == "boltops":
            boltops_kernels()  # Fail at construction, before scheduler activation.
            if state_backend() == "aiter":
                aiter_state_kernel()
        if strategy != "extra_buffer":
            return result
        cache = config.cache_config
        if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
            raise ValueError("KDA extra_buffer requires prefix caching and mamba_cache_mode=align")
        if config.speculative_config is not None:
            raise ValueError("KDA extra_buffer currently requires speculative decoding disabled")
        parallel = config.parallel_config
        if (getattr(parallel, "decode_context_parallel_size", 1) != 1
                or getattr(parallel, "prefill_context_parallel_size", 1) != 1):
            raise ValueError("KDA extra_buffer requires DCP=PCP=1")
        if not hasattr(result, "num_prefill_checkpoint_blocks"):
            raise RuntimeError("vLLM lacks Mamba internal prefill checkpoint support")
        if cache.block_size % CHUNK_SIZE != 0:
            raise ValueError("KDA extra_buffer requires the attention cache block size divisible by 64")
        if cache.prefix_match_unit is not None and (
            cache.prefix_match_unit % CHUNK_SIZE or cache.block_size % cache.prefix_match_unit
        ):
            raise ValueError("KDA extra_buffer prefix_match_unit must divide block_size and be divisible by 64")
        logger.info_once("Kimi KDA cache: extra_buffer via MambaManager; alignment=64, extra blocks=1")
        # The cache config's original mamba_block_size is the 64-token update
        # stride. Align-mode's allocated checkpoint column, however, is indexed
        # by the scheduler/hash block (1536 for the K3 runtime layout). The
        # deployed MambaManager checkpoint index and Mamba forward block table
        # both consume MambaSpec.block_size, so bind it to that resolved block.
        configured = replace(
            result, block_size=cache.block_size,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=CHUNK_SIZE,
        )
        logger.info_once(
            "Kimi KDA cache spec: mode=extra_buffer, prefix_caching=True, "
            "attention_and_mamba_block=%d, state_update_chunk=%d, checkpoint_pages=%d",
            configured.block_size, CHUNK_SIZE,
            configured.num_prefill_checkpoint_blocks,
        )
        return configured

    spec._hcu_kimi_kda_checkpoint_spec = True
    layer.get_kv_cache_spec = spec


def validate_loaded_kda_model(model):
    """Fail closed if an explicitly selected KDA backend was not installed."""
    if prefill_backend() != "boltops":
        return
    import sys

    module = sys.modules.get("vllm.models.kimi_k3.amd.kda")
    layer = getattr(module, "KimiK3DeltaAttention", None)
    if (not getattr(module, "_vllm_hcu_kimi_reference_conv_applied", False)
            or layer is None
            or not getattr(layer.get_kv_cache_spec,
                           "_hcu_kimi_kda_checkpoint_spec", False)):
        raise RuntimeError("BoltOps KDA was selected but its model patch is not active")
    count = sum(isinstance(child, layer) for child in model.modules())
    if not count:
        raise RuntimeError("BoltOps KDA was selected but the model has no local KDA layers")
    logger.info("Kimi KDA selected: %d local layers, prefill=%s, state=%s, cache=%s",
                count, prefill_backend(), state_backend(), cache_strategy())


@dataclass
class Checkpoints:
    # Indices refer to the prefill-only recurrent chunk list; convolution
    # offsets refer to the non-spec decode-first input before convolution.
    slots: torch.Tensor
    h_indices: torch.Tensor
    conv_ends: tuple[int, ...]
    positions: tuple[int, ...]
    conv_snapshots: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    recurrent_target: torch.Tensor | None = None
    recurrent_values: torch.Tensor | None = None


def checkpoint_plan(boundaries, ends, *, num_decodes, block_size, hash_size):
    """CPU-only plan using exactly the scheduler/manager validity predicate."""
    from vllm.v1.kv_cache_interface import (
        get_mamba_prefill_checkpoint_position, is_mamba_prefill_checkpoint_valid,
    )
    rows, cols, h_indices, conv_ends, positions = [], [], [], [], []
    chunk_base = 0
    for row in range(num_decodes, len(boundaries) - 1):
        length = boundaries[row + 1] - boundaries[row]
        end = ends[row]
        start = end - length
        pos = get_mamba_prefill_checkpoint_position(end, hash_size, False)
        if is_mamba_prefill_checkpoint_valid(
            start, end, pos, hash_size, block_size, CHUNK_SIZE
        ):
            rows.append(row)
            cols.append((end + block_size - 1) // block_size - 2)
            # h is the state at the START of each chunk, so delta/64 is
            # the next chunk's index, not delta/64 - 1.
            h_indices.append(chunk_base + (pos - start) // CHUNK_SIZE)
            conv_ends.append(boundaries[row] + pos - start)
            positions.append(pos)
        chunk_base += (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    return rows, cols, h_indices, conv_ends, positions


def attach_checkpoints(builder, common, metadata):
    if cache_strategy() != "extra_buffer" or metadata.num_prefills == 0:
        return
    if metadata.spec_sequence_masks is not None:
        raise RuntimeError("KDA extra_buffer received speculative metadata")
    # Upper bounds are exact on prefill rows. Never synchronize GPU metadata.
    ends = getattr(common, "_seq_lens_cpu", None)
    if ends is None:
        ends = common.seq_lens_cpu_upper_bound
    if ends is None:
        raise RuntimeError("KDA extra_buffer requires CPU prefill sequence lengths")
    cache = builder.vllm_config.cache_config
    rows, cols, hi, ce, positions = checkpoint_plan(
        common.query_start_loc_cpu[:metadata.num_decodes + metadata.num_prefills + 1].tolist(),
        ends.tolist(),
        num_decodes=metadata.num_decodes,
        block_size=builder.kv_cache_spec.block_size,
        hash_size=cache.prefix_match_unit or cache.block_size,
    )
    if not rows:
        return
    device = common.block_table_tensor.device
    row_ids = torch.tensor(rows, dtype=torch.long, device=device)
    col_ids = torch.tensor(cols, dtype=torch.long, device=device)
    payload = Checkpoints(
        common.block_table_tensor[row_ids, col_ids],
        torch.tensor(hi, dtype=torch.long, device=device), tuple(ce), tuple(positions),
    )
    if os.environ.get("VLLM_KIMI_KDA_DEBUG", "0") == "1":
        logger.warning(
            "Kimi KDA checkpoint plan rows=%s cols=%s h_indices=%s positions=%s "
            "slots=%s boundaries=%s ends=%s",
            rows, cols, hi, positions, payload.slots.detach().cpu().tolist(),
            common.query_start_loc_cpu[:metadata.num_decodes + metadata.num_prefills + 1].tolist(),
            ends.tolist(),
        )
    if metadata.nums_dict is None:
        metadata.nums_dict = {}
    metadata.nums_dict[CHECKPOINT_KEY] = payload


def save_conv_checkpoints(x, conv_states, metadata):
    payload = (getattr(metadata, "nums_dict", None) or {}).get(CHECKPOINT_KEY)
    if payload is None:
        return
    # Kimi's forward normalizes the physical cache layout before splitting Q/K/V.
    # The convolution hook always receives [slot, channels, window].
    width = conv_states.shape[-1]
    # Save raw projected inputs before causal_conv1d may update them in place.
    indices = torch.tensor(
        [[end - width + i for i in range(width)] for end in payload.conv_ends],
        device=x.device, dtype=torch.long,
    )
    # causal_conv1d updates the running-state slots after this hook. Keep a
    # private snapshot and restore it only after the complete layer forward.
    values = x[:, indices].permute(1, 0, 2).to(conv_states.dtype).clone()
    payload.conv_snapshots.append((conv_states, values))


def finalize_checkpoints(metadata, layer=None):
    """Publish snapshots after normal conv and recurrent state writes finish."""
    payload = (getattr(metadata, "nums_dict", None) or {}).get(CHECKPOINT_KEY)
    if payload is None:
        return
    if os.environ.get("VLLM_KIMI_KDA_DEBUG", "0") == "1":
        recurrent_overwrite = None
        if payload.recurrent_values is not None:
            current = payload.recurrent_target[payload.slots].float()
            saved = payload.recurrent_values.float()
            recurrent_overwrite = float((current - saved).abs().max().item())
        state_indices = getattr(metadata, "prefill_state_indices", None)
        initial_states = getattr(metadata, "prefill_has_initial_state", None)
        logger.warning(
            "Kimi KDA checkpoint finalize layer=%s slots=%s positions=%s "
            "h_indices=%s recurrent_pre_restore_delta=%s conv_caches=%d "
            "prefill_state_indices=%s has_initial=%s",
            getattr(layer, "prefix", None), payload.slots.detach().cpu().tolist(),
            payload.positions, payload.h_indices.detach().cpu().tolist(),
            recurrent_overwrite, len(payload.conv_snapshots),
            state_indices.detach().cpu().tolist() if state_indices is not None else None,
            initial_states.detach().cpu().tolist() if initial_states is not None else None,
        )
    for conv_target, conv_values in payload.conv_snapshots:
        conv_target[payload.slots] = conv_values
    payload.conv_snapshots.clear()
    if payload.recurrent_values is not None:
        payload.recurrent_target[payload.slots] = payload.recurrent_values
        payload.recurrent_target = None
        payload.recurrent_values = None


def _chunk_gated_delta_rule_fwd_h_fp32(
    state, *, k, w, u, gk, initial_state, cu_seqlens, chunk_indices
):
    """Run the BoltOps state kernel with FP32 chunk-state storage.

    The kernel accumulates recurrent state in FP32. Its public helper normally
    allocates ``h`` like ``k`` (BF16 for Kimi), which rounds internal chunk
    boundaries before prefix-cache export. This path keeps the exported states
    FP32. The KDA output kernel casts loaded ``h`` to the query dtype before its
    dot product, preserving the existing BF16 output calculation.
    """
    B, T, Hg, K = k.shape
    H, V = u.shape[-2:]
    chunk_size = CHUNK_SIZE
    N = len(cu_seqlens) - 1
    NT = len(chunk_indices)
    chunk_offsets = state.prepare_chunk_offsets(cu_seqlens, chunk_size)
    state_shape = (N, H, V, K)
    h = torch.empty(
        (B, NT, H, V, K), device=k.device, dtype=torch.float32
    )
    v_new = torch.empty_like(u)
    final_state = torch.empty(state_shape, device=k.device, dtype=torch.float32)
    state.launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
        k=k,
        u=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=None,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        N=N,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=chunk_size,
        use_exp2=True,
        transpose_state_layout=True,
        inplace_final_state=False,
        null_state_index=-1,
        kernel_cfg=None,
    )
    return h, v_new, final_state


def chunk_kda(**kwargs):
    """Full BoltOps chain, exposing chunk-start h without global monkeypatches."""
    chunk, state = boltops_kernels()
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    if kwargs.get("use_qk_l2norm_in_kernel", False):
        q, k = (chunk.l2norm_fwd(x.contiguous()) for x in (q, k))
    else:
        q, k = q.contiguous(), k.contiguous()
    cu = kwargs["cu_seqlens"]
    ci = kwargs.get("chunk_indices")
    if ci is None:
        raise RuntimeError("BoltOps KDA requires scheduler-owned chunk_indices")
    scale = kwargs.get("scale") or k.shape[-1] ** -0.5
    g, beta = chunk.fused_kda_gate_chunk_cumsum(
        kwargs["raw_g"].contiguous(), raw_beta=kwargs["raw_beta"],
        A_log=kwargs["A_log"], g_bias=kwargs.get("g_bias"),
        lower_bound=kwargs.get("lower_bound"), cu_seqlens=cu,
        chunk_indices=ci, chunk_size=CHUNK_SIZE,
    )
    aqk, a = chunk.chunk_kda_fwd_intra(
        q=q, k=k, gk=g, beta=beta, scale=scale, cu_seqlens=cu,
        chunk_indices=ci, chunk_size=CHUNK_SIZE,
        safe_gate=kwargs.get("lower_bound") is not None,
    )
    w, u, _, kg = chunk.recompute_w_u_fwd(
        k=k, v=v.contiguous(), beta=beta, A=a, gk=g,
        cu_seqlens=cu, chunk_indices=ci,
    )
    active = ACTIVE_LAYER.get()
    payload = (
        (getattr(active[1], "nums_dict", None) or {}).get(CHECKPOINT_KEY)
        if active is not None else None
    )
    initial_state = kwargs["initial_state"].contiguous()
    if state_backend() == "aiter":
        h, vn, final = _chunk_gated_delta_rule_fwd_h_aiter(
            k=kg, w=w, u=u, gk=g, initial_state=initial_state,
            cu_seqlens=cu, chunk_indices=ci,
        )
    elif payload is not None:
        h, vn, final = _chunk_gated_delta_rule_fwd_h_fp32(
            state,
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            cu_seqlens=cu,
            chunk_indices=ci,
        )
    else:
        h, vn, final = state.chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g, initial_state=initial_state,
            output_final_state=True, cu_seqlens=cu, chunk_indices=ci,
            chunk_size=CHUNK_SIZE, use_exp2=True, transpose_state_layout=True,
        )
    output = chunk.chunk_gla_fwd_o_gk(
        q=q, v=vn, g=g, A=aqk, h=h, o=torch.empty_like(v), scale=scale,
        cu_seqlens=cu, chunk_indices=ci, chunk_size=CHUNK_SIZE,
    )
    if payload is not None:
        layer, _ = active
        recurrent = layer.kv_cache[1]
        payload.recurrent_target = recurrent
        payload.recurrent_values = h[0, payload.h_indices].to(
            recurrent.dtype
        ).clone()
    return output, final
