# SPDX-License-Identifier: Apache-2.0
"""MiniMax-specific PCP: local queries/MLPs, replicated logical KV, EP tokens.

The generic upstream DualChunkSwap planner and sampling restoration are reused.
Only MiniMax cache writes gather current K/V/index-K; queries remain rank-local.
"""
import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from .minimax31_prefill import (enabled as _prefill_enabled,
                                validate_policy as _validate_prefill_policy,
                                reserve_pack_rows)

from vllm.distributed.parallel_state import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd import model as stock
from vllm.models.minimax_m3.amd.model_state import MiniMaxM3ModelState
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from .minimax31_static_runtime import guard_model_forward

OWNER = "minimax31-pcp-ep"
ARCH = "MiniMaxM3SparseForConditionalGeneration"
_MODEL_SHA = "ffd0d5724579a68640a377a482a77e79c3d17e3114e5f38e117183179faf0f68"
_original_attention = stock.Attention
_original_dense = stock.MiniMaxM3Attention
_original_sparse = stock.MiniMaxM3SparseAttention
_original_indexer = stock.MiniMaxM3Indexer
_INDEXER_SHA = "318df992d4c216e7ae48655d879bbd8231d838eeca7567dab37f473b67a65d7b"
_backends = {}
_registration = None


def is_target(config):
    return (getattr(config.parallel_config, "prefill_context_parallel_size", 1) > 1
            and ARCH in (getattr(config.model_config.hf_config, "architectures", ()) or ()))


def validate_scope(config, *, allow_text_model=False):
    pc = config.parallel_config
    actual = tuple(getattr(pc, field) for field in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "prefill_context_parallel_size", "decode_context_parallel_size"))
    if actual != (1, 1, 1, 8, 1) or not pc.enable_expert_parallel:
        raise ValueError("MiniMax PCP extension requires TP1 PP1 DP1 PCP8 EP8 DCP1")
    text_target = (allow_text_model and "MiniMaxM3SparseForCausalLM" in
                   (getattr(config.model_config.hf_config, "architectures", ()) or ()))
    if not (is_target(config) or text_target) or not config.use_v2_model_runner:
        raise ValueError("MiniMax PCP requires the actual MiniMax model and V2 runner")
    if not config.model_config.enforce_eager:
        raise ValueError("MiniMax PCP is validated for eager only")
    mm = config.model_config.multimodal_config
    if mm is None or not mm.language_model_only:
        raise ValueError("MiniMax PCP requires explicit --language-model-only")
    if config.lora_config is not None or config.speculative_config is not None:
        raise ValueError("MiniMax PCP does not support LoRA or speculative decoding")
    if pc.enable_eplb or pc.expert_placement_strategy != "linear":
        raise ValueError("MiniMax PCP requires linear placement without EPLB")
    if pc.all2all_backend != "allgather_reducescatter":
        raise ValueError("MiniMax PCP requires allgather_reducescatter")
    allowed = ("auto", "bfloat16", "fp8")
    if config.cache_config.cache_dtype not in allowed:
        raise ValueError("MiniMax PCP KV dtype is outside this experiment's scope")


def pcp_backend(base):
    if base not in _backends:
        # This declaration is paired with the model's actual cache-input gather.
        # The original backend math and metadata builders are retained.
        _backends[base] = type(
            "MiniMaxPCP" + base.__name__, (base,),
            {"supports_pcp": classmethod(lambda cls: True), "__module__": __name__})
    return _backends[base]


class MiniMax31PCPAttentionLayer(_original_attention):
    def __init__(self, *args, **kwargs):
        if kwargs.get("attn_backend") is not None:
            raise ValueError("unexpected foreign MiniMax dense attention backend")
        kwargs["attn_backend"] = pcp_backend(TritonAttentionBackend)
        super().__init__(*args, **kwargs)


def gather_cache_rows(rows, slots):
    # GroupCoordinator.all_gather consumes rows on the calling stream before
    # a later pack may overwrite scratch; no async handle/source escapes here.
    group = get_pcp_group()
    if group.world_size != 8 or rows.ndim != 2 or slots.ndim != 1:
        raise ValueError("invalid PCP cache-input group or row/slot layout")
    if slots.shape[0] != rows.shape[0] * group.world_size:
        raise ValueError("PCP gathered slot mapping does not match padded local rows")
    return group.all_gather(rows.contiguous(), dim=0)


def validate_fp8_sparse_cache(layer, slots):
    """Validate FP8 main-KV and the selected BF16 or FP8 index cache.

    Quantization is performed by ``triton_reshape_and_cache_flash`` after
    the PCP gather.  The stock sparse attention implementation reads the same
    per-layer descales. Check values on every call: changing a scale while
    a cache is bound would invalidate values already quantized into that cache.
    """
    if layer.kv_cache_dtype != "fp8" or layer.kv_cache.numel() == 0:
        return
    cache = layer.kv_cache
    expected_shape = (layer.num_kv_heads, 2 * layer.head_dim)
    if (cache.ndim != 4 or cache.shape[1] != expected_shape[0]
            or cache.shape[3] != expected_shape[1]
            or cache.dtype != layer.kv_cache_torch_dtype):
        raise ValueError("FP8 MiniMax PCP main KV cache layout/dtype mismatch")
    if slots.ndim != 1 or slots.device != cache.device:
        raise ValueError("FP8 MiniMax PCP slot mapping layout/device mismatch")
    index_cache = layer.indexer.index_cache
    index_dtype = torch.float8_e4m3fn if getattr(layer, "_hcu_fp8_index", False) else torch.bfloat16
    if (index_cache.dtype != index_dtype
            or (index_cache.kv_cache.numel()
                and index_cache.kv_cache.dtype != index_dtype)):
        raise ValueError("MiniMax PCP indexer cache dtype does not match index mode")
    index_storage = index_cache.kv_cache
    if index_dtype == torch.float8_e4m3fn and index_storage.numel():
        if (index_storage.ndim != 3 or tuple(index_storage.shape[1:]) != (128, 128)
                or index_storage.device != cache.device or index_storage.stride(-1) != 1):
            raise ValueError("FP8 index side cache layout/device mismatch")
        binding = (index_storage.data_ptr(), tuple(index_storage.shape),
                   tuple(index_storage.stride()), index_storage.device, index_storage.dtype)
        previous = getattr(layer, "_hcu_fp8_index_binding", None)
        if previous is not None and previous != binding:
            raise ValueError("FP8 index cache changed; construct a fresh engine and prefix")
        layer._hcu_fp8_index_binding = binding
    scales = []
    for name in ("_k_scale", "_v_scale"):
        scale = getattr(layer, name, None)
        if (not isinstance(scale, torch.Tensor) or scale.numel() != 1
                or scale.dtype != torch.float32
                or scale.device != cache.device):
            raise ValueError("FP8 MiniMax PCP %s must be a local FP32 scalar" % name)
        scales.append(scale)
    if getattr(layer, "_hcu_native_msa", False) and _prefill_enabled():
        from .minimax31_prefill import check_bound_scales
        if check_bound_scales(layer):
            return
    values = tuple(float(scale.item()) for scale in scales)
    for name, value in zip(("_k_scale", "_v_scale"), values):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("FP8 MiniMax PCP %s must be finite and positive" % name)
    checked = getattr(layer, "_hcu_fp8_checked", None)
    if checked is not None and checked[0] == cache.data_ptr() and checked[1] != values:
        raise ValueError("FP8 MiniMax PCP scales changed while KV cache is bound")
    layer._hcu_fp8_checked = (cache.data_ptr(), values)
    if getattr(layer, "_hcu_native_msa", False) and _prefill_enabled():
        from .minimax31_prefill import bind_validated_scales
        bind_validated_scales(layer)


class MiniMax31PCPDenseAttention(_original_dense):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        stock.ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, self.q_norm.weight, self.k_norm.weight,
            self.rotary_emb.cos_sin_cache, positions,
            self.num_heads, self.num_kv_heads, self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon, kv_cache_dtype="auto")
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        slots_by_layer = get_forward_context().slot_mapping
        if isinstance(slots_by_layer, dict) and self.attn.layer_name in slots_by_layer:
            slots = slots_by_layer[self.attn.layer_name]
            kv = gather_cache_rows(qkv[:, self.q_size:], slots)
            k, v = kv.split(self.kv_size, dim=-1)
        output = self.attn(q, k, v)
        output, _ = self.o_proj(output)
        return output


class MiniMax31NativeIndexCache(_original_indexer):
    """Scoped metadata/cache owner; FP8 score is exclusively native MSA.

    The vLLM neutral indexer rejects FP8 on ROCm because its compute path is
    CUDA-only. Here the existing Triton metadata builder and side-cache spec
    are retained; the Triton forward is deliberately unavailable for FP8.
    """
    def __init__(self, **kwargs):
        dtype = kwargs.get("indexer_kv_dtype", "bf16")
        if dtype not in ("fp8", "fp8_e4m3"):
            self._hcu_native_index_only = False
            super().__init__(**kwargs)
            return
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        validate_scope(config, allow_text_model=True)
        from .minimax31_config import artifacts
        artifacts(config).require("msa_extension_dir")
        if config.cache_config.cache_dtype != "fp8":
            raise ValueError("FP8 index requires FP8 main cache")
        if (kwargs.get("topk_blocks") != 16 or kwargs.get("sparse_block_size") != 128
                or kwargs.get("num_kv_heads") != 4 or kwargs.get("num_index_heads") != 4
                or kwargs.get("index_head_dim") != 128 or kwargs.get("score_type") != "max"):
            raise ValueError("FP8 index shape/score outside native MSA contract")
        from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerTritonImpl
        torch.nn.Module.__init__(self)
        self._hcu_native_index_only = True
        self.impl = MiniMaxM3IndexerTritonImpl(**kwargs)

    def forward(self, *args, **kwargs):
        if self._hcu_native_index_only:
            raise RuntimeError("FP8 index cache may only be read by native MSA")
        return super().forward(*args, **kwargs)


class MiniMax31PCPSparseAttention(_original_sparse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .minimax31_config import artifacts
        deployment = artifacts()
        self._hcu_msa_extension_dir = deployment.msa_extension_dir
        self._hcu_msa_stage2_receipt = deployment.msa_stage2_receipt
        self._hcu_native_msa = self._hcu_msa_extension_dir is not None
        self._hcu_fp8_index = self.indexer.index_cache.dtype == torch.float8_e4m3fn
        if self._hcu_fp8_index and not self._hcu_native_msa:
            raise ValueError("FP8 index requires a matching native MSA artifact")
        if self._hcu_native_msa and (self.kv_cache_dtype != "fp8" or self.use_aiter_sparse_pa):
            raise ValueError("native MSA requires FP8 PCP and the scoped non-AITER cache")
        if getattr(self, "use_nvfp4_kv", False) or self.indexer_emits_table:
            raise ValueError("MiniMax PCP preserves the existing non-NVFP4 Triton indexer")
        self.attn_backend = pcp_backend(self.attn_backend)
        cache = self.indexer.index_cache
        cache.backend_cls = pcp_backend(cache.backend_cls)

    def _try_compact_cache_write(self, qkv, slots):
        # Both maps already describe the rank-major PCP gather. Keep the main
        # and index slot maps independent, including their -1 padding.
        rows = qkv.shape[0]
        if (rows not in (2048, 4096, 6144, 8192) or not self._hcu_native_msa
                or not self._hcu_fp8_index or not _prefill_enabled()
                or self.skip_index_topk or self.use_aiter_sparse_pa
                or not self.kv_cache.numel() or qkv.dtype != torch.bfloat16
                or qkv.stride(1) != 1 or self.kv_size != 512
                or qkv.size(1) != self.q_size + self.index_q_size + 1152):
            return False
        from .minimax31_msa import ensure_native_cache
        from .minimax31_prefill_ops import pcp_write_rows, prefill_fp8_transport

        side = self.indexer.index_cache
        binding = ensure_native_cache(self)
        main_slots, index_slots = slots[self.layer_name], slots[side.prefix]
        if not pcp_write_rows.supports_layout(
            qkv, rows * 8, main_slots, index_slots, binding.key_writer, side.kv_cache,
        ):
            return False
        encoded = prefill_fp8_transport.gather_fp8(self, qkv, main_slots, gather_cache_rows)
        pcp_write_rows.write(encoded, main_slots, index_slots, binding.key_writer,
                             side.kv_cache, tile_rows=16, kv_groups=8, warps=8)
        return True

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        n = qkv.shape[0]
        slots = get_forward_context().slot_mapping
        if not isinstance(slots, dict) or self.layer_name not in slots:
            return qkv.new_zeros((n, self.hidden_size))
        validate_fp8_sparse_cache(self, slots[self.layer_name])
        query = qkv.new_empty((n, self.q_size))
        index_query = None
        common = dict(num_index_heads=self.num_idx_heads, q_out=query,
                      kv_cache_dtype=self.kv_cache_dtype)
        if self.skip_index_topk:
            common["skip_index_branch"] = True
        else:
            # The FP8 candidate converts the BF16 fused output explicitly.
            index_query = qkv.new_empty((n, self.index_q_size),
                dtype=torch.bfloat16 if self._hcu_fp8_index else self.indexer.index_cache.dtype)
            common.update(index_q_norm_weight=self.index_q_norm.weight,
                          index_k_norm_weight=self.index_k_norm.weight,
                          index_q_out=index_query)
        # The native kernel writes normalized/RoPE K and index-K back into qkv.
        # Cache insertion is deliberately deferred until all PCP rows are gathered.
        stock.ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, self.q_norm.weight, self.k_norm.weight,
            self.rotary_emb.cos_sin_cache, positions,
            self.num_heads, self.num_kv_heads, self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon, **common)
        if not self._try_compact_cache_write(qkv, slots):
            kv_end = self.q_size + 2 * self.kv_size
            rows = qkv[:, self.q_size:kv_end]
            if not self.skip_index_topk:
                index_k_start = kv_end + self.index_q_size
                if self._hcu_native_msa and _prefill_enabled():
                    from .minimax31_prefill import pack_rows
                    rows = pack_rows((rows, qkv[:, index_k_start:]))
                else:
                    rows = torch.cat((rows, qkv[:, index_k_start:]), dim=-1)
            gathered = gather_cache_rows(rows, slots[self.layer_name])
            k = gathered[:, :self.kv_size].view(-1, self.num_kv_heads, self.head_dim)
            v = gathered[:, self.kv_size:2*self.kv_size].view_as(k)
            index_k = None if self.skip_index_topk else gathered[:, 2*self.kv_size:]
            if self.use_aiter_sparse_pa:
                key_cache, value_cache = self.get_aiter_sparse_pa_kv_cache()
                page_slots = self._get_aiter_sparse_pa_slot_mapping(
                    slots[self.layer_name], key_cache, value_cache)
                self._insert_aiter_sparse_pa_kv(
                    k, v, index_k, page_slots,
                    None if index_k is None else slots[self.indexer.index_cache.prefix],
                    key_cache, value_cache)
            elif self.kv_cache.numel():
                from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
                    triton_reshape_and_cache_flash)
                if self._hcu_native_msa:
                    from .minimax31_msa import ensure_native_cache
                    binding = ensure_native_cache(self)
                    keys, values = binding.key_writer, binding.value_writer
                else:
                    keys, values = self.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
                fused_store = (self._hcu_native_msa and index_k is not None
                               and _prefill_enabled())
                if fused_store:
                    from .minimax31_prefill_triton import store_pcp_cache
                    store_pcp_cache(gathered, slots[self.layer_name],
                        slots[self.indexer.index_cache.prefix], keys, values,
                        self.indexer.index_cache.kv_cache, self._k_scale, self._v_scale)

                else:
                    triton_reshape_and_cache_flash(
                        k, v, keys, values, slots[self.layer_name], self.kv_cache_dtype,
                        self._k_scale, self._v_scale)
                if index_k is not None and not fused_store:
                    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
                        minimax_m3_insert_index_cache)
                    if self._hcu_fp8_index:
                        # Unit dequantization scale; saturate before the stock
                        # scatter converts BF16 to E4M3FN. Prefix and append use
                        # this same path, including PCP gathered rows.
                        index_k = index_k.clamp(-448.0, 448.0)
                    minimax_m3_insert_index_cache(
                        index_k, self.indexer.index_cache.kv_cache,
                        slots[self.indexer.index_cache.prefix])
        if self._hcu_native_msa:
            from .minimax31_msa import run_native_attention
            if self._hcu_fp8_index and index_query is not None:
                index_query = index_query.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            output = run_native_attention(self, query, index_query)
        else:
            output = self._run_attention(query, index_query, torch.empty_like(query))
        output, _ = self.o_proj(output)
        return output


class MiniMax31PCPManager(PCPManager):
    def __init__(self, *args, **kwargs):
        # The explicit legacy EP runner gathers before routing. Its padding
        # view must therefore use the same rank-major global row layout.
        kwargs["moe_gathers_tokens"] = False
        self._hcu_native_msa = False
        self._hcu_static = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def validate_config(vllm_config, supports_mm_inputs):
        # This is a new GQA/sparse implementation, not an MLA capability bypass.
        # The explicit language-only model configuration rejects media inputs.
        validate_scope(vllm_config)

    def prepare_inputs_for_capture_or_profile(self, input_batch):
        _validate_prefill_policy()
        local = super().prepare_inputs_for_capture_or_profile(input_batch)
        if self._hcu_static:
            from .minimax31_static_runtime import set_forward_layout
            set_forward_layout(local)
        return local

    def partition_batch(self, input_batch, batch_desc):
        _validate_prefill_policy()
        local = super().partition_batch(input_batch, batch_desc)
        # Reserve the static frame before mutating shared MSA metadata.
        if self._hcu_static:
            from .minimax31_static_runtime import set_forward_layout
            set_forward_layout(local)
        if self._hcu_native_msa:
            from .minimax31_msa import set_forward_layout
            set_forward_layout(local)
        return local


def empty_rank_attention_batch(batch):
    """A masked one-query attention view for a rank with zero real tokens.

    It changes only metadata building, never the planner, routing padding,
    token ownership, saved global batch, or sampling restoration.
    """
    if batch.num_tokens != 0:
        return batch
    if batch.num_tokens_after_padding < 1 or batch.num_reqs != 1:
        raise ValueError("unexpected empty-rank PCP padding contract")
    query_start = batch.query_start_loc.new_tensor([0, 1])
    return replace(
        batch, num_tokens=1, num_scheduled_tokens=np.ones(1, dtype=np.int32),
        query_start_loc=query_start, query_start_loc_np=np.array([0, 1], dtype=np.int32),
        seq_lens=batch.seq_lens.new_ones(1),
        seq_lens_cpu_upper_bound=torch.ones(1, dtype=torch.int32),
        num_computed_tokens_np=np.zeros(1, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(1, dtype=np.int32),
        is_prefilling_np=np.zeros(1, dtype=np.bool_), has_prefill=False,
        max_query_len=1)


class MiniMax31PCPModelState(MiniMaxM3ModelState):
    def prepare_attn(self, input_batch, *args, **kwargs):
        return super().prepare_attn(empty_rank_attention_batch(input_batch),
                                    *args, **kwargs)


class MiniMax31PCPForConditionalGeneration(stock.MiniMaxM3SparseForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        validate_scope(vllm_config)
        _validate_prefill_policy()  # freeze/report the selected model policy before loading
        from .minimax31_static_contract import model_construction_scope
        with model_construction_scope(vllm_config.model_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def get_model_state_cls():
        return MiniMax31PCPModelState


def profile_local_token_budget(global_tokens, max_num_reqs, world_size=8):
    """Conservative per-rank DualChunkSwap rows for PCP-only execution.

    Each prefill contributes at most 2*ceil(q/(2*world_size)) local
    rows; owned single-token decodes fit the same 2-row/request slack.
    Extra 256-row rounding keeps the memory probe conservative. This
    changes only the synthetic no-attention probe, never the scheduler.
    """
    if (type(global_tokens) is not int or global_tokens < 1
            or type(max_num_reqs) is not int or max_num_reqs < 1
            or world_size != 8):
        raise ValueError("unsupported MiniMax PCP profiling budget")
    bound = math.ceil(global_tokens / world_size) + 2 * min(max_num_reqs, global_tokens)
    return min(global_tokens, math.ceil(bound / 256) * 256)


class MiniMax31PCPModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .minimax31_config import artifacts
        from .minimax31_msa import prepare_native_prefill
        from .minimax31_static_contract import from_model_config
        if self.pcp_manager is not None:
            deployment = artifacts(self.vllm_config)
            prepare_native_prefill(deployment)
            self.pcp_manager._hcu_native_msa = deployment.msa_extension_dir is not None
            self.pcp_manager._hcu_static = from_model_config(self.vllm_config.model_config) is not None

    @guard_model_forward
    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)

    @torch.inference_mode()
    @guard_model_forward
    def _dummy_run(self, num_tokens, *args, **kwargs):
        if kwargs.get("is_profile", False):
            _validate_prefill_policy()
            manager = self.pcp_manager
            if (args or kwargs.get("skip_attn") is not True
                    or kwargs.get("uniform_decode", False)
                    or manager.pcp_world_size != 8 or manager.dcp_world_size != 1
                    or not manager.shard_decode_requests
                    or self.speculator is not None):
                raise RuntimeError("unsupported MiniMax PCP synthetic memory probe")
            local_tokens = profile_local_token_budget(num_tokens, self.max_num_reqs)
            if manager._hcu_native_msa and _prefill_enabled():
                # Only the memory probe may grow persistent pack scratch.
                # The shared dummy/capture preparation also runs after sizing KV.
                reserve_pack_rows(local_tokens, self.device)
            print("MINIMAX_PCP_PROFILE_BUDGET " + json.dumps({
                "global_scheduler_budget": num_tokens, "local_probe_rows": local_tokens,
                "max_num_reqs": self.max_num_reqs, "scheduler_unchanged": True}), flush=True)
            num_tokens = local_tokens
        return super()._dummy_run(num_tokens, *args, **kwargs)

    @property
    def pcp_manager_cls(self):
        return MiniMax31PCPManager


def install_minimax31_pcp():
    global _registration
    from vllm import ModelRegistry
    from hcu_adapter.bridges.vllm_model_runner import register_model_runner_factory
    if hashlib.sha256(Path(inspect.getfile(stock)).read_bytes()).hexdigest() != _MODEL_SHA:
        raise RuntimeError("MiniMax PCP model source changed; revalidate cache gather semantics")
    if _registration is not None:
        check_health()
        return
    from vllm.models.minimax_m3.common import indexer as indexer_module
    if (hashlib.sha256(Path(inspect.getfile(indexer_module)).read_bytes()).hexdigest() != _INDEXER_SHA
            or stock.MiniMaxM3Indexer is not _original_indexer):
        raise RuntimeError("MiniMax indexer ABI changed")
    if (stock.Attention is not _original_attention
            or stock.MiniMaxM3Attention is not _original_dense
            or stock.MiniMaxM3SparseAttention is not _original_sparse):
        raise RuntimeError("foreign MiniMax model factory modification")
    # Only this model module's explicit factories are changed, at startup.
    # No global Attention implementation or community source file is modified.
    from .minimax31_ep_runner import install_minimax31_ep_runner
    install_minimax31_ep_runner(stock)
    stock.Attention = MiniMax31PCPAttentionLayer
    stock.MiniMaxM3Attention = MiniMax31PCPDenseAttention
    stock.MiniMaxM3SparseAttention = MiniMax31PCPSparseAttention
    stock.MiniMaxM3Indexer = MiniMax31NativeIndexCache
    ModelRegistry.register_model(ARCH, MiniMax31PCPForConditionalGeneration)
    _registration = register_model_runner_factory(
        OWNER, matches=is_target, create=MiniMax31PCPModelRunner)
    check_health()


def check_health():
    from vllm import ModelRegistry
    from hcu_adapter.bridges.vllm_model_runner import require_model_runner_factory
    if _registration is None:
        raise RuntimeError("MiniMax PCP not installed")
    require_model_runner_factory(_registration)
    from .minimax31_ep_runner import check_minimax31_ep_runner
    check_minimax31_ep_runner(stock)
    registered = ModelRegistry.models.get(ARCH)
    if stock.MiniMaxM3Indexer is not MiniMax31NativeIndexCache:
        raise RuntimeError("MiniMax native index cache factory ownership changed")
    if (getattr(registered, "model_cls", None) is not MiniMax31PCPForConditionalGeneration
            or stock.Attention is not MiniMax31PCPAttentionLayer
            or stock.MiniMaxM3Attention is not MiniMax31PCPDenseAttention
            or stock.MiniMaxM3SparseAttention is not MiniMax31PCPSparseAttention):
        raise RuntimeError("MiniMax PCP factory ownership changed")
