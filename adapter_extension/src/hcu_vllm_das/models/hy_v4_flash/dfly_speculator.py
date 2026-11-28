# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from typing import Any, NamedTuple

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    has_compiled_submodule,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

from .dfly import HYV4DFlyForCausalLM
from .dfly_inputs import prepare_dfly_inputs, write_dfly_query_slots
from .dfly_target_reuse import HYV4DFlyTargetReuse
from .fullgraph_runner import _admits_hisparse_graph

logger = init_logger(__name__)


def _dfly_capture_sizes(
    target_sizes: list[int] | None, query_width: int, max_num_reqs: int
) -> list[int]:
    """Core4ba PW sizes are NOT rounded by the manager; own aligned buckets.

    Always include draft capacity: short target prefills may expand to a larger
    query block. This does not change target admission or its capture buckets.
    """
    if query_width < 2 or max_num_reqs < 1:
        raise ValueError("DFly requires a positive capacity and query width >= 2")
    capacity = query_width * max_num_reqs
    return sorted({
        min(((n + query_width - 1) // query_width) * query_width, capacity)
        for n in (target_sizes or []) if n > 0
    } | {query_width, capacity})


def _dfly_padded_batch(count: int, query_width: int, max_num_reqs: int) -> int:
    if query_width < 2 or max_num_reqs < 1 or count <= 0 or count % query_width:
        raise ValueError("DFly graph token bucket must contain whole query blocks")
    batch = count // query_width
    if batch > max_num_reqs:
        raise ValueError("DFly padded query block exceeds allocated draft capacity")
    return batch


class DFlyCudaGraphManager(CudaGraphManager):
    """Keep captured metadata alive exactly as long as its owning graphs.

    Core4ba drops speculator manager attributes after throwaway-pool profiling.
    Storing this bank on the speculator would retain stale profiling buffers.
    """

    def __init__(self, config: VllmConfig, device: torch.device,
                 mode: CUDAGraphMode, decode_query_len: int):
        super().__init__(config, device, mode, decode_query_len)
        self.full_attention_states: dict[
            BatchExecutionDescriptor, tuple[dict[str, Any], dict[str, torch.Tensor]]
        ] = {}

    def release_graphs(self) -> None:
        super().release_graphs()
        self.full_attention_states.clear()


class DFlyProposeInputs(NamedTuple):
    """The `propose` inputs a DFly query block is built from: the target step's
    results and per-request sampling state."""

    query_start_loc: torch.Tensor
    positions: torch.Tensor
    idx_mapping: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    last_hidden_states: torch.Tensor
    # Copied to the speculator's fixed buffers, which graphs read.
    temperature: torch.Tensor
    seeds: torch.Tensor


class HYV4DFlyQueryModel(nn.Module):
    """Fixed-shape query computation; attention keeps its existing eager breaks."""

    def __init__(
        self,
        model: HYV4DFlyForCausalLM,
        draft_tokens: torch.Tensor,
        sample_fn: Callable[[int, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.query_width = model.model.query_width
        self.sample_fn = sample_fn
        self.register_buffer("draft_tokens", draft_tokens, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch = active.shape[0]
        embeddings = self.model.embed_input_ids(input_ids)
        embeddings[:: self.query_width] = self.model.model.mtp_fusion(
            embeddings[:: self.query_width], target_hidden
        )
        hidden = self.model(
            input_ids=input_ids, positions=positions, inputs_embeds=embeddings
        ).view(batch, self.query_width, -1)
        previous = input_ids.view(batch, self.query_width)[:, 0].long()
        for step in range(self.query_width - 1):
            corrected = self.model.apply_hidden_correction(
                hidden[:, step + 1], previous
            )
            previous = self.sample_fn(step, corrected)
            self.draft_tokens[:batch, step] = torch.where(active, previous, -1)
        return self.draft_tokens[:batch]


class HYV4DFlySpeculator(DraftModelSpeculator):
    can_reuse_target_attn_metadata = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        hisparse = _admits_hisparse_graph(vllm_config)
        if vllm_config.attention_config.hisparse_config is not None and not hisparse:
            raise NotImplementedError("HiSparse + DFly joint runtime remains unqualified")
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        spec = vllm_config.speculative_config
        assert spec is not None
        draft_cache_dtype = spec.kv_cache_dtype or cache_dtype
        target_dtype = "bfloat16" if cache_dtype == "auto" else cache_dtype
        draft_dtype = "bfloat16" if draft_cache_dtype == "auto" else draft_cache_dtype
        if draft_dtype != target_dtype:
            raise ValueError(
                "DFly draft and target must use the same MLA KV cache dtype; "
                "omit speculative_config.kv_cache_dtype to inherit the target"
            )
        self._hy4_pd_role = None
        if vllm_config.kv_transfer_config is not None and not hisparse:
            from ..hy4_standard.pd_policy import classify_standard_hy4_role
            self._hy4_pd_role = classify_standard_hy4_role(vllm_config)
        if (
            vllm_config.lora_config is not None
            or vllm_config.parallel_config.use_ubatching
        ):
            raise ValueError("DFly does not support LoRA or microbatching")
        # A target step can admit max_num_seqs short prefills, but DFly
        # expands every request into K+1 simultaneous queries. Size *all*
        # inherited draft buffers and the draft model/attention configuration
        # for that expansion before the Core constructor allocates them.
        # Never widen the target scheduler or its token admission budget.
        self._requested_cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        draft_vllm_config = copy(vllm_config)
        draft_vllm_config.scheduler_config = copy(vllm_config.scheduler_config)
        query_capacity = (
            vllm_config.scheduler_config.max_num_seqs * (spec.num_speculative_tokens + 1)
        )
        draft_vllm_config.scheduler_config.max_num_batched_tokens = (
            query_capacity if self._hy4_pd_role == "producer" else max(
                vllm_config.scheduler_config.max_num_batched_tokens, query_capacity,
            )
        )
        super().__init__(draft_vllm_config, device)
        self._hisparse_config_owner = vllm_config if hisparse else None
        config = copy(super().attn_vllm_config)
        config.model_config = copy(self.draft_model_config)
        config.parallel_config = copy(vllm_config.parallel_config)
        config.parallel_config.enable_expert_parallel = False
        config.parallel_config.is_moe_model = False
        if self._hy4_pd_role == "producer":
            from ..hy4_standard.pcp_draft import make_producer_draft_parallel
            config.parallel_config = make_producer_draft_parallel(vllm_config.parallel_config)
        if self._hy4_pd_role is not None:
            # The connector belongs to the actual target runner. Dense draft
            # construction and metadata must never acquire a second PD owner.
            config.kv_transfer_config = None
        config.attention_config = copy(config.attention_config)
        config.attention_config.sparse_mla_force_mqa = True
        config.attention_config.use_non_causal = False
        config.compilation_config = copy(config.compilation_config)
        # Do not share mutable bucket lists with the target.
        sizes = config.compilation_config.cudagraph_capture_sizes
        config.compilation_config.cudagraph_capture_sizes = (
            list(sizes) if sizes is not None else None
        )
        config.cache_config = vllm_config.cache_config
        config.quant_config = None
        self._draft_vllm_config = config
        self.num_query_per_req = self.num_speculative_steps + 1
        self.max_model_len_gpu = torch.tensor(self.max_model_len, dtype=torch.int32, device=device)
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.target_hidden_states = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self.active_requests = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        # The target batch row of each request's last accepted token, and the
        # position of its bonus (anchor) query.
        self.source_rows = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.anchors = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.mrope_positions = (
            torch.zeros(
                self.draft_model_config.mrope_num_dims,
                self.max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if self.uses_mrope
            else None
        )
        # Draft logits cache column of each step, as 0-d views.
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self.query_model: HYV4DFlyQueryModel | None = None
        self.cudagraph_manager: DFlyCudaGraphManager | None = None
        self.reuse: HYV4DFlyTargetReuse | None = None
        self._storage_validated = False
        self._slot_mappings: torch.Tensor | None = None

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        config = self._draft_vllm_config
        config.quant_config = VllmConfig.get_quantization_config(
            self.draft_model_config, get_draft_load_config(config)
        )
        scope = nullcontext()
        if self._hisparse_config_owner is not None:
            from .hisparse_planner_compat import hisparse_residency_scope

            scope = hisparse_residency_scope(
                self._hisparse_config_owner, construction_config=config, draft=True,
            )
        context = (
            set_current_vllm_config(config)
            if self._hy4_pd_role is not None else nullcontext()
        )
        with scope, context:
            draft = get_model(
                vllm_config=config,
                model_config=self.draft_model_config,
                load_config=get_draft_load_config(config),
                prefix="dfly",
            )
        if not isinstance(draft, HYV4DFlyForCausalLM):
            raise TypeError("DFly speculator requires HYV4DFlyForCausalLM")
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        maybe_share_target_embed(draft, draft.model, language_model.model)
        # v0.33 has the same sharing behavior inline in load_eagle_model.
        # This draft explicitly owns no LM head and requests no head quantization.
        head = get_target_lm_head(target_model, language_model)
        if head is not None:
            draft.lm_head = head
        if head is None or draft.model.embed_tokens is None:
            raise ValueError("DFly requires compatible target embedding and LM head")
        if (
            head.weight.dtype != self.dtype
            or draft.model.embed_tokens.weight.dtype != self.dtype
        ):
            raise ValueError("DFly requires BF16 shared embedding and LM head weights")
        draft.logits_processor.head_dtype = self.vllm_config.model_config.head_dtype
        self.reuse = HYV4DFlyTargetReuse(
            target_model, draft, self.vllm_config,
            capture_target_indices=self._hy4_pd_role != "producer",
        )
        self.query_model = HYV4DFlyQueryModel(
            draft, self.draft_tokens, self._sample_step
        )
        config.quant_config = draft.model.quant_config
        config.model_config.hf_config = draft.config
        config.model_config.hf_text_config = draft.config
        logger.info(
            "HYV4 DFly: %d layers, N=%d, TP=%d, DP=%d (TP-local draft), "
            "target index/KV=%s/%s, %s writable KV aliases, "
            "max length=%d, draft quantization=%s, draft sampling=%s",
            len(draft.model.layers),
            self.num_speculative_steps,
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.data_parallel_size,
            draft.config.target_index_layer,
            draft.config.target_kv_layers,
            config.cache_config.cache_dtype,
            self.max_model_len,
            self.draft_model_config.quantization or "none",
            self.speculative_config.draft_sample_method,
        )
        return draft

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return self._draft_vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        context = (
            set_current_vllm_config(self._draft_vllm_config)
            if self._hy4_pd_role is not None else nullcontext()
        )
        with context:
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
        assert self.reuse is not None
        self.reuse.validate_groups(kv_cache_config)
        self._storage_validated = False
        cache_groups = kv_cache_config.kv_cache_groups
        if self.vllm_config.attention_config.hisparse_config is not None:
            slots = block_tables.slot_mappings
            if (slots.ndim != 2 or slots.dtype != torch.int64
                    or slots.shape[0] != len(cache_groups)):
                raise ValueError("HiSparse DFly slot mapping layout mismatch")
            # Core binds the writers after set_attn. Give target and draft the
            # same persistent storage even when draft queries outnumber the
            # target scheduling budget.
            if slots.shape[1] < self.max_num_tokens:
                slots = slots.new_full((len(cache_groups), self.max_num_tokens), PAD_SLOT_ID)
                block_tables.slot_mappings = slots
            self._slot_mappings = slots
            # Resident groups need draft slots even without a draft attention layer.
            self._query_slot_groups = tuple(gid for gid, group in enumerate(cache_groups)
                                            if group.kv_cache_spec.uses_slot_mapping)
        else:
            self._slot_mappings = torch.full(
                (len(cache_groups), self.max_num_tokens),
                PAD_SLOT_ID,
                dtype=torch.int64,
                device=self.device,
            )
            self._query_slot_groups = tuple(gid for gid, groups in enumerate(self.attn_groups) if groups)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Called AFTER set_attn: the draft has its own backend support/capacity.
        requested = self._requested_cudagraph_mode
        if (not self.vllm_config.model_config.enforce_eager
                and requested.has_piecewise_cudagraphs()
                and not cudagraph_mode.has_piecewise_cudagraphs()):
            raise RuntimeError(
                "DFly target PIECEWISE was downgraded; enable a compiled target "
                "or set VLLM_USE_BREAKABLE_CUDAGRAPH=1 before worker startup"
            )
        if self.draft_model_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and self.attn_cg_support.min_cg_support.value
                < AttentionCGSupport.UNIFORM_BATCH.value):
            raise RuntimeError("DFly draft attention does not support uniform FULL graphs")
        config = self._draft_vllm_config
        compilation = config.compilation_config
        # Both config and the list are draft-owned. Core's PW candidates retain
        # sizes literally; non-W-aligned buckets would truncate request rows.
        compilation.cudagraph_mode = cudagraph_mode
        compilation.cudagraph_capture_sizes = _dfly_capture_sizes(
            self.vllm_config.compilation_config.cudagraph_capture_sizes,
            self.num_query_per_req,
            self.max_num_reqs,
        )
        compilation.max_cudagraph_capture_size = max(compilation.cudagraph_capture_sizes)
        self.cudagraph_manager = DFlyCudaGraphManager(
            config, self.device, cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )
        if cudagraph_mode.has_piecewise_cudagraphs():
            assert self.query_model is not None
            if self.cudagraph_manager.use_breakable_cg:
                # Capture the complete query wrapper with its eager breaks.
                self.cudagraph_manager.init_breakable_cg_runner(self.query_model)
            elif not has_compiled_submodule(self.query_model):
                # Match Core's actual PW capability test. A compiled inner
                # backbone is also valid, but fusion/correction/sampling outside
                # that module remain eager in that route (not wrapper graphs).
                raise RuntimeError(
                    "DFly PIECEWISE requires a torch-compiled query submodule "
                    "or VLLM_USE_BREAKABLE_CUDAGRAPH=1"
                )

    @torch.inference_mode()
    def capture(self) -> None:
        manager = self.cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        assert self.reuse is not None
        self.reuse.validate_storage()

        def create_forward_fn(
            desc: BatchExecutionDescriptor, warmup: bool
        ) -> Callable[[CUDAGraphMode], None]:
            count = desc.num_tokens
            batch = _dfly_padded_batch(
                count, self.num_query_per_req, self.max_num_reqs
            )
            self.block_tables.get_dummy_block_tables(batch)
            # An all-padding block: no K/V writes, no draft logits writes.
            self._prepare_query(
                self._dummy_propose_inputs(), 0, batch, write_slots=True
            )
            # A FULL graph keeps the metadata it was captured with, so capture
            # it for the longest sequences it may replay.
            self.draft_max_seq_len = (
                self.max_model_len
                if desc.cg_mode == CUDAGraphMode.FULL
                else self.num_query_per_req
            )
            metadata, slots = self._query_attention_state(
                desc, torch.zeros(batch, dtype=torch.int32, device="cpu")
            )
            if desc.cg_mode == CUDAGraphMode.FULL and not warmup:
                # The manager requests fresh metadata AFTER warmup, then
                # captures forward(NONE). NONE is not evidence of eager mode.
                manager.full_attention_states[desc] = (metadata, slots)

            def forward(cg_mode: CUDAGraphMode) -> None:
                self._run_query_model(count, metadata, slots, cg_mode)

            return forward

        try:
            manager.capture(
                create_forward_fn, progress_bar_desc="Capturing DFly CUDA graphs"
            )
        except Exception:
            # Do not retain partially captured graphs/scheduler owners or allow
            # accidental recapture over a live bank. Initialization must fail.
            manager.graphs.clear()
            if manager.breakable_cg_runner is not None:
                manager.breakable_cg_runner.clear_graphs()
            manager.full_attention_states.clear()
            raise

    def _dummy_propose_inputs(self) -> DFlyProposeInputs:
        """Placeholder inputs for dummy runs and capture; never read with
        `num_reqs=0`."""
        return DFlyProposeInputs(
            query_start_loc=self.input_buffers.query_start_loc,
            positions=self.input_buffers.positions,
            idx_mapping=self.idx_mapping,
            num_sampled=self.source_rows,
            num_rejected=self.source_rows,
            last_sampled=self.draft_tokens,
            last_hidden_states=self.target_hidden_states,
            temperature=self.temperature,
            seeds=self.seeds,
        )

    def _prepare_query(
        self,
        inputs: DFlyProposeInputs,
        num_reqs: int,
        padded_batch: int,
        write_slots: bool,
    ) -> None:
        """Fill the query block, its slots and the draft layers' sparse indices.
        Rows past `num_reqs` are padding, so `num_reqs=0` gives an all-padding
        block."""
        assert self.reuse is not None
        rope_state = getattr(getattr(self, "model_state", None), "rope_state", None)
        self.max_model_len_gpu.fill_(self.max_model_len)
        prepare_dfly_inputs(
            input_ids=self.input_buffers.input_ids,
            positions=self.input_buffers.positions,
            is_padding=self.input_buffers.is_padding,
            query_start_loc=self.input_buffers.query_start_loc,
            seq_lens=self.input_buffers.seq_lens,
            active=self.active_requests,
            source_rows=self.source_rows,
            anchors=self.anchors,
            target_hidden_out=self.target_hidden_states,
            mrope_positions=self.mrope_positions,
            draft_idx_mapping=self.idx_mapping,
            out_temperature=self.temperature,
            out_seeds=self.seeds,
            target_query_start_loc=inputs.query_start_loc,
            target_positions=inputs.positions,
            idx_mapping=inputs.idx_mapping,
            num_sampled=inputs.num_sampled,
            num_rejected=inputs.num_rejected,
            last_sampled=inputs.last_sampled,
            target_hidden=inputs.last_hidden_states,
            mrope_delta=rope_state.prefill_delta.gpu if rope_state else None,
            temperature=inputs.temperature,
            seeds=inputs.seeds,
            max_model_len=self.max_model_len_gpu,
            mask_token_id=self.model.config.mask_token_id,
            query_width=self.num_query_per_req,
            num_reqs=num_reqs,
            padded_batch=padded_batch,
        )
        if write_slots:
            self._write_query_slots(padded_batch)
        self.reuse.update_indices(
            self.active_requests, self.source_rows, self.anchors, padded_batch
        )

    def _write_query_slots(self, padded_batch: int) -> None:
        assert self._slot_mappings is not None
        for gid in self._query_slot_groups:
            write_dfly_query_slots(
                self._slot_mappings[gid],
                self.input_buffers.positions,
                self.active_requests,
                self.block_tables.input_block_tables[gid],
                self.block_tables.block_sizes[gid],
                self.max_model_len_gpu,
                self.num_query_per_req,
                padded_batch,
            )

    def _sample_step(self, step: int, hidden: torch.Tensor) -> torch.Tensor:
        """The draft tokens at query offset `step + 1` of every row."""
        if self.draft_logits is None:
            return self.model.compute_logits(hidden).argmax(dim=-1)
        batch = hidden.shape[0]
        # Draws are keyed by the position before the predicted token.
        pos = self.anchors[:batch] + step
        return gumbel_sample(
            self.model.compute_logits(hidden),
            self.idx_mapping[:batch],
            self.temperature,
            self.seeds,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _query_attention_state(
        self, desc: BatchExecutionDescriptor, seq_lens_cpu_upper_bound: torch.Tensor
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
        assert self._slot_mappings is not None
        count = desc.num_tokens
        batch = _dfly_padded_batch(count, self.num_query_per_req, self.max_num_reqs)
        full = desc.cg_mode == CUDAGraphMode.FULL
        # Unlike the inherited helper, this passes the actual Core4ba capture
        # hook. FULL executes forward(NONE), so its Python bounds must already
        # cover future contexts; runtime batch objects cannot change the graph.
        qsl_cpu = torch.arange(batch + 1, dtype=torch.int32) * self.num_query_per_req
        upper_bound = (seq_lens_cpu_upper_bound[:batch] + self.num_query_per_req).clamp(
            max=self.max_model_len
        )
        if full:
            upper_bound.fill_(self.max_model_len)
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=batch,
            num_tokens=count,
            query_start_loc_gpu=self.input_buffers.query_start_loc[:batch + 1],
            query_start_loc_cpu=qsl_cpu,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:batch],
            max_seq_len=self.max_model_len if full else self.draft_max_seq_len,
            block_tables=[x[:batch] for x in self.block_tables.input_block_tables],
            slot_mappings=self._slot_mappings[:, :count],
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=upper_bound,
            positions=self.input_buffers.positions[:count],
            is_prefilling=self.draft_is_prefilling[:batch],
            for_cudagraph_capture=full,
            causal=True,
        )
        slots = build_slot_mappings_by_layer(
            self._slot_mappings[:, :count], self.kv_cache_config
        )
        return metadata, slots

    def _refresh_full_graph_attention(
        self, desc: BatchExecutionDescriptor, metadata: dict[str, Any] | None,
        slots: dict[str, torch.Tensor] | None,
    ) -> None:
        manager = self.cudagraph_manager
        if (manager is None or desc not in manager.full_attention_states
                or metadata is None or slots is None):
            raise RuntimeError("DFly FULL graph has no captured attention state")
        captured, captured_slots = manager.full_attention_states[desc]
        if captured.keys() != metadata.keys() or captured_slots.keys() != slots.keys():
            raise RuntimeError("DFly FULL graph attention layer ownership changed")
        # Rebuilding metadata refreshes shared builder buffers (e.g. req IDs)
        # after another bucket. Copy any non-alias tensor into the CAPTURED
        # tensor; never swap pointers or replace its shape-specific scheduler.
        def refresh(old: torch.Tensor, new: torch.Tensor) -> None:
            if (old.shape != new.shape or old.dtype != new.dtype
                    or old.device != new.device or old.stride() != new.stride()):
                raise RuntimeError("DFly FULL graph metadata tensor contract changed")
            if old.data_ptr() != new.data_ptr():
                old.copy_(new)

        for name, old in captured.items():
            new = metadata[name]
            if type(old) is not type(new):
                raise RuntimeError("DFly FULL graph metadata type changed")
            for field in ("num_reqs", "num_actual_tokens", "max_query_len", "max_seq_len",
                          "num_decodes", "num_prefills", "num_decode_tokens",
                          "decode_max_query_len", "fp8_use_mixed_batch"):
                if getattr(old, field, None) != getattr(new, field, None):
                    raise RuntimeError(f"DFly FULL graph metadata changed: {field}")
            for field in ("query_start_loc", "seq_lens", "slot_mapping",
                          "block_table", "req_id_per_token"):
                old_tensor, new_tensor = getattr(old, field, None), getattr(new, field, None)
                if isinstance(old_tensor, torch.Tensor):
                    if not isinstance(new_tensor, torch.Tensor):
                        raise RuntimeError(f"DFly FULL graph tensor disappeared: {field}")
                    refresh(old_tensor, new_tensor)
        for name, old in captured_slots.items():
            refresh(old, slots[name])

    def _run_query_model(
        self,
        count: int,
        metadata: dict[str, Any] | None,
        slots: dict[str, torch.Tensor] | None,
        cg_mode: CUDAGraphMode,
    ) -> None:
        assert self.query_model is not None
        batch = _dfly_padded_batch(
            count, self.num_query_per_req, self.max_num_reqs
        )
        positions = (
            self.input_buffers.positions[:count]
            if self.mrope_positions is None
            else self.mrope_positions[:, :count]
        )
        model_inputs = dict(
            input_ids=self.input_buffers.input_ids[:count],
            positions=positions,
            target_hidden=self.target_hidden_states[:batch],
            active=self.active_requests[:batch],
        )
        with set_forward_context(
            metadata,
            self._draft_vllm_config,
            num_tokens=count,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=count),
            slot_mapping=slots,
            is_padding=self.input_buffers.is_padding[:count],
        ):
            if cg_mode == CUDAGraphMode.PIECEWISE:
                assert self.cudagraph_manager is not None
                self.cudagraph_manager.run_pw_graph(self.query_model, model_inputs)
            else:
                self.query_model(**model_inputs)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # Core's dummy target step is capped by the target token budget.
        # Profile the independent worst-case draft shape, including temporary
        # activations/workspaces, rather than only those target dummy requests.
        if self._hy4_pd_role == "producer":
            from ..hy4_standard.pcp_draft import require_producer_zero_drafts
            require_producer_zero_drafts(dummy_run=dummy_run, is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens)
            # The owned scheduler admits only K=0 producer requests. Dummy
            # profiling must represent that same path, not allocate/execute
            # a multi-step draft which no live producer request can reach.
            if dummy_run:
                num_speculative_tokens = 0
        batch = self.max_num_reqs if dummy_run and is_profile else input_batch.num_reqs
        from .dfly_topology import validate_rank_local_dfly_step
        validate_rank_local_dfly_step(
            self.dp_size, self.dp_rank, dp_sync_state,
            local_reqs=input_batch.num_reqs, dummy_run=dummy_run)
        query_width = self.num_query_per_req
        count = batch * query_width
        if batch > self.max_num_reqs or count > self.max_num_tokens:
            raise ValueError("DFly query block exceeds allocated draft capacity")
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        assert self.reuse is not None
        descriptor = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch)
        if not dummy_run and self.cudagraph_manager is not None:
            descriptor = self.cudagraph_manager.dispatch(
                batch, count, uniform_token_count=query_width, num_active_loras=0
            )
        padded_count = descriptor.num_tokens
        padded_batch = _dfly_padded_batch(
            padded_count, query_width, self.max_num_reqs
        )
        if padded_batch > self.max_num_reqs or padded_count > self.max_num_tokens:
            raise ValueError("DFly padded query block exceeds allocated draft capacity")
        if not dummy_run and not self._storage_validated:
            self.reuse.validate_storage()
            self._storage_validated = True

        with_attn = not (dummy_run and skip_attn_for_dummy_run)
        if dummy_run:
            inputs, num_reqs = self._dummy_propose_inputs(), 0
        else:
            inputs = DFlyProposeInputs(
                query_start_loc=input_batch.query_start_loc,
                positions=input_batch.positions,
                idx_mapping=input_batch.idx_mapping,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                last_hidden_states=last_hidden_states,
                temperature=temperature,
                seeds=seeds,
            )
            num_reqs = batch
        self._prepare_query(inputs, num_reqs, padded_batch, write_slots=with_attn)

        metadata = None
        draft_slots = None
        if with_attn:
            upper_bound = torch.zeros(padded_batch, dtype=torch.int32, device="cpu")
            target_batch = input_batch.num_reqs
            upper_bound[:target_batch].copy_(
                input_batch.seq_lens_cpu_upper_bound[:target_batch]
            )
            self.draft_max_seq_len = min(
                int(upper_bound.max()) + query_width, self.max_model_len
            )
            metadata, draft_slots = self._query_attention_state(descriptor, upper_bound)

        if descriptor.cg_mode == CUDAGraphMode.FULL:
            # The inputs, slots, indices and metadata above fill the persistent
            # buffers the graph reads.
            assert self.cudagraph_manager is not None
            self._refresh_full_graph_attention(descriptor, metadata, draft_slots)
            self.cudagraph_manager.run_fullgraph(descriptor)
        else:
            self._run_query_model(
                padded_count, metadata, draft_slots, descriptor.cg_mode
            )
        return result
