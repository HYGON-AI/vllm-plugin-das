# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU integration boundary for vLLM's Model Runner V2."""

import functools
import os
from contextlib import nullcontext

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm_hcu.runtime_compat.step5 import is_step5_config
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.model_executor.layers.attention.pcp import (
    replicated_mtp_batch_scope,
)
from vllm_hcu.forward_context_runtime import (
    deepep_auto_request_phase_scope,
    set_deepep_auto_request_phase,
)
from vllm_hcu.v1.pcp_manager import make_hcu_pcp_manager_cls
from vllm_hcu.patch.platform.core_fix.patch_vllm_config import is_glm53_pcp

logger = init_logger(__name__)

# Exact host mirror of num_computed_tokens / block tables (as vllm_hcu_21's V1
# runner keeps). Without it a host-scheduling attention builder (step4 DSA)
# reads seq_lens back from the device, which waits for the whole previous step
# -- the MTP drafter included -- and leaves the GPU idle at every step start.
_USE_HOST_SEQ_LENS = os.environ.get("VLLM_HCU_HOST_SEQ_LENS", "1") == "1"
# Debug: compare the mirror against the device seq_lens every step.
_CHECK_HOST_SEQ_LENS = os.environ.get("VLLM_HCU_CHECK_HOST_SEQ_LENS", "0") == "1"


def record_pp_spec_draft_index_stream(
    model_runner: object, input_batch: object | None
) -> bool:
    """Backport upstream PP draft index stream ownership fix (#55745)."""
    if getattr(model_runner, "_vllm_hcu_suppress_pp_spec_draft_sync", False):
        return False
    pp_handler = getattr(model_runner, "pp_handler", None)
    num_speculative_steps = int(getattr(model_runner, "num_speculative_steps", 0))
    if (
        pp_handler is None
        or num_speculative_steps == 0
        or input_batch is None
        or not getattr(pp_handler, "is_last_rank")
    ):
        return False

    getattr(input_batch, "idx_mapping").record_stream(
        getattr(pp_handler, "broadcast_stream")
    )
    return True


class HcuGPUModelRunnerV2(GPUModelRunner):
    """HCU compatibility adapter around upstream v0.25.1 Model Runner V2."""

    def __init__(self, vllm_config, device):
        self._hcu_pcp_manager_cls = None
        super().__init__(vllm_config, device)
        # Upstream creates the selected manager during KV-cache
        # initialization. Keep the adapter attribute available before that
        # lifecycle point without allocating a second manager.
        if not hasattr(self, "pcp_manager"):
            self.pcp_manager = None
        self._hsl_enabled = False
        self._hsl_pending = None
        self._hsl_batch = None

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load and pack GLM router weights in the worker weights pool."""

        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm_hcu.model_executor.layers.fused_moe.glm53_router_gemv import (
            bind_glm53_router_gates,
        )

        self._vllm_hcu_num_router_gates = bind_glm53_router_gates(
            getattr(self, "model", None)
        )

    @property
    def pcp_manager_cls(self):
        if self._hcu_pcp_manager_cls is None:
            self._hcu_pcp_manager_cls = make_hcu_pcp_manager_cls(self.vllm_config)
        return self._hcu_pcp_manager_cls

    def initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling=False,
        kv_cache_allocation_context=None,
    ):
        pcp_size = int(
            self.vllm_config.parallel_config.prefill_context_parallel_size
        )
        if (
            pcp_size > 1
            and len(kv_cache_config.kv_cache_groups) != 1
            and not is_glm53_pcp(self.vllm_config)
        ):
            raise ValueError(
                "HCU PCP requires exactly one KV cache group."
            )
        from vllm_hcu.v1.kv_cache import use_hcu_flash_kv_cache_allocator

        if is_glm53_pcp(self.vllm_config):
            from vllm_hcu.v1.glm53_pcp import install_glm53_pcp

            # Install before upstream discovers backends and checks PCP support.
            install_glm53_pcp(self)
        with use_hcu_flash_kv_cache_allocator(self):
            super().initialize_kv_cache(
                kv_cache_config,
                is_profiling=is_profiling,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
        if pcp_size > 1 and self.pcp_manager is None:
            raise RuntimeError(
                "official MRV2 did not initialize the selected HCU PCP manager"
            )
        self._init_host_seq_lens()

    # ------------------------------------------------------------------
    # Host mirror of num_computed_tokens.
    #
    # The device value changes in exactly two places on this path:
    # add_request (exact) and post_update (+= query_len - num_rejected). The
    # mirror replays both. num_rejected is copied to pinned memory right after
    # rejection sampling and applied lazily at the next attention metadata
    # build, so the host waits for the rejection sampler of the previous step
    # only, while the drafter keeps running. Configurations where the device
    # value moves in other ways (PP, pooling, adaptive verification
    # reallocating drafts on the device, PCP/DCP) keep the device read-back.
    # ------------------------------------------------------------------
    def _init_host_seq_lens(self) -> None:
        self._hsl_enabled = (
            _USE_HOST_SEQ_LENS
            and is_step5_config(self.vllm_config)
            and self.pcp_manager is None
            and not getattr(self, "use_dcp", False)
            and getattr(self, "pp_handler", None) is None
            and getattr(self, "pooling_runner", None) is None
            and getattr(self, "adaptive_verification", None) is None
        )
        self._hsl_pending = None
        self._hsl_batch = None
        if not self._hsl_enabled:
            return
        n = self.max_num_reqs
        self._hsl_exact = np.zeros(n, dtype=np.int64)
        # Slots (re)added since the pending correction was recorded: the
        # device post_update ran before their add_request, so skip them.
        self._hsl_fresh = np.zeros(n, dtype=bool)
        pin = torch.cuda.is_available()
        self._hsl_rejected = torch.zeros(n, dtype=torch.int32, pin_memory=pin)
        self._hsl_event = torch.cuda.Event() if pin else None
        self.block_tables.enable_host_mirror()
        self._hsl_wrap_model_state()

    def _hsl_wrap_model_state(self) -> None:
        model_state = self.model_state
        if getattr(model_state, "_vllm_hcu_hsl_wrapped", False):
            return
        original_prepare_attn = model_state.prepare_attn

        @functools.wraps(original_prepare_attn)
        def prepare_attn(input_batch, *args, **kwargs):
            if input_batch is self._hsl_batch and input_batch.seq_lens_cpu is None:
                self._hsl_resolve()
                self._hsl_attach(input_batch)
            return original_prepare_attn(input_batch, *args, **kwargs)

        model_state.prepare_attn = prepare_attn
        model_state._vllm_hcu_hsl_wrapped = True

    def _hsl_resolve(self) -> None:
        pending = self._hsl_pending
        if pending is not None:
            self._hsl_pending = None
            idx, delta, has_rejected = pending
            if has_rejected:
                if self._hsl_event is not None:
                    self._hsl_event.synchronize()
                delta = delta - self._hsl_rejected.numpy()[: idx.shape[0]]
            keep = ~self._hsl_fresh[idx]
            self._hsl_exact[idx[keep]] += delta[keep]
        self._hsl_fresh[:] = False

    def _hsl_record(self, input_batch, num_rejected) -> None:
        # One correction per sampled batch; a leftover one is applied first so
        # the pinned buffer is free to reuse.
        self._hsl_resolve()
        n = input_batch.num_reqs
        query_lens = np.diff(input_batch.query_start_loc_np[: n + 1]).astype(np.int64)
        has_rejected = input_batch.num_draft_tokens > 0 and num_rejected is not None
        if has_rejected:
            self._hsl_rejected[:n].copy_(num_rejected[:n], non_blocking=True)
            if self._hsl_event is not None:
                self._hsl_event.record()
        self._hsl_pending = (
            input_batch.idx_mapping_np.copy(),
            query_lens,
            has_rejected,
        )

    def _hsl_attach(self, input_batch) -> None:
        n = input_batch.num_reqs
        n_pad = input_batch.num_reqs_after_padding
        idx = input_batch.idx_mapping_np
        query_lens = np.diff(input_batch.query_start_loc_np[: n + 1])
        computed = np.zeros(n_pad, dtype=np.int32)
        seq_lens = np.zeros(n_pad, dtype=np.int32)
        computed[:n] = self._hsl_exact[idx]
        seq_lens[:n] = computed[:n] + query_lens
        if _CHECK_HOST_SEQ_LENS:
            ref = input_batch.seq_lens[:n_pad].cpu().numpy()
            if not np.array_equal(ref, seq_lens):
                bad = np.flatnonzero(ref != seq_lens)
                logger.error(
                    "host seq_lens mirror mismatch at rows %s: host %s device %s",
                    bad.tolist(),
                    seq_lens[bad].tolist(),
                    ref[bad].tolist(),
                )
                # Resync so a single slip does not persist.
                self._hsl_exact[idx] = ref[:n] - query_lens
                seq_lens[:] = ref
                computed[:n] = ref[:n] - query_lens
        input_batch.seq_lens_cpu = torch.from_numpy(seq_lens)
        input_batch.num_computed_tokens_cpu = torch.from_numpy(computed)

    def add_requests(self, scheduler_output) -> None:
        super().add_requests(scheduler_output)
        if not self._hsl_enabled:
            return
        req_id_to_index = self.req_states.req_id_to_index
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_index = req_id_to_index[new_req_data.req_id]
            self._hsl_exact[req_index] = new_req_data.num_computed_tokens
            self._hsl_fresh[req_index] = True

    def sample(self, hidden_states, input_batch, grammar_output):
        out = super().sample(hidden_states, input_batch, grammar_output)
        if self._hsl_enabled and input_batch is self._hsl_batch:
            self._hsl_batch = None
            self._hsl_record(input_batch, out[2])
        return out

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        input_batch = super().prepare_inputs(
            scheduler_output,
            batch_req_state,
            batch_desc,
        )
        if self._hsl_enabled:
            # The mirror is resolved lazily in model_state.prepare_attn (see
            # _hsl_wrap_model_state), so all host work before the metadata
            # build overlaps the previous step's rejection sampler.
            input_batch.seq_lens_cpu = None
            input_batch.num_computed_tokens_cpu = None
            self._hsl_batch = input_batch
        set_deepep_auto_request_phase(input_batch.is_prefilling_np)
        if is_glm53_pcp(self.vllm_config):
            from vllm_hcu.v1.glm53_pcp import bind_glm53_pcp_batch

            bind_glm53_pcp_batch(self.pcp_manager, input_batch)
        return input_batch

    @functools.wraps(GPUModelRunner.execute_model)
    def execute_model(self, *args, **kwargs):
        scope = nullcontext()
        dummy_scope = nullcontext()
        if is_glm53_pcp(self.vllm_config):
            from vllm_hcu.v1.glm53_pcp import glm53_pcp_step_scope

            scope = glm53_pcp_step_scope()
            if kwargs.get("dummy_run", False):
                dummy_scope = replicated_mtp_batch_scope()
        with deepep_auto_request_phase_scope(), scope, dummy_scope:
            if is_step5_config(getattr(self, "vllm_config", None)):
                from vllm_hcu.v1.step5_execution import execute_model

                return execute_model(self, *args, **kwargs)
            return super().execute_model(*args, **kwargs)

    def profile_run(self) -> None:
        """Profile PCP with the per-rank token budget plus routing slack."""

        pcp_size = int(self.vllm_config.parallel_config.prefill_context_parallel_size)
        # Upstream profiles the target and MTP prefill from the same dummy
        # batch. Runtime MTP sampling restores the global PCP batch before
        # propose(), so shrinking this shared budget would under-profile the
        # replicated draft prefill even though it is safe for the target.
        if (
            pcp_size <= 1
            or getattr(self, "speculator", None) is not None
            # GLM5Next KDA and K-pool see the full, replicated token batch.
            or is_glm53_pcp(self.vllm_config)
        ):
            return super().profile_run()

        original_max = self.max_num_tokens
        # Prefill tokens are partitioned, but decode tokens are replicated on
        # every PCP rank. A prefill request can also occupy two
        # DualChunkSwap segments, so reserve two tokens of per-request layout
        # headroom when that exceeds the decode width. This bounds every
        # rank-local mix while retaining the 12.5% prefill imbalance slack.
        partitioned_tokens = ((original_max + pcp_size - 1) // pcp_size) * 9 // 8
        replicated_request_tokens = min(
            original_max,
            int(self.max_num_reqs) * max(int(self.decode_query_len), 2),
        )
        self.max_num_tokens = min(
            original_max,
            max(1, partitioned_tokens + replicated_request_tokens),
        )
        try:
            return super().profile_run()
        finally:
            self.max_num_tokens = original_max

    def prepare_attn(self, input_batch):
        if self.pcp_manager is None:
            out = super().prepare_attn(input_batch)
            if self._hsl_enabled and input_batch is self._hsl_batch:
                input_batch.block_tables_np = self.block_tables.gather_block_tables_np(
                    input_batch.idx_mapping_np, input_batch.num_reqs_after_padding
                )
            return out
        return self.pcp_manager.prepare_attn(input_batch)

    def prepare_dummy_attn(self, input_batch, valid_state_slots=False):
        if is_glm53_pcp(self.vllm_config):
            manager = self.pcp_manager
            self.pcp_manager = None
            try:
                return super().prepare_dummy_attn(input_batch, valid_state_slots)
            finally:
                self.pcp_manager = manager
        if self.pcp_manager is None:
            return super().prepare_dummy_attn(input_batch, valid_state_slots)
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
            state_slots = torch.arange(
                1,
                input_batch.num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            for block_table in block_tables:
                block_table[:, 0].copy_(state_slots)
        slot_mappings = self.pcp_manager.get_dummy_slot_mappings(input_batch.num_tokens)
        return block_tables, slot_mappings

    def sample_tokens(self, grammar_output):
        execute_model_state = self.execute_model_state
        use_replicated_mtp_batch = (
            self.pcp_manager is not None
            and execute_model_state is not None
            and getattr(self, "speculator", None) is not None
        )
        restore_nonfinal_pp_batch = (
            self.pcp_manager is not None
            and execute_model_state is not None
            and not getattr(self, "is_last_pp_rank", True)
        )
        if use_replicated_mtp_batch or restore_nonfinal_pp_batch:
            (
                restored_hidden_states,
                restored_input_batch,
            ) = self.pcp_manager.restore_for_sampling(execute_model_state.hidden_states)
            execute_model_state = execute_model_state._replace(
                hidden_states=restored_hidden_states,
                input_batch=restored_input_batch,
            )
            self.execute_model_state = execute_model_state
        input_batch = (
            None if execute_model_state is None else execute_model_state.input_batch
        )
        scope = (
            replicated_mtp_batch_scope() if use_replicated_mtp_batch else nullcontext()
        )
        pcp_manager = self.pcp_manager
        record_pp_spec_draft_index_stream(self, input_batch)
        with scope:
            if use_replicated_mtp_batch:
                assert execute_model_state is not None
                assert input_batch is not None
                block_tables, slot_mappings = pcp_manager.prepare_global_attn()
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self.model_state.prepare_attn(
                    input_batch,
                    CUDAGraphMode.NONE,
                    block_tables,
                    slot_mappings,
                    self.attn_groups,
                    self.kv_cache_config,
                )
                execute_model_state = execute_model_state._replace(
                    attn_metadata=attn_metadata,
                    slot_mappings_by_layer=slot_mappings_by_layer,
                )
                self.execute_model_state = execute_model_state
            if use_replicated_mtp_batch:
                # The HCU path restored PCP state above so it could rebuild
                # global draft metadata. Upstream sample_tokens() now performs
                # the same restore itself, so hide the manager to avoid a
                # second gather/reorder of the global batch.
                self.pcp_manager = None
                try:
                    output = super().sample_tokens(grammar_output)
                finally:
                    self.pcp_manager = pcp_manager
            else:
                # Upstream restores PCP only on the final PP stage. Non-final
                # ranks need the global batch restored before PP receive.
                output = super().sample_tokens(grammar_output)
        return output


__all__ = [
    "HcuGPUModelRunnerV2",
    "record_pp_spec_draft_index_stream",
]
