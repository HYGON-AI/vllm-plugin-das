# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU integration boundary for vLLM's Model Runner V2."""

import functools
from contextlib import nullcontext

import torch

from vllm.config.compilation import CUDAGraphMode
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


def record_pp_spec_draft_index_stream(
    model_runner: object, input_batch: object | None
) -> bool:
    """Backport upstream PP draft index stream ownership fix (#55745)."""
    if getattr(model_runner, "_vllm_hcu_suppress_pp_spec_draft_sync", False):
        return False
    pp_handler = getattr(model_runner, "pp_handler", None)
    num_speculative_steps = int(
        getattr(model_runner, "num_speculative_steps", 0)
    )
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

    @property
    def pcp_manager_cls(self):
        if self._hcu_pcp_manager_cls is None:
            self._hcu_pcp_manager_cls = make_hcu_pcp_manager_cls(
                self.vllm_config
            )
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
        if pcp_size > 1 and len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError(
                "HCU PCP requires exactly one KV cache group."
            )
        from vllm_hcu.v1.kv_cache import use_hcu_flash_kv_cache_allocator

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

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        input_batch = super().prepare_inputs(
            scheduler_output,
            batch_req_state,
            batch_desc,
        )
        set_deepep_auto_request_phase(input_batch.is_prefilling_np)
        return input_batch

    @functools.wraps(GPUModelRunner.execute_model)
    def execute_model(self, *args, **kwargs):
        with deepep_auto_request_phase_scope():
            return super().execute_model(*args, **kwargs)

    def prepare_attn(self, input_batch):
        if self.pcp_manager is None:
            return super().prepare_attn(input_batch)
        return self.pcp_manager.prepare_attn(input_batch)

    def prepare_dummy_attn(self, input_batch, valid_state_slots=False):
        if self.pcp_manager is None:
            return super().prepare_dummy_attn(input_batch, valid_state_slots)
        block_tables = self.block_tables.get_dummy_block_tables(
            input_batch.num_reqs
        )
        if valid_state_slots:
            state_slots = torch.arange(
                1,
                input_batch.num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            for block_table in block_tables:
                block_table[:, 0].copy_(state_slots)
        slot_mappings = self.pcp_manager.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample_tokens(self, grammar_output):
        execute_model_state = self.execute_model_state
        use_replicated_mtp_batch = (
            self.pcp_manager is not None
            and execute_model_state is not None
            and getattr(self, "speculator", None) is not None
        )
        if use_replicated_mtp_batch:
            (
                restored_hidden_states,
                restored_input_batch,
            ) = self.pcp_manager.restore_for_sampling(
                execute_model_state.hidden_states
            )
            execute_model_state = execute_model_state._replace(
                hidden_states=restored_hidden_states,
                input_batch=restored_input_batch,
            )
            self.execute_model_state = execute_model_state
        input_batch = (
            None
            if execute_model_state is None
            else execute_model_state.input_batch
        )
        scope = (
            replicated_mtp_batch_scope()
            if use_replicated_mtp_batch
            else nullcontext()
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
                # Current upstream owns the ordinary PCP restore lifecycle.
                output = super().sample_tokens(grammar_output)
        return output


__all__ = [
    "HcuGPUModelRunnerV2",
    "record_pp_spec_draft_index_stream",
]
