# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    CUDAGraphMode,
    ExecuteModelState,
    InputBatch,
    IntermediateTensors,
    MambaSpec,
    ModelRunnerOutput,
    SchedulerOutput,
    UBatchState,
    build_slot_mappings_by_layer,
    get_num_active_loras_for_dispatch,
    make_cudagraph_stats,
    requires_raw_input_tokens,
    set_active_mm_loras,
    set_dummy_context,
    set_forward_context,
    torch,
)

from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp


@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: IntermediateTensors | None = None,
    dummy_run: bool = False,
    skip_attn_for_dummy_run: bool = False,
    is_profile: bool = False,
    context_len: int = 0,
    valid_dummy_state_slots: bool = False,
) -> ModelRunnerOutput | IntermediateTensors | None:
    if not dummy_run:
        # Update the request states.
        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()
        if scheduler_output.total_num_scheduled_tokens == 0:
            # No need to run the model.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)

    # Get batch descriptor and sync across DP ranks.
    num_reqs = len(scheduler_output.num_scheduled_tokens)
    num_toks = scheduler_output.total_num_scheduled_tokens
    max_query_len = max(scheduler_output.num_scheduled_tokens.values())
    batch_req_state, uniform_tok_count = self.gather_batch_req_state(
        scheduler_output, dummy_run
    )
    if batch_req_state is not None:
        num_toks = batch_req_state.num_tokens
        if self.pcp_manager is not None:
            num_toks = self.pcp_manager.get_num_tokens_for_dispatch(
                batch_req_state.num_scheduled_tokens,
                batch_req_state.is_prefilling_np,
            )

    num_active_loras = 0
    if self.lora_config:
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        num_active_loras = get_num_active_loras_for_dispatch(
            self.lora_config, self.lora_state, req_ids, dummy_run
        )

    skip_compiled = False
    if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
        # Encoder-decoder models such as Whisper should run eager/non-compiled
        # when encoder inputs are scheduled, because this step updates
        # cross-attention cache with dynamic encoder outputs.
        skip_compiled = True

    batch_desc, dp_sync = dispatch_cg_and_sync_dp(
        self.cudagraph_manager,
        num_reqs,
        num_toks,
        uniform_tok_count,
        self.dp_size,
        self.dp_rank,
        max_query_len=max_query_len,
        need_eager=is_profile or skip_compiled,
        num_active_loras=num_active_loras,
        parallel_config=self.parallel_config,
        allow_ubatching=(
            self.ubatch_runner is not None and not skip_attn_for_dummy_run
        ),
        uniform_decode=uniform_tok_count == self.decode_query_len,
    )

    if batch_desc.num_tokens == 0:
        # All DP ranks have zero tokens to run.
        empty_output = self.kv_connector.no_forward(scheduler_output)
        return self._merge_ec_connector_no_forward(scheduler_output, empty_output)

    cudagraph_stats = None
    if not dummy_run:
        # Common case.
        # Prepare all the inputs and copy to the input buffers.
        if self.observability_config.cudagraph_metrics:
            cudagraph_stats = make_cudagraph_stats(batch_desc, num_toks)
        assert batch_req_state is not None
        input_batch = self.prepare_inputs(scheduler_output, batch_req_state, batch_desc)
        block_tables, slot_mappings = self.prepare_attn(input_batch)
        # Mamba "align" pre-copy: migrate recurrent state across block
        # boundaries before the forward. Runs only on real batches, and
        # before model_state.prepare_attn gathers num_accepted_tokens so the
        # boundary reset is visible to the attention metadata.
        self.model_state.preprocess_state(
            input_batch,
            block_tables,
            self.kv_cache_config,
            self.req_states.num_computed_tokens.gpu,
        )

        if self.lora_config:
            # Activate LoRA adapters.
            lora_inputs = self.lora_state.make_lora_inputs(
                input_batch.req_ids,
                input_batch.idx_mapping_np,
                input_batch.num_scheduled_tokens,
            )
            self._set_active_loras(*lora_inputs)
    else:
        # No actual tokens to run. A dummy run for DP or memory profiling.
        dummy_num_reqs = batch_desc.num_reqs or num_reqs
        input_batch = InputBatch.make_dummy(
            dummy_num_reqs,
            batch_desc.num_tokens,
            self.input_buffers,
            max_query_len=batch_desc.max_query_len,
        )
        if not skip_attn_for_dummy_run:
            block_tables, slot_mappings = self.prepare_dummy_attn(
                input_batch, valid_dummy_state_slots
            )
            if context_len:
                set_dummy_context(
                    input_batch,
                    self.block_tables,
                    context_len,
                    self.kv_cache_config.num_blocks,
                    self.max_model_len,
                )
        else:
            assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                "Attention metadata must be prepared for dummy runs when using "
                "FULL cudagraph mode."
            )
            block_tables = None
            slot_mappings = None

    # Embeddings do not read the attention metadata. Building them first
    # lets their host work overlap the previous step on the device, ahead
    # of metadata builders that may wait for it (exact seq_lens).
    input_ids = input_batch.input_ids
    inputs_embeds = None
    ec_connector_output = None
    if self.uses_inputs_embeds and self.is_first_pp_rank:
        # Prepare inputs_embeds (MM encoder outputs and/or prompt_embeds
        # overlay). Only first PP rank prepares them.
        if dummy_run:
            # Obtain embeddings of correct shape for compiled model.
            inputs_embeds = self.model_state.dummy_inputs_embeds(
                input_batch.num_tokens_after_padding
            )
        else:
            scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
            if self.supports_mm_inputs and self.lora_config is not None:
                set_active_mm_loras(
                    model=self.model,
                    lora_manager=self.lora_manager,
                    encoder_cache=self.encoder_cache,
                    req_id_to_index=self.req_states.req_id_to_index,
                    lora_state=self.lora_state,
                    scheduled_encoder_inputs=scheduled_encoder_inputs,
                )
            with self.ec_connector.maybe_get_output(
                scheduler_output
            ) as ec_connector_output:
                inputs_embeds = self.model_state.prepare_inputs_embeds(
                    scheduled_encoder_inputs, input_batch, self.req_states
                )
        if inputs_embeds is not None and not requires_raw_input_tokens(self.model):
            input_ids = None

    attn_metadata = None
    slot_mappings_by_layer = None
    ubatch_state: UBatchState | None = None
    if batch_desc.num_ubatches > 1:
        assert self.ubatch_runner is not None
        assert block_tables is not None and slot_mappings is not None
        ubatch_state = self.ubatch_runner.prepare(
            input_batch, block_tables, slot_mappings
        )
    elif not (dummy_run and skip_attn_for_dummy_run):
        assert slot_mappings is not None
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        assert block_tables is not None
        attn_groups = self.attn_groups
        if dummy_run and is_profile:
            # Mamba layers take a cheap warmup path with no metadata;
            # attention metadata is still built so those kernels tune.
            attn_groups = [
                [g for g in groups if not isinstance(g.kv_cache_spec, MambaSpec)]
                for groups in attn_groups
            ]
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            batch_desc.cg_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            self.kv_cache_config,
            # FULL replay reads capture-time metadata buffers. Re-stage them
            # from the zeroed dummy block tables instead of retaining state
            # indices from the previous real batch.
            for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
        )

    model_inputs = {
        "input_ids": input_ids,
        "positions": input_batch.positions,
        "inputs_embeds": inputs_embeds,
        "intermediate_tensors": None,
        # NOTE: Values returned by `prepare_inputs` will override the default
        # values above.
        **self.model_state.prepare_inputs(input_batch, self.req_states),
    }
    if not self.is_first_pp_rank:
        # Update for non-first PP ranks.
        model_inputs["input_ids"] = None
        model_inputs["inputs_embeds"] = None

        # Prepare the intermediate tensors.
        assert intermediate_tensors is not None
        assert self.intermediate_tensors is not None
        n = input_batch.num_tokens_after_padding
        new_tensors = {
            k: v[:n] if dummy_run else v[:n].copy_(intermediate_tensors.tensors[k][:n])
            for k, v in self.intermediate_tensors.tensors.items()
        }
        model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
        del intermediate_tensors

    # Update the EPLB meta.
    ubatch_slices = ubatch_state.slices if ubatch_state is not None else None
    self.eplb.prepare_forward(self.model_config, input_batch.num_tokens, ubatch_slices)

    self.step_timing.record_batch(input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL)
    self.step_timing.forward_start()

    # Run model.
    if batch_desc.cg_mode == CUDAGraphMode.FULL:
        # Use explicit cudagraph replay for FULL mode.
        # NOTE(woosuk): Here, we don't need to pass the input tensors,
        # because they are already copied to the CUDA graph input buffers.
        assert self.cudagraph_manager is not None
        self.kv_connector.pre_forward(scheduler_output)
        model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
    else:
        # For piecewise and eager mode, just call model().
        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=self.lora_config is not None,
            num_active_loras=batch_desc.num_active_loras,
        )

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=batch_desc.cg_mode,
            num_tokens_across_dp=(
                dp_sync.num_tokens_across_dp if dp_sync is not None else None
            ),
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
            slot_mapping=slot_mappings_by_layer,
            skip_compiled=skip_compiled,
            is_padding=input_batch.is_padding,
        ):
            self.kv_connector.pre_forward(scheduler_output)
            if ubatch_state is not None:
                assert self.ubatch_runner is not None
                model_output = self.ubatch_runner.run(
                    self.model, model_inputs, ubatch_state
                )
            elif batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                # cudagraph, chosen inside run_pw_graph). cg_mode is only
                # PIECEWISE after the cudagraph manager exists.
                assert self.cudagraph_manager is not None
                model_output = self.cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                model_output = self.model(**model_inputs)

    if self.is_last_pp_rank:
        if self.use_aux_hidden_state_outputs:
            assert isinstance(model_output, tuple)
            hidden_states, aux_hidden_states = model_output
        else:
            assert isinstance(model_output, torch.Tensor)
            hidden_states = model_output
            aux_hidden_states = None
        output_intermediate_tensors = None
    else:
        assert isinstance(model_output, IntermediateTensors)
        hidden_states = None
        aux_hidden_states = None
        output_intermediate_tensors = model_output

    routed_experts = None
    if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
        assert slot_mappings is not None
        routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)

    finished_req_ids = scheduler_output.finished_req_ids
    self.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=attn_metadata,
        slot_mappings_by_layer=slot_mappings_by_layer,
        hidden_states=hidden_states,
        aux_hidden_states=aux_hidden_states,
        dp_sync=dp_sync,
        finished_req_ids=finished_req_ids,
        ec_connector_output=ec_connector_output,
        routed_experts=routed_experts,
        cudagraph_stats=cudagraph_stats,
    )

    if not self.is_last_pp_rank:
        # Non-last PP rank: return IntermediateTensors for sending.
        assert output_intermediate_tensors is not None
        assert self.pp_handler is not None
        return self.pp_handler.relay_aux_hidden_states(
            model_inputs["intermediate_tensors"], output_intermediate_tensors
        )
    return None
