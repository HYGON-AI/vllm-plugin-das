# SPDX-License-Identifier: Apache-2.0
"""Preserve native multimodal preparation across HCU PCP token layouts."""

from contextlib import contextmanager

import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class HYV4VLModelState(DefaultModelState):
    """Run native image/cache/RoPE preparation globally, then select local tokens.

    Canonical InputBatch positions remain untouched. Only the separate model
    positions and embeddings are gathered, into storage distinct from native
    global buffers. Request admission, eviction and attention stay native.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        device = torch.device(device)
        before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        super().__init__(vllm_config, model, encoder_cache, device)
        self.pcp_manager = None
        self._pcp_size = int(vllm_config.parallel_config.prefill_context_parallel_size)
        self._dummy_run = False
        if self._pcp_size > 1:
            # Keep global capacity: real preparation uses G tokens, while the
            # profile/capture caller already supplies its per-rank token count.
            self._local_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.inputs_embeds_size),
                dtype=self.dtype,
                device=device,
            )
            self._local_positions = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device=device,
            )
        after = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        if after < before:
            raise RuntimeError("HYV4VL model-state allocation delta is negative")
        # Native load_model measures weights before constructing model_state.
        self._hcu_vl_persistent_memory_bytes = after - before

    def bind_pcp_manager(self, manager):
        self.pcp_manager = manager

    @contextmanager
    def dummy_run_scope(self, dummy_run):
        previous = self._dummy_run
        self._dummy_run = bool(dummy_run)
        try:
            yield
        finally:
            self._dummy_run = previous

    def _global_batch(self, local_batch):
        if self.pcp_manager is None:
            raise RuntimeError(
                "HYV4VL real PCP preparation requires a bound PCP manager"
            )
        return self.pcp_manager.get_global_batch(local_batch)

    def dummy_inputs_embeds(self, num_tokens):
        if self._pcp_size <= 1:
            # Graph replay must use the same native storage as real execution.
            embeds = super().dummy_inputs_embeds(num_tokens)
        else:
            embeds = self._local_inputs_embeds[:num_tokens]
        embeds.zero_()
        return embeds

    def _dummy_positions(self, num_tokens):
        if self._pcp_size <= 1:
            positions = self.rope_state.get_positions(num_tokens)
        else:
            positions = self._local_positions[:, :num_tokens]
        positions.zero_()
        return positions

    def get_mm_embeddings(self, scheduled_encoder_inputs, input_batch, req_states):
        if self._dummy_run:
            return self.dummy_inputs_embeds(input_batch.num_tokens_after_padding)
        if self._pcp_size <= 1:
            return super().get_mm_embeddings(
                scheduled_encoder_inputs, input_batch, req_states
            )
        global_batch = self._global_batch(input_batch)
        global_embeds = super().get_mm_embeddings(
            scheduled_encoder_inputs, global_batch, req_states
        )
        return self.pcp_manager.gather_global_token_tensor(
            global_embeds,
            input_batch,
            token_dim=0,
            out=self._local_inputs_embeds[: input_batch.num_tokens_after_padding],
        )

    def prepare_inputs(self, input_batch, req_states):
        if self._dummy_run:
            return {
                "positions": self._dummy_positions(input_batch.num_tokens_after_padding)
            }
        if self._pcp_size <= 1:
            return super().prepare_inputs(input_batch, req_states)
        global_batch = self._global_batch(input_batch)
        model_inputs = super().prepare_inputs(global_batch, req_states)
        model_inputs["positions"] = self.pcp_manager.gather_global_token_tensor(
            model_inputs["positions"],
            input_batch,
            token_dim=1,
            out=self._local_positions[:, : input_batch.num_tokens_after_padding],
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        return {
            "inputs_embeds": self.dummy_inputs_embeds(num_tokens),
            "positions": self._dummy_positions(num_tokens),
        }
