# SPDX-License-Identifier: Apache-2.0
"""Build genuine global VL state, then select PCP-local token rows."""

from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .pcp_manager import HYV4PCPProducerManager


class HYV4PCPProducerModelState(DefaultModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hy4_pcp_manager = None

    def configure_pcp(self, manager):
        if not isinstance(manager, HYV4PCPProducerManager):
            raise TypeError("HY4 producer state requires its owned PCP manager")
        if self._hy4_pcp_manager not in (None, manager):
            raise RuntimeError("HY4 producer state PCP owner changed")
        self._hy4_pcp_manager = manager

    def _manager(self):
        if self._hy4_pcp_manager is None:
            raise RuntimeError("HY4 producer state has no attached PCP layout owner")
        return self._hy4_pcp_manager

    def prepare_inputs(self, input_batch, req_states):
        manager = self._manager()
        if manager.local_layout is None:
            return super().prepare_inputs(input_batch, req_states)
        global_batch, layout = manager.require_live_batch(input_batch)
        inputs = super().prepare_inputs(global_batch, req_states)
        if inputs.get("positions") is not None:
            inputs["positions"] = layout.select(inputs["positions"], token_dim=1)
        return inputs

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        manager = self._manager()
        if manager.local_layout is None:
            return super().prepare_inputs_embeds(
                scheduled_encoder_inputs, input_batch, req_states
            )
        global_batch, layout = manager.require_live_batch(input_batch)
        embeddings = super().prepare_inputs_embeds(
            scheduled_encoder_inputs, global_batch, req_states
        )
        return layout.select(embeddings, token_dim=0)
