# SPDX-License-Identifier: Apache-2.0
"""Outer model boundaries; official inner loops/loaders/logits stay inherited.

These classes are explicit candidates. Registry publication and complete model
qualification are separate from importing or constructing these boundaries.
"""
import copy
import os

from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.models.glm4_moe import Glm4MoeForCausalLM
from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM
from vllm.sequence import IntermediateTensors

from .runtime_sp_attention import (
    check_attention_layers, install_attention_layers, runtime_sp_model_construction,
)
from .runtime_sp_batch import RuntimeSPBatch, _BATCH_KEY
from .runtime_sp_moe import check_moe_factory_policy, install_moe_factory_policy
from .runtime_sp_policy import custom_sp_requested


class RuntimeSPModelBoundary:
    def __init__(self, *, vllm_config, prefix=""):
        enabled = custom_sp_requested(vllm_config)
        if enabled:
            group = get_tp_group()
            if not callable(getattr(group.device_communicator, "all_to_all_single", None)):
                raise RuntimeError("runtime SP model requires its admitted communicator")
            install_attention_layers()
            install_moe_factory_policy()
            with runtime_sp_model_construction():
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._runtime_sp_group = group
            self._runtime_sp_size = group.world_size
            self._runtime_sp_rank = group.rank_in_group
            self._runtime_sp_pid = os.getpid()
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._runtime_sp_enabled = enabled

    def _check_runtime_sp(self):
        group = get_tp_group()
        if (os.getpid() != self._runtime_sp_pid or group is not self._runtime_sp_group
                or group.world_size != self._runtime_sp_size
                or group.rank_in_group != self._runtime_sp_rank):
            raise RuntimeError("runtime SP model process or TP ownership changed")
        check_attention_layers()
        check_moe_factory_policy()
        return group

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if not self._runtime_sp_enabled:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        group = self._check_runtime_sp()
        context = get_forward_context()
        if _BATCH_KEY in context.additional_kwargs:
            raise RuntimeError("nested runtime SP model boundaries are not supported")
        batch = RuntimeSPBatch.create(group.world_size, group.rank_in_group,
            positions.shape[0], context.is_padding, positions.device)
        pp = get_pp_group()
        if pp.is_first_rank:
            # Vocabulary-TP embedding all-reduces corresponding token rows.
            # It must see the SAME full IDs on every rank before token sharding.
            if inputs_embeds is None:
                if input_ids is None or input_ids.shape[0] != batch.num_tokens:
                    raise ValueError("runtime SP model requires full input IDs or embeddings")
                inputs_embeds = self.embed_input_ids(input_ids)
            local_embeds = batch.shard(inputs_embeds)
            local_intermediate = None
        else:
            if intermediate_tensors is None:
                raise ValueError("runtime SP PP stage requires hidden states and residual")
            local_intermediate = IntermediateTensors({
                key: batch.shard(intermediate_tensors[key])
                for key in ("hidden_states", "residual")
            })
            local_embeds = None

        local_context = copy.copy(context)
        local_context.additional_kwargs = dict(context.additional_kwargs, **{_BATCH_KEY: batch})
        local_context.is_padding = batch.local_padding
        try:
            with override_forward_context(local_context):
                # Attention QKV exchanges restore global token rows, so positions
                # stay global while hidden states and BOTH residual streams are local.
                result = super().forward(None, batch.positions(positions),
                                         local_intermediate, local_embeds)
        finally:
            # Preserve Core's custom-op name cursor for subsequent consumers of
            # the same forward context, including partial/failed execution.
            context.moe_layer_index = local_context.moe_layer_index
        if isinstance(result, IntermediateTensors):
            return IntermediateTensors({key: batch.gather(value, group)
                                        for key, value in result.items()})
        return batch.gather(result, group)


class RuntimeSPGlm4MoeForCausalLM(RuntimeSPModelBoundary, Glm4MoeForCausalLM):
    pass


class RuntimeSPHYV3ForCausalLM(RuntimeSPModelBoundary, HYV3ForCausalLM):
    pass
