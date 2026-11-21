# SPDX-License-Identifier: Apache-2.0
"""Route with the mask matching Core's local or dispatched token layout."""
import copy

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from .runtime_sp_batch import current_sp_batch


class RuntimeSPMoERunner(MoERunner):
    def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input,
                            input_ids=None, shared_experts_overlapping=False):
        batch = current_sp_batch()
        if batch is None or not self.do_naive_dispatch_combine:
            return super()._apply_quant_method(hidden_states, router_logits,
                shared_experts_input, input_ids, shared_experts_overlapping)
        context = get_forward_context()
        ep = get_ep_group()
        if ep.world_size == batch.size and context.dp_metadata is None:
            # DP1: Core's gathered EP order is exactly the full TP token order.
            padding = batch.global_padding
        else:
            # Core already entered sp_local_sizes before its dispatch. Gather
            # the mask over the same EP group and exact per-rank chunk sizes.
            sizes = (context.dp_metadata.get_chunk_sizes_across_dp_rank()
                     if context.dp_metadata is not None
                     else [batch.local_padding.shape[0]] * ep.world_size)
            if (sizes is None or len(sizes) != ep.world_size
                    or sizes[ep.rank_in_group] != batch.local_padding.shape[0]):
                raise RuntimeError("runtime SP MoE padding does not match dispatch sizes")
            padding = ep.all_gatherv(batch.local_padding, dim=0, sizes=sizes)
        if padding.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("runtime SP MoE padding does not match dispatched tokens")
        routed_context = copy.copy(context)
        routed_context.is_padding = padding
        try:
            with override_forward_context(routed_context):
                return super()._apply_quant_method(hidden_states, router_logits,
                    shared_experts_input, input_ids, shared_experts_overlapping)
        finally:
            context.moe_layer_index = routed_context.moe_layer_index
