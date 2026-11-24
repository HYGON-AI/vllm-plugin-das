# SPDX-License-Identifier: Apache-2.0
"""Profile real PCP expert ownership without shrinking the padded activation probe."""

from dataclasses import replace

import numpy as np
import torch

from hcu_adapter.bridges.vllm_worker import Worker
from vllm import envs
from vllm.distributed import get_pcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states
from vllm.v1.worker.workspace import current_workspace_manager

from .profile_worker import HT_FIXED_CAPACITY_KEY
from ..moe.ct_w4a8_prepare import fixed_ht_batch_capacity
from ..moe.ct_w4a8_runtime import HcuCTContiguousExperts, contiguous_aligned_rows
from ..quantization.ct_w4a8 import NAME
from ..quantization.ct_w4a8_method import HcuCTW4A8MoEMethod
from ..quantization.dsv41_channel_fp8 import _is_v41


logger = init_logger(f"vllm.{__name__}")


def uses_w4a8_pcp_profile(config):
    p = config.parallel_config
    model = config.model_config
    return (
        envs.VLLM_USE_V2_MODEL_RUNNER
        and envs.VLLM_MOE_SKIP_PADDING
        and (p.tensor_parallel_size, p.data_parallel_size,
             p.pipeline_parallel_size, p.decode_context_parallel_size) == (1, 1, 1, 1)
        and p.prefill_context_parallel_size in (4, 8)
        and p.enable_expert_parallel
        and p.all2all_backend == "deepep_high_throughput"
        and config.kernel_config.moe_backend == "deep_gemm"
        and model.enforce_eager
        and model.quantization == NAME
        and _is_v41(model.hf_config)
    )


def target_profile_padding(manager, batch):
    """Keep the normal dummy tensor extent, but dispatch only the PCP-owned rows."""
    counts = batch.num_scheduled_tokens
    prefill = np.ones(len(counts), dtype=np.bool_)
    padding = np.ones(batch.num_tokens_after_padding, dtype=np.bool_)
    for req, offset, count in manager._iter_rank_chunks(manager.pcp_rank, counts, prefill):
        start = int(batch.query_start_loc_np[req]) + offset
        padding[start:start + count] = False
    return torch.from_numpy(padding).to(batch.input_ids.device), int((~padding).sum())


def routed_token_bound(max_tokens, max_reqs, decode_query_len, pcp_size, shard_decodes):
    """A prefill row routes once. An unsharded decode query routes on every rank."""
    replicated = 0 if shard_decodes else min(max_tokens, max_reqs * decode_query_len)
    return max_tokens + (pcp_size - 1) * replicated


def contiguous_workspace_bytes(num_tokens, topk, local_experts, intermediate, hidden):
    """Upper bound when all admitted expert assignments reach the same rank."""
    rows = contiguous_aligned_rows(num_tokens, topk, local_experts)
    # Both scratch views use BF16. The shared output view is bounded by
    # the first scratch view because every received row has a local route.
    return rows * (max(intermediate, hidden) + max(2 * intermediate, hidden)) * 2


class W4A8ProfilePCPManager(PCPManager):
    profile_target = False
    profile_batch = None

    def prepare_inputs_for_capture_or_profile(self, input_batch):
        batch = super().prepare_inputs_for_capture_or_profile(input_batch)
        if not self.profile_target:
            return batch
        padding, owned = target_profile_padding(self, batch)
        self.profile_batch = batch
        logger.info("W4A8 PCP profile rank %d: %d/%d target rows routed to MoE",
                    self.pcp_rank, owned, batch.num_tokens)
        return replace(batch, is_padding=padding)


class W4A8PcpProfileRunner(GPUModelRunner):
    @property
    def pcp_manager_cls(self):
        return W4A8ProfilePCPManager

    @torch.inference_mode()
    def profile_run(self, randomize_inputs=False):
        manager = self.pcp_manager
        assert isinstance(manager, W4A8ProfilePCPManager)
        routed_tokens = routed_token_bound(
            self.max_num_tokens, self.max_num_reqs, self.decode_query_len,
            manager.pcp_world_size, manager.shard_decode_requests,
        )
        workspace = max((
            contiguous_workspace_bytes(
                routed_tokens, target.top_k, target.local_num_experts,
                target.moe_config.intermediate_size_per_partition,
                target.moe_config.hidden_dim,
            )
            for target in self.model.modules()
            if isinstance(target, RoutedExperts)
            and isinstance(target.quant_method, HcuCTW4A8MoEMethod)
        ), default=0)
        if not workspace:
            raise RuntimeError("W4A8 PCP profiling requires loaded native W4A8 experts")
        # Reserve before the dummy forward so the transient peak is measured
        # with the capacity-limited runtime workspace already resident.
        current_workspace_manager().get_simultaneous(((workspace,), torch.uint8))
        logger.info("W4A8 PCP profile rank %d: reserved %.2f GiB for at most %d routed rows",
                    manager.pcp_rank, workspace / 2**30, routed_tokens)
        return super().profile_run(randomize_inputs=False)

    def execute_model(self, scheduler_output, *args, **kwargs):
        manager = self.pcp_manager
        profile_target = (kwargs.get("is_profile") and kwargs.get("dummy_run")
                and kwargs.get("skip_attn_for_dummy_run")
                and isinstance(manager, W4A8ProfilePCPManager))
        capacity = self._fixed_ht_capacity(scheduler_output, profile_target, kwargs)
        if not profile_target:
            with fixed_ht_batch_capacity(capacity):
                return super().execute_model(scheduler_output, *args, **kwargs)
        manager.profile_target = True
        try:
            with fixed_ht_batch_capacity(capacity):
                result = super().execute_model(scheduler_output, *args, **kwargs)
            state = self.execute_model_state
            original = manager.profile_batch
            assert state is not None and original is not None
            padding = state.input_batch.is_padding

            def restore(tensor):
                # All target rows still have their global sequence positions.
                # Only the PCP owner contributes each row to the draft/sampler.
                tensor.masked_fill_(padding.view(-1, *([1] * (tensor.ndim - 1))), 0)
                return get_pcp_group().all_reduce(tensor)

            if self.speculator is not None:
                draft_states = get_drafter_hidden_states(self.model, state.hidden_states)
                if draft_states is not state.hidden_states:
                    draft_states.copy_(restore(draft_states))
            self.execute_model_state = state._replace(
                input_batch=original,
                hidden_states=restore(state.hidden_states),
                aux_hidden_states=(None if state.aux_hidden_states is None else
                                   [restore(aux) for aux in state.aux_hidden_states]),
            )
            return result
        finally:
            manager.profile_target = False
            manager.profile_batch = None

    def _fixed_ht_capacity(self, scheduler_output, profile_target, execute_options):
        config = getattr(self, "vllm_config", None)
        extra = getattr(config, "additional_config", None) or {}
        if not extra.get(HT_FIXED_CAPACITY_KEY):
            return None
        if profile_target:
            tokens, requests = self.max_num_tokens, self.max_num_reqs
        elif not execute_options.get("dummy_run") and not execute_options.get("is_profile"):
            # All PCP ranks see the same scheduler batch before partitioning.
            # Rank-local tensor shapes alone cannot bound other senders.
            tokens = scheduler_output.total_num_scheduled_tokens
            requests = len(scheduler_output.num_scheduled_tokens)
        else:
            return None
        manager = self.pcp_manager
        # Use fixed dispatch for prefill-sized batches. For small batches,
        # the fixed expert workspace and launches on otherwise empty ranks
        # dominate; keep Core's dynamic receive and empty-rank fast path.
        if tokens < manager.pcp_world_size * HcuCTContiguousExperts.ALIGNMENT:
            return None
        return routed_token_bound(tokens, requests, self.decode_query_len,
                                  manager.pcp_world_size, manager.shard_decode_requests)


class W4A8PcpProfileWorker(Worker):
    def _make_model_runner(self):
        try:
            from ..dsa_prefill_runner import wrap_profile_runner
        except ModuleNotFoundError as error:
            if error.name != "hcu_vllm_das.dsa_prefill_runner":
                raise
            return W4A8PcpProfileRunner(self.vllm_config, self.device)
        return wrap_profile_runner(W4A8PcpProfileRunner, self.vllm_config, self.device)
