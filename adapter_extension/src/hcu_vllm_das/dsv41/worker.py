# SPDX-License-Identifier: Apache-2.0
"""Reuse Core profiling with rank-local HCU V4.1 PCP dummy inputs."""
from vllm.logger import init_logger
from hcu_adapter.bridges.vllm_worker import Worker

from .profile_worker import local_token_bound
from ..quantization.dsv41_channel_fp8 import _is_v41

# vLLM configures a non-propagating handler on the ``vllm`` namespace only.
logger = init_logger(f"vllm.{__name__}")


def uses_pcp_channel_fp8(config):
    """Whether this resolved config profiles rank-local PCP Channel-FP8 batches."""
    cfg = config
    return (
        cfg.parallel_config.prefill_context_parallel_size > 1
        and cfg.kernel_config.moe_backend in ("deep_gemm_mega_moe", "deep_gemm")
        and _is_v41(cfg.model_config.hf_config)
    )


class HcuDSV41PcpWorker(Worker):
    def _uses_pcp_channel_fp8(self):
        return uses_pcp_channel_fp8(self.vllm_config)

    @property
    def randomize_dummy_inputs(self) -> bool:
        return super().randomize_dummy_inputs or self._uses_pcp_channel_fp8()

    def determine_available_memory(self):
        runner = self.model_runner
        if not self._uses_pcp_channel_fp8() or runner.pcp_manager is None:
            return super().determine_available_memory()
        global_tokens = runner.max_num_tokens
        local_tokens = local_token_bound(
            global_tokens, runner.max_num_reqs,
            self.parallel_config.prefill_context_parallel_size)
        if self.rank == 0:
            logger.info(
                "HCU V4.1 PCP memory profile: global budget=%d, local bound=%d; "
                "Core workspace/cache budgets remain unchanged",
                global_tokens, local_tokens)
        # Core's profile_run uses this runner field to size its dummy batch.
        # The scheduler config and all Core cache/workspace preparation retain
        # their global budget. Real inference restores the original runner bound.
        runner.max_num_tokens = local_tokens
        try:
            return super().determine_available_memory()
        finally:
            runner.max_num_tokens = global_tokens
