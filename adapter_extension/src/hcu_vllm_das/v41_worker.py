# SPDX-License-Identifier: Apache-2.0
"""Reuse Core profiling with rank-local HCU V4.1 PCP dummy inputs."""
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _local_profile_token_bound(global_tokens, max_reqs, pcp_size):
    # Core gives a prefill rank two ceil(query_len / (2 * PCP)) chunks.
    # Two extra rows per request cover their rounding and replicated decodes.
    return min(global_tokens, (global_tokens + pcp_size - 1) // pcp_size + 2 * max_reqs)


class HcuV41PcpWorker(Worker):
    def _uses_pcp_channel_fp8(self):
        cfg = self.vllm_config
        return (
            cfg.parallel_config.prefill_context_parallel_size > 1
            and cfg.kernel_config.moe_backend in ("deep_gemm_mega_moe", "deep_gemm")
            and getattr(cfg.model_config.hf_config, "model_type", None) == "deepseek_v41"
        )

    @property
    def randomize_dummy_inputs(self) -> bool:
        return super().randomize_dummy_inputs or self._uses_pcp_channel_fp8()

    def determine_available_memory(self):
        runner = self.model_runner
        if not self._uses_pcp_channel_fp8() or runner.pcp_manager is None:
            return super().determine_available_memory()
        global_tokens = runner.max_num_tokens
        local_tokens = _local_profile_token_bound(
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
