import torch
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

import vllm_hcu.platforms.envs as henvs


def _use_hcu_topk_topp_sampler() -> bool:
    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER
    )


class HcuTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        # TopKTopPSampler selects a platform implementation by assigning an
        # instance-level ``forward`` in its constructor. Select the HCU
        # dispatcher at the same boundary so the base assignment cannot mask
        # it on CPU/ROCm/XPU.
        self.forward = self._forward_hcu

    def _forward_hcu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (k is None and p is None) or generators:
            return self.forward_native(logits, generators, k, p)
        if _use_hcu_topk_topp_sampler():
            try:
                from lightop import sampling
            except ImportError as exc:
                raise RuntimeError(
                    "HCU top-k/top-p sampler is enabled but lightop is unavailable"
                ) from exc

            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = sampling.top_k_top_p_sampling_from_probs(
                probs, k, p, deterministic=True
            )
            return next_token_ids.view(-1), None
        return self.forward_native(logits, generators, k, p)


class HcuSampler(Sampler):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        if _use_hcu_topk_topp_sampler():
            self.topk_topp_sampler = HcuTopKTopPSampler(logprobs_mode)
