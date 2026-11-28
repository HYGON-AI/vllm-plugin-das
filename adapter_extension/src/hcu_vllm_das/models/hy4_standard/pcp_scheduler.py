# SPDX-License-Identifier: Apache-2.0
"""Reject non-prefill producer requests before native queue mutation."""

from copy import deepcopy

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler

from .pd_policy import classify_standard_hy4_role


def validate_prefill_only_request(request):
    params = request.kv_transfer_params
    if (
        request.sampling_params is None
        or request.pooling_params is not None
        or type(request.max_tokens) is not int
        or request.max_tokens != 1
        or type(request.sampling_params.max_tokens) is not int
        or request.sampling_params.max_tokens != request.max_tokens
        or request.resumable
        or not isinstance(params, dict)
        or params.get("do_remote_decode") is not True
        or params.get("do_remote_prefill", False) is not False
        or not isinstance(params.get("transfer_id"), str)
        or not params["transfer_id"]
    ):
        raise ValueError(
            "HY4 PD producer requires a non-resumable max_tokens=1 remote-decode request and transfer_id"
        )

    # P's sampled token is discarded by PD. Native check_stop checks cached
    # EOS/stop IDs before the length cap, so ignore_eos alone is insufficient.
    # Rebuild the real struct, including request-local processor metadata;
    # clone() may return the original when skip_clone=True.
    values = {
        name: deepcopy(getattr(request.sampling_params, name))
        for name in SamplingParams.__struct_fields__
    }
    values.update(
        ignore_eos=True,
        stop=[],
        stop_token_ids=[],
        min_tokens=0,
        _eos_token_id=None,
        _all_stop_token_ids=set(),
        output_text_buffer_length=0,
        skip_clone=False,
    )
    values["extra_args"] = values["extra_args"] or {}
    values["extra_args"]["kv_transfer_params"] = deepcopy(params)
    return SamplingParams(**values)


class HYV4PCPProducerScheduler(Scheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        if (
            classify_standard_hy4_role(vllm_config) != "producer"
            or vllm_config.scheduler_config.async_scheduling
            or vllm_config.scheduler_config.get_scheduler_cls() is not type(self)
        ):
            raise ValueError(
                "HY4 producer requires its explicit synchronous scheduler owner"
            )
        super().__init__(vllm_config, *args, **kwargs)

    def add_request(self, request):
        prepared = validate_prefill_only_request(request)
        if request.request_id in self.requests:
            raise ValueError(
                "HY4 one-token producer does not accept active request/session updates"
            )
        request.sampling_params = prepared
        request.kv_transfer_params = prepared.extra_args["kv_transfer_params"]
        return super().add_request(request)
