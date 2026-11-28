# SPDX-License-Identifier: Apache-2.0
"""Seal model-level PD admission to the explicit connector and runner owners."""

from .pd_policy import classify_standard_hy4_role, is_standard_vl_target


def runtime_role(config):
    parallel = config.parallel_config
    if (
        config.kv_transfer_config is not None
        or parallel.prefill_context_parallel_size != 1
    ):
        if not is_standard_vl_target(config.model_config.hf_config):
            raise ValueError("HY4 PCP/PD requires the explicit standard VL target")
        from ...hy4_pcp_dfly_activation import require_active

        require_active()
        role = classify_standard_hy4_role(config)
        if role == "producer":
            from .pcp_manager import HYV4PCPProducerManager

            HYV4PCPProducerManager.validate_config(config, True)
        return role
    from ..hy_v4_flash.dfly_topology import validate_standard_hy4_parallel

    validate_standard_hy4_parallel(
        parallel, enforce_eager=config.model_config.enforce_eager
    )
    return "decode"
