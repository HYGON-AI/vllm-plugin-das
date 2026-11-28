# SPDX-License-Identifier: Apache-2.0
"""Explicit standard VL roles shared by runtime and direct Mooncake owners.

This module has no framework imports: selection does not initialize a device,
register a model, or turn an unrelated text/Flash/draft config into a VL model.
"""

DIRECT_MOONCAKE_MODULE = "hcu_vllm_das.hy4_pcp_mooncake"


def is_standard_vl_target(hf_config):
    kind = getattr(hf_config, "model_type", None)
    if kind == "hy_v4":
        # init_vllm_registered_model replaces the outer config with its text
        # config. Derived hy_v4_dfly configs may inherit this marker, but their
        # distinct model_type must never enter the target/producer branch.
        return getattr(hf_config, "hy4_standard_vl", False) is True
    if kind != "hy_v4_vl":
        return False
    return (
        getattr(hf_config, "architectures", None) == ["HYV4VLForConditionalGeneration"]
        and getattr(getattr(hf_config, "text_config", None), "model_type", None)
        == "hy_v4"
    )


def is_standard_dfly_pd(config):
    return (
        is_standard_vl_target(config.model_config.hf_config)
        and getattr(getattr(config, "speculative_config", None), "method", None)
        == "dfly"
        and getattr(config, "kv_transfer_config", None) is not None
    )


def validate_standard_pd_parallel(parallel, *, kv_role, enforce_eager):
    producer = kv_role == "kv_producer"
    if kv_role not in ("kv_producer", "kv_consumer"):
        raise ValueError("Standard VL PD requires a producer or consumer role")
    expected = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 16 if producer else 1,
        "data_parallel_size": 1 if producer else 16,
    }
    backends = (
        ("deepep_high_throughput",)
        if producer
        else ("deepep_high_throughput", "deepep_low_latency")
    )
    if (
        any(
            type(getattr(parallel, key, None)) is not int
            or getattr(parallel, key) != value
            for key, value in expected.items()
        )
        or not getattr(parallel, "enable_expert_parallel", False)
        or getattr(parallel, "all2all_backend", None) not in backends
        or not enforce_eager
        or any(
            getattr(parallel, field, False)
            for field in ("enable_dbo", "enable_eplb", "enable_elastic_ep")
        )
        or type(getattr(parallel, "data_parallel_rank", None)) is not int
        or not 0 <= parallel.data_parallel_rank < expected["data_parallel_size"]
    ):
        raise ValueError(
            "Standard VL requires eager fixed EP16: TP1/PP1/DCP1, "
            "PCP16/DP1 HT producer or PCP1/DP16 HT/LL decode; no DBO/EPLB/elastic EP"
        )


def classify_standard_hy4_role(config):
    """Return producer/decode only for an explicit standard VL target.

    Local DP16 target-only VL remains a decode role. A connector, if present,
    must be this installed direct Mooncake owner. No other connector or PCP
    topology is silently granted producer admission.
    """
    if not is_standard_vl_target(config.model_config.hf_config):
        raise ValueError("Standard VL role requires the explicit VL target identity")
    transfer = getattr(config, "kv_transfer_config", None)
    role = "kv_consumer"
    if transfer is not None:
        if (
            getattr(transfer, "kv_connector", None) != "MooncakeConnector"
            or getattr(transfer, "kv_connector_module_path", None)
            != DIRECT_MOONCAKE_MODULE
        ):
            raise ValueError(
                "Standard VL PD requires the owned direct Mooncake connector"
            )
        role = getattr(transfer, "kv_role", None)
    validate_standard_pd_parallel(
        config.parallel_config,
        kv_role=role,
        enforce_eager=getattr(config.model_config, "enforce_eager", False),
    )
    return "producer" if role == "kv_producer" else "decode"


def requires_last_prompt_replay(config):
    return (
        is_standard_dfly_pd(config)
        and getattr(config.kv_transfer_config, "kv_role", None) == "kv_consumer"
    )


def committed_remote_prompt_tokens(num_prompt_tokens):
    """Commit N-1 real prompt tokens so D recomputes the final hidden state."""
    if type(num_prompt_tokens) is not int or num_prompt_tokens < 1:
        raise ValueError(
            "DFly remote prefill requires a nonempty integer prompt length"
        )
    return num_prompt_tokens - 1
