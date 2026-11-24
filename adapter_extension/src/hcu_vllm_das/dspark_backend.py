# SPDX-License-Identifier: Apache-2.0
"""Default to the admitted W4A8 draft LL path, with an explicit AG/RS opt-out."""

KEY = "hcu_dsv41_dspark_draft_backend"


def native_draft_ll_requested(config):
    extra = getattr(config, "additional_config", None)
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("DSpark additional_config must be a dict")
    explicit = KEY in extra
    backend = extra.get(KEY, "deepep_low_latency")
    if backend == "allgather_reducescatter":
        return False
    if backend != "deepep_low_latency":
        raise ValueError("Unsupported HCU DSpark draft backend: " + str(backend))
    if not explicit:
        # Other models and deployment geometries retain the native AG/RS
        # policy. Explicit requests keep the strict diagnostic error path.
        try:
            return _validate(config)
        except (AttributeError, ImportError, OSError, ValueError):
            return False
    return _validate(config)


def _validate(config):
    from .dspark_agrs import _owned_quant_kind, draft_config, is_owned_draft
    from .dspark_graph import native_decode_dspark

    if (_owned_quant_kind(config) != "ct_w4a8"
            or not native_decode_dspark(config) or is_owned_draft(config)):
        raise ValueError("W4A8 draft LL requires the native CT TP1 DP/EP target")
    # Reuse all default feature, topology and graph capacity checks without
    # constructing any model, communicator or transport resource.
    draft_config(config)
    from deep_ep import Buffer
    from .dspark_ll import validate_legacy_ipc

    from .dspark_ll_quant import receiver_quantizer_available

    parallel = getattr(config.speculative_config, 'target_parallel_config', None)
    if parallel is None:
        parallel = config.parallel_config
    validate_legacy_ipc(Buffer, parallel.data_parallel_size)
    if not receiver_quantizer_available():
        raise ValueError("W4A8 draft LL requires vLLM public per-token INT8 quantization")
    return True
