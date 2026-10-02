# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep the DSpark drafter on EP without inheriting target DeepEP LL.

For the DeepSeek-V4 MoE draft, SGLang's speculative ``a2a=none`` path still
gathers the DP-local token shards before local EP expert execution and combines
the partial results afterwards.  In vLLM this is the naive AG-RS P/F path;
``NoDPEP`` would be a different, non-equivalent local-token-only algorithm.
"""

from __future__ import annotations

import contextvars
import functools
from types import FunctionType, MethodType, ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.utils"
PATCH_ID = "worker.core_fix.deepseek_v4_dspark.draft_ag_rs"
TARGET_SYMBOL = f"{TARGET_MODULE}._get_dspark_parallel_config"
_WRAPPER_MARKER = "_vllm_hcu_dspark_draft_ag_rs_wrapper"
_LOAD_MARKER = "_vllm_hcu_dspark_draft_load_wrapper"
_IN_DSPARK_DRAFT_FORWARD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vllm_hcu_in_dspark_draft_forward", default=False
)


def in_dspark_draft_forward() -> bool:
    return _IN_DSPARK_DRAFT_FORWARD.get()


def _install_draft_run_model_scope() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator

    cls = speculator.DFlashSpeculator
    if getattr(cls, "_vllm_hcu_draft_run_model_applied", False):
        return
    original = cls._run_model

    @functools.wraps(original)
    def hcu_run_model(self, *args, **kwargs):
        token = _IN_DSPARK_DRAFT_FORWARD.set(True)
        try:
            return original(self, *args, **kwargs)
        finally:
            _IN_DSPARK_DRAFT_FORWARD.reset(token)

    cls._run_model = hcu_run_model
    setattr(cls, "_vllm_hcu_draft_run_model_applied", True)


def _make_draft_ag_rs_manager(all2all_manager):
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    manager = AgRsAll2AllManager(
        all2all_manager.cpu_group,
        all2all_manager.tcp_store_group,
    )

    def get_sizes(num_local_tokens, comm_group):
        # DSpark's profile/full-graph draft batches are padded uniformly across
        # DP ranks. Target DP metadata describes the target batch, which has a
        # different token count from the draft query batch and cannot be reused.
        # Avoid an extra device collective on HCU as well.
        return [num_local_tokens] * comm_group.world_size

    manager._get_sizes = get_sizes
    return manager


def _bind_draft_manager(prepare_finalize, all2all_manager):
    """Bind the naive P/F instance to a private AG-RS manager."""
    if getattr(prepare_finalize, "_vllm_hcu_ag_rs_manager", None) is not None:
        return prepare_finalize
    manager = _make_draft_ag_rs_manager(all2all_manager)
    for name in ("prepare", "finalize"):
        original = getattr(prepare_finalize, name).__func__
        # Preserve upstream scale/LoRA/reduction handling, but bind only this
        # instance's communication dependency. No global mutation in capture.
        bound = FunctionType(
            original.__code__,
            dict(original.__globals__, get_ep_group=lambda: manager),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        bound.__kwdefaults__ = original.__kwdefaults__
        functools.update_wrapper(bound, original)
        setattr(prepare_finalize, name, MethodType(bound, prepare_finalize))
    setattr(prepare_finalize, "_vllm_hcu_ag_rs_manager", manager)
    return prepare_finalize


def maybe_bind_draft_ag_rs(prepare_finalize, all2all_manager):
    """Use AG-RS when a layer requests it but the process manager is DeepEP.

    ``NoDPEP`` is intentionally not selected here.  For the DeepSeek-V4 MoE
    draft, SGLang's speculative ``a2a=none`` still gathers DP token shards and
    combines the local EP partial results; it is not a local-token-only path.
    """
    prepare = getattr(prepare_finalize, "prepare", None)
    finalize = getattr(prepare_finalize, "finalize", None)
    class_name = type(prepare_finalize).__name__
    if (
        not callable(prepare)
        or not callable(finalize)
        or "NaiveDPEP" not in class_name
    ):
        return prepare_finalize
    if all2all_manager is None:
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            get_ep_all2all_manager,
        )

        all2all_manager = get_ep_all2all_manager()
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    # A native AG-RS deployment already has the correct global manager. The
    # special case is a draft layer configured for AG-RS inside a target
    # process whose communicator was created for DeepEP LL.
    if isinstance(all2all_manager, AgRsAll2AllManager):
        return prepare_finalize
    if getattr(prepare_finalize, "_vllm_hcu_ag_rs_manager", None) is not None:
        return prepare_finalize
    return _bind_draft_manager(prepare_finalize, all2all_manager)


def apply_to_module(module: ModuleType) -> bool:
    dspark_utils = load_exact_module(TARGET_MODULE, module)
    original_parallel = require_callable(
        dspark_utils, "_get_dspark_parallel_config", TARGET_SYMBOL
    )
    original_load = require_callable(
        dspark_utils, "load_dspark_model", f"{TARGET_MODULE}.load_dspark_model"
    )
    _install_draft_run_model_scope()

    if getattr(dspark_utils, "_vllm_hcu_dspark_draft_ag_rs_applied", False):
        current = getattr(dspark_utils, "_get_dspark_parallel_config", None)
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        if not getattr(original_load, _LOAD_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE}.load_dspark_model is stale"
            )
        return False

    require_exact_signature(
        original_parallel,
        TARGET_SYMBOL,
        positional=("parallel_config", "tensor_parallel_size"),
    )
    require_exact_signature(
        original_load,
        f"{TARGET_MODULE}.load_dspark_model",
        positional=("target_model", "vllm_config"),
    )

    @functools.wraps(original_parallel)
    def hcu_get_dspark_parallel_config(parallel_config, tensor_parallel_size):
        result = original_parallel(parallel_config, tensor_parallel_size)
        if not getattr(result, "enable_expert_parallel", False):
            return result
        from vllm.config import replace

        # Keep expert sharding across EP8. Only replace the
        # communication backend so it does not allocate a second DeepEP LL heap.
        return replace(result, all2all_backend="allgather_reducescatter")

    @functools.wraps(original_load)
    def hcu_load_dspark_model(target_model, vllm_config):
        from vllm.config import replace
        from vllm_hcu.patch.config import get_hcu_config

        backend = getattr(vllm_config.speculative_config, "moe_backend", None)
        backend = backend or "triton"
        # Override both config owners on a draft-only copy. HCU 'auto'
        # delegates explicit official backends such as Triton to vLLM.
        sidecar = get_hcu_config(vllm_config).with_updates(
            moe_backend="deep_gemm" if backend == "deep_gemm" else "auto",
            deepep_auto=False,
        )
        additional = dict(vllm_config.additional_config or {})
        additional["hcu"] = sidecar.to_dict()
        draft_config = replace(
            vllm_config,
            kernel_config=replace(vllm_config.kernel_config, moe_backend=backend),
            additional_config=additional,
        )
        return original_load(target_model, draft_config)

    setattr(hcu_get_dspark_parallel_config, _WRAPPER_MARKER, True)
    setattr(hcu_load_dspark_model, _LOAD_MARKER, True)
    setattr(
        dspark_utils, "_vllm_hcu_original_get_dspark_parallel_config", original_parallel
    )
    setattr(dspark_utils, "_vllm_hcu_original_load_dspark_model", original_load)
    setattr(dspark_utils, "_get_dspark_parallel_config", hcu_get_dspark_parallel_config)
    setattr(dspark_utils, "load_dspark_model", hcu_load_dspark_model)
    setattr(dspark_utils, "_vllm_hcu_dspark_draft_ag_rs_applied", True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
    "maybe_bind_draft_ag_rs",
]
