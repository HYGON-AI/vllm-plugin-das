# SPDX-License-Identifier: Apache-2.0
"""DSpark draft scope and FULL-graph bridge for the HCU V41 plugin.

The 0.32 path allowed the native V41 DP8/EP8 DSpark target to capture the
DFlash-backed draft as FULL graphs while the target used PIECEWISE graphs. The
0.33 CT AG-RS migration retained the private draft config and MoE manager but
left out the loader scope and draft-context handoff. This module restores only
that reviewed path through adapter-owned wrappers; it does not replace the
native target communicator or create process groups.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
import functools
import inspect
from importlib import import_module
from threading import RLock, get_ident

_BACKEND = "allgather_reducescatter"
_ACTIVE_SCOPE = ContextVar("hcu_dsv41_dspark_draft_scope", default=None)
_SCOPE_LOCK = RLock()
_LOADER_MARKER = "_hcu_dsv41_dspark_graph_loader"
_CONTEXT_MARKER = "_hcu_dsv41_dspark_graph_context"
_BATCH_MARKER = "_hcu_dsv41_dspark_batch_capacity"
_BUFFER_MARKER = "_hcu_dsv41_dspark_buffer_capacity"
_installed = False
_loader_module = None
_speculator_module = None
_loader_original = None
_context_original = None
_failure: str | None = None


@dataclass(frozen=True)
class TokenCapacity:
    """Per-pass row limits, independently bounded by the runner token budget."""
    query_per_request: int
    query_rows: int
    verify_rows: int
    budget: int


def native_decode_dspark(config) -> bool:
    """Select TP1 V41 DSpark with matching DP/EP construction policy.

    Buffer and collective sizes are derived from the actual group width.
    The transport owner separately validates single-node legacy IPC support.
    """
    spec = getattr(config, "speculative_config", None)
    parallel = getattr(spec, "target_parallel_config", None) if spec else None
    if parallel is None:
        parallel = getattr(config, "parallel_config", None)
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    return bool(
        spec is not None
        and getattr(spec, "method", None) == "dspark"
        and getattr(hf, "model_type", None) in ("deepseek_v4", "deepseek_v41")
        and parallel is not None
        and (
            getattr(parallel, "tensor_parallel_size", None),
            getattr(parallel, "prefill_context_parallel_size", None),
            getattr(parallel, "pipeline_parallel_size", None),
            getattr(parallel, "decode_context_parallel_size", None),
        )
        == (1, 1, 1, 1)
        and type(getattr(parallel, "data_parallel_size", None)) is int
        and parallel.data_parallel_size > 1
        and bool(getattr(parallel, "enable_expert_parallel", False))
        and getattr(parallel, "all2all_backend", None) == "deepep_low_latency"
    )


def token_capacity(config) -> TokenCapacity:
    spec = config.speculative_config
    scheduler = config.scheduler_config
    n = spec.num_speculative_tokens
    requests = scheduler.max_num_seqs
    budget = scheduler.max_num_batched_tokens
    active = getattr(scheduler, 'max_num_active_seqs', None)
    if active is None:
        # Match the native scheduler when it leaves the active limit unset.
        active = requests
    if any(type(value) is not int or value <= 0 for value in (n, requests, budget)):
        raise ValueError(
            "DSpark requires positive speculative tokens, request slots and capacity"
        )
    anchor = getattr(spec.draft_model_config.hf_config, "sample_from_anchor", True)
    if type(anchor) is not bool:
        raise ValueError("DSpark sample_from_anchor must be bool")
    query = n if anchor else n + 1
    if budget < requests:
        raise ValueError(
            "DSpark request slots exceed the native scheduler token budget: "
            f"requests={requests}, budget={budget}"
        )
    if type(active) is not int or not 0 < active <= requests:
        raise ValueError('DSpark active request bound must fit native request slots')
    # The scheduler bounds target rows, but a separate draft pass emits query
    # rows for every scheduled request, including short prefills. Allocate that
    # pass independently rather than changing the user's scheduling budget.
    return TokenCapacity(
        query, max(active * query, budget), min(active * (n + 1), budget), budget
    )


def draft_buffer_config(config):
    """Size private draft buffers without mutating target scheduling policy."""
    capacity = token_capacity(config)
    if capacity.query_rows == config.scheduler_config.max_num_batched_tokens:
        return config
    local = copy(config)
    local.scheduler_config = copy(config.scheduler_config)
    local.scheduler_config.max_num_batched_tokens = capacity.query_rows
    return local


def validate_graph_capacity(config, moe=None) -> None:
    """Validate graph buckets against scheduler and optional MoE capacity."""
    if config.model_config.enforce_eager:
        return
    compilation = getattr(config, "compilation_config", None)
    sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    mode = getattr(getattr(compilation, "cudagraph_mode", None), "name", None)
    if (
        mode
        not in ("FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "PIECEWISE")
        or not sizes
        or any(type(size) is not int or size <= 0 for size in sizes)
    ):
        raise ValueError("DSpark graph capture requires known positive token buckets")
    budget = config.scheduler_config.max_num_batched_tokens
    if max(sizes) > budget:
        raise ValueError("DSpark graph capture exceeds scheduler capacity")
    if moe is not None:
        declared = getattr(moe, "max_num_tokens", None)
        if type(declared) is not int or max(sizes) > declared or declared > budget:
            raise ValueError("DSpark graph capture does not match MoE capacity")


class DSparkDraftScope:
    """Per-draft config/EP scope; borrowed groups are never destroyed."""

    def __init__(
        self,
        target_config,
        config,
        target_ep,
        manager,
        parallel_state,
        current_config,
        capacity,
        static_ep,
    ):
        self.target_config = target_config
        self.config = config
        self.target_ep = target_ep
        self.manager = manager
        self.draft_ep = _BorrowedEpView(target_ep, manager)
        self.parallel_state = parallel_state
        self.current_config = current_config
        self.capacity = capacity
        self.static_ep = static_ep
        self.owner_thread = get_ident()

    @contextmanager
    def activate(self):
        if get_ident() != self.owner_thread:
            raise RuntimeError("DSpark draft scope must run on its owning worker thread")
        with _SCOPE_LOCK:
            previous = self.parallel_state._EP
            if previous is not self.target_ep and previous is not self.draft_ep:
                raise RuntimeError("DSpark EP owner changed outside the draft scope")
            token = _ACTIVE_SCOPE.set(self)
            if not self.static_ep:
                self.parallel_state._EP = self.draft_ep
            try:
                with self.current_config(self.config):
                    yield
            finally:
                self.parallel_state._EP = previous
                _ACTIVE_SCOPE.reset(token)


class _BorrowedCommunicatorView:
    def __init__(self, target, manager):
        self._target = target
        self.all2all_manager = manager
        self.all2all_backend = _BACKEND

    def __getattr__(self, name):
        return getattr(self._target, name)

    def dispatch_router_logits(self, *args, **kwargs):
        return type(self._target).dispatch_router_logits(self, *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        return type(self._target).dispatch(self, *args, **kwargs)

    def combine(self, *args, **kwargs):
        return type(self._target).combine(self, *args, **kwargs)

    def allocate_combine_input(self, *args, **kwargs):
        return type(self._target).allocate_combine_input(self, *args, **kwargs)

    def combine_into_output(self, *args, **kwargs):
        return type(self._target).combine_into_output(self, *args, **kwargs)

    def _borrowed_resource(self, *args, **kwargs):
        raise RuntimeError("DSpark communicator view does not own borrowed resources")

    destroy = checkpoint_prepare = checkpoint_restore = suspend = resume = _borrowed_resource


class _BorrowedEpView:
    def __init__(self, target, manager):
        self._target = target
        self.device_communicator = _BorrowedCommunicatorView(
            target.device_communicator, manager
        )

    def __getattr__(self, name):
        return getattr(self._target, name)

    def dispatch_router_logits(self, *args, **kwargs):
        return type(self._target).dispatch_router_logits(self, *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        return type(self._target).dispatch(self, *args, **kwargs)

    def combine(self, *args, **kwargs):
        return type(self._target).combine(self, *args, **kwargs)

    def allocate_combine_input(self, *args, **kwargs):
        return type(self._target).allocate_combine_input(self, *args, **kwargs)

    def combine_into_output(self, *args, **kwargs):
        return type(self._target).combine_into_output(self, *args, **kwargs)

    def destroy(self):
        raise RuntimeError("DSpark EP view does not own borrowed process groups")


def _load_with_scope(original, target_model, vllm_config):
    from .dspark_agrs import _owned_quant_kind, draft_config
    from .layer_split_config import KEY, layer_split_enabled

    if layer_split_enabled(vllm_config):
        # Core validates the PCP1 draft config before constructing the draft
        # model. Drop the target-only switch on a private loader config first.
        vllm_config = copy(vllm_config)
        vllm_config.additional_config = dict(vllm_config.additional_config)
        del vllm_config.additional_config[KEY]

    spec = getattr(vllm_config, "speculative_config", None)
    # CT W4A8 and Channel-FP8 both own a private AG-RS draft scope so Graph
    # capture does not inherit the target's DeepEP LL communicator.
    if _owned_quant_kind(vllm_config) is None:
        return original(target_model, vllm_config)
    if spec is None or getattr(spec, "method", None) != "dspark":
        return original(target_model, vllm_config)
    from .dspark_backend import native_draft_ll_requested

    if native_draft_ll_requested(vllm_config):
        return original(target_model, vllm_config)
    if vllm_config.parallel_config.data_parallel_size <= 1:
        return original(target_model, vllm_config)

    config = draft_config(vllm_config)
    capacity = token_capacity(config)
    parallel_state = import_module("vllm.distributed.parallel_state")
    from vllm.config import set_current_vllm_config
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    target_ep = parallel_state.get_ep_group()
    if target_ep.device_communicator is None:
        raise RuntimeError("DSpark target EP group has no device communicator")
    if target_ep.world_size != config.parallel_config.data_parallel_size:
        raise ValueError("DSpark TP1 EP group must match draft DP membership")
    static_ep = native_decode_dspark(vllm_config)
    with _SCOPE_LOCK:
        with set_current_vllm_config(config):
            manager = AgRsAll2AllManager(target_ep.cpu_group)
        if manager.world_size != target_ep.world_size:
            raise ValueError("DSpark AG-RS manager does not match target EP group")

    if static_ep:
        manager._hcu_dsv41_static_ownership = True

        def padded_sizes(num_local_tokens, comm_group):
            if not 0 <= num_local_tokens <= capacity.query_rows:
                raise ValueError("DSpark AG-RS local token capacity exceeded")
            return [num_local_tokens] * comm_group.world_size

        manager._get_sizes = padded_sizes

    scope = DSparkDraftScope(
        vllm_config,
        config,
        target_ep,
        manager,
        parallel_state,
        set_current_vllm_config,
        capacity,
        static_ep,
    )
    with scope.activate():
        model = original(target_model, config)
    model._hcu_dsv41_dspark_draft_scope = scope
    forward = model.forward

    @functools.wraps(forward)
    def draft_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None and input_ids.shape[0] > capacity.query_rows:
            raise ValueError("DSpark draft forward exceeds checked token capacity")
        with scope.activate():
            return forward(*args, **kwargs)

    model.forward = draft_forward
    return model


def _patch_loader() -> None:
    global _loader_module, _loader_original
    module = import_module("vllm.v1.worker.gpu.spec_decode.dspark.utils")
    caller = import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    original = getattr(module, "load_dspark_model", None)
    if not callable(original):
        raise RuntimeError("vLLM DSpark loader is unavailable")
    if getattr(module, _LOADER_MARKER, False):
        _loader_module = module
        _loader_original = getattr(module, "_hcu_dsv41_original_load_dspark_model")
        return
    if tuple(inspect.signature(original).parameters) != ("target_model", "vllm_config"):
        raise RuntimeError("vLLM DSpark loader ABI changed")
    alias = getattr(caller, "load_dspark_model", None)
    if alias is not None and alias is not original:
        raise RuntimeError("vLLM DSpark speculator loader alias changed")

    @functools.wraps(original)
    def wrapped(target_model, vllm_config):
        return _load_with_scope(original, target_model, vllm_config)

    module._hcu_dsv41_original_load_dspark_model = original
    setattr(wrapped, _LOADER_MARKER, True)
    module.load_dspark_model = wrapped
    caller.load_dspark_model = wrapped
    setattr(module, _LOADER_MARKER, True)
    _loader_module = module
    _loader_original = original


def _patch_draft_batch(cls) -> None:
    """Reject an oversized actual draft before native input preparation writes."""
    if getattr(cls, _BATCH_MARKER, False):
        return
    original = getattr(cls, "propose", None)
    if not callable(original):
        raise RuntimeError("vLLM DFlash propose entry is unavailable")
    parameters = tuple(inspect.signature(original).parameters)
    required = ("input_batch", "dummy_run", "dp_sync_state")
    if any(name not in parameters for name in required):
        raise RuntimeError("vLLM DFlash propose capacity ABI changed")
    positions = {name: parameters.index(name) - 1 for name in required}

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_speculator_name", None) == "DSpark":
            def argument(name, default=None):
                position = positions[name]
                return kwargs.get(name, args[position] if len(args) > position else default)

            batch = argument("input_batch")
            if batch is None:
                raise ValueError("DSpark input batch is missing")
            requests = batch.num_reqs
            sync = argument("dp_sync_state")
            if argument("dummy_run", False) and sync is not None:
                requests = min(requests, sync.num_reqs)
            if not 0 <= requests <= self.max_num_reqs:
                raise ValueError("DSpark actual batch exceeds request slots")
            target_capacity = getattr(self, "_hcu_target_token_budget", self.max_num_tokens)
            if not 0 <= batch.num_tokens <= target_capacity:
                raise ValueError("DSpark actual target rows exceed runner capacity")
            query_rows = requests * self.num_query_per_req
            if query_rows > self.max_num_tokens:
                raise ValueError(
                    "DSpark actual draft query exceeds runner capacity before input preparation: "
                    f"query={query_rows}, capacity={self.max_num_tokens}, requests={requests}"
                )
        return original(self, *args, **kwargs)

    cls._hcu_dsv41_original_propose = original
    setattr(wrapped, _BATCH_MARKER, True)
    cls.propose = wrapped
    setattr(cls, _BATCH_MARKER, True)


def _patch_draft_buffers(cls) -> None:
    if getattr(cls, _BUFFER_MARKER, False):
        return
    original_init, original_attn = cls.__init__, cls.set_attn
    if tuple(inspect.signature(original_init).parameters) != ("self", "vllm_config", "device"):
        raise RuntimeError("vLLM DSpark constructor capacity ABI changed")

    @functools.wraps(original_init)
    def initialize(self, vllm_config, device):
        from .dspark_agrs import _owned_quant_kind
        local = vllm_config
        if native_decode_dspark(vllm_config) and _owned_quant_kind(vllm_config) is not None:
            local = draft_buffer_config(vllm_config)
            self._hcu_target_token_budget = vllm_config.scheduler_config.max_num_batched_tokens
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(local):
            original_init(self, local, device)

    @functools.wraps(original_attn)
    def set_attn(self, model_state, kv_cache_config, block_tables,
                 target_input_buffers, target_attn_groups):
        if hasattr(self, "_hcu_target_token_budget"):
            # This happens before graph capture. Keep the live native block
            # tables and their ownership; only enlarge the shared output rows.
            mappings = block_tables.slot_mappings
            if mappings.shape[1] < self.max_num_tokens:
                import torch
                from vllm.v1.attention.backends.utils import PAD_SLOT_ID
                block_tables.slot_mappings = torch.full(
                    (mappings.shape[0], self.max_num_tokens), PAD_SLOT_ID,
                    dtype=mappings.dtype, device=mappings.device,
                )
        return original_attn(self, model_state, kv_cache_config, block_tables,
                             target_input_buffers, target_attn_groups)

    cls.__init__, cls.set_attn = initialize, set_attn
    setattr(cls, _BUFFER_MARKER, True)


def _patch_draft_context() -> None:
    global _speculator_module, _context_original
    module = import_module("vllm.v1.worker.gpu.spec_decode.dflash.speculator")
    cls = getattr(module, "DFlashSpeculator", None)
    original = getattr(cls, "_run_model", None)
    if cls is None or not callable(original):
        raise RuntimeError("vLLM DFlash speculator context entry is unavailable")
    _patch_draft_batch(cls)
    dspark_cls = import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator").DSparkSpeculator
    _patch_draft_buffers(dspark_cls)
    if getattr(cls, _CONTEXT_MARKER, False):
        _speculator_module = module
        _context_original = getattr(cls, "_hcu_dsv41_original_run_model")
        return
    if tuple(inspect.signature(original).parameters) != (
        "self",
        "num_tokens",
        "attn_metadata",
        "slot_mappings",
        "num_tokens_across_dp",
        "cudagraph_runtime_mode",
    ):
        raise RuntimeError("vLLM DFlash speculator context ABI changed")

    mode_none = module.CUDAGraphMode.NONE

    @functools.wraps(original)
    def wrapped(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=mode_none,
    ):
        if getattr(self, "_speculator_name", None) == "DSpark":
            if not 0 <= num_tokens <= self.max_num_tokens:
                raise ValueError("DSpark padded draft rows exceed runner capacity")
        scope = getattr(getattr(self, "model", None), "_hcu_dsv41_dspark_draft_scope", None)
        if scope is None:
            return original(
                self,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        if getattr(self, "_speculator_name", None) != "DSpark":
            raise ValueError("DSpark draft scope is attached to a non-DSpark speculator")
        if cudagraph_runtime_mode != mode_none:
            if not scope.static_ep or cudagraph_runtime_mode != module.CUDAGraphMode.FULL:
                raise ValueError("DSpark draft Graph scope supports FULL capture only")
        if scope.static_ep:
            if num_tokens > scope.capacity.query_rows:
                raise ValueError("DSpark draft Graph exceeds checked token capacity")
            if num_tokens_across_dp is None:
                import torch

                num_tokens_across_dp = torch.full(
                    (scope.config.parallel_config.data_parallel_size,),
                    num_tokens,
                    dtype=torch.int32,
                )
            elif any(int(value) != num_tokens for value in num_tokens_across_dp):
                raise ValueError("DSpark Graph requires uniform padded token slots")
        with scope.activate():
            previous = self.vllm_config
            self.vllm_config = scope.config
            try:
                return original(
                    self,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode,
                )
            finally:
                self.vllm_config = previous

    cls._hcu_dsv41_original_run_model = original
    setattr(wrapped, _CONTEXT_MARKER, True)
    cls._run_model = wrapped
    setattr(cls, _CONTEXT_MARKER, True)
    _speculator_module = module
    _context_original = original


def install_dspark_graph_support() -> None:
    global _installed, _failure
    if _failure is not None:
        raise RuntimeError(_failure)
    if _installed:
        return
    try:
        _patch_loader()
        _patch_draft_context()
    except BaseException as error:
        _failure = f"DSpark Graph support installation failed: {error}"
        raise RuntimeError(_failure) from error
    _installed = True
    print("HCU DSV41: DSpark FULL-graph bridge installed", flush=True)


def check_dspark_graph_support() -> None:
    if _failure is not None:
        raise RuntimeError(_failure)
    if not _installed:
        raise RuntimeError("DSpark Graph bridge is not installed")
    if not getattr(_loader_module, _LOADER_MARKER, False):
        raise RuntimeError("DSpark Graph loader wrapper lost its owner")
    cls = getattr(_speculator_module, "DFlashSpeculator", None)
    if not getattr(cls, _CONTEXT_MARKER, False):
        raise RuntimeError("DSpark Graph context wrapper lost its owner")
    if not getattr(cls, _BATCH_MARKER, False):
        raise RuntimeError("DSpark actual-batch capacity wrapper lost its owner")
    dspark_cls = import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator").DSparkSpeculator
    if not getattr(dspark_cls, _BUFFER_MARKER, False):
        raise RuntimeError("DSpark private-buffer capacity wrapper lost its owner")


__all__ = [
    "TokenCapacity",
    "check_dspark_graph_support",
    "install_dspark_graph_support",
    "native_decode_dspark",
    "token_capacity",
    "validate_graph_capacity",
]
