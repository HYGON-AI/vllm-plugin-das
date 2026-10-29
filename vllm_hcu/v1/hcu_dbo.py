# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dual-batch overlap (DBO) for the DP/EP prefill node.

HCU_DBO_MIN_TOKENS=N (0, the default, disables) splits a step into two
microbatches when every DP rank has at least N tokens, so one microbatch's MoE
dispatch broadcast / combine reduce (modular_kernel._dp_pipe_apply) overlaps
the other microbatch's compute.

vLLM rejects DBO with speculative decoding and CUDA graphs at config time, and
enable_dbo is part of the compile hash. Both are sidestepped by turning
enable_dbo on in the worker only (after config validation) and keeping it out
of ParallelConfig.compute_hash: microbatched steps run eager anyway, so the
compiled graphs are the same. Not a VLLM_HCU_* name for the same reason.

What the framework's V2 microbatching misses for Step5, fixed here:
- the sliced InputBatch keeps the full batch's host mirrors (seq_lens_cpu,
  num_computed_tokens_cpu, block_tables_np) and max_query_len; the DSA
  schedule reads block-table rows [0, num_reqs) of the mirror, which is
  another request's pages for the second microbatch.
- Step4DSAState caches one schedule per step keyed on the metadata object and
  steps its pending-row table when it builds one. Two microbatches alternate
  layer by layer, so it would rebuild (and re-step) on every layer. Here the
  first microbatch builds the step's schedule as usual and the second builds
  its own without stepping the table; its closed regions are released with the
  first's.
- a microbatch that ends up all padding or decode-only would take paths that
  are not microbatch-aware, so every rank checks the actual query boundaries
  at the shared split point and agrees to split only if both halves have a
  prefill row.
- the microbatch threads run on GPU 0 unless given this rank's HIP context
  (_patch_thread_device), and the second microbatch's workspace is never sized.

DP8 prefill node, 32 x 65536 random / real-text prompts (2026-10-15, one server,
interleaved): ~22.93k -> ~23.85k tok/s random (+3.9%), 23.2k -> 24.1k real
(+3.7%); prompt logprobs bit-identical to unsplit steps.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import os
import sys

import numpy as np
import torch
from vllm_hcu.runtime_compat.step5 import is_step5_config
from vllm_hcu.platforms import envs as henvs

from vllm.logger import init_logger

logger = init_logger(__name__)

DBO_MIN_TOKENS = (
    int(os.environ.get("HCU_DBO_MIN_TOKENS", "0"))
    if henvs.optional_custom_op_enabled() else 0
)
# Minimum real tokens every rank must have in the second (padded) half.
# This count alone does not exclude decode-only halves in mixed batches;
# _has_prefill_in_both_ubatches also checks the sliced query lengths.
_MIN_TAIL = 64
_DBO_BATCH = contextvars.ContextVar("hcu_step5_dbo_batch", default=None)


def enable_in_worker(vllm_config) -> bool:
    if not is_step5_config(vllm_config) or not henvs.optional_custom_op_enabled(
        DBO_MIN_TOKENS > 0
    ):
        return False
    pc = vllm_config.parallel_config
    if DBO_MIN_TOKENS <= 0 or pc.data_parallel_size <= 1 or pc.enable_dbo:
        return False
    _patch_compute_hash()
    _patch_revalidation()
    pc.enable_dbo = True
    pc._hcu_step5_dbo = True
    _patch_dp_threshold()
    _patch_slicing()
    # No CU partitioning: the NCCL kernels already take few CUs
    # (NCCL_MAX_NCHANNELS), and its assert trips on HCU.
    import vllm.v1.worker.gpu.ubatch_utils as uu

    uu.create_sm_control_context = lambda parallel_config: contextlib.nullcontext()
    # One workspace for both microbatches. The second's is never sized (warmup
    # runs unsplit, then the manager locks) and would cost ~4.3 GB. Sharing is
    # safe because the workspace is scratch inside compute-stream segments --
    # no view lives across a microbatch handoff (MoE results go to the pipe's
    # own buffers) -- and both microbatches' compute is ordered on one stream.
    import vllm.v1.worker.workspace as ws

    ws.dbo_current_ubatch_id = lambda: 0
    _patch_thread_device()
    logger.warning(
        "HCU DBO on: steps with >= %d tokens on every DP rank run as two "
        "microbatches",
        DBO_MIN_TOKENS,
    )
    return True


def _patch_compute_hash() -> None:
    from vllm.config.parallel import ParallelConfig

    orig = ParallelConfig.compute_hash
    if getattr(orig, "_hcu_dbo", False):
        return

    def compute_hash(self):
        if not self.enable_dbo or not getattr(self, "_hcu_step5_dbo", False):
            return orig(self)
        self.enable_dbo = False
        try:
            return orig(self)
        finally:
            self.enable_dbo = True

    compute_hash._hcu_dbo = True
    ParallelConfig.compute_hash = compute_hash


def _patch_revalidation() -> None:
    """Configs derived in the worker (with_hf_config -> replace) re-run
    VllmConfig.__post_init__, whose microbatching checks would reject this
    setup; run it as validated, with DBO off."""
    from vllm.config.vllm import VllmConfig

    orig = VllmConfig.__post_init__
    if getattr(orig, "_hcu_dbo", False):
        return

    def __post_init__(self):
        pc = self.parallel_config
        if not pc.enable_dbo or not is_step5_config(self):
            return orig(self)
        pc.enable_dbo = False
        try:
            return orig(self)
        finally:
            pc.enable_dbo = True

    __post_init__._hcu_dbo = True
    VllmConfig.__post_init__ = __post_init__


@contextlib.contextmanager
def dbo_batch_scope(runner, batch_req_state):
    """Make only this dispatch's ordered request metadata visible to the gate."""
    token = _DBO_BATCH.set((runner, batch_req_state))
    try:
        yield
    finally:
        _DBO_BATCH.reset(token)


def _has_prefill_in_both_ubatches(
    query_lens, num_tokens: int, split_point: int, decode_threshold: int | None
) -> bool:
    """Match DSA's query-length classification after truncating at the split.

    BatchReqState is already in the order used by prepare_inputs. A request
    straddling the split contributes a separate, shorter query to each half.
    Padding is deliberately absent here: it cannot provide a prefill row.
    """
    if (
        query_lens is None
        or not isinstance(decode_threshold, (int, np.integer))
        or isinstance(decode_threshold, (bool, np.bool_))
        or decode_threshold < 1
    ):
        return False
    try:
        query_lens = np.asarray(query_lens)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        query_lens.ndim != 1
        or not np.issubdtype(query_lens.dtype, np.integer)
        or np.any(query_lens <= 0)
        or np.any(query_lens > num_tokens)
    ):
        return False
    ends = np.cumsum(query_lens, dtype=np.int64)
    if ends.size == 0 or int(ends[-1]) != num_tokens:
        # Missing metadata, dummy batches and PCP-transformed token layouts
        # must fall back rather than validate a different batch's boundaries.
        return False
    starts = ends - query_lens
    left = np.minimum(ends, split_point) - np.minimum(starts, split_point)
    right = np.maximum(ends, split_point) - np.maximum(starts, split_point)
    return bool(np.any(left > decode_threshold) and np.any(right > decode_threshold))


def _dsa_decode_threshold(runner) -> int | None:
    # Use the initialized builders, not runner.decode_query_len: DSA's
    # threshold also depends on parallel drafting and decode context parallelism.
    mod = sys.modules.get("vllm_hcu.model_executor.layers.step4_dsa")
    builder_cls = getattr(mod, "Step4DSAMetadataBuilder", None)
    if builder_cls is None:
        return None
    thresholds = [
        getattr(builder, "reorder_batch_threshold", None)
        for groups in getattr(runner, "attn_groups", ())
        for group in groups
        for builder in getattr(group, "metadata_builders", ())
        if isinstance(builder, builder_cls)
    ]
    if not thresholds or any(
        not isinstance(t, (int, np.integer))
        or isinstance(t, (bool, np.bool_))
        or t < 1
        for t in thresholds
    ):
        return None
    return int(max(thresholds))


def _patch_dp_threshold() -> None:
    import vllm.v1.worker.gpu.dp_utils as dpu

    if getattr(dpu.check_ubatch_thresholds, "_hcu_dbo", False):
        return

    original_threshold = dpu.check_ubatch_thresholds

    def check_ubatch_thresholds(config, num_tokens, uniform_decode):
        if not getattr(config, "_hcu_step5_dbo", False):
            return original_threshold(config, num_tokens, uniform_decode)
        if num_tokens < DBO_MIN_TOKENS:
            return False
        # The caller passes the smallest rank's count; the split point is half
        # the largest, which only the caller's all-reduced tensor knows.
        caller = sys._getframe(1)
        across = caller.f_locals.get("num_tokens_across_dp")
        if across is None:
            return False
        split_point = int(across.max()) // 2
        if num_tokens - split_point < _MIN_TAIL:
            return False

        # InputBatch is built only after this decision. The Step5 execution
        # route supplies its ordered BatchReqState for this dispatch explicitly.
        current = _DBO_BATCH.get()
        runner, batch = (None, None) if current is None else current
        eligible = False
        try:
            eligible = (
                not uniform_decode
                and runner is not None
                and getattr(runner, "parallel_config", None) is config
                # These modes rewrite query boundaries after dispatch; their
                # original BatchReqState cannot prove the eventual split safe.
                and getattr(runner, "pcp_manager", None) is None
                and getattr(runner, "adaptive_verification", None) is None
                and _has_prefill_in_both_ubatches(
                    getattr(batch, "num_scheduled_tokens", None),
                    caller.f_locals.get("num_tokens", -1),
                    split_point,
                    _dsa_decode_threshold(runner),
                )
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            # Invalid or unavailable local metadata is a veto, never a local
            # exception that could strand the other ranks at the vote below.
            eligible = False
        # Totals and the allow_ubatching vote above are already DP-unanimous,
        # so every rank reaches this collective. Query distributions need not
        # match: one unsafe (or unavailable) local layout vetoes DBO everywhere.
        vote = torch.tensor([int(eligible)], dtype=torch.int32, device="cpu")
        dpu.dist.all_reduce(
            vote, op=dpu.dist.ReduceOp.MIN, group=dpu.get_dp_group().cpu_group
        )
        return bool(vote.item())

    check_ubatch_thresholds._hcu_dbo = True
    dpu.check_ubatch_thresholds = check_ubatch_thresholds


def _patch_thread_device() -> None:
    """Microbatch threads are new threads: no current device and no HIP
    context. torch.cuda.set_device does not make a context current, and
    Inductor's static Triton launcher, finding none, makes device 0's primary
    context current -- switching the thread to GPU 0 for every later kernel
    (VMFault on this rank's memory). Make the step device's primary context
    current up front."""
    import ctypes

    from vllm.v1.worker.ubatching import UBatchContext

    orig_enter = UBatchContext.__enter__
    if getattr(orig_enter, "_hcu_dbo", False):
        return
    hip = ctypes.CDLL("libamdhip64.so")
    ctxs: dict[int, ctypes.c_void_p] = {}

    def set_context(index: int) -> None:
        ctx = ctxs.get(index)
        if ctx is None:
            ctx = ctypes.c_void_p()
            err = hip.hipDevicePrimaryCtxRetain(ctypes.byref(ctx), index)
            if err:
                raise RuntimeError(f"hipDevicePrimaryCtxRetain({index}) failed: {err}")
            ctxs[index] = ctx
        err = hip.hipCtxSetCurrent(ctx)
        if err:
            raise RuntimeError(f"hipCtxSetCurrent failed: {err}")

    def __enter__(self):
        index = self.compute_stream.device.index
        torch.cuda.set_device(index)
        set_context(index)
        return orig_enter(self)

    __enter__._hcu_dbo = True
    UBatchContext.__enter__ = __enter__

    orig_exit = UBatchContext.__exit__

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The runner only reports a microbatch's error after joining both
        # threads, and the sibling waits forever for a handoff: log it now.
        if exc_type is not None:
            logger.error("HCU DBO: microbatch %d failed", self.id, exc_info=exc_val)
        return orig_exit(self, exc_type, exc_val, exc_tb)

    UBatchContext.__exit__ = __exit__


def _patch_slicing() -> None:
    import vllm.v1.worker.gpu.ubatch_utils as uu

    orig = uu._slice_input_batch
    if getattr(orig, "_hcu_dbo", False):
        return

    def _slice_input_batch(input_batch, ubatch_slice, query_start_loc_buf, seq_lens_buf):
        ub = orig(input_batch, ubatch_slice, query_start_loc_buf, seq_lens_buf)
        start = ubatch_slice.request_slice.start
        stop = ubatch_slice.request_slice.stop
        qlens = np.diff(ub.query_start_loc_np).astype(np.int32)
        upd = {"max_query_len": int(qlens.max(initial=0))}
        mirrors = {}
        if getattr(input_batch, "seq_lens_cpu", None) is not None:
            seq_lens = input_batch.seq_lens_cpu[start:stop].clone()
            # Same truncation as _slice_seq_lens: only tokens past this
            # microbatch's end are missing from the straddling request.
            cut = int(input_batch.query_start_loc_np[stop]) - ubatch_slice.token_slice.stop
            if cut > 0:
                seq_lens[-1] -= cut
            mirrors["seq_lens_cpu"] = seq_lens
            mirrors["num_computed_tokens_cpu"] = seq_lens - torch.from_numpy(qlens)
        else:
            mirrors["seq_lens_cpu"] = None
            mirrors["num_computed_tokens_cpu"] = None
        tables = getattr(input_batch, "block_tables_np", None)
        mirrors["block_tables_np"] = (
            tuple(t[start:stop] for t in tables) if tables is not None else None
        )
        # The clean vLLM InputBatch dataclass has no HCU host-mirror fields.
        # Attach them after replacement, as HcuGPUModelRunnerV2 does, rather
        # than requiring a source-patched upstream dataclass constructor.
        ub = dataclasses.replace(ub, **upd)
        for name, value in mirrors.items():
            setattr(ub, name, value)
        return ub

    _slice_input_batch._hcu_dbo = True
    uu._slice_input_batch = _slice_input_batch


def install_runner(runner) -> None:
    """Per-runner hooks, once the ubatch runner and the model exist."""
    config = getattr(runner, "vllm_config", None)
    pc = getattr(config, "parallel_config", None)
    if (DBO_MIN_TOKENS <= 0
            or not is_step5_config(config)
            or not getattr(pc, "enable_dbo", False)
            or getattr(pc, "_hcu_step5_dbo", False) is not True
            or not henvs.optional_custom_op_enabled(True)):
        return
    # Explicit native DBO skips enable_in_worker(), so it has neither the HCU
    # split-safety gate nor slicing/device setup. Only wrap our owned runners.
    ur = getattr(runner, "ubatch_runner", None)
    if ur is None or getattr(ur, "_hcu_dbo", False):
        return
    orig_prepare = ur.prepare

    def prepare(input_batch, block_tables, slot_mappings):
        # The host seq_lens mirror is attached lazily, by the full batch's
        # prepare_attn -- which a microbatched step never calls.
        if (
            runner._hsl_enabled
            and input_batch is runner._hsl_batch
            and getattr(input_batch, "seq_lens_cpu", None) is None
        ):
            runner._hsl_resolve()
            runner._hsl_attach(input_batch)
        out = orig_prepare(input_batch, block_tables, slot_mappings)
        if not getattr(runner, "_hcu_dbo_announced", False):
            runner._hcu_dbo_announced = True
            logger.warning(
                "HCU DBO: first microbatched step, %d tokens", input_batch.num_tokens
            )
        return out

    ur.prepare = prepare

    orig_run = ur.run
    from vllm.utils.torch_utils import current_stream

    def run(model, model_inputs, ubatch_state):
        # run() hands vLLM's thread-local current_stream() to the microbatch
        # threads as their compute stream, and switching to it sets their
        # device. In the runner thread that value can be a stale stream made
        # on cuda:0 before set_device, while the step really runs on this
        # device's current stream: the microbatches then run on GPU 0
        # against this GPU's memory (VMFault).
        real = torch.cuda.current_stream()
        if current_stream() != real:
            torch.cuda.set_stream(real)
        assert ur.comm_stream.device == real.device, (ur.comm_stream, real)
        return orig_run(model, model_inputs, ubatch_state)

    ur.run = run
    ur._hcu_dbo = True
    _patch_dsa()


def _noop() -> None:
    pass


def _patch_dsa() -> None:
    # Step5 registers its plugin-owned DSA backend directly. There is no
    # requirement for a corresponding module in the installed vLLM tree.
    mod = sys.modules.get("vllm_hcu.model_executor.layers.step4_dsa")
    if mod is None:
        return
    state_cls = mod.Step4DSAState
    orig_schedule = state_cls.schedule
    if getattr(orig_schedule, "_hcu_dbo", False):
        return
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

    def schedule(self, md):
        if not dbo_enabled():
            # An unsplit forward starts a separate cache lifetime. Do not keep
            # microbatch metadata/tensors alive or carry its active cache into
            # a later split forward.
            self.__dict__.pop("_hcu_dbo_sched", None)
            self.__dict__.pop("_hcu_dbo_cache_uid", None)
            return orig_schedule(self, md)
        uid = dbo_current_ubatch_id()
        per_ub = self.__dict__.get("_hcu_dbo_sched")
        if per_ub is None:
            per_ub = self._hcu_dbo_sched = [None, None]
        previous_uid = self.__dict__.get("_hcu_dbo_cache_uid")
        if previous_uid is not None and per_ub[previous_uid] is not None:
            previous = per_ub[previous_uid]
            # Step4Indexer.run resets step_cache *after* schedule() on the
            # first DSA layer. Save the actual dict at the next handoff rather
            # than retaining the empty dict installed before that reset.
            per_ub[previous_uid] = (previous[0], previous[1], self.step_cache)
        hit = per_ub[uid]
        if hit is not None and hit[0] is md:
            self.step_cache = hit[2]
            self._hcu_dbo_cache_uid = uid
            return hit[1]
        if md.schedule is not None or md.num_prefills == 0:
            raise RuntimeError(
                "HCU DBO: Step-4 DSA microbatch without prefill rows "
                f"(ubatch {uid}, num_prefills={md.num_prefills}, "
                f"num_decodes={md.num_decodes})"
            )
        if uid == 0:
            # Microbatch 0 always reaches a layer first: it is this step's
            # schedule, built (and the pending table stepped) as usual.
            sched = orig_schedule(self, md)
            per_ub[1] = None
        else:
            self.end_step = _noop
            self.pending.begin_step = _noop
            try:
                sched = self._general_schedule(md)
            finally:
                del self.end_step
                del self.pending.begin_step
            main = self._schedule
            if main is not None and main is not sched:
                main.completed_slots.extend(sched.completed_slots)
                sched.completed_slots = []
        self.step_cache = {}
        per_ub[uid] = (md, sched, self.step_cache)
        self._hcu_dbo_cache_uid = uid
        return sched

    schedule._hcu_dbo = True
    state_cls.schedule = schedule
