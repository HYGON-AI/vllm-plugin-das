# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Sequence-parallel MoE (SP-MoE) for the Kimi-K3 AMD attn-residual layer.

Opt-in with ``VLLM_HCU_K3_SP_MOE=1``. Unset or ``0`` (the default) leaves the
target module untouched: the patch only sets its marker and installs nothing.

Problem
=======
Deployment: Kimi-K3 W4A8, DP8 x TP4 x EP32, DeepEP low-latency, RCCL, decode
captured with ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``. ``o_proj`` is a
``RowParallelLinear(reduce_results=True)``, so after attention all four TP
ranks hold the same ``N`` rows. Each rank then routes and dispatches all
``N`` rows through DeepEP. That means 4x the a2a traffic, 4x the masked
GEMM ``M`` and 4x the latent down/up-proj and gate work, which is the
duplication the upstream ``batched_deep_gemm_moe.py`` FIXME describes.

Design (layer-internal SP, applied to MoE layers only)
======================================================
The attn->MoE boundary moves from the replicated row domain to the
row-sharded domain, using ``AR(x) == AG(RS(x))``. ``N_pad = TP*ceil(N/TP)``,
``S = N_pad/TP``, rank ``r`` owns global rows ``[r*S, r*S+S)``, and ``v_r``
is how many of those rows are real (``< N``)::

    full   prefix_sum[N]  (layer input, replicated, bit-identical on all ranks)
    full   pre-attn attn_res + input_layernorm + block write   (unchanged)
    full   attention (TP head-sharded) -> o_proj PARTIAL sums  (no AR)
    pad    partial -> [N_pad]                     (only when N % TP != 0)
    RS     -> attn_local[S]                       (collective 1)
    shard  prefix_local = prefix_sum[rows_r] + attn_local
    shard  mlp attn_res(prefix_local, block_residual[rows_r]) + post-norm
    shard  KimiMoE: gate, latent down-proj, DeepEP dispatch of S rows,
           masked GEMM, combine, latent up-proj
           shared expert: AG(x_local)->TP-sharded MLP->RS  (collectives 2,3)
    shard  prefix_local += moe_local
    AG     -> prefix_sum[N_pad] -> [:N]           (collective 4)

Per MoE layer this is RS+AG+RS+AG. Today it is AR(o_proj) + AR(shared
expert), and an AR moves the same bytes as RS+AG, so TP communication
volume stays the same. Each rank's DeepEP dispatch drops from ``N`` to
``ceil(N/TP)`` rows. The trade is 4 collective launches instead of 2, which
matters when decode is latency-bound (see risk R1).

Dense layers (``first_k_dense_replace``) keep the original forward. Their
TP-sharded ``KimiMLP`` would need the same AG/RS pair and gains nothing from
SP, since no dispatch happens there. Layers without attn residuals, such as
the MTP block (``KimiDecoderLayer`` built with attn-res disabled, and the
DSpark draft which reuses it), are also left alone: their eager warmup
collectives ride the same stock TP group APIs as before. Every layer still
returns a full-row ``prefix_sum``, so SP and non-SP layers mix freely.

``KimiDecoderLayer.forward`` is not replaced: when ``use_attn_residuals`` is
set it only dispatches to ``self.forward_attn_residual`` (the patched
method). Its non-attn-res branch is never taken by an SP layer, because SP is
enabled only for layers with ``use_attn_residuals``.

Hidden prerequisite: the shared expert
--------------------------------------
``KimiMoE.shared_experts`` is a TP-sharded ``KimiMLP`` with
``down_proj.reduce_results=False``. The HCU ``MoERunner`` all-reduces its
partial output over TP in ``_maybe_reduce_shared_expert_output``. Once TP
ranks hold different rows, an all-reduce over that sharded domain would sum
index-aligned partials of DIFFERENT global rows (rank r's index u is global
row r*S+u), so the result would be wrong with no error raised. For SP
layers this patch therefore, at the instance level only:

* wraps the shared expert ``forward`` as ``AG(x_local) -> MLP -> RS``, which
  gives exact rows for this rank's shard
  (``RS(MLP_r(AG(x)))[rows_r] == AR(MLP_r(x))[rows_r]``);
* makes that runner's ``_maybe_reduce_shared_expert_output`` a passthrough;
* checks on layer construction (Python only, before any capture) that the
  runner has no other TP reduction left: ``moe_config.tp_size == 1`` (EP),
  the combine output is already reduced (DeepEP LL/HT), there is no EP fused
  combine, the runner is not in upstream SP mode, and DBO is off. If any
  check fails it raises instead of producing wrong numerics.

Upstream solves this with ``ReplicatedLinear`` shared experts when
``is_sequence_parallel`` is set (deepseek_v2). That is not possible here
without re-sharding the loaded W4A8 shared-expert weights.

Trap 1: the attn-res residual domain
====================================
``block_residual`` is ``[N, B, H]``, allocated by ``KimiLinearModel.forward``
with full rows and carried across layers. Ruling:

* Writes (``block_residual[:, idx, :].copy_(prefix_sum)``) happen at layer
  entry, before attention, with the full, replicated ``prefix_sum``. They stay
  in the full domain and are unchanged.
* The pre-attention ``attn_res`` reads all ``N`` rows in the full domain and
  is unchanged.
* The MLP-side ``attn_res`` runs in the shard domain. Its ``prefix_local`` row
  ``i`` is global row ``r*S+i``, so it is called with the aligned view
  ``block_residual[r*S : r*S+v_r]``. A dim-0 slice of a contiguous tensor is
  still contiguous (only the storage offset changes), and attn_res is
  row-independent (a per-row softmax over blocks plus a weighted sum), so
  per-row indexing stays valid. Padding rows (``i >= v_r``) have no
  ``block_residual`` row. They are never passed to the kernel; their hidden
  state is zero, which ``RMSNorm`` maps to zero.
* The adds ``prefix + attn`` and ``prefix + moe`` are elementwise. They run on
  the shard and are bit-identical to the same rows of the full computation.
* Returned to the next layer: ``prefix_sum`` is full ``[N, H]``, restored by
  this layer's final AG (a view of the ``[N_pad, H]`` gather buffer).
  ``block_residual`` is the same full object, never sharded, never
  copied. Rows produced by AG are bit-identical across TP ranks, so the
  replicated-domain invariant the next layer relies on still holds.
  ``AR`` gave the same guarantee.
* The model-level final ``attn_res`` and the aux hidden-state taps
  (EAGLE/DSpark) see full rows exactly as before.

Trap 2: graph capture and padding
=================================
* ``gpu_model_runner._pad_for_sequence_parallelism`` keys on
  ``compilation_config.pass_config.enable_sp``, not on
  ``parallel_config.use_sequence_parallel_moe``. The latter is already True in
  this deployment (DeepEP LL + EP + TP>1 + DP>1), but it never triggers row
  padding. The HCU model runner inherits that method unchanged, and
  ``enable_sp`` is off for this model, so nothing upstream guarantees
  ``N % TP == 0``. The HCU ``enable_custom_sp`` feature
  (``platform/core_fix/patch_compilation_config.py``) only rounds the
  *capture sizes* through upstream's ``enable_sp`` branch of
  ``adjust_cudagraph_sizes_for_sp``. It does not pad eager batches and is not
  relied upon here.
* So padding is done inside this layer: zero rows go onto the un-reduced
  ``o_proj`` output (zero partial sums reduce to zero), and the AG output is
  sliced back to ``N``. ``N``, ``S``, ``N_pad`` and ``v_r`` are Python ints
  derived from ``hidden_states.shape[0]`` and the static TP rank. Under capture
  ``N`` is the captured batch size, so every shape and branch is baked once per
  graph and identical on every replay. No data-dependent shape, no host sync.
* torch.compile removes "eager path" from that reasoning: a graph first
  traced with ``N % TP == 0`` specializes Python shape branches away and vLLM
  drops the resulting guards (the documented dropped-guard behavior behind
  the always-pad precedent in ``patch_qwen4_exp_sp``). The layer body is
  therefore branch-free in ``N``: the partial sums are padded
  unconditionally (identity when already padded), the shard windows are
  materialized with ``index_select`` over an index clamped into range plus a
  multiplicative ``keep`` mask (padding rows alias the last real row, and the
  row-wise kernels always receive exactly ``S`` finite rows -- softmax of
  zeros is never computed), and every ``N``-derived conditional
  (``valid == shard``, ``padded != num_tokens``, zero-row skips) is gone.
  The MoE padding-mask window is likewise applied inside the router (which
  always runs inside the opaque MoE custom op) instead of being published by
  a Python attribute write the layer used to perform.
* Pure decode with ``num_spec_tokens=7`` gives ``N = 8 * num_reqs`` (padded to
  the capture size), which is always divisible by 4, so padding costs nothing.
  Admission steps that mix spec and non-spec requests, DP dummy batches
  (``N=1``) and arbitrary prefill chunks do not satisfy the assumption. They
  take the pad-and-slice path, which is eager, so the extra pad copy and slice
  are acceptable. When ``N < TP``, trailing ranks own only padding rows
  (``v_r == 0``) and still join every collective and DeepEP dispatch,
  sending zero rows.
* Padding rows are routed and dispatched (at most TP-1 rows per DP rank).
  Their latent and shared outputs are about zero and get sliced away.

Trap 3: expert-side expected_m
==============================
``BatchedDeepGemmExperts.estimate_expected_m`` uses either
(a) ``sum(num_tokens_across_dp)`` when ``DPMetadata`` exists, or
(b) ``topk_ids.size(0) * num_dispatchers``. ``topk_ids`` is the pre-dispatch
local routing of this rank.

* (a) ``num_tokens_across_dp`` counts each DP rank's ``N`` once. Before SP the
  real dispatch was ``TP * sum(N)`` rows, so the estimate was already 4x too
  LOW. After SP the real dispatch is ``sum(N_pad) ~= sum(N)``, so the
  estimate becomes accurate. Nothing is overestimated.
* (b) ``local`` changes from ``N`` to ``S`` together with the real per-rank
  dispatch, so ``S * 32`` is exactly the dispatched total. It stays accurate
  and it is still a Python int, which is cudagraph-safe.

``expected_m`` is only a tile/wave hint for the masked GEMM. Correctness
comes from the dispatch counts (``masked_m``), and the per-rank buffer
capacity ``max_num_tokens`` only needs ``S <= N``. Ruling: no change. The
minimal-change rule forbids touching ``batched_deep_gemm_moe.py`` without a
demonstrated overestimate, and there is none.

Collectives
===========
SP collectives use this deployment's stock single-comm topology -- there is
no second communicator here and none is created:

* capturing -> ``get_tp_group().device_communicator.pynccl_comm`` -- the
  same comm the stock captured TP collectives use;
* eager (prefill, DP dummies, warmup) -> the stock
  ``tensor_model_parallel_*`` ``reduce_scatter`` / ``_all_gather`` group
  API. Concretely the eager reduce-scatter lands on
  ``device_communicator.reduce_scatter`` -- i.e. the TP pynccl comm, the
  same one captured segments ride -- and the eager all-gather on c10d.
  This is exactly where the stock eager TP collectives of the deployment
  already go, so SP adds no new comm and no new comm pairing; every rank
  issues the same sequence in the same stream order, so there is no
  cross-comm ordering to break. A warn-once (per message) keeps the
  routing observable.

Consequence for the role-separation workstream: eager SP reduce-scatters
SHARE the TP pynccl comm with captured work (today shared with all other
stock eager RS traffic too). The moment a role-separated deployment gives
eager collectives their own comm, this patch's eager routing must join it
or refuse to enable -- that is that workstream's scope, deliberately not
probed here (no role-separation accessors, no resolution latch).

* torch.compile is NOT a fallback case: both collectives are invoked through
  registered custom ops (``vllm::hcu_k3_sp_reduce_scatter`` /
  ``vllm::hcu_k3_sp_all_gather``). Dynamo treats them as opaque and executes
  only the shape fake while tracing, so the routing predicate above never
  runs at trace time and never bakes into the compiled graph; it runs at
  every real invocation -- during capture (the capturing-stream check fires)
  and on each eager or compiled runtime call alike. The op entry points
  resolve this module through ``sys.modules``, so the already-registered ops
  keep calling the live code even after a reload of the patch module.

Breakable cudagraph: the MoE body (including the shared expert's AG/RS) is
captured by the HCU MoE replacement, so every SP collective inside a decode
step is captured alongside it. The per-call predicate re-evaluates at
capture time and selects the TP pynccl comm for captured segments, so
capture and replay use the same communicator. Eager steps keep the stock
routing (reduce-scatter on TP pynccl, all-gather on c10d). Each collective's
comm is fixed per phase, so interleaving prefill and decode never changes
which comm a given call uses.

Composition and provenance
==========================
This wrapper must apply after ``worker.op_opt.kimi_k3.reference_paths`` (it
captures that patch's ``hcu_residual_forward`` as the non-SP chain, and its
registration sits right below it in ``_OP_CALLBACKS``). ``_maybe_enable_layer``
verifies per layer at construction that the class method is still this
wrapper, and raises otherwise, because a later overwrite would run
un-reduced ``o_proj`` through the non-SP path.

Dual-implementation debate provenance: this file merges two independent
implementations (GLM main session, claude code review channel). The shared
expert (Trap-S) diagnosis, the phase-routed comm selection, init-time
enablement with runner validation, and the A1-A4 acceptance criteria are the
claude design, adopted after its AR index-alignment proof refuted the GLM
draft's runner-reduction stance. The custom-ops discipline (SP enablement
requires ``henvs.custom_ops_enabled()``, matching the reference patch's
runtime gate), the TP-pynccl usability preflight, and the warn-once on
stock-API routing are the GLM corrections, adopted after review. The
role-separation support (aux-comm routing) that earlier drafts carried was
removed from this PR by review decision; it belongs to the role-separation
workstream.

Known risks
===========
R1 latency: decode does 4 RCCL collectives per MoE layer instead of 2 ARs,
   and the displaced ARs were most likely custom quick-reduce/aiter IPC
   kernels (``CudaCommunicator.all_reduce`` prefers them over pynccl on
   ROCm), while the RS/AG here go to raw pynccl. At ``N <= 128`` both the
   a2a and the GEMM are latency-bound, so the net gain must be measured.
   The compile-safe rewrite adds small constant costs even on already
   padded batches: one unconditional F.pad copy of the ``o_proj`` partial
   (aten returns a clone rather than the input) and an ``index_select``
   gather of ``S x B x H`` rows from ``block_residual`` where the old code
   took a view. Follow-up: use
   ``ca_comm.custom_reduce_scatter`` / ``custom_all_gather``
   for the captured phase (same approach as ``patch_qwen4_exp_sp``, which
   tries custom kernels first), or cross-layer SP that keeps ``prefix_sum``
   and ``block_residual`` sharded and AGs only the normed attention input.
   That needs ``KimiLinearModel.forward`` changes and is out of scope here.
R2 numerics: RS+AG versus AR can sum in a different order, so results match
   within fp tolerance, not bit-for-bit. Row-wise kernels (attn_res, RMSNorm,
   router GEMM) may choose different tiles for ``S`` rows than for ``N`` rows.
R3 shared-expert collectives inside the DeepEP MK window: ``reference_paths``
   pins the shared expert to the main stream, but DeepEP LL supports async
   finalize, so the shared expert (now including its AG and RS) runs in the
   MK_INTERNAL_OVERLAPPED window between ``finalize_async`` and the receive
   hook, where previously no TP collective ran. Every rank runs the same
   order and DeepEP LL does not use RCCL, so this is stream-safe, but two
   RCCL kernels now sit between the captured DeepEP send and recv kernels:
   treat long prefill/decode interleave soak as a mandatory validation
   (see A4). Correcting the earlier claim that the window could not host
   these collectives.
R4 coupling: depends on the HCU ``MoERunner`` reduction layout. The
   construction-time check fails hard if that layout changes. Also depends on
   ``o_proj`` being the last op of ``self_attn`` and on ``RowParallelLinear``
   reading ``reduce_results`` at call time.
R5 ordering: see "Composition and provenance" above.

Rollback
========
Leave ``VLLM_HCU_K3_SP_MOE`` unset (it defaults to off) or set it to ``0``
and restart the workers.
Nothing is installed in that case, and model state, weights and graphs are
the same as before this patch.
"""

from __future__ import annotations

import functools
import sys
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.sp_moe"
TARGETS = (
    f"{TARGET_MODULE}.KimiDecoderLayer.__init__",
    f"{TARGET_MODULE}.KimiDecoderLayer.forward_attn_residual",
)
ENV_SWITCH = "VLLM_HCU_K3_SP_MOE"

_MARKER = "_vllm_hcu_kimi_k3_sp_moe_applied"
_INIT_WRAPPER = "_vllm_hcu_kimi_k3_sp_moe_init_wrapper"
_FORWARD_WRAPPER = "_vllm_hcu_kimi_k3_sp_moe_forward_wrapper"
_LAYER_STATE = "_vllm_hcu_kimi_k3_sp_moe_state"


def _switch_enabled() -> bool:
    # Delegate verbatim to the envs EFFECTIVE policy (master AND child).
    # It is the install gate for this patch and the same helper feeds the
    # HcuFeatureConfig persistence that partitions vLLM's persistent
    # compilation cache, so enablement and the cache key can never drift
    # apart. Using the raw child request here instead would keep the
    # wrapper installed when an operator turns VLLM_HCU_USE_CUSTOM_OPS
    # off to roll back: apply_to_module would still wrap the decoder and
    # every SP-eligible layer would die in _maybe_enable_layer with
    # "HCU custom ops are disabled" instead of falling back to the SP
    # off path (review P2: skip SP installation under a disabled master).
    try:
        from vllm_hcu.platforms import envs as henvs

        return henvs.k3_sp_moe_enabled()
    except Exception:
        # Fail closed like _custom_ops_enabled: a foreign tree without the
        # envs module has not deployed the SP patch set either.
        return False


def _custom_ops_enabled() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.custom_ops_enabled())
    except Exception:
        # Fail closed: without the envs module this is not the HCU plugin
        # runtime that deploys the boltops attn_res kernel.
        return False


class _SpState:
    """Per-layer static SP facts; plain Python, baked into captured graphs."""

    __slots__ = ("tp_size", "tp_rank", "runner", "checked")

    def __init__(self, tp_size: int, tp_rank: int, runner) -> None:
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.runner = runner
        self.checked = False


# ---------------------------------------------------------------------------
# Collectives: capture-time pynccl / eager-time stock group API -- the
# deployment's stock single-comm topology, no second comm is created.
# ---------------------------------------------------------------------------


def _usable(comm) -> bool:
    return (
        comm is not None
        and not getattr(comm, "disabled", True)
        and getattr(comm, "available", True)
    )


def _phase_comm():
    """TP pynccl comm for captured calls, None for eager ones.

    Only executed at real invocation time. While dynamo traces, the two SP
    collectives are opaque custom ops whose fake impl answers shapes, so this
    predicate never runs at trace time and no routing decision is baked into
    a compiled graph (P1: a compile-time fallback would mix communicator
    roles during prefill/decode interleaving). Eager calls intentionally
    return None (the stock group API): a dedicated eager-role communicator
    is role-separation workstream scope, not this PR's.
    """
    if torch.cuda.is_current_stream_capturing():
        from vllm.distributed import get_tp_group

        comm = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
        return comm if _usable(comm) else None
    return None


def _warn_fallback(phase: str) -> None:
    # warning_once dedupes per message, so no module-level "already warned"
    # flag is needed (and none survives reloads). The stock-API route is the
    # designed eager topology here; the warning exists so an operator who
    # expected a dedicated eager comm notices where the collectives went.
    from vllm.logger import init_logger

    init_logger(__name__).warning_once(
        "SP-MoE %s uses the stock TP group API (expected without a "
        "role-separated eager comm).",
        phase,
    )


def _sp_reduce_scatter(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    x = x.contiguous()
    comm = _phase_comm()
    if comm is None:
        _warn_fallback("reduce_scatter")
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        out = tensor_model_parallel_reduce_scatter(x, dim=0)
    else:
        out = x.new_empty((x.shape[0] // tp_size, *x.shape[1:]))
        comm.reduce_scatter(out, x)
    # Custom-op outputs must never alias their inputs.
    assert out.data_ptr() != x.data_ptr()
    return out


def _sp_all_gather(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    x = x.contiguous()
    comm = _phase_comm()
    if comm is None:
        _warn_fallback("all_gather")
        from vllm.distributed import tensor_model_parallel_all_gather

        out = tensor_model_parallel_all_gather(x, dim=0)
    else:
        out = x.new_empty((x.shape[0] * tp_size, *x.shape[1:]))
        comm.all_gather(out, x)
    assert out.data_ptr() != x.data_ptr()
    return out


# Custom-op wrappers: opaque to dynamo so the phase-routed communicator
# selection above survives torch.compile (P1 fix). The entry points bounce
# through sys.modules so an already-registered op always calls the code of
# the module object currently occupying this name, even after a reload.

_SP_OPS_REGISTERED = False


def _sp_reduce_scatter_op_entry(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    live = sys.modules[__name__]
    return live._sp_reduce_scatter(x, tp_size)


def _sp_all_gather_op_entry(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    live = sys.modules[__name__]
    return live._sp_all_gather(x, tp_size)


def _sp_reduce_scatter_fake(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    return x.new_empty((x.shape[0] // tp_size, *x.shape[1:]))


def _sp_all_gather_fake(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    return x.new_empty((x.shape[0] * tp_size, *x.shape[1:]))


def _ensure_custom_ops_registered() -> None:
    global _SP_OPS_REGISTERED
    if _SP_OPS_REGISTERED:
        return
    # Registry-based guard: re-running direct_register_custom_op on the same
    # names would re-define the schema and raise.
    if not hasattr(torch.ops.vllm, "hcu_k3_sp_reduce_scatter"):
        from vllm.utils.torch_utils import direct_register_custom_op

        direct_register_custom_op(
            op_name="hcu_k3_sp_reduce_scatter",
            op_func=_sp_reduce_scatter_op_entry,
            mutates_args=[],
            fake_impl=_sp_reduce_scatter_fake,
        )
        direct_register_custom_op(
            op_name="hcu_k3_sp_all_gather",
            op_func=_sp_all_gather_op_entry,
            mutates_args=[],
            fake_impl=_sp_all_gather_fake,
        )
    _SP_OPS_REGISTERED = True


# ---------------------------------------------------------------------------
# Layer forward.
# ---------------------------------------------------------------------------


def _attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj,
    norm,
    num_valid_blocks: int,
) -> torch.Tensor:
    # Same kernel as op_opt.patch_kimi_k3_reference_paths (the deployed path).
    if num_valid_blocks <= 0:
        return prefix_sum

    from boltops.generic import attn_res

    return attn_res(
        prefix_sum,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        num_valid_blocks,
        norm.variance_epsilon,
    )


def _check_runner(runner) -> None:
    """Prove the runner performs no TP reduction besides the shared AR."""

    moe_config = runner.moe_config
    problems = []
    if moe_config.tp_size != 1:
        problems.append(f"moe tp_size={moe_config.tp_size} (EP required)")
    if moe_config.is_sequence_parallel:
        problems.append("runner already in upstream sequence-parallel mode")
    if getattr(moe_config, "skip_final_all_reduce", False):
        problems.append("skip_final_all_reduce is set")
    if not runner._fused_output_is_reduced:
        problems.append("combine output is not reduced (DeepEP LL/HT required)")
    # The runner can additionally fuse the shared-partials AR and the routed
    # output into one TP collective (_fuse_kimi_k3_latent_all_reduce), but its
    # own preconditions require tp_size > 1 AND a non-reduced combine here --
    # both of which the checks above already forbid. The attribute on the
    # routed transform is set True by reference_paths on every layer, so it
    # must not be treated as a problem on its own.
    if getattr(runner, "_ep_fused_combine", False):
        problems.append("EP fused combine is enabled")
    if getattr(runner, "enable_dbo", False):
        problems.append("DBO is enabled")
    if problems:
        raise RuntimeError(
            f"{ENV_SWITCH}=1 is incompatible with this MoE runner: "
            + "; ".join(problems)
            + f". Unset {ENV_SWITCH} to restore the replicated MoE path."
        )


def _sp_forward_attn_residual(
    self,
    state: _SpState,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = state.tp_size
    num_tokens = hidden_states.shape[0]
    shard = -(-num_tokens // tp_size)
    padded = shard * tp_size
    start = state.tp_rank * shard

    # Branch-free window arithmetic (P1: dynamo specializes shape branches
    # to the first traced token count and vLLM drops the guards). Rows
    # [start, start + shard) of the padded domain, with padding rows aliased
    # onto the last real row (always defined for N >= 1) and zeroed through
    # the multiplicative mask after every row-wise kernel. No slice is ever
    # taken past N, so no guard depends on the token count.
    idx = torch.arange(start, start + shard, device=hidden_states.device)
    keep = (idx < num_tokens).unsqueeze(-1).to(hidden_states.dtype)
    # Padding rows alias the last real row (always defined for N >= 1).
    idx = idx.clamp(max=num_tokens - 1)

    # Full (replicated) domain: identical to the reference path.
    prefix_sum = hidden_states
    hidden_states = _attn_res(
        prefix_sum,
        block_residual,
        self.self_attention_res_proj,
        self.self_attention_res_norm,
        self.prev_valid_blocks,
    )

    if self.is_block_write_layer:
        block_residual[:, self.block_write_idx, :].copy_(prefix_sum)
        prefix_sum = None

    hidden_states = self.input_layernorm(hidden_states)
    # o_proj.reduce_results is False on SP layers: TP partial sums. Pad is
    # unconditional and free when N is already padded.
    attn_partial = self._run_self_attn(positions, hidden_states)
    attn_full = torch.nn.functional.pad(
        attn_partial, (0, 0, 0, padded - num_tokens)
    )
    attn_local = torch.ops.vllm.hcu_k3_sp_reduce_scatter(attn_full, tp_size)

    # Row-sharded domain: local row i is global row start + i. Padding rows
    # are exactly zero: the global padding suffix carries zero partial sums
    # on every rank (same N everywhere), and `keep` zeroes the rows the
    # gather aliased.
    if prefix_sum is None:
        prefix_local = attn_local
    else:
        prefix_local = prefix_sum.index_select(0, idx) * keep + attn_local

    mlp_valid_blocks = self.prev_valid_blocks + (
        1 if self.is_block_write_layer else 0
    )
    if mlp_valid_blocks <= 0:
        hidden_local = prefix_local
    else:
        # The kernel sees exactly `shard` finite rows (padding rows enter as
        # zero prefix + a real row's residual and are zeroed by `keep`
        # afterwards; the kernel is row-independent, so the aliasing is
        # invisible and softmax-of-zero-rows is never computed).
        res_window = block_residual.index_select(0, idx)
        hidden_local = (
            _attn_res(
                prefix_local,
                res_window,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
            )
            * keep
        )

    hidden_local = self.post_attention_layernorm(hidden_local)
    if not state.checked:
        _check_runner(state.runner)
        state.checked = True
    # The router windows the global VLLM_MOE_SKIP_PADDING suffix bitmap to
    # this rank's rows itself (see _attach_padding_window: the window lives
    # on MoERunner.router, consumed inside the opaque MoE op).
    hidden_local = self.mlp(hidden_local)
    prefix_local = prefix_local + hidden_local

    # Back to the full domain for the next layer. The gathered tensor has
    # `padded` rows; the slice is always in range.
    prefix_sum = torch.ops.vllm.hcu_k3_sp_all_gather(prefix_local, tp_size)
    return prefix_sum[:num_tokens], block_residual


# ---------------------------------------------------------------------------
# Per-layer enablement (instance level, at model construction).
# ---------------------------------------------------------------------------


def _find_o_proj(self_attn: torch.nn.Module, tp_size: int):
    found = {
        id(module): module
        for name, module in self_attn.named_modules()
        if name.rsplit(".", 1)[-1] == "o_proj" and hasattr(module, "reduce_results")
    }
    if len(found) != 1:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: expected exactly one o_proj RowParallelLinear in "
            f"{type(self_attn).__name__}, found {len(found)}"
        )
    (o_proj,) = found.values()
    if "RowParallelLinear" not in {cls.__name__ for cls in type(o_proj).__mro__}:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: o_proj is {type(o_proj).__name__}, not RowParallelLinear"
        )
    if o_proj.reduce_results is not True or getattr(o_proj, "tp_size", tp_size) != tp_size:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: o_proj must be a TP-{tp_size} reducing RowParallelLinear"
        )
    return o_proj


def _attach_padding_window(moe, tp_rank: int, tp_size: int) -> None:
    """Publish the SP row window to the runner's router.

    The router consumes the global ``VLLM_MOE_SKIP_PADDING`` suffix bitmap
    and needs this rank's window coordinates. Its ``_compute_routing`` runs
    inside the opaque MoE custom op, i.e. at every real invocation; the
    layer-side attribute write this replaces is invisible to dynamo under
    torch.compile.
    """
    runner = getattr(moe, "experts", None)
    router = getattr(runner, "router", None)
    if router is None or not callable(getattr(router, "_compute_routing", None)):
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: MoERunner.router exposes no _compute_routing; "
            "cannot route the SP padding-mask window to the router"
        )
    router._hcu_sp_padding_window = (tp_rank, tp_size)


def _wrap_shared_experts(shared: torch.nn.Module, tp_size: int) -> None:
    down_proj = getattr(shared, "down_proj", None)
    if getattr(down_proj, "reduce_results", None) is not False:
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: shared expert down_proj must emit TP partial sums"
        )
    # KimiMLP.forward has no x_and_scale_quanted parameter, so the HCU
    # SharedExperts fast path (_run_layer with a pre-quantized tuple) is
    # structurally unreachable for this layer: it always calls
    # self._layer(x), i.e. exactly the instance forward wrapped below.
    original_forward = shared.forward

    def sp_shared_forward(x: torch.Tensor) -> torch.Tensor:
        # RS(MLP_r(AG(x_local))) == rows of AR(MLP_r(x_full)) for this rank.
        partial = original_forward(_sp_all_gather(x, tp_size))
        return _sp_reduce_scatter(partial, tp_size)

    shared.forward = sp_shared_forward


def _tp_pynccl_usable() -> bool:
    """Whether the shared TP pynccl comm can actually move data.

    Everything the SP forward reduces or gathers in the capture phase, and
    the stock fallback in the eager phase, routes through
    ``device_communicator.reduce_scatter``, which has NO c10d fallback
    (unlike all_gather/all_reduce): with pynccl disabled the PyNcclCommunicator
    collectives no-op and the caller's ``torch.empty`` output is returned
    unwritten. ``disabled`` is fixed at communicator construction
    (VLLM_DISABLE_PYNCCL / missing NCCL lib / world_size == 1), so an
    enablement-time probe is sound for eager and captured calls alike.
    """
    from vllm.distributed import get_tp_group

    comm = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
    return _usable(comm)


def _maybe_enable_layer(layer, decoder_cls, logger) -> _SpState | None:
    if not getattr(layer, "use_attn_residuals", False):
        return None
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is None or layer.mlp is not moe:
        return None

    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None

    parallel_config = get_current_vllm_config().parallel_config
    problems = []
    if not _custom_ops_enabled():
        problems.append(
            "HCU custom ops are disabled; the SP layer body needs the "
            "boltops attn_res kernel deployed by worker.op_opt."
            "kimi_k3.reference_paths"
        )
    if not parallel_config.use_sequence_parallel_moe:
        problems.append("requires EP + all2all backend + TP>1 + DP>1")
    if getattr(parallel_config, "use_ubatching", False):
        problems.append("DBO/ubatching is unsupported")
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        problems.append("prefill context parallel is unsupported")
    if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
        problems.append("decode context parallel is unsupported")
    if not _tp_pynccl_usable():
        problems.append(
            "the TP pynccl communicator is unavailable (VLLM_DISABLE_PYNCCL, "
            "missing NCCL library); device_communicator.reduce_scatter has "
            "no c10d fallback, so every SP reduce-scatter -- capture path "
            "included -- would return unwritten output"
        )
    if not getattr(vars(decoder_cls).get("forward_attn_residual"), _FORWARD_WRAPPER, False):
        problems.append("KimiDecoderLayer.forward_attn_residual was re-patched after SP-MoE")
    if problems:
        raise RuntimeError(
            f"{ENV_SWITCH}=1 cannot be applied: " + "; ".join(problems)
        )

    from vllm_hcu.model_executor.layers.fused_moe.moe_runner import MoERunner

    runner = moe.experts
    if not isinstance(runner, MoERunner):
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: KimiMoE.experts is {type(runner).__name__}, "
            "not the HCU MoERunner"
        )
    require_callable(
        runner,
        "_maybe_reduce_shared_expert_output",
        "MoERunner._maybe_reduce_shared_expert_output",
    )
    shared = getattr(moe, "shared_experts", None)
    runner_shared = getattr(runner, "_shared_experts", None)
    if (shared is None) != (runner_shared is None) or (
        shared is not None and runner_shared._layer is not shared
    ):
        raise PatchCompatibilityError(
            f"{ENV_SWITCH}: runner shared expert is not KimiMoE.shared_experts"
        )

    o_proj = _find_o_proj(layer.self_attn, tp_size)
    o_proj.reduce_results = False
    if shared is not None:
        _wrap_shared_experts(shared, tp_size)

        def shared_output_already_sharded(shared_output, fused_output_is_reduced=None):
            return shared_output

        runner._maybe_reduce_shared_expert_output = shared_output_already_sharded

    # Compile-safe SP collectives (P1) + router-side padding-mask window.
    # The window is consumed by the runner's router inside _compute_routing
    # (which always runs inside the opaque MoE custom op); the layer body no
    # longer publishes a windowed mask through an attribute write (invisible
    # to dynamo). The gate linear is a ReplicatedRouting with no routing
    # hook -- the window must live on the router (review P0).
    _ensure_custom_ops_registered()
    _attach_padding_window(moe, get_tensor_model_parallel_rank(), tp_size)

    logger.info_once(
        "Kimi-K3 SP-MoE enabled (%s=1): o_proj reduce-scatter, MoE dispatch "
        "of ceil(N/%d) rows per TP rank, all-gather after MoE.",
        ENV_SWITCH,
        tp_size,
    )
    return _SpState(tp_size, get_tensor_model_parallel_rank(), runner)


# ---------------------------------------------------------------------------
# Patch entry points.
# ---------------------------------------------------------------------------


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    decoder = require_class(target, "KimiDecoderLayer", f"{TARGET_MODULE}.KimiDecoderLayer")
    state = getattr(target, _MARKER, None)
    if state is not None:
        if state == "enabled" and not (
            getattr(vars(decoder).get("__init__"), _INIT_WRAPPER, False)
            and getattr(vars(decoder).get("forward_attn_residual"), _FORWARD_WRAPPER, False)
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[1]} is stale; restart the process"
            )
        return False
    if not _switch_enabled():
        setattr(target, _MARKER, "disabled")
        return True

    original_init = require_callable(decoder, "__init__", TARGETS[0])
    original_forward = require_callable(decoder, "forward_attn_residual", TARGETS[1])
    # inspect.signature follows __wrapped__, so against the
    # reference_paths wrapper this resolves to the upstream True signature.
    require_exact_signature(
        original_forward,
        TARGETS[1],
        positional=("self", "positions", "hidden_states", "block_residual"),
    )
    require_callable(decoder, "_run_self_attn", f"{TARGET_MODULE}.KimiDecoderLayer._run_self_attn")

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        setattr(self, _LAYER_STATE, _maybe_enable_layer(self, decoder, logger))

    @functools.wraps(original_forward)
    def hcu_forward_attn_residual(self, positions, hidden_states, block_residual):
        sp_state = getattr(self, _LAYER_STATE, None)
        if sp_state is None:
            return original_forward(self, positions, hidden_states, block_residual)
        return _sp_forward_attn_residual(
            self, sp_state, positions, hidden_states, block_residual
        )

    setattr(hcu_init, _INIT_WRAPPER, True)
    setattr(hcu_forward_attn_residual, _FORWARD_WRAPPER, True)
    setattr(decoder, "_vllm_hcu_sp_moe_original_init", original_init)
    setattr(decoder, "_vllm_hcu_sp_moe_original_forward_attn_residual", original_forward)
    decoder.__init__ = hcu_init
    decoder.forward_attn_residual = hcu_forward_attn_residual
    setattr(target, _MARKER, "enabled")
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["ENV_SWITCH", "PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]


# ---------------------------------------------------------------------------
# Acceptance criteria (not run here; for the validation deployment).
#
# A1 dispatch rows: with VLLM_HCU_K3_SP_MOE=1, DeepEP LL dispatch per rank goes
#    from N to ceil(pad(N)/4) rows, where pad(N) = 4*ceil(N/4). The
#    router-side padding-window fix (MoERunner.router._hcu_sp_padding_window,
#    consumed by window_sp_padding_mask inside the opaque MoE op) is REQUIRED
#    for imbalanced DP steps: without windowed masking, every rank dispatches
#    its padding rows and an 8-real-token step in a 128-row graph would
#    dispatch 104 rows instead of 32 (and 8 after the mask fix). The window
#    applies on the LightOp routing path (HcuGroupedTopKRouter._compute_routing);
#    super()._compute_routing fallbacks keep the global suffix masking and
#    are safe but skip the saving. Measure topk_ids.size(0) / DeepEP send
#    counts on both balanced and imbalanced DP steps.
# A2 numerics: same prompts with greedy sampling, switch 0 vs 1. Per-layer
#    prefix_sum and final logits must match within bf16 tolerance (AR vs
#    RS+AG ordering only). Greedy tokens should match, and spec-decode
#    acceptance rate should not change. Include a prefill with N % 4 != 0
#    and a DP dummy batch with N=1.
# A3 graphs: with VLLM_USE_BREAKABLE_CUDAGRAPH=1, capture every decode size
#    with the switch on, then replay mixed batch sizes. No shape mismatch, no
#    capture error, and the output for each size equals an eager run of that
#    size. All SP shapes are Python ints of N and the static TP rank.
# A4 soak: the shared expert's AG/RS now run between the captured DeepEP
#    send and recv kernels, which is a new position for RCCL work on these
#    comms. Soak a long prefill/decode interleave (hours) with no RCCL hang.
#    Eager SP collectives deliberately ride the same single-comm topology as
#    every other eager TP collective in the deployment; when role-separated
#    eager collectives land (a later workstream), repeat this soak with its
#    env switch and strict collective tracing.
# ---------------------------------------------------------------------------
