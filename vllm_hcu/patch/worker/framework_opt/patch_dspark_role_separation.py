# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep DSpark draft-logits all_gather out of the captured CUDA graph.

Port of the vLLM functional-branch fix (kimi-k3-dspark-liuwc, commit
877e6996) to the v0.25.1 paired runtime: with TP > 1 the only collective
inside the DSpark draft graph is the draft-logits all_gather, and this
port moves it out of the graph.  Docking placement differs from the source
branch BY ROLE: the wheel's c10d TP communicator is a captured-role comm
(the target decode graph captures its own logits all_gather there), so the
eager replay tail must NOT ride it -- it publishes on the auxiliary eager
communicator instead (``role_separation.eager_all_gather``).  Every
communicator then keeps a single role, the property whose violation
deadlocks RCCL.

Three adaptations were required by the paired runtime (verified against
the installed vLLM 0.25.1 sources, md5 bf382b4ea89487bbf7b27e5fa82da9bf):

1. The speculator still subclasses ``DFlashSpeculator`` and the FULL-graph
   entry point is the inherited ``query_cudagraph_manager.run_fullgraph``
   (``dflash/speculator.py``); there is no speculator-owned
   ``run_fullgraph`` to override, so the eager tail hooks the manager
   instance attribute instead (identical blast radius: only THIS draft
   graph's replay).
2. ``_generate_draft`` is interposed as a whole because it is the only
   seam that receives ``cudagraph_runtime_mode``; the fork's
   capture-aware dispatch is reproduced VERBATIM on it
   (``in_full_graph = capturing or mode == FULL``), so replay-mode
   invocations take the shard path too -- ``_sample_sequential`` (whose
   eager all_gather rides the captured-role c10d comm here) must never
   re-run a sampling that the replay tail already performed.
3. The replay-time padding rule mirrors the capture driver exactly:
   ``num_reqs = desc.num_reqs or min(desc.num_tokens, max_num_reqs)``
   (``dflash/cudagraph.py`` ``create_forward_fn``).

Communicator role separation itself (the auxiliary eager all_reduce
communicator AND the eager-role docking point for the tail all_gather)
lives in ``vllm_hcu.models.kimi_k3.role_separation``; the AR reroute is
applied inside the plugin-owned Kimi-K3 model forwards.  Both halves honor
``VLLM_ROLE_SEP`` / ``VLLM_ROLE_SEP_STRICT``.
"""

from __future__ import annotations

import functools
import os
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.dspark_role_separation"
TARGETS = (
    f"{TARGET_MODULE}.DSparkSpeculator.load_draft_model",
    f"{TARGET_MODULE}.DSparkSpeculator._generate_draft",
    f"{TARGET_MODULE}.DSparkSpeculator.init_cudagraph_manager",
)
_MARKER = "_vllm_hcu_dspark_role_separation_applied"
_LOAD_WRAPPER = "_vllm_hcu_dspark_role_separation_load_wrapper"
_GEN_WRAPPER = "_vllm_hcu_dspark_role_separation_generate_wrapper"
_INIT_WRAPPER = "_vllm_hcu_dspark_role_separation_init_wrapper"
_REPLAY_WRAPPER = "_vllm_hcu_dspark_replay_tail_wrapper"
_SHARD_STATE = "_vllm_hcu_dspark_shard_enabled"

# Snapshot of the paired-runtime sampling loop this port reproduces.
_MARKOV_SOURCE_LINES = "v0.25.1 dspark/speculator.py _sample_sequential L112-150"


def _role_separation_enabled() -> bool:
    # Mirrors kimi_k3.role_separation's env contract; kept local so this
    # DSpark patch does not depend on a Kimi-K3 model module.
    return os.getenv("VLLM_ROLE_SEP", "1").lower() not in ("false", "0")


def _role_separation_strict() -> bool:
    # Validation deployments (VLLM_ROLE_SEP_STRICT=1): a captured draft
    # logits all_gather must be a hard error instead of a warning.  Every
    # communicator here is single-role (captured ARs and the captured
    # target-logits AG on their graph comms, eager collectives on the
    # auxiliary eager comm), so a draft all_gather baked into the capture
    # re-creates the captured-x-eager same-communicator pair that
    # deadlocks RCCL.
    return os.getenv("VLLM_ROLE_SEP_STRICT", "0").lower() not in (
        "",
        "0",
        "false",
    )


def _guard_captured_draft_all_gather(where: str) -> None:
    from vllm.logger import init_logger

    msg = (
        "DSpark: the draft-logits all_gather would stay inside the captured "
        "draft graph (%s). Every communicator here is single-role (the c10d "
        "TP comm is captured-role in this runtime), so an in-graph gather "
        "re-creates the captured-x-eager same-communicator pair that "
        "deadlocks RCCL. Fix the "
        "cause (VLLM_ROLE_SEP=1 + a sharded, non-replicated lm_head with "
        "compute_draft_logits_shard), or set VLLM_ROLE_SEP_STRICT=0 to "
        "downgrade this to a warning." % where
    )
    if _role_separation_strict():
        raise RuntimeError(msg)
    # The paired v0.25.1 Logger has no error_once; warning_once keeps the
    # one-time report and the strict mode carries the hard-error contract.
    init_logger(__name__).warning_once(msg)


def _shard_logits_in_graph(speculator, num_reqs: int, head_hidden) -> None:
    """Collective-free capture segment: write the per-rank draft logits
    shard into a persistent buffer (replaces the captured all_gather)."""
    n_spec = speculator.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[speculator.sample_indices[:num_sample]]
    shard = speculator.model.compute_draft_logits_shard(sample_hidden)
    expected = speculator._vllm_hcu_shard_width
    assert shard.shape[-1] == expected, (
        "lm_head shard width %s != expected %s" % (shard.shape[-1], expected)
    )
    speculator._vllm_hcu_shard_buf[: shard.shape[0]].copy_(shard)


def torch_is_capturing() -> bool:
    import torch

    return bool(torch.cuda.is_current_stream_capturing())


def _gather_and_markov(speculator, num_reqs: int) -> None:
    """Eager replay companion: all_gather the shard, then run the Markov
    sampling loop.  Executed after every FULL-graph replay."""
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    from vllm_hcu.models.kimi_k3.role_separation import (
        eager_all_gather,
        eager_ar_scope,
    )

    if torch.cuda.is_current_stream_capturing():
        # The whole point of this method is to run OUTSIDE the graph; if it
        # ever runs during a capture, the all_gather is baked in again.
        _guard_captured_draft_all_gather("_gather_and_markov while capturing")
    n_spec = speculator.num_speculative_steps
    num_sample = num_reqs * n_spec
    assert speculator._vllm_hcu_shard_buf is not None
    shard = speculator._vllm_hcu_shard_buf[:num_sample]
    # Same shapes and dim semantics as the runtime LogitsProcessor gather
    # -> bit-identical logits; the eager-role auxiliary communicator carries
    # it so the captured-role c10d TP comm is never joined eagerly.
    full = eager_all_gather(shard, dim=-1)
    vocab_size = speculator._vllm_hcu_draft_vocab_size
    base_logits = full[..., :vocab_size].view(num_reqs, n_spec, vocab_size)

    # Markov tail reproduced verbatim from the paired runtime's
    # _sample_sequential loop (snapshot: _MARKOV_SOURCE_LINES).  The sharded
    # runtime Markov head emits one collective per iteration: markov_embed
    # all_reduces through VocabParallelEmbedding on the DEFAULT TP group
    # (the engine c10d comm that also carries the captured decode-logits
    # all_gather inside the FULL graph); markov_bias is routed to the eager
    # auxiliary comm inside the model (shard GEMM + eager_all_gather).  The
    # scope pins the embed all_reduce to the same auxiliary comm, so the
    # whole per-step tail stays off the comm whose captured members exist.
    # (Convicted exposure: 20261009 19:41 boot wedged exactly on this
    # alternation -- 16-rank runTreeUpDown/ProtoSimple spin -- before the
    # scope was added.)
    idx_map = speculator.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = speculator.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = speculator.input_buffers.input_ids[speculator._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        with eager_ar_scope():
            markov_embed = speculator.model.markov_embed(prev)
            bias = speculator.model.markov_bias(markov_embed)
        logits_i = base_logits[:, i] + bias
        if speculator.draft_logits is not None:
            if speculator._d2t_scatter_index is not None:
                assert speculator._draft_scatter_buf is not None
                buf = speculator._draft_scatter_buf[:num_reqs]
                buf.index_copy_(
                    1, speculator._d2t_scatter_index, logits_i.to(buf.dtype)
                )
                logits_i = buf
            draft_sampled_i = gumbel_sample(
                logits_i,
                idx_map[:, i],
                speculator.temperature,
                speculator.seeds,
                sample_pos[:, i] - 1,
                apply_temperature=True,
                output_processed_logits=speculator.draft_logits,
                output_processed_logits_col=speculator._step_cols[i],
                use_fp64=speculator.use_fp64_gumbel,
            )
        else:
            draft_sampled_i = speculator.model.map_draft_to_target(
                logits_i.argmax(dim=-1)
            )
        speculator.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i


def _install_shard_outcome(speculator, model) -> None:
    """After load_draft_model: enable the shard path or fall back with a
    guard report (mirrors the source branch's load_draft_model block)."""
    import torch

    from vllm.logger import init_logger

    if not _role_separation_enabled():
        # Rollback contract: VLLM_ROLE_SEP=0 restores the legacy single-
        # communicator behavior including the in-graph draft all_gather.
        setattr(speculator, _SHARD_STATE, False)
        return

    shard_fn = getattr(model, "compute_draft_logits_shard", None)
    lm_head = getattr(model, "lm_head", None)
    shard_width = getattr(lm_head, "num_embeddings_per_partition", None)
    if (
        shard_fn is None
        or shard_width is None
        or getattr(lm_head, "replicated", False)
    ):
        # Shard API missing or the head is replicated (no AG to split out):
        # the legacy in-graph all_gather would come back -- captured-x-eager
        # same-communicator pair with it.
        _guard_captured_draft_all_gather(
            "draft model has no compute_draft_logits_shard with a "
            "non-replicated lm_head"
        )
        setattr(speculator, _SHARD_STATE, False)
        return

    draft_vocab_size = (
        getattr(speculator.draft_model_config.hf_config, "draft_vocab_size", None)
        or speculator.vocab_size
    )
    device = (
        speculator.draft_logits.device
        if speculator.draft_logits is not None
        else speculator.device
    )
    setattr(speculator, "_vllm_hcu_shard_width", int(shard_width))
    setattr(speculator, "_vllm_hcu_draft_vocab_size", int(draft_vocab_size))
    setattr(
        speculator,
        "_vllm_hcu_shard_buf",
        torch.zeros(
            speculator.max_num_tokens,
            int(shard_width),
            dtype=speculator.dtype,
            device=device,
        ),
    )
    setattr(speculator, _SHARD_STATE, True)
    # [fork 877e6996 load_draft_model success log, verbatim]
    init_logger(__name__).info_once(
        "DSpark: draft-logits all_gather moved out of the captured graph "
        "(VLLM_ROLE_SEP); shard width %d",
        int(shard_width),
    )


def _resolve_target_lm_head_fork_parity(target_model):
    """Fork parity for ``get_target_lm_head``: a *ForConditionalGeneration
    target keeps its lm_head on ``get_language_model()``, not on the wrapper.
    The wheel share block only probes ``target_model.lm_head`` and therefore
    silently skips the share for K3 targets, starving the guard's
    non-replicated-head premise (fork 877e6996 dspark/utils.py)."""

    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    return getattr(language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def _share_target_lm_head_fork_parity(model, target_model) -> None:
    """Re-run the wheel lm_head share with the fork-resolved target head."""

    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    target_lm_head = _resolve_target_lm_head_fork_parity(target_model)
    if target_lm_head is None:
        return
    draft_lm_head = getattr(model, "lm_head", None)
    if not _should_share(model, "has_own_lm_head", draft_lm_head, target_lm_head):
        return
    if draft_lm_head is not None:
        del model.lm_head
    model.lm_head = target_lm_head


def apply_to_module(module: ModuleType) -> bool:
    """Interpose the three DSparkSpeculator methods on the paired runtime."""
    spec = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        spec, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator"
    )
    wrapped = (
        (cls, "load_draft_model", TARGETS[0], _LOAD_WRAPPER),
        (cls, "_generate_draft", TARGETS[1], _GEN_WRAPPER),
        (cls, "init_cudagraph_manager", TARGETS[2], _INIT_WRAPPER),
    )
    if already_applied(spec, _MARKER, wrapped):
        return False

    original_load = require_callable(cls, "load_draft_model", TARGETS[0])
    require_exact_signature(
        original_load,
        TARGETS[0],
        positional=("self", "target_model", "target_attn_layer_names"),
    )
    original_generate = require_callable(cls, "_generate_draft", TARGETS[1])
    from vllm.config.compilation import CUDAGraphMode

    require_exact_signature(
        original_generate,
        TARGETS[1],
        positional=(
            "self",
            "num_reqs",
            "num_tokens_padded",
            "attn_metadata",
            "slot_mappings",
            "num_tokens_across_dp",
            "cudagraph_runtime_mode",
        ),
        defaults={"cudagraph_runtime_mode": CUDAGraphMode.NONE},
    )
    original_init = require_callable(cls, "init_cudagraph_manager", TARGETS[2])
    require_exact_signature(
        original_init,
        TARGETS[2],
        positional=("self", "cudagraph_mode"),
    )

    @functools.wraps(original_load)
    def hcu_load_draft_model(self, target_model, target_attn_layer_names):
        model = original_load(self, target_model, target_attn_layer_names)
        # Wheel share miss: for a *ForConditionalGeneration target the wheel
        # resolves target_model.lm_head (absent on the wrapper) and skips the
        # sharing, leaving the K3 draft without a head. Re-run the share with
        # the fork-resolved head BEFORE the guard evaluates its premise.
        _share_target_lm_head_fork_parity(model, target_model)
        if self.__dict__.get(_SHARD_STATE) is None:
            _install_shard_outcome(self, model)
        return model

    @functools.wraps(original_generate)
    def hcu_generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=None,
    ):
        from vllm.config.compilation import CUDAGraphMode

        # Full draft step (captured under CUDA graph): parallel backbone
        # forward then sequential Markov sampling over its hidden state
        # outputs.  [fork 877e6996 _generate_draft, verbatim]
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        # Capture-aware branch. NOTE: the capture driver calls
        # forward_fn(CUDAGraphMode.NONE) *inside* the torch.cuda.graph block
        # (gpu/cudagraph_utils.py: `with torch.cuda.graph(graph, self.pool):
        # forward_fn(CUDAGraphMode.NONE)`), so `cudagraph_runtime_mode` is
        # NONE during capture even for a FULL graph. Keying off the mode
        # alone made this branch dead during capture: the draft all_gather
        # stayed in the graph (captured role on c10d) while
        # run_fullgraph_with_tail ran an eager all_gather on the same
        # communicator on every replay -- the captured-x-eager
        # same-communicator pair is exactly the RCCL deadlock. Detect
        # capture directly instead.  [fork 877e6996, verbatim]
        capturing = torch_is_capturing()
        in_full_graph = capturing or (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        if getattr(self, _SHARD_STATE, False) and in_full_graph:
            # Collective-free capture: stop the graph at the sharded logits
            # write; the all_gather + Markov sampling execute eagerly right
            # after each replay (run_fullgraph_with_tail in
            # init_cudagraph_manager).
            _shard_logits_in_graph(self, num_reqs, head_hidden)
        else:
            if in_full_graph:
                # Reached when the shard path is unavailable: this enqueue
                # happens *inside* the capture, so the all_gather would be
                # baked into the graph.
                _guard_captured_draft_all_gather(
                    "_sample_sequential while capturing the draft FULL graph"
                )
            # This eager branch runs OUTSIDE any capture (the capturing case
            # took the collective-free shard branch above).  The sharded
            # markov head's embed all_reduce inside _sample_sequential must
            # not land on the engine c10d comm that carries the captured
            # decode all_gather, so route the whole eager sampling step to
            # the auxiliary comm (the no-op-at-capture guard inside
            # eager_ar_scope is belt-and-braces for this branch being
            # capture-free by construction).
            from vllm_hcu.models.kimi_k3.role_separation import (
                eager_ar_scope,
            )

            with eager_ar_scope():
                self._sample_sequential(num_reqs, head_hidden)

    @functools.wraps(original_init)
    def hcu_init_cudagraph_manager(self, cudagraph_mode):
        original_init(self, cudagraph_mode)
        if not getattr(self, _SHARD_STATE, False):
            return
        mgr = self.query_cudagraph_manager
        if mgr is None or getattr(mgr, _REPLAY_WRAPPER, False):
            return
        original_replay = mgr.run_fullgraph

        @functools.wraps(original_replay)
        def run_fullgraph_with_tail(desc, _orig=original_replay, _spec=self):
            _orig(desc)
            # The captured draft graph ends at the sharded-logits write
            # (collective-free capture); finish the draft step eagerly,
            # still before propose() returns draft_tokens.
            num_reqs = desc.num_reqs
            if num_reqs is None:
                # Same padding rule the capture driver uses
                # (dflash/cudagraph.py create_forward_fn), so the eager tail
                # reads exactly the rows the graph wrote.
                num_reqs = min(desc.num_tokens, _spec.max_num_reqs)
            _gather_and_markov(_spec, num_reqs)

        setattr(run_fullgraph_with_tail, _REPLAY_WRAPPER, True)
        mgr.run_fullgraph = run_fullgraph_with_tail  # type: ignore[method-assign]

    setattr(hcu_load_draft_model, _LOAD_WRAPPER, True)
    setattr(hcu_generate_draft, _GEN_WRAPPER, True)
    setattr(hcu_init_cudagraph_manager, _INIT_WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_dspark_load_draft_model", original_load)
    setattr(cls, "_vllm_hcu_original_dspark_generate_draft", original_generate)
    setattr(
        cls, "_vllm_hcu_original_dspark_init_cudagraph_manager", original_init
    )
    setattr(cls, "load_draft_model", hcu_load_draft_model)
    setattr(cls, "_generate_draft", hcu_generate_draft)
    setattr(cls, "init_cudagraph_manager", hcu_init_cudagraph_manager)
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
