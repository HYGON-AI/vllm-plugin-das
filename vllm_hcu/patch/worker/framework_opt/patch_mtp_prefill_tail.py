# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run the multi-module MTP drafter only on the tail of each prefill chunk.

Every MTP module of a sliding-window drafter (window W) runs over all scheduled
tokens, but nothing reads the result except its KV cache slots inside the window
of a later query, and K/V are per-position.  With N >= steps * (W - 1) + 1 kept
tokens at the end of each request's query, module k's KV is exact for the last
N - k * (W - 1) positions, so the last token's drafts and every later query
(next chunk or decode) see exactly what a full run would have written. Decode
and verification queries are always kept intact; prefill chunks no longer
than the retained tail are untouched as well.
For differing windows, retain 1 + sum(W - 1) across executed modules, counting
reused modules on every invocation.

HCU_MTP_PREFILL_TAIL=N enables it (0, the default, disables).  Requires
prefix caching off and a finite sliding window in every executed MTP layer:
a cached block's MTP slots outside every chunk tail were never written.
Not a VLLM_HCU_* name: eager-only, no compile-hash change.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from types import ModuleType

import numpy as np
import torch

from vllm_hcu.platforms import envs as henvs

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.mtp_prefill_tail"
TARGETS = (f"{TARGET_MODULE}.MultiModuleMTPSpeculator.propose",)
_MARKER = "_vllm_hcu_mtp_prefill_tail_applied"
_WRAPPER = "_vllm_hcu_mtp_prefill_tail_wrapper"

_TAIL = (
    int(os.environ.get("HCU_MTP_PREFILL_TAIL", "0"))
    if henvs.custom_ops_enabled() else 0
)


def _mtp_tail_required_tokens(self) -> int | None:
    """Bound dependencies of the exact modules selected by Step5's MTP loop."""
    hf = self.draft_model_config.hf_text_config
    predictor = getattr(getattr(self, "model", None), "model", None)
    start = getattr(predictor, "mtp_start_layer_idx", None)
    count = getattr(predictor, "num_mtp_layers", None)
    steps = self.num_speculative_steps
    layer_types = getattr(hf, "layer_types_with_mtp", None)
    if (
        type(start) is not int or start < 0
        or type(count) is not int or count <= 0
        or type(steps) is not int or steps <= 0
        or start != getattr(hf, "num_hidden_layers", None)
        or count != getattr(hf, "num_nextn_predict_layers", None)
        or not isinstance(layer_types, (list, tuple))
        or len(layer_types) < start + count
    ):
        return None
    layers = getattr(predictor, "layers", None)
    need = 1
    for step in range(steps):
        # Step5MultiTokenPredictor._layer reuses modules modulo their count.
        # Count repeated windows again: each invocation expands dependencies.
        idx = start + step % count
        if layer_types[idx] != "sliding_attention":
            return None
        try:
            layer = layers[str(idx)]
        except (KeyError, IndexError, TypeError):
            return None
        attention = getattr(
            getattr(getattr(layer, "mtp_block", None), "self_attn", None),
            "attn", None,
        )
        window = getattr(attention, "sliding_window", None)
        if type(window) is not int or window <= 0:
            return None
        need += window - 1
    return need


def _tail_enabled(self) -> bool:
    if not henvs.custom_ops_enabled():
        return False
    ok = getattr(self, "_hcu_mtp_tail_ok", None)
    if ok is None:
        from vllm_hcu.runtime_compat.step5 import is_step5_config

        if not is_step5_config(self.vllm_config):
            self._hcu_mtp_tail_ok = False
            return False
        cache_config = self.vllm_config.cache_config
        ok = _TAIL > 0 and not cache_config.enable_prefix_caching
        if ok:
            need = _mtp_tail_required_tokens(self)
            if need is None or _TAIL < need:
                from vllm.logger import init_logger

                init_logger(__name__).warning(
                    "HCU_MTP_PREFILL_TAIL=%d ignored: executed MTP layers must "
                    "all have finite sliding windows and require >= %s tokens",
                    _TAIL,
                    need,
                )
                ok = False
        self._hcu_mtp_tail_ok = ok
    return ok


def _trim(self, input_batch, last_hidden_states, dp_sync):
    """Return (input_batch, last_hidden_states, dp_sync) restricted to the last
    _TAIL tokens of prefill requests, keeping decode/verification intact."""
    num_reqs = input_batch.num_reqs
    prefilling = getattr(input_batch, "is_prefilling_np", None)
    if (
        not isinstance(prefilling, np.ndarray)
        or prefilling.dtype != np.bool_
        or prefilling.shape != (num_reqs,)
    ):
        # Query length does not distinguish chunked prefill from speculative
        # verification. Missing request-phase metadata cannot safely opt in.
        return None
    qsl = input_batch.query_start_loc_np[: num_reqs + 1].astype(np.int64)
    q = np.diff(qsl)
    n = q.copy()
    n[prefilling] = np.minimum(q[prefilling], _TAIL)
    if np.array_equal(n, q):
        return None
    if self.dp_size > 1 and (dp_sync is None or not dp_sync.eager):
        # Only reuse an eager agreement: anything else would need a new
        # collective that ranks without long queries would not join.
        return None
    new_qsl = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(n, out=new_qsl[1:])
    total = int(new_qsl[-1])
    idx = (
        np.arange(total, dtype=np.int64)
        - np.repeat(new_qsl[:-1], n)
        + np.repeat(qsl[1:] - n, n)
    )
    host = torch.from_numpy(np.concatenate([idx, new_qsl]).astype(np.int32))
    dev = host.pin_memory().to(self.device, non_blocking=True)
    gidx, qsl_gpu = dev[:total], dev[total:]

    new_batch = dataclasses.replace(
        input_batch,
        num_scheduled_tokens=n.astype(input_batch.num_scheduled_tokens.dtype),
        num_tokens=total,
        num_tokens_after_padding=total,
        query_start_loc=qsl_gpu,
        query_start_loc_np=new_qsl.astype(input_batch.query_start_loc_np.dtype),
        input_ids=input_batch.input_ids[gidx],
        positions=input_batch.positions[gidx],
        max_query_len=int(n.max()),
    )
    # Clean vLLM batches carry the HCU mirrors as dynamic attributes, so
    # dataclasses.replace does not preserve them. The retained tail ends at
    # the same sequence position; its computed prefix grows by the dropped
    # query tokens. Keep this metadata on the plugin-owned path as well.
    seq_lens_cpu = getattr(input_batch, "seq_lens_cpu", None)
    new_batch.seq_lens_cpu = seq_lens_cpu
    new_batch.num_computed_tokens_cpu = (
        seq_lens_cpu - torch.from_numpy(n.astype(np.int32))
        if seq_lens_cpu is not None else None
    )
    new_batch.block_tables_np = getattr(input_batch, "block_tables_np", None)
    if dp_sync is not None:
        across = dp_sync.num_tokens_across_dp.clone()
        across[self.dp_rank] = total
        dp_sync = dataclasses.replace(dp_sync, num_tokens_across_dp=across)
    return new_batch, last_hidden_states[gidx], dp_sync


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        spec, "MultiModuleMTPSpeculator", f"{TARGET_MODULE}.MultiModuleMTPSpeculator"
    )
    wrapped = ((cls, "propose", TARGETS[0], _WRAPPER),)
    if already_applied(spec, _MARKER, wrapped):
        return False
    original = require_callable(cls, "propose", TARGETS[0])

    @functools.wraps(original)
    def hcu_propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        *args,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        **kwargs,
    ):
        if (
            _TAIL > 0
            and not dummy_run
            and not is_profile
            and aux_hidden_states is None
            and mm_inputs is None
            and _tail_enabled(self)
        ):
            trimmed = _trim(self, input_batch, last_hidden_states, dp_sync)
            if trimmed is not None:
                input_batch, last_hidden_states, dp_sync = trimmed
        return original(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            *args,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            **kwargs,
        )

    setattr(hcu_propose, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_propose", original)
    setattr(cls, "propose", hcu_propose)
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
