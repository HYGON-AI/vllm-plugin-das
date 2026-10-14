# SPDX-License-Identifier: Apache-2.0
"""Step-4 DSA (DeepSeek-style top-k sparse attention) attention backend.

Step-4 has 23 ``full_attention`` layers that the reference release runs through
DSA and 69 ``sliding_attention`` layers that run on plain SDPA. Only the 23 are
routed here; the sliding layers keep whatever backend the platform picks.

What DSA does, per query token: an *indexer* scores every 8-token region of the
context, the top ``topk`` regions are selected, and attention runs over only
those regions. At ``topk * region_size == 4096`` the selected set covers the
whole causal history, so DSA is provably identical to full causal attention up
to 4096 tokens and only starts to differ above that. That invariant is what the
``run_e2e_dsa`` operator gate checks, and it is also how this backend is
bootstrapped: with ``VLLM_STEP4_DSA_DENSE_EQUIV=1`` the planner selects *every*
visible region, so any output difference against the dense backend is a wiring
bug in the paging/metadata path rather than a difference in what was attended to.

The numeric kernels are not implemented here. They are called from
``step4_dsa_kernels``, a verbatim copy of the reference ``inference/kernel.py``
that ``operator_parity`` gates; see ``check_dsa_vendor_drift.py``. This module is
only the driver: it owns the parts the reference implementation gets for free
because it assumes one contiguous cache per request, and vLLM does not.

``VLLM_STEP4_DSA_BOLTOPS=decode,prefill`` routes the named sparse-attention
stages to ``boltops.step4_dsa`` instead, for A/B-ing the operator team's library
against the vendored copy end to end. See ``BOLTOPS_STAGES``.

Three pieces of state, stored three different ways:

* **K/V** -- vLLM's own paged cache. ``get_kv_cache_shape`` declares NHD so that
  ``kv_cache[0].flatten(0, 1)`` is exactly the ``[tokens, num_kv_groups,
  head_dim]`` layout the sparse kernels consume. The stock HCU FlashAttention
  backend stores V transposed, which those kernels cannot read, which is why
  this backend declares its own shape instead of reusing that one.
* **Region summaries** -- ``Step4DSAState.summary``, self-allocated and indexed
  by the *same physical block id* as the KV cache. See that class for why this
  is not an ``MLAAttentionSpec``.
* **Pending partial regions** -- ``Step4PendingTable`` plus a small dense buffer,
  keyed by the region's physical summary slot rather than by a batch row. A
  region only ever gets a summary once it is *complete*, so the tokens of the
  region a request is currently filling have to be held somewhere across steps
  and across prefill chunks; that is what pending is.
"""

from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
    REGION_BLOCK_SIZE,
    _NO_FP_FUSION,
    build_rope_cache,
    csa_compress_regions,
    decode_rows_meta,
    decode_sparse_meta,
    indexer_logits,
    indexer_norm_rope,
    merge_split_states,
    prefill_sparse_meta,
    round_activations_e4m3,
    sparse_attention_decode,
    sparse_attention_prefill,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm.step5.step4_dsa")

# Step-4's head_dim. Declared as a supported-head-size list so that a checkpoint
# with different geometry fails at layer construction rather than inside a
# Triton kernel with a shape error.
STEP4_HEAD_DIM = 192

# The qualified name of the indexer op. Duplicated as a literal in
# transformers_utils/configs/step4.py, which cannot import this module (it runs
# during config resolution, long before torch/Triton would be wanted). If the
# two ever drift apart, _require_splitting_op fails at startup and says so.
STEP4_DSA_INDEX_OP = "vllm::step4_dsa_index"

# Every Step4Indexer on this rank, keyed by the name of the attention layer it
# feeds. The custom op below can only be handed tensors and strings, so the
# object itself has to be looked up rather than passed.
_INDEXERS: dict[str, "Step4Indexer"] = {}

# Bootstrap/debug switch, see the module docstring. Selecting every region makes
# DSA mathematically equal to full causal attention, at the cost of a
# [tokens, all_regions] metadata tensor, so this is for validation only.
DENSE_EQUIV = os.environ.get("VLLM_STEP4_DSA_DENSE_EQUIV", "0") == "1"

# How many query tokens are scored against their region summaries at a time.
# The scoring tensor is [rows, regions] fp32, and at 64K a request has 8192
# regions: a whole 16384-token prefill chunk scored at once would be 536 MB.
# 2048 rows brings that to 67 MB, which is the only reason this knob exists --
# selection is per row, so the chunking changes nothing about the result.
# Query-axis chunking for prefill indexer scoring. Off by default, and the default
# is the point: it used to be 2048, which at a 65536-token prefill meant 32
# ``indexer_logits`` launches per request per DSA layer -- 6716 launches in one
# recorded 9-step trace -- plus two activation quantizations and four group copies
# each. The chunk existed only to bound the ``[groups * count, nscore]`` fp32
# temporary that ``indexer_logits`` used to return; now that the kernel scores
# directly into ``prefill_logits`` there is no temporary to bound, and the query
# axis was always grid dimension 0 anyway, so one launch does the same work.
#
# Measured on one 65536-token, 1023-region request (groups=4), per DSA layer:
#
#     old, chunk=2048   9.60 ms   +47 MiB transient
#     new, no chunk     6.57 ms  +359 MiB
#     new, chunk=8192   6.72 ms   +23 MiB
#
# So nearly all of the 3.0 ms is the copies and the redundant key quantization,
# not the launch count -- collapsing 32 launches to 1 is only the last 0.15 ms,
# and it costs ~340 MiB of transient peak because the e4m3 copy of ``index_q`` is
# then made at full length. That peak is bounded by --max-num-batched-tokens and
# freed immediately, which is why the default is still 0; set
# VLLM_STEP4_DSA_Q_CHUNK to a positive token count to trade it back.
INDEXER_Q_CHUNK = int(os.environ.get("VLLM_STEP4_DSA_Q_CHUNK", "0"))

# Decode score width, in regions, for the CUDA-graph-captured path. A graph
# replays the launch arguments it recorded, so the [rows, regions] score tensor
# and the selector's scan bound must be the same on every step rather than
# following the batch's longest context. The default covers max_model_len, which
# is correct but pessimistic: the selector scans the whole declared width, so a
# server started with --max-model-len 65536 does an eighth of the region scan a
# 512K one does. Set this only to go *below* max_model_len deliberately -- a
# batch whose real history exceeds it is rejected loudly, never truncated.
GRAPH_REGIONS_OVERRIDE = int(os.environ.get("VLLM_STEP4_DSA_GRAPH_REGIONS", "0"))

# Debug switch for the host-side slot derivation in _decode_schedule. When set,
# the slots computed from the block table's host mirror are checked against the
# device tensor they replace -- which reintroduces exactly the device-to-host
# sync the mirror exists to remove, so this is for bring-up and for bisecting a
# suspected mirror/device divergence, never for a timed run.
CHECK_SLOTS = os.environ.get("VLLM_STEP4_DSA_CHECK_SLOTS", "0") == "1"

# Verify-batch indexer scores with one program per request (spec_indexer_logits)
# instead of one per query token. Bit-identical; "0" restores the per-token kernel.
_SPEC_INDEXER_PER_REQUEST = (
    os.environ.get("VLLM_STEP4_DSA_SPEC_INDEXER_PER_REQUEST", "1") == "1"
)

# ...and with the request's (token, head) pairs stacked into one fp8 x fp8 dot
# (_spec_indexer_stacked_kernel). Same products, summed over heads in the same
# order, but the dot's internal fp32 accumulation order differs: not
# bit-identical. "0" restores the per-head bf16 dots.
_SPEC_INDEXER_FP8_DOT = (
    os.environ.get("VLLM_STEP4_DSA_SPEC_INDEXER_FP8_DOT", "1") == "1"
)

# Depth of the pinned staging ring used to upload the per-step pending rows.
# One buffer per vector is not enough: the upload is asynchronous, so the next
# step's host write can land in the buffer before the previous step's copy has
# read it out of it. Four is a step of run-ahead well past anything the
# scheduler currently sustains; the events below make the depth a bound rather
# than an assumption.
_STAGING_DEPTH = 4

# Which sparse-attention stages to run out of ``boltops.step4_dsa`` (the
# operator team's library, contract ``step4-dsa-ops-v1``) instead of the
# vendored reference copy. Comma-separated, empty means "vendored everywhere":
#
#   VLLM_STEP4_DSA_BOLTOPS=decode,prefill
#
# Only these two stages are switchable. The selector, indexer, CSA compression
# and metadata builders stay on the vendored copy in either case: they produce
# the *inputs* both libraries consume, so keeping one producer means a
# difference in output is attributable to the attention kernel alone.
#
# Two things the two libraries do not share:
#
# * boltops reads an fp8 cache but does not do fp8 *arithmetic*. As of its
#   commit 28239f4 both stages take ``k_scale``/``v_scale`` and dequantize a
#   ``float8_e4m3fn`` cache on load -- ``k_scale`` folded into ``softmax_scale``
#   on the host, ``v_scale`` applied to the fp32 output under an
#   ``APPLY_V_SCALE`` constexpr -- then widen to the query dtype before
#   ``tl.dot``. That matches the vendored copy at
#   ``VLLM_STEP4_DSA_FP8_MATH=off`` and nothing else: boltops has no equivalent
#   of ``qk_fp8``/``pv_fp8``, so an A/B against the server default compares two
#   different arithmetics, not two implementations of one.
#   boltops also keys its occupancy JSON on the *cache* dtype, because the query
#   stays bfloat16 on the fp8 path and the cache dtype is the only thing that
#   distinguishes it. ``_boltops_decode_config`` therefore has to pass both.
# * boltops decode owns its split count. ``num_splits=1`` there means "split to
#   fill the CUs and merge for me", so the ``_decode_num_splits`` policy and the
#   explicit ``merge_split_states`` pass are skipped on that path.
_BOLTOPS_STAGE_NAMES = frozenset({"decode", "prefill"})
BOLTOPS_STAGES = frozenset(
    name.strip()
    for name in os.environ.get("VLLM_STEP4_DSA_BOLTOPS", "").split(",")
    if name.strip()
)
if BOLTOPS_STAGES - _BOLTOPS_STAGE_NAMES:
    raise ValueError(
        f"VLLM_STEP4_DSA_BOLTOPS: unknown stage(s) "
        f"{sorted(BOLTOPS_STAGES - _BOLTOPS_STAGE_NAMES)}; "
        f"expected a comma-separated subset of {sorted(_BOLTOPS_STAGE_NAMES)}"
    )

# Which sparse-attention stages to run through
# flash-attention-cutlass's own sparse kernels (``fa_sparse_attention_prefill``/
# ``fa_sparse_attention_decode``, and their ``_fp8`` variants) instead of the
# vendored Triton reference copy. Same shape as ``VLLM_STEP4_DSA_BOLTOPS``:
#
#   VLLM_STEP4_DSA_FLASH_ATTN=decode,prefill
#
# The selector, indexer, CSA compression and metadata builders stay on the
# vendored copy regardless -- only the attention-compute call is swapped, same
# as the BoltOps switch above.
#
# Sliding-window (dense) attention on HCU already routes through flash_attn
# when ``VLLM_HCU_USE_FLASH_ATTN``/``VLLM_HCU_USE_FLASH_ATTN_UNIFIED`` is set
# (see ``vllm_hcu.platforms.hcu._get_backend_priorities``), but that selector
# hardcodes the sparse-MLA backends whenever ``use_sparse`` is true and never
# consults those two flags for DSA layers -- and DSA layers bypass backend
# auto-selection entirely, wiring straight to ``Step4SparseAttentionBackend``
# in ``step4.py``. So without this switch, setting those two env vars makes
# sliding layers call flash_attn but leaves DSA on Triton. Default DSA to
# flash_attn under exactly the same two flags, so one launch config moves both
# attention families together; ``VLLM_STEP4_DSA_FLASH_ATTN`` still overrides
# explicitly (e.g. to stage prefill out while decode is validated).
_FLASH_ATTN_STAGE_NAMES = frozenset({"decode", "prefill"})
_flash_attn_stages_raw = os.environ.get("VLLM_STEP4_DSA_FLASH_ATTN")
if _flash_attn_stages_raw is not None:
    FLASH_ATTN_DSA_STAGES = frozenset(
        name.strip() for name in _flash_attn_stages_raw.split(",") if name.strip()
    )
else:
    import vllm_hcu.platforms.envs as _henvs

    def _flash_attn_dsa_ops_available() -> bool:
        """Whether flash_attn here exposes the sparse ops at all.

        DCU builds ship ``fa_sparse_attention_{prefill,decode}`` but not their
        ``_fp8`` siblings, and the fp8 variants are what an fp8-KV step5 run
        needs. Probing once here turns "VLLM_HCU_USE_FLASH_ATTN is set" into an
        availability decision instead of an ImportError several minutes into
        model load, and keeps the env var meaning what it says for the sliding
        layers (which use a different flash_attn entry point and are unaffected).
        """
        # Probe exactly what _flash_attn_dsa_ops() imports, including the _fp8
        # variants. Checking only the bf16 pair is not enough: a DCU build can
        # ship fa_sparse_attention_{prefill,decode} and omit their _fp8
        # siblings, which is precisely the case this guard exists for.
        try:
            from flash_attn.flash_attn_interface import (  # noqa: F401
                fa_sparse_attention_decode,
                fa_sparse_attention_decode_fp8,
                fa_sparse_attention_prefill,
                fa_sparse_attention_prefill_fp8,
            )
        except ImportError:
            return False
        return True

    _flash_attn_requested = (
        _henvs.VLLM_HCU_USE_FLASH_ATTN or _henvs.VLLM_HCU_USE_FLASH_ATTN_UNIFIED
    )
    if _flash_attn_requested and _flash_attn_dsa_ops_available():
        FLASH_ATTN_DSA_STAGES = frozenset(_FLASH_ATTN_STAGE_NAMES)
    else:
        if _flash_attn_requested:
            logger.info(
                "step4 DSA: flash_attn sparse attention was requested by "
                "VLLM_HCU_USE_FLASH_ATTN/VLLM_HCU_USE_FLASH_ATTN_UNIFIED but "
                "this flash_attn build does not expose "
                "fa_sparse_attention_prefill/decode; keeping the vendored "
                "Triton DSA kernels."
            )
        FLASH_ATTN_DSA_STAGES = frozenset()
if FLASH_ATTN_DSA_STAGES - _FLASH_ATTN_STAGE_NAMES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FLASH_ATTN: unknown stage(s) "
        f"{sorted(FLASH_ATTN_DSA_STAGES - _FLASH_ATTN_STAGE_NAMES)}; "
        f"expected a comma-separated subset of {sorted(_FLASH_ATTN_STAGE_NAMES)}"
    )
if FLASH_ATTN_DSA_STAGES & BOLTOPS_STAGES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FLASH_ATTN and VLLM_STEP4_DSA_BOLTOPS both claim "
        f"stage(s) {sorted(FLASH_ATTN_DSA_STAGES & BOLTOPS_STAGES)}; a stage "
        f"can only run through one alternate sparse-attention implementation"
    )

# How much of the sparse attention arithmetic runs in fp8 on an fp8 cache. The
# vendored kernels honour this for both decode and prefill; boltops decode reads
# an fp8 cache but always widens before the dot, and boltops prefill has no fp8
# MMA path at all, so a boltops stage behaves as ``off`` regardless.
#
#   VLLM_STEP4_DSA_FP8_MATH=off             K/V widen to the query dtype
#   VLLM_STEP4_DSA_FP8_MATH=qk              Q quantized, QK dot runs on the fp8 MMA
#   VLLM_STEP4_DSA_FP8_MATH=full  (default) PV dot runs on the fp8 MMA as well
#
# ``full`` is the fastest setting and, as of 2026-09-21, the server default by
# explicit instruction. Per-layer cost at the production shape:
#
#   decode (topk=512):    117 us off, 83 us qk, 53 us full
#   prefill (8K queries): 9.2 ms off, 11.5 ms qk, 6.3 ms full
#
# ``full`` is the only one where an fp8 cache costs less latency than bf16 (77 us
# decode, 7.3 ms prefill) rather than more, and for prefill ``qk`` alone is the
# slowest of the four because the loop body holds both an fp8 K tile and a
# bf16-widened V tile and blows through the 256-VGPR budget (91 spills). ``full``
# is what removes bf16 from the loop, and only then does it fit (0 spills).
#
# Read this before trusting that default. ``off`` is the only setting inside the
# acceptance envelopes (stage 6 for decode, stage 5 for prefill), so the operator
# suite does not bless what now runs by default. The fp8 cache by itself stays in
# the envelope because kernel and reference read the same fp8 bytes and the
# storage error cancels; quantizing Q, and much more so P, is error no fp32
# reference models -- measured max relative error against an fp32 reference goes
# 4.0e-3 (off) -> 7.1e-3 (qk) -> 3.2e-2 (full) for decode, similar for prefill.
# No end-to-end score has been run at this setting. ``VLLM_STEP4_DSA_FP8_MATH=off``
# restores the gated path.
_FP8_MATH_MODES = {"off": (False, False), "qk": (True, False), "full": (True, True)}
# HCU default is 'off', not the CUDA-tuned 'full'. The fp8-math branches cast the
# query to e4m3 and feed the fp8 MMA, and DCU's Triton backend cannot lower them:
# the tiled hd192 prefill kernel fails in the MLIR pipeline with
# "Unsupported conversion from 'f8E4M3FN' to 'f32' with rounding mode rtne".
# That is a compile-time failure, not a numeric one, so the setting is not a
# tuning knob here -- 'qk' and 'full' simply do not build. 'off' is also the
# only mode inside the acceptance envelopes, so this costs only speed.
_fp8_math_default = "off" if current_platform.is_rocm() else "full"
_fp8_math = (
    os.environ.get("VLLM_STEP4_DSA_FP8_MATH", _fp8_math_default).strip().lower()
    or _fp8_math_default
)
if _fp8_math not in _FP8_MATH_MODES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FP8_MATH={_fp8_math!r}: expected one of "
        f"{sorted(_FP8_MATH_MODES)}"
    )
DECODE_QK_FP8, DECODE_PV_FP8 = _FP8_MATH_MODES[_fp8_math]
PREFILL_QK_FP8, PREFILL_PV_FP8 = _FP8_MATH_MODES[_fp8_math]
if _fp8_math != "off":
    # Loud, once, at import: the thing running is not the thing the ten gates
    # accepted, and a reader of these logs should not have to know that from a
    # source comment.
    logger.warning(
        "step4 DSA: fp8 math = %r. This is faster than 'off' but sits "
        "outside the acceptance envelopes (stage 6 decode, stage 5 prefill), "
        "so operator_parity does not cover it and no end-to-end score has been "
        "recorded at this setting. Set VLLM_STEP4_DSA_FP8_MATH=off for the gated path.",
        _fp8_math,
    )


@functools.lru_cache(maxsize=1)
def _boltops_attention() -> tuple[Any, Any]:
    """``(prefill, decode)`` from boltops, checked for an API version we know.

    Imported lazily: the module is only needed when a stage is switched to it,
    and an unavailable library should fail where the switch was asked for.
    """
    try:
        import boltops.step4_dsa as ops
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "VLLM_STEP4_DSA_BOLTOPS asked for boltops sparse attention but "
            "boltops.step4_dsa could not be imported"
        ) from exc
    version = getattr(ops, "STEP4_DSA_OPS_API_VERSION", None)
    if version != "step4-dsa-ops-v1":
        raise RuntimeError(
            f"boltops.step4_dsa speaks {version!r}, this backend calls "
            f"'step4-dsa-ops-v1'. The prefill/decode signatures may have moved."
        )
    logger.info(
        "step4 DSA: %s from boltops.step4_dsa (%s)",
        ", ".join(sorted(BOLTOPS_STAGES)),
        version,
    )
    return ops.triton_sparse_attention_prefill, ops.triton_sparse_attention_decode


@functools.lru_cache(maxsize=1)
def _flash_attn_dsa_ops() -> tuple[Any, Any, Any, Any]:
    """``(prefill_bf16, prefill_fp8, decode_bf16, decode_fp8)`` from flash_attn.

    Imported lazily, same reasoning as ``_boltops_attention``: only needed once
    a stage is actually switched to it, and an unavailable/stale build should
    fail at the switch, not at import time for everyone who never sets
    ``VLLM_STEP4_DSA_FLASH_ATTN``/``VLLM_HCU_USE_FLASH_ATTN*``.

    These four ops come from flash-attention-cutlass's own Stage-5/6 sparse
    port (``fa_sparse_attention_{prefill,decode}[_fp8]`` in
    ``flash_attn_interface.py``), which speaks the same ``packed_regions``/
    ``region_counts`` bit-packing (``phys_region | (valid_tokens << 24)``) and
    ``[tokens, num_kv_groups, head_dim]`` flat KV-cache layout as the vendored
    Triton kernels here, at the same ``head_dim=192`` Step-4 uses throughout.
    """
    try:
        from flash_attn.flash_attn_interface import (
            fa_sparse_attention_decode,
            fa_sparse_attention_decode_fp8,
            fa_sparse_attention_prefill,
            fa_sparse_attention_prefill_fp8,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "VLLM_STEP4_DSA_FLASH_ATTN (or VLLM_HCU_USE_FLASH_ATTN/"
            "VLLM_HCU_USE_FLASH_ATTN_UNIFIED) asked for flash_attn sparse "
            "attention but flash_attn.flash_attn_interface could not be "
            "imported, or does not expose the fa_sparse_attention_* ops"
        ) from exc
    logger.info(
        "step4 DSA: %s from flash_attn.flash_attn_interface",
        ", ".join(sorted(FLASH_ATTN_DSA_STAGES)),
    )
    return (
        fa_sparse_attention_prefill,
        fa_sparse_attention_prefill_fp8,
        fa_sparse_attention_decode,
        fa_sparse_attention_decode_fp8,
    )


def _flash_attn_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float | None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter matching ``sparse_attention_prefill``'s call in ``forward()``.

    ``_ignored`` swallows the vendored kernel's tiled-prefill-only kwargs
    (``logical_regions``, ``q_positions``, ``request_indices``, ``block_q``)
    and the fp8-math toggles (``qk_fp8``, ``pv_fp8``, ``q_scale``): the
    flash-attention-cutlass port has no query-tiled path (it raises
    ``NotImplementedError`` if ``logical_regions`` is passed) and always
    quantizes/dequantizes K/V by ``k_scale``/``v_scale`` on its own fp8 MMA
    path, with no separate query-quantization knob.
    """
    prefill_bf16, prefill_fp8, _, _ = _flash_attn_dsa_ops()
    if key_cache.dtype == torch.float8_e4m3fn:
        return prefill_fp8(
            query,
            key_cache,
            value_cache,
            packed_regions,
            region_counts,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return prefill_bf16(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
    )


def _flash_attn_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float | None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter matching ``sparse_attention_decode``'s call in ``forward()``.

    Always calls with ``num_splits=1``: the C++ entry point ignores the
    argument and always returns final merged BF16 output plus natural-log
    LSE internally (see its docstring), so unlike the vendored Triton decode
    path this adapter never calls ``merge_split_states`` -- there is nothing
    left to merge. ``_ignored`` swallows ``qk_fp8``/``pv_fp8``/``q_scale``,
    which have no equivalent here (see ``_flash_attn_prefill``).
    """
    _, _, decode_bf16, decode_fp8 = _flash_attn_dsa_ops()
    if key_cache.dtype == torch.float8_e4m3fn:
        return decode_fp8(
            query,
            key_cache,
            value_cache,
            packed_regions,
            region_counts,
            kv_seqlens,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            softmax_scale=softmax_scale,
            num_splits=1,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return decode_bf16(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
        num_splits=1,
    )


@functools.lru_cache(maxsize=64)
def _boltops_decode_config(
    num_reqs: int,
    num_heads: int,
    head_dim: int,
    num_kv_groups: int,
    topk: int,
    region_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> dict[str, Any]:
    """boltops' tuned decode launch config, resolved without reading the device.

    boltops keys its decode JSON partly on the batch's longest context and
    reads that with ``kv_seqlens.max().item()`` -- a blocking device-to-host
    sync on every call, and illegal inside a CUDA graph capture, which is
    exactly the path decode takes here. Passing ``config=`` explicitly skips
    that lookup, so do the lookup on the host instead: everything else the
    selector reads is a shape, and the longest context is already on the host in
    ``seq_lens_cpu``. Meta tensors stand in for the real ones.

    ``cache_dtype`` is passed separately from the query ``dtype`` because the
    selector keys its ``dtype`` entry off the KV cache: the query stays bfloat16
    on the fp8 serving path, so folding the two would resolve the bfloat16 tile
    for an fp8 cache.

    Returns ``{}`` -- which boltops reads as "keep the built-in tile" -- when
    nothing matches or the private selector has moved. Note that nothing matches
    on a device with no shipped JSON: the files are named after
    ``torch.cuda.get_device_name()``, and boltops ships ``BW1100`` -- aliased
    from ``BW1101`` since its commit 28239f4 -- plus ``BW200`` and ``K100_AI``.
    """
    from boltops.step4_dsa.triton import decode as bolt_decode

    def meta(*shape: int, dt: torch.dtype = dtype) -> torch.Tensor:
        return torch.empty(shape, device="meta", dtype=dt)

    try:
        config = bolt_decode._load_sparse_attn_decode_config(
            meta(num_reqs, num_heads, head_dim),
            meta(1, num_kv_groups, head_dim, dt=cache_dtype),
            meta(num_kv_groups * num_reqs, topk, dt=torch.int64),
            torch.tensor([max_seq_len], dtype=torch.int32),  # host-side on purpose
            1,
            region_size,
        )
    except Exception:  # noqa: BLE001 - private API, any failure means "untuned"
        logger.warning_once(
            "step4 DSA: could not resolve a boltops decode config; running its "
            "built-in tile. Check whether _load_sparse_attn_decode_config still "
            "exists in boltops.step4_dsa.triton.decode."
        )
        return {}
    return config or {}


# --------------------------------------------------------------------------
# Driver-local state updates
# --------------------------------------------------------------------------


@triton.jit
def _store_live_summary_rows_kernel(
    dst_ptr,
    src_ptr,
    slots_ptr,
    live_ptr,
    dst_row_stride,
    dst_proxy_stride,
    src_row_stride,
    src_proxy_stride,
    PROXY_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
) -> None:
    """Copy completed region summaries without compacting live rows on host."""
    row = tl.program_id(0)
    if tl.load(live_ptr + row) == 0:
        return

    slot = tl.load(slots_ptr + row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_P)
    mask = offsets < PROXY_DIM
    values = tl.load(
        src_ptr + row * src_row_stride + offsets * src_proxy_stride,
        mask=mask,
    )
    tl.store(
        dst_ptr + slot * dst_row_stride + offsets * dst_proxy_stride,
        values,
        mask=mask,
    )


def _store_live_summary_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    slots: torch.Tensor,
    live: torch.Tensor,
) -> None:
    """Store fixed-shape FP8 rows; dead records leave the cache untouched."""
    rows = source.shape[0]
    if rows == 0:
        return
    proxy = source.shape[-1]
    _store_live_summary_rows_kernel[(rows,)](
        destination,
        source,
        slots,
        live,
        destination.stride(0),
        destination.stride(2),
        source.stride(0),
        source.stride(2),
        PROXY_DIM=proxy,
        BLOCK_P=triton.next_power_of_2(proxy),
        num_warps=4,
    )


@triton.jit
def _spec_summaries_kernel(
    tok_k_ptr,
    tok_z_ptr,
    pend_k_ptr,
    pend_z_ptr,
    summary_ptr,
    query_start_loc_ptr,
    head_ptr,
    tail_ptr,
    pending_in_ptr,
    pending_out_ptr,
    tail_reg_ptr,
    counts_ptr,
    slots_ptr,
    live_ptr,
    keep_rows_ptr,
    num_tokens,
    trash_row,
    stride_tok,
    stride_tok_z,
    stride_pend_row,
    stride_pend_pos,
    stride_summary_row,
    NRM: tl.constexpr,
    REGION: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """``_update_summaries_spec`` in one launch: one program per buffer region.

    The unfused path materialises the per-request compression buffer (pending
    head, then the request's query tokens scattered in at ``head``), compresses
    it, stores the live summaries and writes the tail / keep rows back -- about
    28 launches per layer, nearly all of them launch-bound. Here each program
    gathers its own region of that buffer straight from ``pending`` and the
    token rows, so nothing is materialised.

    The compression arithmetic is ``_csa_compress_regions_kernel``'s, statement
    for statement and at the same tile shape and launch options, so summaries
    are bit-identical. The only cross-program hazard is a ``pending`` row read
    as one request's head and written by another program; the schedule rules it
    out (``Step4PendingTable.acquire`` never recycles a row touched this step),
    and the one row a program may both read and write -- its own head row,
    reused as tail or keep row -- is read before its values are stored.
    """
    g = tl.program_id(0)
    req = g // NRM
    j = g % NRM

    head = tl.load(head_ptr + req)
    q_start = tl.load(query_start_loc_ptr + req)
    # Graph-padding requests may carry a stale, non-monotonic entry: their
    # regions are dead (count 0, not live, trash rows), so only the loads have
    # to stay in bounds.
    q_len = tl.maximum(tl.load(query_start_loc_ptr + req + 1) - q_start, 0)

    t = tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    t_ok = t < REGION
    dim_ok = dims < PROXY_DIM
    off = j * REGION + t
    i = off - head
    tok = (q_start + i).to(tl.int64)
    from_tok = t_ok & (i >= 0) & (i < q_len) & (tok >= 0) & (tok < num_tokens)
    from_pend = t_ok & (off < head)

    p_in = tl.load(pending_in_ptr + req)
    tok_offs = tok[:, None] * stride_tok + dims[None, :]
    pend_offs = p_in * stride_pend_row + off[:, None] * stride_pend_pos + dims[None, :]
    tok_mask = from_tok[:, None] & dim_ok[None, :]
    pend_mask = from_pend[:, None] & dim_ok[None, :]
    key = tl.where(
        tok_mask,
        tl.load(tok_k_ptr + tok_offs, mask=tok_mask, other=0.0),
        tl.load(pend_k_ptr + pend_offs, mask=pend_mask, other=0.0),
    )
    zed = tl.where(
        tok_mask,
        tl.load(
            tok_z_ptr + tok[:, None] * stride_tok_z + dims[None, :],
            mask=tok_mask,
            other=0.0,
        ),
        tl.load(pend_z_ptr + pend_offs, mask=pend_mask, other=0.0),
    )

    if tl.load(live_ptr + g) != 0:
        count = tl.load(counts_ptr + g)
        mask = (t < count)[:, None] & dim_ok[None, :]
        logits = tl.where(mask, zed, float("-inf")).to(tl.float32)
        values = tl.where(mask, key, 0.0).to(tl.float32)
        shift = tl.max(tl.where(mask, logits, float("-inf")))
        weights = tl.where(mask, tl.exp(logits - shift), 0.0)
        denominator = tl.sum(weights, axis=0)
        numerator = tl.sum(weights * values, axis=0)
        summary = tl.where(
            denominator > 0.0, numerator / tl.maximum(denominator, 1e-20), 0.0
        )
        slot = tl.load(slots_ptr + g)
        tl.store(
            summary_ptr + slot * stride_summary_row + dims,
            summary.to(tl.bfloat16).to(tl.float8e4nv),
            mask=dim_ok,
        )

    row_offs = t[:, None] * stride_pend_pos + dims[None, :]
    if g == tl.load(tail_reg_ptr + req):
        p_out = tl.load(pending_out_ptr + req)
        if p_out != trash_row:
            tail_mask = (t < tl.load(tail_ptr + req))[:, None] & dim_ok[None, :]
            tl.store(
                pend_k_ptr + p_out * stride_pend_row + row_offs, key, mask=tail_mask
            )
            tl.store(
                pend_z_ptr + p_out * stride_pend_row + row_offs, zed, mask=tail_mask
            )
    keep = tl.load(keep_rows_ptr + g)
    if keep != trash_row:
        row_mask = t_ok[:, None] & dim_ok[None, :]
        tl.store(pend_k_ptr + keep * stride_pend_row + row_offs, key, mask=row_mask)
        tl.store(pend_z_ptr + keep * stride_pend_row + row_offs, zed, mask=row_mask)


# Fold _update_summaries_spec into _spec_summaries_kernel. Bit-identical; "0"
# restores the unfused torch path.
_SPEC_FUSED_SUMMARIES = (
    os.environ.get("VLLM_STEP4_DSA_SPEC_FUSED_SUMMARIES", "1") == "1"
)

# One GEMM for the four indexer projections (Step4Indexer._merged_projection).
_MERGED_PROJ = os.environ.get("STEP5_DSA_MERGED_PROJ", "1") == "1"


@triton.jit
def _store_kv_rows_kernel(
    k_dst_ptr,
    v_dst_ptr,
    k_src_ptr,
    v_src_ptr,
    slots_ptr,
    k_dst_row_stride: tl.int64,
    v_dst_row_stride: tl.int64,
    k_dst_group_stride: tl.int64,
    v_dst_group_stride: tl.int64,
    k_src_row_stride: tl.int64,
    v_src_row_stride: tl.int64,
    k_scale,
    v_scale,
    ROW_ELEMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    QUANTIZE: tl.constexpr,
) -> None:
    """Store dense K/V rows while ignoring padded slots on device.

    The destination row is addressed as ``[group, HEAD_DIM]`` rather than as one
    flat run. vLLM 0.28 hands each layer a *fused* cache page with K and V side
    by side in the content axis, so one KV group's head sits ``2 * HEAD_DIM``
    elements from the next and the two strides cannot be collapsed into one.
    With ``k_dst_group_stride == HEAD_DIM`` -- the split layout this file was
    written against -- the decomposition below is exactly the flat run.
    """
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot < 0:
        return

    offsets = tl.arange(0, BLOCK_ELEMS)
    mask = offsets < ROW_ELEMS
    group = offsets // HEAD_DIM
    dim = offsets % HEAD_DIM
    k_dst = k_dst_ptr + slot * k_dst_row_stride + group * k_dst_group_stride + dim
    v_dst = v_dst_ptr + slot * v_dst_row_stride + group * v_dst_group_stride + dim
    key = tl.load(
        k_src_ptr + row * k_src_row_stride + offsets,
        mask=mask,
    )
    value = tl.load(
        v_src_ptr + row * v_src_row_stride + offsets,
        mask=mask,
    )
    if QUANTIZE:
        # Divide by the scale, then clamp before the cast. Letting tl.store do the
        # narrowing on its own is what makes an fp8 cache quietly wrong: e4m3
        # saturates at 448, so every activation above it would land on the same
        # number with no indication anything happened. This is the same division
        # the sparse kernels undo, and the same clamp DSALayerCache._store_dtype
        # applies in the gated source. The bound is spelled as a literal rather
        # than FP8_E4M3_MAX because Triton cannot close over a plain float global;
        # act_quant in the kernel source writes -448.0/448.0 for the same reason.
        key = tl.clamp(key.to(tl.float32) / k_scale, -448.0, 448.0)
        value = tl.clamp(value.to(tl.float32) / v_scale, -448.0, 448.0)
    tl.store(k_dst, key, mask=mask)
    tl.store(v_dst, value, mask=mask)


def _store_kv_rows(
    k_destination: torch.Tensor,
    v_destination: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Store K/V rows; negative slots leave the cache untouched.

    Each ``[num_kv_heads, head_size]`` row must be internally dense. K and V
    may have different token-row strides, as they do when unpacked from Step-4's
    fused QKV projection.

    When the destination is ``float8_e4m3`` the rows are quantized on the way in,
    dividing by the scale the sparse kernels later multiply back. The destination
    dtype is what selects that path, not a flag: the cache tensor is allocated by
    the runner from ``kv_cache_dtype``, so it is the one thing that cannot get out
    of step with what the kernels will read.
    """
    rows = key.shape[0]
    if rows == 0:
        return
    row_elems = key.shape[1] * key.shape[2]
    head_dim = key.shape[2]
    # vLLM 0.28's fused page interleaves K and V in the content axis, so a group
    # is `stride(1)` elements away rather than `head_dim`. The kernels take that
    # as an argument; a split layout passes head_dim and reduces to the flat run.
    k_group_stride = k_destination.stride(1)
    v_group_stride = v_destination.stride(1)
    _store_kv_rows_kernel[(rows,)](
        k_destination,
        v_destination,
        key,
        value,
        slots,
        k_destination.stride(0),
        v_destination.stride(0),
        k_group_stride,
        v_group_stride,
        key.stride(0),
        value.stride(0),
        k_scale,
        v_scale,
        ROW_ELEMS=row_elems,
        HEAD_DIM=head_dim,
        BLOCK_ELEMS=triton.next_power_of_2(row_elems),
        QUANTIZE=k_destination.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )


@triton.jit
def _decode_indexer_logits_kernel(
    index_q_ptr,
    summary_ptr,
    block_table_ptr,
    weights_ptr,
    out_ptr,
    nscore_ptr,
    num_reqs,
    stride_q_req,
    stride_q_group,
    stride_q_head,
    stride_s_row,
    stride_bt_req,
    stride_w_req,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    regions_per_page: tl.constexpr,
    query_dtype: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Batched decode form of ``_indexer_logits_kernel``.

        score[r, j] = sum_h relu(dot(q_rh, k_rj)) * w_rh

    Same maths, same accumulation order, one difference: the key set is indexed
    per request rather than shared, so requests move from the *host* loop that
    ``_score`` runs into the program grid. That is the whole point -- at decode
    every request contributes one query, so the released ``indexer_logits`` gets
    called once per request with ``seq_q == 1``, and 23 layers x num_reqs of those
    launches is most of the DSA path's eager-launch count.

    **Why this is bit-identical to the reference and not merely close.** Each
    output element is an independent dot product over ``proxy_dim``, so
    ``tl.dot``'s accumulation order along K does not depend on the M extent. The
    reference runs M = BLOCK_Q = 64 with one valid query and 63 masked rows; this
    runs M = 16 the same way. Keeping the per-head loop (rather than folding heads
    into M and reducing) preserves the fp32 add order of ``accumulator``, which is
    what a tree reduction over heads would have changed. Selection is a top-k over
    these scores, so a last-ULP difference is not cosmetic: it reorders ties.
    """
    req = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    # One query per request at decode; the remaining rows exist only so that
    # tl.dot has a legal M extent, exactly as in the reference.
    query_ok = queries < 1
    key_ok = keys < tl.load(nscore_ptr + req)

    # Address the fp8 summary pool the same way the selector does: physical KV
    # block `b` owns summary rows [b * rpp, (b + 1) * rpp). Reading it in place
    # rather than taking a pre-gathered key tensor is what keeps every argument
    # here a persistent buffer -- and it drops a [num_reqs, width, proxy] gather
    # plus its fp8->bf16 cast from every one of the 23 layers.
    pages = tl.load(
        block_table_ptr + req * stride_bt_req + keys // regions_per_page,
        mask=key_ok,
        other=0,
    ).to(tl.int64)
    slots = pages * regions_per_page + keys % regions_per_page
    key_tile = tl.load(
        summary_ptr + slots[None, :] * stride_s_row + dims[:, None],
        mask=key_ok[None, :],
        other=0.0,
    ).to(query_dtype)

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + req * stride_q_req
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :]
            + queries[:, None] * 0,
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + req * stride_w_req + group * stride_w_group + head
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight

    rows = group * num_reqs + req
    tl.store(
        out_ptr + rows * stride_out_row + keys[None, :] + queries[:, None] * 0,
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def decode_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    nscore: torch.Tensor,
    out: torch.Tensor,
    *,
    regions_per_page: int,
    block_k: int = 128,
    q_prerounded: bool = False,
) -> None:
    """Score every decode request against its own key set, in one launch.

    Args:
        index_q: ``[num_reqs, groups, heads_per_group, proxy_dim]``.
        weights: ``[num_reqs, groups, heads_per_group]``, prescale already applied.
        summary: the fp8 region-summary pool, ``[rows, 1, proxy_dim]``. Read in
            place through ``block_table``; nothing is gathered.
        block_table: ``[num_reqs, pages]`` logical->physical map, the same one
            the region selector reads.
        nscore: ``[num_reqs]`` int32, complete regions visible to each request.
        out: ``[groups * num_reqs, width]`` float32, group-major. Only the first
            ``nscore[req]`` columns of a row are written. That is exactly the
            range the region selector then reads -- its ``history_lengths`` is
            the same ``(seq_len - 1) // region_size`` -- and every load past it
            is masked off, so the columns this kernel skips are never observed
            and the caller does not have to clear them. Do not narrow either
            bound on the assumption that the other one still covers you.

    Only ``index_q`` is rounded here. The pool is already e4m3, so the
    reference's key-side ``round_activations_e4m3`` is the identity on it: every
    finite e4m3 value is exact in bf16, and widening then re-rounding returns
    the same byte.
    """
    num_reqs, groups, heads_per_group, proxy_dim = index_q.shape
    width = out.shape[1]
    if summary.shape[-1] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} pool={summary.shape[-1]}")
    if block_table.shape[0] < num_reqs:
        raise ValueError(
            f"block table has {block_table.shape[0]} rows, need {num_reqs}"
        )
    if out.shape[0] != groups * num_reqs:
        raise ValueError(f"out has {out.shape[0]} rows, expected {groups * num_reqs}")
    if width > block_table.shape[1] * regions_per_page:
        raise ValueError(
            f"score width {width} exceeds the {block_table.shape[1]} pages x "
            f"{regions_per_page} regions the block table can address"
        )

    # q_prerounded: index_q came from indexer_norm_rope(round_q=True).
    quantized_q = index_q if q_prerounded else round_activations_e4m3(index_q)

    _decode_indexer_logits_kernel[(num_reqs, triton.cdiv(width, block_k), groups)](
        quantized_q,
        summary,
        block_table,
        weights,
        out,
        nscore,
        num_reqs,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        summary.stride(0),
        block_table.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        regions_per_page=regions_per_page,
        query_dtype=tl.bfloat16 if quantized_q.dtype is torch.bfloat16 else tl.float16,
        BLOCK_Q=16,
        BLOCK_K=block_k,
    )


# Query tokens per request the verify scorer below can hold: one dot row each.
SPEC_INDEXER_MAX_QUERY = 16


@triton.jit
def _spec_indexer_logits_kernel(
    index_q_ptr,
    summary_ptr,
    block_table_ptr,
    weights_ptr,
    out_ptr,
    nscore_ptr,
    query_start_loc_ptr,
    num_tokens,
    stride_q_tok,
    stride_q_group,
    stride_q_head,
    stride_s_row,
    stride_bt_req,
    stride_w_tok,
    stride_w_group,
    stride_out_row,
    weight_scale,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    regions_per_page: tl.constexpr,
    query_dtype: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Speculative-verify form of ``_decode_indexer_logits_kernel``.

    One program per (request, key block, group) instead of per query token:
    the request's query tokens are the rows of the dot, so each key tile is
    read from the summary pool once for all of them rather than once per token
    (with 1 of 16 dot rows doing work). 3.7x faster at 16 requests x 4 tokens.

    Bit-identical to the per-token kernel for the reason given there: every
    output element is an independent dot over ``proxy_dim``, whichever row it
    sits in, and the per-head loop keeps the fp32 add order of ``accumulator``.
    Each token still stops at its own ``nscore``; the tile is loaded up to the
    widest of them.

    ``weight_scale`` multiplies the widened head weight, so the caller can pass
    the raw bf16 head-weight projection instead of launching the widening and
    the prescale as separate kernels -- the same IEEE ops the torch path runs.
    The query's e4m3 round trip is *not* done here: converting to fp8 in
    software on every key block made the kernel 2.6x slower on gfx938. It is
    done once, where the query is produced (``indexer_norm_rope``'s
    ``round_q``).
    """
    req = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + req)
    q_len = tl.load(query_start_loc_ptr + req + 1) - q_start
    queries = tl.arange(0, BLOCK_Q)
    # Graph-padding requests have q_len 0 (or a stale, non-monotonic entry):
    # no row is live and nothing is written.
    query_ok = queries < q_len
    tokens = q_start + queries
    nscore = tl.load(nscore_ptr + tokens, mask=query_ok, other=0)
    widest = tl.max(nscore, axis=0)
    if k_block * BLOCK_K >= widest:
        return

    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    key_any = keys < widest
    pages = tl.load(
        block_table_ptr + req * stride_bt_req + keys // regions_per_page,
        mask=key_any,
        other=0,
    ).to(tl.int64)
    slots = pages * regions_per_page + keys % regions_per_page
    key_tile = tl.load(
        summary_ptr + slots[None, :] * stride_s_row + dims[:, None],
        mask=key_any[None, :],
        other=0.0,
    ).to(query_dtype)

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + tokens[:, None] * stride_q_tok
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + tokens * stride_w_tok + group * stride_w_group + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        head_weight = head_weight * weight_scale
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    rows = group * num_tokens + tokens
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & (keys[None, :] < nscore[:, None]),
    )


@triton.jit
def _spec_indexer_stacked_kernel(
    index_q_ptr,
    summary_ptr,
    block_table_ptr,
    weights_ptr,
    out_ptr,
    nscore_ptr,
    query_start_loc_ptr,
    num_tokens,
    stride_q_tok,
    stride_q_group,
    stride_q_head,
    stride_s_row,
    stride_bt_req,
    stride_w_tok,
    stride_w_group,
    stride_out_row,
    weight_scale,
    proxy_dim: tl.constexpr,
    regions_per_page: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """``_spec_indexer_logits_kernel`` with the heads stacked into the dot rows.

    For 4 heads per group: row ``t * 4 + h`` of the single dot is token ``t``,
    head ``h``, so a 4-token request fills one 16-row MFMA tile instead of four
    quarter-full ones, and the dot runs fp8 x fp8 on the pool as stored. The
    query is already e4m3-valued (``q_prerounded``), so its cast to fp8 here is
    exact and every product is the same as in the bf16 dot; the per-head ReLU,
    weight and the head sum ``(((0 + h0) + h1) + h2) + h3`` are kept as well.
    Only the dot's internal fp32 summation order over ``proxy_dim`` differs
    (~1e-5 relative). 99 -> 43us at 16 requests x 4 tokens, 8704 regions.
    """
    req = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + req)
    q_len = tl.load(query_start_loc_ptr + req + 1) - q_start
    queries = tl.arange(0, BLOCK_Q)
    query_ok = queries < q_len
    tokens = q_start + queries
    nscore = tl.load(nscore_ptr + tokens, mask=query_ok, other=0)
    widest = tl.max(nscore, axis=0)
    if k_block * BLOCK_K >= widest:
        return

    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    key_any = keys < widest
    pages = tl.load(
        block_table_ptr + req * stride_bt_req + keys // regions_per_page,
        mask=key_any,
        other=0,
    ).to(tl.int64)
    slots = pages * regions_per_page + keys % regions_per_page
    key_tile = tl.load(
        summary_ptr + slots[None, :] * stride_s_row + dims[:, None],
        mask=key_any[None, :],
        other=0.0,
    )

    heads = tl.arange(0, 4)
    rows_ok = query_ok[:, None] & (heads[None, :] >= 0)
    query_tile = tl.load(
        index_q_ptr
        + tokens[:, None, None] * stride_q_tok
        + group * stride_q_group
        + heads[None, :, None] * stride_q_head
        + dims[None, None, :],
        mask=rows_ok[:, :, None],
        other=0.0,
    )
    query_tile = tl.reshape(query_tile, (BLOCK_Q * 4, proxy_dim))
    query_tile = query_tile.to(summary_ptr.dtype.element_ty)
    head_weight = tl.load(
        weights_ptr
        + tokens[:, None] * stride_w_tok
        + group * stride_w_group
        + heads[None, :],
        mask=rows_ok,
        other=0.0,
    ).to(tl.float32)
    head_weight = head_weight * weight_scale
    scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0) * tl.reshape(head_weight, (BLOCK_Q * 4,))[:, None]
    # [BLOCK_Q * 4, BLOCK_K] -> per-head [BLOCK_Q, BLOCK_K] slices.
    scores = tl.permute(tl.reshape(scores, (BLOCK_Q, 4, BLOCK_K)), (0, 2, 1))
    scores = tl.reshape(scores, (BLOCK_Q, BLOCK_K, 2, 2))
    heads_02, heads_13 = tl.split(scores)
    head0, head2 = tl.split(heads_02)
    head1, head3 = tl.split(heads_13)
    accumulator = (((0.0 + head0) + head1) + head2) + head3

    rows = group * num_tokens + tokens
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & (keys[None, :] < nscore[:, None]),
    )


def spec_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    nscore: torch.Tensor,
    out: torch.Tensor,
    *,
    num_reqs: int,
    regions_per_page: int,
    block_k: int = 128,
    weight_scale: float = 1.0,
    q_prerounded: bool = False,
    max_query: int | None = None,
) -> None:
    """``decode_indexer_logits`` for verify batches, one program per request.

    Args:
        index_q: ``[num_tokens, groups, heads_per_group, proxy_dim]``.
        weights: ``[num_tokens, groups, heads_per_group]``, multiplied by
            ``weight_scale`` (in fp32, after widening) inside the kernel.
        q_prerounded: ``index_q`` already went through
            ``round_activations_e4m3`` (``indexer_norm_rope(round_q=True)``),
            so the separate round-trip pass is skipped. The rounding is
            idempotent, so this only saves the launches.
        summary: the fp8 region-summary pool, read in place.
        block_table: ``[num_reqs, pages]``, one row per *request* (not the
            per-token copy the per-token kernel takes).
        query_start_loc: ``[num_reqs + 1]`` int32; request ``r`` owns tokens
            ``[query_start_loc[r], query_start_loc[r + 1])``, at most
            ``SPEC_INDEXER_MAX_QUERY`` of them.
        nscore: ``[num_tokens]`` int32, regions visible to each token.
        out: ``[groups * num_tokens, width]`` float32, group-major; written
            exactly where ``decode_indexer_logits`` writes it.
        max_query: upper bound on any request's query length. When given
            (with 4 heads per group, an fp8 pool and ``q_prerounded``), the
            stacked fp8 kernel is used; see ``_spec_indexer_stacked_kernel``.
    """
    num_tokens, groups, heads_per_group, proxy_dim = index_q.shape
    width = out.shape[1]
    if summary.shape[-1] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} pool={summary.shape[-1]}")
    if block_table.shape[0] < num_reqs or query_start_loc.shape[0] < num_reqs + 1:
        raise ValueError(
            f"block table / query_start_loc too short for {num_reqs} requests"
        )
    if out.shape[0] != groups * num_tokens:
        raise ValueError(f"out has {out.shape[0]} rows, expected {groups * num_tokens}")
    if width > block_table.shape[1] * regions_per_page:
        raise ValueError(
            f"score width {width} exceeds the {block_table.shape[1]} pages x "
            f"{regions_per_page} regions the block table can address"
        )

    if (
        _SPEC_INDEXER_FP8_DOT
        and max_query is not None
        and max_query <= SPEC_INDEXER_MAX_QUERY
        and heads_per_group == 4
        and q_prerounded
        and summary.dtype == torch.float8_e4m3fn
    ):
        # 32-key blocks on one warp: the dot is tiny, so more, smaller programs
        # keep more CUs busy (bk32/nw1 43us vs bk128/nw4 53us).
        _spec_indexer_stacked_kernel[(num_reqs, triton.cdiv(width, 32), groups)](
            index_q,
            summary,
            block_table,
            weights,
            out,
            nscore,
            query_start_loc,
            num_tokens,
            index_q.stride(0),
            index_q.stride(1),
            index_q.stride(2),
            summary.stride(0),
            block_table.stride(0),
            weights.stride(0),
            weights.stride(1),
            out.stride(0),
            float(weight_scale),
            proxy_dim=proxy_dim,
            regions_per_page=regions_per_page,
            # >= 4 so the dot has >= 16 rows.
            BLOCK_Q=max(4, triton.next_power_of_2(int(max_query))),
            BLOCK_K=32,
            num_warps=1,
        )
        return

    quantized_q = index_q if q_prerounded else round_activations_e4m3(index_q)

    _spec_indexer_logits_kernel[(num_reqs, triton.cdiv(width, block_k), groups)](
        quantized_q,
        summary,
        block_table,
        weights,
        out,
        nscore,
        query_start_loc,
        num_tokens,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        summary.stride(0),
        block_table.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        float(weight_scale),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        regions_per_page=regions_per_page,
        query_dtype=tl.bfloat16 if quantized_q.dtype is torch.bfloat16 else tl.float16,
        BLOCK_Q=SPEC_INDEXER_MAX_QUERY,
        BLOCK_K=block_k,
        num_warps=4,
    )


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@dataclass
class Step4DSAMetadata:
    """Everything the DSA path needs that vLLM already computes.

    This is a repackaging of ``CommonAttentionMetadata``, not a computation. The
    region selection itself is *not* here: it depends on ``hidden_states``, which
    the attention layer never sees, so the indexer computes it model-side and
    hands it over through the layer object (see ``Step4DSAImpl.forward``).
    """

    # Batch split. With reorder_batch_threshold=1 decodes are at the front.
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    num_actual_tokens: int

    block_table: torch.Tensor  # [num_reqs, max_blocks] int32
    slot_mapping: torch.Tensor  # [num_tokens] int64, -1 = padding
    seq_lens: torch.Tensor  # [num_reqs] int32, context incl. current token
    query_start_loc: torch.Tensor  # [num_reqs + 1] int32
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    # Absolute position of every query token; the region selector needs it to
    # derive each row's causal horizon. May be None on a dummy/profiling run.
    positions: torch.Tensor | None
    max_seq_len: int
    block_size: int
    # Host mirror of ``block_table``, when the runner has one. The block table is
    # built on the host and uploaded, so this is the same data one step earlier
    # in the pipeline, not a read-back -- which is the whole point: the pending
    # bookkeeping below is keyed by physical summary slot, and deriving the slot
    # from the device tensor costs a sync that stalls the step. ``None`` means
    # "no mirror available", and every consumer keeps its device-side fallback.
    # Rows at or past num_reqs are stale rather than NULL_BLOCK_ID; they carry
    # seq_len == 0, which is what the masks below key on.
    block_table_np: np.ndarray | None = None
    # The decode-only compression schedule, when the metadata builder was able
    # to construct it. Building it there rather than inside the indexer op is
    # what makes the decode path capturable: the schedule is where all the host
    # control flow and the one device-to-host read live, and the builder runs
    # eagerly on every step -- including the steps that are only a graph replay.
    schedule: Step4DSASchedule | None = None


class Step4DSAMetadataBuilder(AttentionMetadataBuilder[Step4DSAMetadata]):
    # Decode-only steps are capturable; steps with a prefill row are not, which
    # is what this level means. Three things used to rule capture out, all of
    # them inside the indexer op: a per-request Python loop, a device-to-host
    # read of the block table, and a score tensor whose width followed the
    # longest context in the batch. build() below now does the first two, into
    # buffers whose addresses outlive the capture, and the width is pinned to
    # Step4DSAState.graph_regions. A prefill row still sizes its logits by its
    # own query length, so mixed and prefill steps stay on the piecewise path
    # where the op runs eagerly -- see _require_splitting_op.
    #
    # UNIFORM_BATCH rather than single-token: a speculative verify step (every
    # request scoring 1 + num_speculative_tokens queries) is a decode step too,
    # run one query token per row -- see Step4DSAState._spec_decode_schedule.
    # At UNIFORM_SINGLE_TOKEN_DECODE vLLM downgrades the whole model to
    # PIECEWISE as soon as spec decode is on, which put all attention calls
    # and this op back on the host every step (and left the MTP drafter with
    # no graph at all).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Pull short queries to the front so the batch splits cleanly into the
    # reference implementation's two paths: decode reads one query per token
    # row, prefill reads a ragged run. Raised to 1 + num_speculative_tokens in
    # __init__ so verify rows count as decodes.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self._state: Step4DSAState | None = None
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        # Builders are made by initialize_kv_cache, i.e. whenever a new KV
        # cache is bound. 0.28's CUDA graph memory profiling binds a throwaway
        # cache of a few blocks first and runs the DSA layers against it; a
        # state still sized from that cache would keep a 32-row summary pool
        # and a 32-region decode width, and the first long prefill would write
        # off its end (a kernel VMFault). Drop it so the next forward sizes
        # everything from the cache that is actually in use.
        state = self._dsa_state()
        if state is not None:
            state.reset_allocation()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if DENSE_EQUIV:
            # The dense-equivalence bypass sets topk to the batch's own score
            # width, so the packed region list is [rows, width + 1] -- a shape
            # that moves with the longest context in the batch and cannot be
            # recorded. Pinning it to the graph width instead would mean a
            # 64K-region top-k per layer at the default max_model_len, and this
            # mode exists to check the selector against full attention, not to
            # be fast. Refuse capture and say so rather than quietly pay that.
            logger.info_once(
                "step4 DSA: VLLM_STEP4_DSA_DENSE_EQUIV=1 disables CUDA graph "
                "capture; its region count follows the batch."
            )
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def _dsa_state(self) -> Step4DSAState | None:
        """The model-side state, reached through any of this backend's layers.

        The builder is constructed by vLLM from a config and a device; it has no
        handle on the model. ``_INDEXERS`` is the same registry the custom op
        uses to get from a layer name back to its indexer, so this is not a new
        coupling, just the existing one read from the other end.
        """
        if self._state is None:
            for name in self.layer_names:
                indexer = _INDEXERS.get(name)
                if indexer is not None:
                    self._state = indexer.state
                    break
        return self._state

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step4DSAMetadata:
        m = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=self.reorder_batch_threshold)
        )
        if num_decode_tokens > num_decodes and (num_prefills > 0 or DENSE_EQUIV):
            # Multi-token decode rows only have a path of their own in a
            # decode-only step. Next to a prefill (or under the dense bypass,
            # whose decode branch assumes one query per row) they go through
            # the general schedule as short prefill rows -- which is what every
            # verify row was before the threshold was raised.
            num_decodes, num_prefills = 0, num_decodes + num_prefills
            num_decode_tokens, num_prefill_tokens = 0, m.num_actual_tokens
        md = Step4DSAMetadata(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_actual_tokens=m.num_actual_tokens,
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping,
            seq_lens=m.seq_lens,
            query_start_loc=m.query_start_loc,
            query_start_loc_cpu=m.query_start_loc_cpu,
            seq_lens_cpu=m.seq_lens_cpu,
            positions=m.positions,
            max_seq_len=m.max_seq_len,
            block_size=self.block_size,
            # vLLM 0.28 dropped CommonAttentionMetadata.block_table_np; the host
            # mirror is genuinely gone rather than renamed, so pass the None the
            # metadata class already documents as "no mirror available". Every
            # consumer keeps its device-side fallback (see _general_schedule and
            # _decode_service...), so this costs one D2H copy per step and
            # changes no result.
            block_table_np=getattr(m, "block_table_np", None),
        )

        # Build the decode schedule here rather than in the indexer op. It is
        # the only part of the DSA step that has host control flow and a
        # device-to-host read in it, and this is the last point that still runs
        # on every step -- once the decode path is captured, the op's Python
        # runs at capture only, and a graph replay reaches the kernels with
        # whatever these buffers hold. Writing them in place, from here, is what
        # makes the replay see the current step.
        state = self._dsa_state()
        if (
            state is not None
            and state.graph_decode
            and state.summary is not None  # not yet allocated: first forward
            and num_prefills == 0
            and num_decodes > 0
        ):
            if num_decode_tokens > num_decodes:
                sched = state._spec_decode_schedule(md, persistent=True)
            else:
                sched = state._decode_schedule(md, persistent=True)
            sched.builder_owned = True
            md.schedule = sched
            # Adopt it now, so that end_step() on the *next* build releases this
            # step's rows even if no layer ever calls finish_layer -- which is
            # exactly what happens when the step is a graph replay.
            state._schedule_key = md
            state._schedule = sched
            state._layers_done = 0
        return md


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


@register_backend(AttentionBackendEnum.CUSTOM)
class Step4SparseAttentionBackend(AttentionBackend):
    """DSA backend for Step-4's 23 full_attention layers.

    Registered into the ``CUSTOM`` enum slot, which is the only one left open
    for out-of-tree backends and is otherwise unused here -- ``vllm_hcu``
    registers five backends and all of them take named slots. The slot matters
    because ``Attention.__init__`` resolves ``AttentionBackendEnum[get_name()]``
    even when the backend class was passed in explicitly.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The sparse kernels read the main K/V cache either as raw bf16 or as
    # float8_e4m3, widening to the query dtype right after the load and folding
    # k_scale/v_scale into the fp32 side. (The indexer's proxy keys are fp8
    # independently of this and live in Step4DSAState.summary, not here.)
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16", "fp8_e4m3"]

    # Keep the K/V write inside this impl. The alternative (declaring False and
    # implementing do_kv_cache_update for unified_kv_cache_update) splits the
    # write away from the layout assumption it has to match, for no gain.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Step4DSAImpl"]:
        return Step4DSAImpl

    @staticmethod
    def get_builder_cls() -> type[Step4DSAMetadataBuilder]:
        return Step4DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [STEP4_HEAD_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # A region must not straddle a page: the sparse kernels address a
        # region's tokens as one contiguous run in the flattened cache.
        return [MultipleOf(REGION_BLOCK_SIZE)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # NHD, and no transpose of V: kv_cache[i].flatten(0, 1) must give
        # [num_blocks * block_size, num_kv_heads, head_size], which is the
        # "paged cache already flattened across pages" contract of
        # sparse_attention_prefill / sparse_attention_decode.
        return (2, num_blocks, block_size, num_kv_heads, head_size)


# --------------------------------------------------------------------------
# Impl
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _device_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:  # no device yet, or a platform without the field
        return 64


def _decode_num_splits(rows: int, topk: int) -> int:
    """How many ways to split the decode slot window across compute units.

    The decode kernel's grid is ``(num_reqs, num_kv_groups, num_splits)``, so at
    the recorded serving config -- 4 requests, one local KV group -- an unsplit
    launch occupies 4 of the device's 64 CUs and leaves the other 60 idle.
    Splitting the 512 selected regions K ways multiplies the grid by K and costs
    one extra log-sum-exp merge pass, which is why ``merge_split_states`` exists.

    Measured on one BW1101 at the trace_s6 decode shape: 1.594 ms unsplit,
    0.180 ms at 16 splits, then flat -- the knee is exactly where the grid fills
    the CUs. Beyond that the merge and the per-split prologue eat the gain, so
    aim for full occupancy and stop.
    """
    device_cus = _device_cu_count()
    splits = -(-device_cus // max(rows, 1))
    # A split thinner than one region block does redundant prologue work for
    # nothing; keep at least 16 slots each.
    return max(1, min(splits, topk // 16))


def _require_splitting_op() -> None:
    """Fail at startup if the indexer op would be captured into a *piecewise* graph.

    A captured graph replays recorded kernels, not Python. For a decode-only
    step that is now the intended path: the builder writes this step's schedule
    into fixed addresses before the model runs, so the recorded kernels read
    current values and the op's Python is only needed at capture. See
    ``Step4DSAMetadataBuilder.get_cudagraph_support``.

    Piecewise is the case this guards. There the op has to sit on a split
    boundary so it runs eagerly between the compiled subgraphs; swallowed into
    one, it executes exactly once -- during warm-up, where there is no metadata
    and it publishes no plan -- and every real step afterwards attends to
    whatever that single execution left behind. The three ways it has actually
    gone wrong: no plan at all, a plan from the previous step, and (before the
    op existed) a traced-away no-op. None of them are quiet, but only because
    the plan is checked; a coarser design would have returned plausible text.

    Being in ``splitting_ops`` serves both: it makes the op a boundary for
    piecewise compilation, and a full-graph capture takes the whole forward
    including the eager gaps between subgraphs.

    Step4Config.__init__ normally arranges this by appending the op to
    CompilationConfig._attention_ops before the config is finalised. It is
    bypassed by passing --compilation-config with an explicit splitting_ops
    list, which is why this check exists at all.
    """
    try:
        config = get_current_vllm_config()
    except AssertionError:
        # No config in scope. The offline gates (test_dsa_indexer.py,
        # test_dsa_vllm_wiring.py) construct the impl directly and call it
        # eagerly, so there is no graph to be captured into.
        return
    compilation = config.compilation_config
    if compilation.cudagraph_mode == CUDAGraphMode.NONE:
        return  # nothing is captured, so the op runs every step wherever it sits
    if STEP4_DSA_INDEX_OP in (compilation.splitting_ops or []):
        return
    raise RuntimeError(
        f"Step-4 DSA requires {STEP4_DSA_INDEX_OP!r} in "
        f"CompilationConfig.splitting_ops, but the resolved list is "
        f"{compilation.splitting_ops!r} with cudagraph_mode="
        f"{compilation.cudagraph_mode}. The indexer would be captured into a "
        f"CUDA graph and its region plan would never be recomputed. Either drop "
        f"the explicit --compilation-config splitting_ops (Step4Config registers "
        f"the op automatically), add the op to that list, or run with "
        f"enforce_eager=True."
    )


class Step4DSAImpl(AttentionImpl[Step4DSAMetadata]):
    """Writes K/V, then runs sparse attention over the indexer's chosen regions.

    The region plan is produced model-side by the indexer and attached to the
    ``Attention`` layer object as ``step4_sparse_plan`` immediately before this
    runs. Passing it through the layer rather than a module-level dict keeps the
    producer and consumer in the same object graph, so a missing plan is a
    construction error rather than a silent fallback.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("Step-4 DSA does not use ALiBi")
        if sliding_window is not None:
            raise NotImplementedError(
                "Step-4's sliding layers do not use DSA; only full_attention "
                "layers are routed to this backend."
            )
        if logits_soft_cap:
            raise NotImplementedError("Step-4 DSA does not use logit soft-capping")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("Step-4 DSA does not share KV across layers")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        if BOLTOPS_STAGES:
            stages = _boltops_attention()
            if kv_cache_dtype not in ("auto", "bfloat16"):
                # boltops gained fp8 cache reads in its commit 28239f4. Probing
                # the signature rather than a version string keeps this honest
                # against a library that is installed editable from a checkout:
                # without the scales the kernels silently attend to raw fp8
                # bytes, which is wrong by a factor of k_scale, not noisy.
                missing = [
                    name
                    for name, fn in zip(("prefill", "decode"), stages)
                    if name in BOLTOPS_STAGES
                    and not {"k_scale", "v_scale"}.issubset(
                        inspect.signature(fn).parameters
                    )
                ]
                if missing:
                    raise NotImplementedError(
                        f"VLLM_STEP4_DSA_BOLTOPS={','.join(sorted(BOLTOPS_STAGES))} "
                        f"with --kv-cache-dtype {kv_cache_dtype}: the installed "
                        f"boltops takes no k_scale/v_scale on "
                        f"{', '.join(missing)}, so it cannot dequantize an fp8 "
                        f"cache. Update boltops to 28239f4 or later, run the "
                        f"comparison at bfloat16 on both sides, or keep the "
                        f"vendored kernels for an fp8 cache."
                    )

        _require_splitting_op()

    # -- KV write ---------------------------------------------------------

    @staticmethod
    def _write_kv(
        kv_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter K/V into the paged cache and return the flattened views.

        ``slot_mapping`` uses -1 for padded rows. Indexing with -1 would silently
        write to the last slot of the cache and corrupt whichever request owns
        it, so the mask is load-bearing, not defensive.

        An fp8 cache arrives here as ``uint8``: vLLM maps every fp8 kv-cache dtype
        to a byte buffer in ``STR_DTYPE_TO_TORCH_DTYPE`` and leaves interpretation
        to the backend. The ``view`` below is that interpretation, and it has to
        happen before either the store or the sparse kernels see the tensor --
        they widen K/V with ``.to(query.dtype)``, which on a uint8 tensor would
        convert the *integer* byte values instead of reinterpreting the fp8 bits,
        and produce plausible-looking garbage rather than an error.
        """
        if kv_cache.ndim == 5:
            # Split layout this file was written against: [2, B, N, H, C].
            num_blocks, block_size = kv_cache.shape[1], kv_cache.shape[2]
            num_kv_heads, head_size = kv_cache.shape[3], kv_cache.shape[4]
            k_flat = kv_cache[0].view(num_blocks * block_size, num_kv_heads, head_size)
            v_flat = kv_cache[1].view(num_blocks * block_size, num_kv_heads, head_size)
            if kv_cache.dtype == torch.uint8:
                k_flat = k_flat.view(torch.float8_e4m3fn)
                v_flat = v_flat.view(torch.float8_e4m3fn)
        elif kv_cache.ndim == 4:
            # vLLM 0.28 / HCU fused page, logically [B, H, N, 2 * head_size]:
            # K is the low half of the content axis and V the high half, so the
            # two share token and group strides and differ only by an offset in
            # the content axis. ``create_kv_cache_views`` builds this from the
            # spec and ignores the backend's get_kv_cache_shape entirely, so the
            # layer receives this shape whatever the backend declares.
            #
            # as_strided rather than view/slice: the K half of a row is not a
            # contiguous run, so .view() would fail and reshape() would silently
            # copy into a tensor the caller never reads back.
            num_blocks, num_kv_heads, block_size, content = kv_cache.shape
            if content % 2:
                raise ValueError(
                    "Fused KV cache content axis must be even (K then V), got "
                    f"{content}."
                )
            head_size = content // 2
            # Reinterpret bytes as e4m3 first: slicing the byte view and viewing
            # afterwards would require the slice to be contiguous, which it is not.
            base = (
                kv_cache.view(torch.float8_e4m3fn)
                if kv_cache.dtype == torch.uint8
                else kv_cache
            )
            sizes = (num_blocks * block_size, num_kv_heads, head_size)
            strides = (base.stride(2), base.stride(1), 1)
            k_flat = base.as_strided(sizes, strides)
            v_flat = base.as_strided(sizes, strides, base.storage_offset() + head_size)
        else:
            raise ValueError(
                f"Unsupported KV cache rank {kv_cache.ndim}: {tuple(kv_cache.shape)}"
            )

        slots = slot_mapping[: key.shape[0]]
        _store_kv_rows(k_flat, v_flat, key, value, slots, k_scale, v_scale)
        return k_flat, v_flat

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Step4DSAMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Step-4 DSA does not support fused output quantization"
            )

        # Dummy/profiling run: no metadata, no cache to write, nothing to
        # attend to. The profiler only needs the allocation high-water mark.
        if attn_metadata is None:
            return output

        num_tokens = attn_metadata.num_actual_tokens
        # Both default to 1.0 unless --calculate-kv-scales is on. Keeping them at
        # 1.0 is what puts this path on the same footing as Step-4's 69 sliding
        # layers: their fp8 flash-attention kernels are compiled with
        # FP8_NO_SCALES and ignore descales entirely, so a DSA layer quantizing at
        # some other scale would make the two families disagree about what the
        # cache bytes mean.
        k_scale = float(getattr(layer, "_k_scale_float", 1.0))
        v_scale = float(getattr(layer, "_v_scale_float", 1.0))
        k_flat, v_flat = self._write_kv(
            kv_cache,
            key[:num_tokens],
            value[:num_tokens],
            attn_metadata.slot_mapping,
            k_scale,
            v_scale,
        )

        plan = getattr(layer, "step4_sparse_plan", None)
        if plan is None:
            raise RuntimeError(
                f"{layer.layer_name}: no DSA region plan. The indexer must run "
                f"before the attention layer it feeds."
            )
        if plan.md is not attn_metadata:
            raise RuntimeError(
                f"{layer.layer_name}: the DSA region plan belongs to a different "
                f"forward step. The indexer op and the attention op have been "
                f"reordered, and attending to a stale region set would be silent."
            )

        num_kv_groups = self.num_kv_heads
        nd, ndt = attn_metadata.num_decodes, attn_metadata.num_decode_tokens

        # Decode rows come first (reorder_batch_threshold). A verify step runs
        # one row per query token rather than per request, with each row's own
        # causal length -- see Step4DSAState._spec_decode_schedule.
        if nd > 0:
            packed, counts = plan.decode
            if plan.decode_seq_lens is not None:
                rows = ndt
                seq_lens_i32 = plan.decode_seq_lens
            else:
                rows = nd
                seq_lens_i32 = attn_metadata.seq_lens[:nd].to(torch.int32)
            if "decode" in FLASH_ATTN_DSA_STAGES:
                # flash-attention-cutlass's C++ entry point ignores num_splits
                # and always returns final merged state (see _flash_attn_decode),
                # so neither _decode_num_splits nor merge_split_states runs here.
                out_d, lse_d = _flash_attn_decode(
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            elif "decode" in BOLTOPS_STAGES:
                # num_splits=1 is boltops' "give me final states": it picks the
                # split count from device occupancy and merges internally, so
                # neither _decode_num_splits nor merge_split_states runs here.
                # config= is resolved on the host to keep boltops' own JSON
                # lookup -- which reads kv_seqlens.max() off the device -- out of
                # a graph-captured step.
                out_d, lse_d = _boltops_attention()[1](
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    num_splits=1,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    config=_boltops_decode_config(
                        rows,
                        self.num_heads,
                        self.head_size,
                        num_kv_groups,
                        int(packed.shape[1]),
                        REGION_BLOCK_SIZE,
                        int(attn_metadata.seq_lens_cpu[:nd].max()),
                        query.dtype,
                        k_flat.dtype,
                    ),
                )
            else:
                splits = _decode_num_splits(rows * num_kv_groups, packed.shape[1])
                out_d, lse_d = sparse_attention_decode(
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    num_splits=splits,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    qk_fp8=DECODE_QK_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                    pv_fp8=DECODE_PV_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                )
                if splits > 1:
                    out_d, lse_d = merge_split_states(out_d, lse_d)
            del lse_d
            output[:ndt] = out_d

        if attn_metadata.num_prefills > 0:
            packed, counts, logical_ids, request_indices = plan.prefill
            q_positions = attn_metadata.positions[ndt:num_tokens].to(torch.int32)
            prefill_kwargs = dict(
                num_kv_groups=num_kv_groups,
                region_size=REGION_BLOCK_SIZE,
                softmax_scale=self.scale,
                logical_regions=logical_ids,
                q_positions=q_positions,
                request_indices=request_indices,
                block_q=128,
                block_regions=16,
            )
            if "prefill" in FLASH_ATTN_DSA_STAGES:
                # flash-attention-cutlass has no query-tiled prefill path, so
                # logical_regions/q_positions/request_indices/block_q from
                # prefill_kwargs are dropped by _flash_attn_prefill rather
                # than forwarded (see its docstring).
                out_p, lse_p = _flash_attn_prefill(
                    query[ndt:num_tokens],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            elif "prefill" in BOLTOPS_STAGES:
                out_p, lse_p = _boltops_attention()[0](
                    query[ndt:num_tokens],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    **prefill_kwargs,
                )
            else:
                out_p, lse_p = sparse_attention_prefill(
                    query[ndt:num_tokens],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    qk_fp8=PREFILL_QK_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                    pv_fp8=PREFILL_PV_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                    **prefill_kwargs,
                )
            del lse_p
            output[ndt:num_tokens] = out_p

        return output


@dataclass
class Step4SparsePlan:
    """Selected regions for one layer, one forward step.

    ``prefill`` additionally carries logical region ids and request indices for the
    query-tiled prefill kernel. The packed words remain physical cache addresses and are
    not interchangeable with those logical ids.
    """

    prefill: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    decode: tuple[torch.Tensor, torch.Tensor] | None = None
    # Per-row context lengths for ``decode`` when its rows are query tokens of
    # a verify step rather than requests. None means one row per request, read
    # from ``md.seq_lens``.
    decode_seq_lens: torch.Tensor | None = None
    # The metadata this plan was built from. The impl compares it by identity:
    # the indexer and the attention layer are two separate ops in the compiled
    # graph, and if anything ever reordered them the layer would silently attend
    # to the previous step's regions instead of failing.
    md: Step4DSAMetadata | None = None


def build_plan(
    attn_metadata: Step4DSAMetadata,
    prefill_logits: torch.Tensor | None,
    decode_logits: torch.Tensor | None,
    *,
    topk: int,
    num_kv_groups: int,
    regions_per_page: int,
    decode_seq_lens: torch.Tensor | None = None,
    decode_table: torch.Tensor | None = None,
    step_cache: dict | None = None,
) -> Step4SparsePlan:
    """Turn indexer scores into the two kernels' metadata.

    Both halves are pass-throughs to the reference selectors, which already
    speak vLLM's block table: they take ``[num_reqs, pages]`` logical->physical
    maps plus a per-row request index, which is exactly what
    ``CommonAttentionMetadata`` carries.

    ``decode_seq_lens``/``decode_table`` are a verify step's per-token rows
    (``[num_decode_tokens]`` lengths and ``[num_decode_tokens, pages]`` table):
    each query token is selected for as a single-token decode at its own
    position.

    ``step_cache`` is a dict that lives for one forward step (see
    ``Step4Indexer.run``); the decode rows' request/history vectors depend on
    the step only, so the first layer builds them and the other 22 reuse them.
    """
    plan = Step4SparsePlan()
    plan.md = attn_metadata
    nd = attn_metadata.num_decodes
    ndt = attn_metadata.num_decode_tokens

    if nd > 0 and decode_logits is not None:
        if decode_seq_lens is not None:
            seq_lens, table = decode_seq_lens, decode_table
            plan.decode_seq_lens = decode_seq_lens
        else:
            seq_lens = attn_metadata.seq_lens[:nd].to(torch.int32)
            table = attn_metadata.block_table[:nd]
        rows_key = ("decode_rows", decode_logits.shape[0])
        rows_meta = None if step_cache is None else step_cache.get(rows_key)
        if rows_meta is None:
            # group-major arange(num_reqs), the default, is what we want
            rows_meta = decode_rows_meta(
                seq_lens, decode_logits.shape[0], None, REGION_BLOCK_SIZE
            )
            if step_cache is not None:
                step_cache[rows_key] = rows_meta
        packed, counts = decode_sparse_meta(
            decode_logits,
            seq_lens,
            table,
            topk=topk,
            region_size=REGION_BLOCK_SIZE,
            regions_per_page=regions_per_page,
            rows_meta=rows_meta,
        )
        plan.decode = (packed, counts)

    if attn_metadata.num_prefills > 0 and prefill_logits is not None:
        positions = attn_metadata.positions
        assert positions is not None, (
            "CommonAttentionMetadata.positions is required by the DSA region "
            "selector; it derives each row's causal horizon from it."
        )
        q_positions = positions[ndt : attn_metadata.num_actual_tokens].to(torch.int32)
        # request_indices maps each score row to its block-table row. Rows are
        # group-major, so the per-token request index tiles once per KV group.
        starts = attn_metadata.query_start_loc_cpu
        req_of_token = torch.repeat_interleave(
            torch.arange(
                nd,
                nd + attn_metadata.num_prefills,
                dtype=torch.int32,
                device=positions.device,
            ),
            (starts[nd + 1 :] - starts[nd:-1]).to(positions.device),
        )
        packed, counts, logical_ids = prefill_sparse_meta(
            prefill_logits,
            q_positions.repeat(num_kv_groups),
            attn_metadata.block_table,
            req_of_token.repeat(num_kv_groups),
            topk=topk,
            region_size=REGION_BLOCK_SIZE,
            regions_per_page=regions_per_page,
            return_logical_ids=True,
        )
        plan.prefill = (
            packed,
            counts,
            logical_ids,
            req_of_token.repeat(num_kv_groups),
        )
        _maybe_dump_prefill_plan(packed, counts, q_positions)

    return plan


def _maybe_dump_prefill_plan(packed, counts, q_positions) -> None:
    """Save one real prefill region plan when VLLM_STEP4_DUMP_PLAN names a path.

    Temporary instrumentation for the query-tiling work. Tiling queries only pays
    if neighbouring queries select overlapping regions, and how much they overlap
    is a property of the trained indexer that no synthetic plan can stand in for.
    Writes the first long chunk it sees and then stays quiet.
    """
    import os

    path = os.environ.get("VLLM_STEP4_DUMP_PLAN")
    if not path or getattr(_maybe_dump_prefill_plan, "_done", False):
        return
    if packed.shape[0] < 8192:  # skip warm-up and short chunks
        return
    _maybe_dump_prefill_plan._done = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = f"{path}.rank{rank}"
    torch.save(
        {
            "packed": packed.cpu(),
            "counts": counts.cpu(),
            "positions": q_positions.cpu(),
        },
        path,
    )
    logger.info("step4 DSA: dumped prefill plan %s -> %s", tuple(packed.shape), path)


# --------------------------------------------------------------------------
# Indexer geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step4DSAGeometry:
    """The DSA shape constants, read from the checkpoint's ``sparse_config``.

    Nothing here is hardcoded except the fallbacks, which match the released
    config. ``regions_per_page`` is deliberately absent: it is
    ``kv_block_size // region_size`` and therefore a property of the *runtime*
    cache, not of the checkpoint, so it is derived where the block size is known.
    """

    proxy_dim: int = 256
    topk: int = 512
    region_size: int = REGION_BLOCK_SIZE
    # The full rotated width of an indexer head. The cos/sin tables are half
    # this, because one frequency drives the pair (d, d + span // 2), and it is
    # that halved value the kernels call ``rotary_dim``.
    rope_span: int = 32
    num_indexer_heads: int = 16
    num_provider_groups: int = 4

    def __post_init__(self) -> None:
        for name in ("proxy_dim", "topk", "num_indexer_heads", "num_provider_groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.region_size != REGION_BLOCK_SIZE:
            raise ValueError(
                f"sparse_config.region_block_size={self.region_size} but the "
                f"released kernels are compiled around {REGION_BLOCK_SIZE}"
            )
        if not 0 < self.rope_span <= self.proxy_dim or self.rope_span % 2:
            raise ValueError(
                "rope span must be positive, even, and no larger than proxy_dim; "
                f"got rope_span={self.rope_span}, proxy_dim={self.proxy_dim}"
            )
        if self.num_indexer_heads % self.num_provider_groups:
            raise ValueError(
                f"{self.num_indexer_heads} indexer heads do not divide into "
                f"{self.num_provider_groups} provider groups"
            )

    @classmethod
    def from_config(cls, sparse_config: Any) -> "Step4DSAGeometry":
        def pick(name: str, default: int) -> int:
            if isinstance(sparse_config, dict):
                return int(sparse_config.get(name, default))
            return int(getattr(sparse_config, name, default))

        return cls(
            proxy_dim=pick("proxy_dim", 256),
            topk=pick("topk", 512),
            region_size=pick("region_block_size", REGION_BLOCK_SIZE),
            rope_span=pick("sparse_indexer_rope_dim", 32),
            num_indexer_heads=pick("sparse_indexer_num_heads", 16),
            num_provider_groups=pick("num_provider_groups", 4),
        )


# --------------------------------------------------------------------------
# Pending partial regions
# --------------------------------------------------------------------------


class Step4PendingTable:
    """Which pending buffer row holds which region's not-yet-complete tokens.

    Keyed by the region's *physical* summary slot, never by a batch row. vLLM
    swaps batch rows when it reorders decodes to the front, and a preempted
    request loses its row outright, so a row index is not a stable identity for
    a request across steps. The physical slot is: it stays valid exactly as long
    as the request holds the block, which is exactly as long as its half-filled
    region means anything.

    Rows are recycled least-recently-touched. That is safe because every live
    partial region is touched on every step its owner is scheduled, and a
    request that stops being scheduled has either finished (pending is dead) or
    been preempted (vLLM recomputes it from position 0, so pending is dead
    again). Evicting a row that was touched *this* step would be a capacity bug,
    so it raises instead.

    The table is shared by all 23 DSA layers -- the slot-to-row assignment is
    layer-independent, only the buffered values differ -- so it is stepped once
    per forward pass, not once per layer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._row_of_slot: dict[int, int] = {}
        self._slot_of_row: list[int] = [-1] * capacity
        self._touched: list[int] = [-1] * capacity
        self.step = 0

    def begin_step(self) -> None:
        self.step += 1

    def get(self, slot: int) -> int:
        """Row holding ``slot``'s partial tokens; raises if there is none."""
        row = self._row_of_slot.get(slot)
        if row is None:
            raise RuntimeError(
                f"Step-4 DSA: no pending tokens for region slot {slot}. A request "
                f"resumed in the middle of a region whose leading tokens were "
                f"never buffered -- the summary for that region would be wrong, "
                f"so this fails loudly rather than silently attending to noise."
            )
        self._touched[row] = self.step
        return row

    def acquire(self, slot: int) -> int:
        """Row for ``slot``, allocating or recycling one if needed."""
        row = self._row_of_slot.get(slot)
        if row is None:
            row = self._free_row()
            self._row_of_slot[slot] = row
            self._slot_of_row[row] = slot
        self._touched[row] = self.step
        return row

    def release(self, slot: int) -> None:
        row = self._row_of_slot.pop(slot, None)
        if row is not None:
            self._slot_of_row[row] = -1
            self._touched[row] = -1

    def retire(self, slot: int) -> None:
        """Mark ``slot``'s row first in line for recycling, but keep it mapped.

        Under speculative decoding a region can close on draft tokens that are
        then rejected, rolling the request back into it. The row (re-filled with
        the whole region by the closing step) must survive until that is known,
        so it stays readable through get() and is only recycled -- oldest
        retirement first, ahead of every live row -- when a row is needed.
        """
        row = self._row_of_slot.get(slot)
        if row is not None:
            self._touched[row] = self.step - (1 << 40)

    def _free_row(self) -> int:
        oldest_row, oldest_step = -1, self.step
        for row in range(self.capacity):
            if self._slot_of_row[row] < 0:
                return row
            if self._touched[row] < oldest_step:
                oldest_row, oldest_step = row, self._touched[row]
        if oldest_row < 0:
            raise RuntimeError(
                f"Step-4 DSA: all {self.capacity} pending region buffers are in use "
                f"this step. Capacity is sized from max_num_seqs, so this means "
                f"more concurrent sequences reached the indexer than the scheduler "
                f"declared."
            )
        # Drop the recycled row's forward mapping too. Leaving it behind would
        # leave a stale slot pointing at a row another region now owns, and
        # get() would hand back that other request's buffered tokens instead of
        # raising -- the one failure this table exists to make impossible.
        self._row_of_slot.pop(self._slot_of_row[oldest_row], None)
        self._slot_of_row[oldest_row] = -1
        return oldest_row


# --------------------------------------------------------------------------
# Per-step schedule
# --------------------------------------------------------------------------


@dataclass
class _RequestPlan:
    """Where one request's tokens land in the compression buffer, this step."""

    row: int  # block-table row
    past: int  # tokens already in the KV cache before this step
    qlen: int  # tokens contributed this step
    tok_start: int  # offset of those tokens in the packed batch
    head: int  # leading tokens taken from pending (== past % region_size)
    buf_start: int  # offset in the assembled compression buffer
    pending_row_in: int  # pending row to read the head from, -1 if head == 0
    pending_row_out: int  # pending row to write the tail to, -1 if no tail
    tail: int  # trailing tokens that do not complete a region
    # Complete regions this request's *last* query may score against, i.e.
    # (past + qlen - 1) // region_size. Earlier queries in the same chunk see
    # fewer; the selector bounds each row by its own position, so scoring the
    # whole width and letting it clip is both correct and simpler.
    nscore: int
    score_slots: torch.Tensor | None  # [nscore] int64 physical summary slots
    # (pending row, buffer offset) of regions that closed this step on what may
    # be draft tokens; the whole region is written back so a rollback into it
    # can re-read its head. Empty without speculative decoding.
    keep_rows: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Step4DSASchedule:
    """Layer-independent bookkeeping for one forward step.

    The 23 DSA layers see the same batch, so which regions complete, where their
    summaries go, and which pending rows are involved is computed once and
    reused. Only the values differ per layer.
    """

    requests: list[_RequestPlan] = field(default_factory=list)
    buf_len: int = 0
    # Region compression arguments, over the assembled buffer.
    starts: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    slots: torch.Tensor | None = None  # physical summary slot per region
    live: torch.Tensor | None = None  # counts == region_size
    # Regions whose summary is now final; their pending rows are released after
    # every layer has consumed them.
    completed_slots: list[int] = field(default_factory=list)

    # -- pure-decode fast path ---------------------------------------------
    # Set only when the step is decode-only (num_prefills == 0), where every
    # request contributes exactly one token and therefore touches exactly one
    # region. That collapses the per-request Python above into fixed-shape
    # tensors, which is what lets the 23 layers run batched instead of looping.
    # ``requests`` is left empty in that case: the generic consumers must not
    # silently fall through to a zero-iteration loop, so they check ``decode``.
    decode: bool = False
    num_decodes: int = 0
    head: torch.Tensor | None = None  # [nd] int64, tokens already in the region
    tail: torch.Tensor | None = None  # [nd] int64, tokens to re-buffer, 0 if closed
    nscore: torch.Tensor | None = None  # [nd] int32, complete regions visible
    max_nscore: int = 0
    pending_in: torch.Tensor | None = None  # [nd] int64 row to read the head from
    pending_out: torch.Tensor | None = None  # [nd] int64 row to write the tail to
    # Built by Step4DSAMetadataBuilder rather than by the first layer to reach
    # state.schedule(). The pending rows have then already been assigned by the
    # time the layers run, so finish_layer() must not release them a second time.
    builder_owned: bool = False
    # Fixed score width this schedule was built against, when it came from
    # persistent buffers. Zero means "size the score tensor to max_nscore", the
    # eager behaviour.
    graph_regions: int = 0

    # -- speculative verify fast path --------------------------------------
    # Set when every request scores up to spec_query_len query tokens (see
    # Step4DSAState._spec_decode_schedule). A request then touches up to
    # ``nrm`` consecutive regions, each given a fixed ``nrm * region_size``
    # stride in the compression buffer; the region vectors
    # (``starts``/``counts``/``slots``/``live``/``keep_rows``) are
    # ``[num_decodes * nrm]``, the ``tok_*`` vectors are ``[num_tokens]``.
    # ``head``/``tail``/``pending_in``/``pending_out`` keep their decode
    # meanings, with the tail taken from region ``tail_reg``.
    spec: bool = False
    num_tokens: int = 0
    nrm: int = 0
    tail_reg: torch.Tensor | None = None  # [nd] int64, flat region of the tail
    keep_rows: torch.Tensor | None = None  # [nd * nrm] int64, trash if not kept
    tok_buf: torch.Tensor | None = None  # [ndt] int64, buffer row per token
    tok_seq: torch.Tensor | None = None  # [ndt] int32, causal length per token
    tok_nscore: torch.Tensor | None = None  # [ndt] int32, regions visible
    tok_table: torch.Tensor | None = None  # [ndt, pages] int32, owner's table row
    req_table: torch.Tensor | None = None  # [nd, pages] int32, md.block_table
    query_start_loc: torch.Tensor | None = None  # [nd + 1] int32, md.query_start_loc


# --------------------------------------------------------------------------
# Model-side DSA state
# --------------------------------------------------------------------------


class Step4DSAState:
    """Summaries, pending buffers, and the per-step schedule for one model.

    One instance per ``Step4Model``; every DSA layer holds a reference and its
    own slot index into the layer-major tensors. Allocation is lazy because the
    number of KV blocks is not known until vLLM has profiled memory and built
    the cache -- which is also why this pool comes out of the leftover headroom
    rather than out of the profiled budget.

    **Why the summaries are not an ``MLAAttentionSpec``.** The obvious route --
    registering them with vLLM's KV cache manager at ``compress_ratio =
    region_size``, the way ``DeepseekV4IndexerCache`` does -- does not survive
    Step-4's mix of specs. Page sizes are only unified through the
    DeepSeek-specific branch when a ``SlidingWindowMLASpec`` is present; Step-4
    has none, so it falls through to ``unify_kv_cache_spec_page_size``, which
    pads every spec up to a common page. A 512 B summary page padded to the
    12288 B DSA/SWA page is a 24x blowup: the summary pool would cost as much as
    the entire KV pool. Allocating here instead costs ``proxy_dim / (2 *
    head_dim * region_size)`` of the DSA KV pool -- about 4.2% at Step-4
    geometry -- and the exact figure is logged.

    **Addressing.** Physical KV block ``b`` owns summary rows ``[b *
    regions_per_page, (b + 1) * regions_per_page)``. Reusing the KV block id
    means no second block table and no second allocator.

    **Why block reuse needs no invalidation.** When a block is recycled its
    summary rows are stale, but only rows for *incomplete* regions can be: a
    region that fills is always overwritten as it fills. And an incomplete
    region's summary is never read -- the selector bounds each row's candidates
    to ``(past + len - 1) // region_size``, i.e. complete regions only, and the
    query's own partial region is force-included without consulting a summary.
    That is precisely why the reference refuses to store partial summaries.
    """

    def __init__(
        self,
        num_layers: int,
        geometry: Step4DSAGeometry,
        max_pending: int,
        dtype: torch.dtype,
        max_num_reqs: int = 0,
        max_model_len: int = 0,
        spec_query_len: int = 1,
    ) -> None:
        self.num_layers = num_layers
        self.geometry = geometry
        self.dtype = dtype
        self.pending = Step4PendingTable(max_pending)
        self.max_pending = max_pending
        # Longest query a speculative verify step can have (1 + drafts). A row
        # no longer than this may be rolled back by rejection after the step,
        # so the regions it closes are kept recoverable; 1 disables that.
        self.spec_query_len = int(spec_query_len)
        # One row past the allocator's capacity, never handed out by the table.
        # The vectorized decode path writes every request's tail unconditionally
        # and sends the requests that have no tail here, which keeps the scatter
        # a fixed-shape tensor op instead of a data-dependent row selection.
        self.pending_trash_row = max_pending

        # Capture parameters. Both must be known for the decode schedule to live
        # in persistent buffers at a fixed score width; when either is zero the
        # path still works, it just reallocates per step at the batch's own
        # width and cannot be captured.
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_len = int(max_model_len)
        self.graph_decode = self.max_num_reqs > 0 and self.max_model_len > 0
        # Set by the first Step4Indexer to be constructed; the score pool is
        # group-major and cannot be sized without it.
        self.num_kv_groups = 0
        self.heads_per_group = 0
        self.graph_regions = 0

        self.summary: torch.Tensor | None = None
        self.pending_key: torch.Tensor | None = None
        self.pending_z: torch.Tensor | None = None
        self.regions_per_page = 0
        self.block_size = 0

        # Persistent decode schedule, filled in place every step. Allocated in
        # ensure_allocated; None until then, and always None when graph_decode
        # is off.
        self._buf: dict[str, torch.Tensor] = {}
        self._host: dict[str, list[torch.Tensor]] = {}
        self._stage = 0
        self._stage_ev: list[torch.cuda.Event] = []
        self._decode_logits: torch.Tensor | None = None
        # Persistent verify-step schedule (_spec_decode_schedule): one int64
        # device vector holding every field at a fixed offset, refilled by one
        # H2D copy per step from a pinned ring like the decode path's.
        self._spec_layout: dict[str, tuple[int, int]] = {}
        self._spec_raw: torch.Tensor | None = None
        self._spec_i32: dict[str, torch.Tensor] = {}
        self._spec_starts: torch.Tensor | None = None
        self._spec_table: torch.Tensor | None = None
        self._spec_host: list[torch.Tensor] = []
        self._spec_ev: list[torch.cuda.Event] = []
        self._spec_stage = 0

        self._rope: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._schedule_key: object | None = None
        self._schedule: Step4DSASchedule | None = None
        # Which layer slots actually exist on this rank, so end_step() fires
        # after the last one rather than after a hardcoded count (pipeline
        # parallel would give a rank only some of the 23).
        self._registered: set[int] = set()
        self._layers_done = 0
        self.first_slot = 0
        # Layer-independent tensors derived during one forward. Reset by the
        # first DSA layer of every forward -- eager, warmup or capture alike --
        # so nothing outlives the step (or the graph) it was built in.
        self.step_cache: dict = {}

    def register_layer(self, slot: int) -> None:
        self._registered.add(slot)
        self.first_slot = min(self._registered)

    def reset_allocation(self) -> None:
        """Forget everything sized from, or written against, the bound KV cache.

        Called when a new KV cache is bound (see Step4DSAMetadataBuilder). The
        next ensure_allocated() then sizes the summary pool and the decode
        graph width from the new cache. Pending rows hold tokens of requests
        that lived in the old cache, so they go too.
        """
        if self.summary is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Step-4 DSA: KV cache rebound during CUDA graph capture."
            )
        self.summary = None
        self.pending_key = None
        self.pending_z = None
        self.regions_per_page = 0
        self.block_size = 0
        self.graph_regions = 0
        self.pending = Step4PendingTable(self.max_pending)
        self._buf = {}
        self._host = {}
        self._stage = 0
        self._stage_ev = []
        self._decode_logits = None
        self._spec_layout = {}
        self._spec_raw = None
        self._spec_i32 = {}
        self._spec_starts = None
        self._spec_table = None
        self._spec_host = []
        self._spec_ev = []
        self._spec_stage = 0
        self._schedule_key = None
        self._schedule = None
        self._layers_done = 0

    # -- allocation -------------------------------------------------------

    def ensure_allocated(self, kv_cache: torch.Tensor, device: torch.device) -> None:
        if self.summary is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            # Anything allocated here would come out of the graph's private
            # memory pool and be freed with it. vLLM runs cudagraph_num_of_warmups
            # eager steps before capturing, so reaching this under capture means
            # the warmup never touched the DSA path -- fail loudly rather than
            # hand back memory that goes away.
            raise RuntimeError(
                "Step-4 DSA: summary pool is still unallocated at CUDA graph "
                "capture time. The capture warmup must run at least one eager "
                "step through the DSA layers first."
            )
        # Read the block count and block size from whichever rank the cache
        # arrived in. On 0.28's fused 4-D page ([B, H, N, 2 * head_size]) the
        # naive shape[1]/shape[2] pair reads H (== 1 at TP8) as the block count
        # and N as the block size: block_size still lands right, so the region
        # check below passes, but the summary pool is sized for 1 * regions_per_page
        # rows instead of num_blocks * regions_per_page and every region write
        # runs off the end of it -- a kernel VMFault, not a Python error.
        if kv_cache.ndim == 4:
            num_blocks, block_size = int(kv_cache.shape[0]), int(kv_cache.shape[2])
        else:
            num_blocks, block_size = int(kv_cache.shape[1]), int(kv_cache.shape[2])
        if block_size % self.geometry.region_size:
            raise ValueError(
                f"KV block size {block_size} is not a multiple of the DSA region "
                f"size {self.geometry.region_size}; a region straddling two pages "
                f"would break the contiguity the sparse kernels rely on."
            )
        self.block_size = block_size
        self.regions_per_page = block_size // self.geometry.region_size
        rows = num_blocks * self.regions_per_page
        proxy = self.geometry.proxy_dim

        self.summary = torch.zeros(
            (self.num_layers, rows, 1, proxy),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.pending_key = torch.zeros(
            (self.num_layers, self.max_pending + 1, self.geometry.region_size, proxy),
            dtype=self.dtype,
            device=device,
        )
        self.pending_z = torch.zeros_like(self.pending_key)
        # Measured against the real cache rather than derived from the geometry:
        # the summary is one proxy vector per region regardless of how many KV
        # heads a rank holds, so the ratio moves with the TP degree.
        kv_bytes = kv_cache.numel() * kv_cache.element_size()
        logger.info(
            "step4 DSA: %d region summaries x %d layers = %.1f MiB fp8 "
            "(%.1f%% of one DSA layer's KV pool), plus %.1f MiB of pending buffers.",
            rows,
            self.num_layers,
            self.summary.numel() / 2**20,
            100.0 * self.summary[0].numel() / max(kv_bytes, 1),
            2 * self.pending_key.numel() * self.pending_key.element_size() / 2**20,
        )
        self._allocate_graph_buffers(rows, device)

    def _allocate_graph_buffers(self, pool_rows: int, device: torch.device) -> None:
        """Persistent decode-schedule buffers and the fixed-width score pool.

        A CUDA graph replays the pointers and launch arguments it recorded, so
        every tensor the decode path touches has to keep both its address and
        its shape from one step to the next. These are allocated once, outside
        capture, and refilled in place by ``_decode_schedule``.

        The score pool is deliberately *not* zeroed, then or ever. Each row is
        written for its own ``nscore`` columns and the selector reads exactly
        ``(seq_len - 1) // region_size`` of them -- the same quantity -- so no
        read ever reaches a column this step did not write. Zeroing 64K fp32
        columns per row per layer would cost more than the scoring.
        """
        if not self.graph_decode:
            return
        if self.num_kv_groups <= 0:
            raise RuntimeError(
                "Step-4 DSA: num_kv_groups was never published to the state; "
                "Step4Indexer.__init__ must set it before the first forward."
            )
        region = self.geometry.region_size
        # A request can never see more complete regions than its own history,
        # nor more than the summary pool holds.
        need = -(-self.max_model_len // region)
        self.graph_regions = min(GRAPH_REGIONS_OVERRIDE or need, need, pool_rows)

        nr = self.max_num_reqs
        i64 = dict(dtype=torch.int64, device=device)
        self._buf = {
            "head": torch.zeros(nr, **i64),
            "tail": torch.zeros(nr, **i64),
            "slots": torch.zeros(nr, **i64),
            "pending_in": torch.full((nr,), self.pending_trash_row, **i64),
            "pending_out": torch.full((nr,), self.pending_trash_row, **i64),
            "nscore": torch.zeros(nr, dtype=torch.int32, device=device),
            "counts": torch.zeros(nr, dtype=torch.int32, device=device),
            "starts": torch.arange(nr, dtype=torch.int32, device=device) * region,
            "live": torch.zeros(nr, dtype=torch.bool, device=device),
        }
        # Staged on pinned host memory so the per-step upload of the pending
        # rows is one async copy rather than two torch.tensor(list) round trips.
        #
        # A *ring* of them, because the copy is async: stream ordering protects
        # the device buffer, not this one. With a single buffer per vector, step
        # N+1's host write can overwrite the bytes step N's copy has not read
        # yet -- a write-after-read hazard that shows up as a request picking up
        # another request's pending row, and the best candidate for the
        # non-determinism the copy below used to be made blocking to avoid. The
        # events make reuse wait for the copy that last read the slot, so the
        # depth is a bound rather than a bet on how far ahead the host runs.
        pin = torch.cuda.is_available()
        self._host = {
            k: [
                torch.zeros(nr, dtype=torch.int64, pin_memory=pin)
                for _ in range(_STAGING_DEPTH)
            ]
            for k in ("pending_in", "pending_out")
        }
        self._stage = 0
        self._stage_ev = (
            [torch.cuda.Event() for _ in range(_STAGING_DEPTH)] if pin else []
        )
        for ev in self._stage_ev:
            # Recorded once so the first _STAGING_DEPTH steps can wait on them
            # uniformly instead of branching on "has this slot been used yet".
            ev.record()
        # A verify step scores one row per query token, not per request.
        score_rows = nr * self.spec_query_len
        self._decode_logits = torch.empty(
            (self.num_kv_groups * score_rows, self.graph_regions),
            dtype=torch.float32,
            device=device,
        )
        mib = self._decode_logits.numel() * 4 / 2**20
        logger.info(
            "step4 DSA: decode graph width %d regions (max_model_len %d, "
            "region %d), %d x %d fp32 score pool = %.1f MiB. The region selector "
            "scans the full width every step, so lowering --max-model-len (or "
            "VLLM_STEP4_DSA_GRAPH_REGIONS) cuts decode selection cost "
            "proportionally.",
            self.graph_regions,
            self.max_model_len,
            region,
            self._decode_logits.shape[0],
            self.graph_regions,
            mib,
        )
        self._warm_graph_kernels(device)

    def _warm_graph_kernels(self, device: torch.device) -> None:
        """Compile the width-dependent decode kernels before capture can reach them.

        Triton compiles on first launch, and a compile inside ``torch.cuda.graph``
        loads a module on the capturing stream -- which is how this model's
        trace bubbles were traced to ``hipModuleLoadDataEx`` in the first place.

        vLLM's own capture warmup does not cover this. The warmup and the capture
        share a batch size, so everything specialized on ``num_decodes`` is
        already compiled by the time capture runs; but the *first* warmup step is
        also the step that allocates this pool, so it necessarily ran at the old
        dynamic width. That width picks a different ``BLOCK_R`` in the region
        selector (``min(1024, next_pow2(seq_regions))``), which is a constexpr,
        so the graph-width variant would compile for the first time under
        capture. One throwaway launch at the real width settles it.
        """
        region = self.geometry.region_size
        proxy = self.geometry.proxy_dim
        groups, hpg = self.num_kv_groups, self.heads_per_group
        pages = -(-self.graph_regions // self.regions_per_page)
        # Zero seq lens: every row is masked out and the kernels return without
        # touching the pool. Compilation is the whole point, not the result.
        seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
        table = torch.zeros((1, pages), dtype=torch.int32, device=device)
        logits = self._decode_logits[:groups, : self.graph_regions]
        decode_indexer_logits(
            torch.zeros((1, groups, hpg, proxy), dtype=self.dtype, device=device),
            torch.zeros((1, groups, hpg), dtype=torch.float32, device=device),
            self.summary[0],
            table,
            torch.zeros(1, dtype=torch.int32, device=device),
            logits,
            regions_per_page=self.regions_per_page,
        )
        decode_sparse_meta(
            logits,
            seq_lens,
            table,
            None,
            topk=self.geometry.topk,
            region_size=region,
            regions_per_page=self.regions_per_page,
        )

    def rope_tables(
        self,
        *,
        theta: float,
        scaling: dict | None,
        max_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer cos/sin, shared between layers with identical rope geometry.

        Built with the released ``build_rope_cache`` rather than vLLM's
        ``get_rope`` so that the table is bit-identical to the reference's: the
        two agree on the maths but not on the intermediate precision, and the
        difference shows up in bf16 table entries from position 24 onward.
        """
        key = (float(theta), repr(scaling), int(max_position), str(device), dtype)
        cached = self._rope.get(key)
        if cached is None:
            cached = build_rope_cache(
                rotary_span=self.geometry.rope_span,
                theta=theta,
                max_position=max_position,
                device=device,
                dtype=dtype,
                scaling=scaling,
            )
            self._rope[key] = cached
        return cached

    # -- per-step schedule ------------------------------------------------

    def schedule(self, md: Step4DSAMetadata) -> Step4DSASchedule:
        """Build (or reuse) this step's compression schedule.

        Keyed on the metadata object, which vLLM rebuilds once per forward pass
        and hands unchanged to all 23 layers. The first layer pays for it; the
        other 22 get it for free, including the pending-row assignments, which
        must not be made twice.
        """
        if self._schedule_key is md:
            assert self._schedule is not None
            return self._schedule

        if md.schedule is not None:
            # Built by the metadata builder, before the model ran. Adopt it as
            # this step's schedule so the 23 layers share it exactly as they
            # would a locally built one.
            self._schedule_key = md
            self._schedule = md.schedule
            self._layers_done = 0
            return md.schedule

        if md.num_prefills == 0 and md.num_decode_tokens > md.num_decodes:
            sched = self._spec_decode_schedule(md)
        elif md.num_prefills == 0 and md.num_decodes > 0:
            sched = self._decode_schedule(md)
        else:
            sched = self._general_schedule(md)
        self._schedule_key = md
        self._schedule = sched
        self._layers_done = 0
        return sched

    def _general_schedule(self, md: Step4DSAMetadata) -> Step4DSASchedule:
        """The chunk-shaped schedule: any mix of prefill and decode rows.

        Loops in Python because a prefill row's region count, buffer offset and
        pending rows all depend on its own query length. ``_decode_schedule``
        handles the case where they cannot.
        """
        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        starts_cpu = md.query_start_loc_cpu.tolist()
        seq_lens_cpu = md.seq_lens_cpu.tolist()
        num_reqs = md.num_decodes + md.num_prefills
        # One host copy per step instead of two device reads per request: the
        # slot arithmetic below is host-side control flow either way. Prefer the
        # block table's host mirror, which is the copy vLLM uploaded from and so
        # costs no synchronisation at all; the device read stays as the fallback
        # for the cases that have no mirror (see Step4DSAMetadata.block_table_np).
        table_np = md.block_table_np
        if table_np is not None and table_np.shape[0] >= num_reqs:
            table_cpu = table_np
        else:
            table_cpu = md.block_table[:num_reqs].to("cpu", non_blocking=False).tolist()

        # Release the previous step's closed regions before claiming this
        # step's. finish_layer() normally does it, but a step that was a graph
        # replay never reached finish_layer, so the release has to be anchored
        # here -- at the one point every step passes through. end_step() is
        # idempotent, so doing it twice costs nothing.
        self.end_step()
        self.pending.begin_step()
        sched = Step4DSASchedule()
        starts: list[int] = []
        counts: list[int] = []
        slots: list[int] = []
        buf_off = 0

        # Protect every resumed head before any new tail can recycle a row.
        # Mixed prefill/verify batches can reopen retired regions after MTP
        # rejection; a later batch row is still live even before we visit it.
        pending_heads = [-1] * num_reqs
        for row in range(num_reqs):
            qlen = starts_cpu[row + 1] - starts_cpu[row]
            if qlen <= 0:
                continue
            past = seq_lens_cpu[row] - qlen
            if past % region:
                region_id = past // region
                slot = int(table_cpu[row][region_id // rpp]) * rpp + region_id % rpp
                pending_heads[row] = self.pending.get(slot)

        for row in range(num_reqs):
            tok_start = starts_cpu[row]
            qlen = starts_cpu[row + 1] - tok_start
            if qlen <= 0:
                continue
            past = seq_lens_cpu[row] - qlen
            head = past % region
            first_region = past // region
            nbuf = head + qlen
            nregions = (nbuf + region - 1) // region
            pages = table_cpu[row]

            def slot_of(region_id: int, _pages=pages) -> int:
                return int(_pages[region_id // rpp]) * rpp + region_id % rpp

            for j in range(nregions):
                filled = min(region, nbuf - j * region)
                starts.append(buf_off + j * region)
                # A count below region_size marks the ragged tail; zero makes
                # the compression kernel skip the region entirely. Storing a
                # partial summary is what pending exists to avoid.
                counts.append(region if filled == region else 0)
                slots.append(slot_of(first_region + j))

            first_slot = slots[-nregions]
            last_slot = slots[-1]
            # The head tokens were buffered by a previous step; that buffer is
            # only free once the region it belongs to has actually closed.
            pending_in = pending_heads[row]
            if head and nbuf >= region:
                sched.completed_slots.append(first_slot)

            tail = nbuf % region
            pending_out = self.pending.acquire(last_slot) if tail else -1

            # A speculative verify step may close a region on draft tokens that
            # are rejected afterwards; the next step then resumes inside it and
            # needs its head again. Keep every region closed here recoverable:
            # write it back whole and retire (not release) its row at step end.
            keep_rows: list[tuple[int, int]] = []
            if 1 < qlen <= self.spec_query_len:
                for j in range(nregions):
                    if nbuf - j * region < region:
                        break
                    slot = slots[len(slots) - nregions + j]
                    if j == 0 and head:
                        keep = pending_in  # already listed in completed_slots
                    else:
                        keep = self.pending.acquire(slot)
                        sched.completed_slots.append(slot)
                    keep_rows.append((keep, buf_off + j * region))

            sched.requests.append(
                _RequestPlan(
                    row=row,
                    past=past,
                    qlen=qlen,
                    tok_start=tok_start,
                    head=head,
                    buf_start=buf_off,
                    pending_row_in=pending_in,
                    pending_row_out=pending_out,
                    tail=tail,
                    nscore=(past + qlen - 1) // region,
                    score_slots=None,
                    keep_rows=keep_rows,
                )
            )
            buf_off += nbuf

        # Summary slots the scorer gathers, in logical region order. Built once
        # per step rather than once per layer; the gather itself is per layer.
        for plan in sched.requests:
            if plan.nscore <= 0:
                continue
            ids = torch.arange(plan.nscore, device=device, dtype=torch.int64)
            plan.score_slots = (
                md.block_table[plan.row, ids // rpp].to(torch.int64) * rpp + ids % rpp
            )

        sched.buf_len = buf_off
        if starts:
            sched.starts = torch.tensor(starts, dtype=torch.int32, device=device)
            sched.counts = torch.tensor(counts, dtype=torch.int32, device=device)
            sched.slots = torch.tensor(slots, dtype=torch.int64, device=device)
            sched.live = sched.counts > 0

        return sched

    def _decode_slots_host(
        self,
        md: Step4DSAMetadata,
        nd: int,
        seq_np: np.ndarray,
        slots: torch.Tensor,
    ) -> np.ndarray:
        """This step's physical summary slot per decode row, without a sync.

        The same arithmetic that produced the device ``slots`` tensor, run on
        the block table's host mirror. This is not a shortcut with a different
        answer: ``BlockTable`` maintains the table on the host and uploads it,
        so the mirror is the source and the device tensor is the copy.

        Falls back to reading ``slots`` back when there is no mirror, or when it
        is too short for this batch, so nothing here depends on the mirror
        existing -- it is a fast path, not a new requirement.
        """
        table = md.block_table_np
        if table is None or table.shape[0] < nd:
            return np.asarray(slots.tolist(), dtype=np.int64)

        region = self.geometry.region_size
        rpp = self.regions_per_page
        past = np.maximum(seq_np - 1, 0)
        first_region = past // region
        page = table[np.arange(nd), first_region // rpp].astype(np.int64, copy=False)
        # Block 0 is NULL_BLOCK_ID and is never allocated to a real request, so
        # a zero page is padding however the row got here. Rows past num_reqs
        # hold a stale page rather than NULL_BLOCK_ID -- the runner's padding
        # fill is applied to the device tensor only -- which is why seq_len is
        # part of the mask and not a redundant check.
        real = (seq_np > 0) & (page > 0)
        out = np.where(real, page * rpp + first_region % rpp, -1)

        if CHECK_SLOTS:
            device_slots = np.asarray(slots.tolist(), dtype=np.int64)
            if not np.array_equal(device_slots, out):
                bad = int(np.flatnonzero(device_slots != out)[0])
                raise RuntimeError(
                    "Step-4 DSA: host slot derivation disagrees with the block "
                    f"table on device at decode row {bad}: host "
                    f"{int(out[bad])} vs device {int(device_slots[bad])} "
                    f"(seq_len {int(seq_np[bad])}). The host mirror is supposed "
                    "to be the copy vLLM uploaded from; a mismatch means it was "
                    "reindexed or committed out of order."
                )
        return out

    def _decode_schedule(
        self, md: Step4DSAMetadata, *, persistent: bool = False
    ) -> Step4DSASchedule:
        """Fixed-shape schedule for a decode-only step.

        Every request contributes exactly one token, so it touches exactly one
        region and ``nregions == 1`` unconditionally. That removes the two things
        the general path loops for: the region count per request, and the packed
        buffer offset. Requests get a fixed stride of ``region_size`` in the
        compression buffer instead of being packed end to end -- the compression
        kernel addresses regions through ``starts``/``counts``, so the padding
        between them is never read, and in exchange the layout stops depending on
        the batch's sequence lengths.

        A region only becomes live when this token fills its last slot, i.e.
        ``(seq_len - 1) % region_size == region_size - 1``; otherwise ``counts``
        is zero and the compression kernel skips it, exactly as before.

        The physical summary slot per request comes from the block table, and
        slot-keying is what lets a partial region follow its request across
        batch reordering, so it has to be known on the host. It is *not* read
        back from the device: the block table is built on the host and uploaded,
        so ``md.block_table_np`` is the same data, and the slot arithmetic is
        repeated there. Reading it from the device tensor instead used to be the
        one blocking sync left in a decode step, which drained everything queued
        ahead of it and capped run-ahead at a single step. The device read
        remains as the fallback when no mirror is available.

        **Padded rows.** In a CUDA-graph batch the trailing rows are not
        requests: vLLM gives them ``seq_len = 0`` and a block table of
        ``NULL_BLOCK_ID`` (0, permanently reserved). Naively, ``past = -1`` makes
        ``first_region = -1`` and the gather indexes out of bounds. Every row is
        therefore masked by ``real = (seq_len > 0) & (page > 0)`` and a dead row
        is driven to a state that is inert everywhere downstream: ``counts = 0``
        so compression skips it, ``live = False`` so no summary is stored,
        ``nscore = 0`` so it scores nothing, pending routed to the trash row, and
        no entry in the host loop at all. That one mask covers dummy runs,
        capture, and replay padding alike, which is why
        ``build_for_cudagraph_capture`` needs no override.

        With ``persistent``, the results are written into the state's
        pre-allocated ``[max_num_reqs]`` buffers instead of freshly allocated
        ones, so their addresses survive from capture to replay.
        """
        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        nd = md.num_decodes
        buf = self._buf if persistent else {}
        if persistent and nd > self.max_num_reqs:
            raise RuntimeError(
                f"Step-4 DSA: decode batch of {nd} exceeds max_num_reqs "
                f"{self.max_num_reqs}; the persistent schedule buffers are sized "
                f"from the scheduler's own limit, so this should be unreachable."
            )
        if persistent:
            # The score width is also bounded by what the block table can
            # address: both the scorer and the region selector reach the summary
            # pool through it, so a column past its last page has no page to
            # resolve. vLLM sizes the table from max_model_len, the same figure
            # graph_regions comes from, so in a server this is a no-op -- but the
            # table is the authority, and it is only visible from here.
            cap = md.block_table.shape[1] * rpp
            if cap < self.graph_regions:
                logger.info(
                    "step4 DSA: decode graph width trimmed %d -> %d regions, the "
                    "limit of a %d-page block table.",
                    self.graph_regions,
                    cap,
                    md.block_table.shape[1],
                )
                self.graph_regions = cap

        seq_lens = md.seq_lens[:nd].to(torch.int64)
        past = (seq_lens - 1).clamp_min(0)
        head = past % region
        first_region = torch.div(past, region, rounding_mode="floor")
        page = torch.gather(
            md.block_table[:nd].to(torch.int64),
            1,
            torch.div(first_region, rpp, rounding_mode="floor").unsqueeze(1),
        ).squeeze(1)
        # Block 0 is NULL_BLOCK_ID and is never allocated to a real request, so
        # a zero page is padding however the row got here.
        real = (seq_lens > 0) & (page > 0)
        zero = torch.zeros_like(head)

        slots = torch.where(
            real, page * rpp + first_region % rpp, -torch.ones_like(head)
        )
        closes = real & (head == region - 1)
        head = torch.where(real, head, zero)

        sched = Step4DSASchedule()
        sched.decode = True
        sched.num_decodes = nd
        sched.buf_len = nd * region

        def out(name: str, value: torch.Tensor) -> torch.Tensor:
            """Land ``value`` in the persistent buffer, or pass it through."""
            dst = buf.get(name)
            if dst is None:
                return value
            # Larger captured views retain these rows after the live batch
            # shrinks. Inactive rows must neither compress an old region nor
            # reuse its pending state.
            dst[nd:].fill_(-1 if name == "slots" else 0)
            view = dst[:nd]
            view.copy_(value)
            return view

        # `starts` is the same arange every step, so the persistent copy is
        # prebuilt in _allocate_graph_buffers and only sliced here.
        sched.starts = (
            buf["starts"][:nd]
            if buf
            else torch.arange(nd, device=device, dtype=torch.int32) * region
        )
        sched.counts = out(
            "counts",
            torch.where(closes, torch.full_like(head, region), zero).to(torch.int32),
        )
        sched.slots = out("slots", slots)
        sched.live = out("live", closes)
        sched.head = out("head", head)
        sched.tail = out("tail", torch.where(closes, zero, head + real.to(head.dtype)))
        sched.nscore = out(
            "nscore", torch.where(real, first_region, zero).to(torch.int32)
        )

        # Host-side pending bookkeeping. Both inputs are already on the host:
        # the lengths from vLLM's CPU mirror, the slots derived from the block
        # table's, so the step no longer synchronises here.
        seq_np = md.seq_lens_cpu[:nd].numpy().astype(np.int64, copy=False)
        slots_np = self._decode_slots_host(md, nd, seq_np, slots)
        past_np = np.maximum(seq_np - 1, 0)
        head_np = past_np % region
        first_np = past_np // region
        trash = self.pending_trash_row
        self.end_step()  # see the note in _general_schedule
        self.pending.begin_step()
        rows_in: list[int] = [trash] * nd
        rows_out: list[int] = [trash] * nd
        widest = 0
        # As in the verify and mixed paths, reserve all heads before acquiring.
        for row in range(nd):
            slot = int(slots_np[row])
            if slot >= 0 and head_np[row]:
                rows_in[row] = self.pending.get(slot)
        for row in range(nd):
            slot = int(slots_np[row])
            if slot < 0:  # padding, not a request
                continue
            head_row = int(head_np[row])
            if head_row == region - 1:
                # The region closed this step; its tokens are now in a summary
                # and the pending row is released once every layer has read it.
                sched.completed_slots.append(slot)
            else:
                rows_out[row] = self.pending.acquire(slot)
            widest = max(widest, int(first_np[row]))
        if buf:
            # Staged through pinned memory rather than torch.tensor(list), whose
            # pageable copy is much slower.
            #
            # These two vectors pick the pending row each region's head is read
            # from and its tail written to, and they land in buffers the
            # captured decode graph reads. This copy was blocking for a long
            # time, because with non_blocking=True the run was not reproducible:
            # two identical greedy runs at four concurrent decodes diverged
            # within a few tokens, while single-request decode stayed clean and
            # the piecewise path was bit-exact. That is the signature of a
            # write-after-read hazard on the *host* staging buffer rather than a
            # missing stream order on the device: the copy is issued on the
            # stream that later replays the graph, so the device side was always
            # ordered, but with one buffer per vector the next step's host write
            # could reach it before the previous step's copy had read it out --
            # and a single decode never noticed, because its pending row does
            # not change from step to step.
            #
            # So the copy is async again, against a ring of staging buffers with
            # an event per slot: reuse waits for the copy that last read it.
            # Blocking here is no longer free either, now that the slot
            # derivation above does not already synchronise. The hazard is
            # closed by construction, but it was not re-bisected against the
            # original divergence -- if non-determinism returns at concurrency,
            # set _STAGING_DEPTH to 1 to put the old configuration back and
            # confirm before looking elsewhere.
            stage = self._stage
            self._stage = (stage + 1) % _STAGING_DEPTH
            ev = self._stage_ev[stage] if self._stage_ev else None
            if ev is not None:
                ev.synchronize()
            h_in = self._host["pending_in"][stage]
            h_out = self._host["pending_out"][stage]
            h_in.numpy().fill(trash)
            h_out.numpy().fill(trash)
            h_in.numpy()[:nd] = rows_in
            h_out.numpy()[:nd] = rows_out
            sched.pending_in = buf["pending_in"][:nd]
            sched.pending_out = buf["pending_out"][:nd]
            # Upload the entire backing buffer: replay may read a larger view
            # captured by an earlier batch, including the inactive trash rows.
            buf["pending_in"].copy_(h_in, non_blocking=ev is not None)
            buf["pending_out"].copy_(h_out, non_blocking=ev is not None)
            if ev is not None:
                ev.record()
        else:
            sched.pending_in = torch.tensor(rows_in, dtype=torch.int64, device=device)
            sched.pending_out = torch.tensor(rows_out, dtype=torch.int64, device=device)

        # Widest history in the batch. The scorer reads the summary pool in
        # place through the block table, so this only sizes the logit tensor;
        # columns past a request's own ``nscore`` are masked by the scorer and
        # discarded by the selector's ``history_lengths``.
        sched.max_nscore = widest
        if persistent:
            if widest > self.graph_regions:
                raise RuntimeError(
                    f"Step-4 DSA: a request needs {widest} region scores but the "
                    f"captured decode width is {self.graph_regions}. Raise "
                    f"VLLM_STEP4_DSA_GRAPH_REGIONS or --max-model-len; the width "
                    f"is a graph constant and cannot be grown per step."
                )
            sched.graph_regions = self.graph_regions
        return sched

    def _trim_graph_width(self, md: Step4DSAMetadata) -> None:
        """Clip ``graph_regions`` to what the block table can address.

        See the note in ``_decode_schedule``: vLLM sizes the table from
        max_model_len, so in a server this is a no-op."""
        cap = md.block_table.shape[1] * self.regions_per_page
        if cap < self.graph_regions:
            logger.info(
                "step4 DSA: decode graph width trimmed %d -> %d regions, the "
                "limit of a %d-page block table.",
                self.graph_regions,
                cap,
                md.block_table.shape[1],
            )
            self.graph_regions = cap

    @staticmethod
    def _spec_fields(nr: int, nrm: int, tokens: int) -> dict[str, tuple[int, int]]:
        """Offset and capacity of every field in the packed verify schedule."""
        sizes = {
            "head": nr,
            "tail": nr,
            "pending_in": nr,
            "pending_out": nr,
            "tail_reg": nr,
            "counts": nr * nrm,
            "slots": nr * nrm,
            "live": nr * nrm,
            "keep_rows": nr * nrm,
            "tok_buf": tokens,
            "tok_req": tokens,
            "tok_seq": tokens,
            "tok_nscore": tokens,
        }
        layout: dict[str, tuple[int, int]] = {}
        off = 0
        for name, n in sizes.items():
            layout[name] = (off, n)
            off += n
        return layout

    def _ensure_spec_buffers(self, md: Step4DSAMetadata, nrm: int, q_max: int) -> None:
        """Allocate the persistent verify-schedule buffers, once, outside capture."""
        pages = md.block_table.shape[1]
        if self._spec_raw is not None:
            if self._spec_table.shape[1] != pages:
                raise RuntimeError(
                    f"Step-4 DSA: block table width changed from "
                    f"{self._spec_table.shape[1]} to {pages} pages after the "
                    f"verify schedule buffers were allocated; captured graphs "
                    f"still point at the old ones."
                )
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Step-4 DSA: verify schedule buffers are still unallocated at CUDA "
                "graph capture time; the capture warmup must build one eagerly."
            )
        device = md.block_table.device
        region = self.geometry.region_size
        nr = self.max_num_reqs
        tokens = nr * q_max
        self._spec_layout = self._spec_fields(nr, nrm, tokens)
        total = sum(n for _, n in self._spec_layout.values())
        self._spec_raw = torch.zeros(total, dtype=torch.int64, device=device)
        self._spec_i32 = {
            name: torch.zeros(
                self._spec_layout[name][1], dtype=torch.int32, device=device
            )
            for name in ("counts", "tok_seq", "tok_nscore")
        }
        self._spec_starts = (
            torch.arange(nr * nrm, dtype=torch.int32, device=device) * region
        )
        self._spec_table = torch.zeros(
            (tokens, pages), dtype=md.block_table.dtype, device=device
        )
        pin = torch.cuda.is_available()
        self._spec_host = [
            torch.zeros(total, dtype=torch.int64, pin_memory=pin)
            for _ in range(_STAGING_DEPTH)
        ]
        self._spec_ev = (
            [torch.cuda.Event() for _ in range(_STAGING_DEPTH)] if pin else []
        )
        for ev in self._spec_ev:
            ev.record()
        self._spec_stage = 0

    def _spec_decode_schedule(
        self, md: Step4DSAMetadata, *, persistent: bool = False
    ) -> Step4DSASchedule:
        """Fixed-shape schedule for a speculative verify step.

        Every request scores up to ``Q = spec_query_len`` query tokens. Without
        this path such a step goes through ``_general_schedule`` as Q-token
        prefill rows: a Python loop per request in every one of the 23 layers,
        one ``indexer_logits`` launch per request, the prefill selector's host
        syncs, and -- because the backend could then only capture single-token
        decode -- no CUDA graph for the whole model, nor for the MTP drafter.
        That is what the decode trace's bubbles were.

        Here each query token is a decode row of its own: token ``i`` of a
        request whose cache held ``past`` tokens is scored and attended exactly
        as a single-token decode at ``seq_len = past + i + 1`` would be, with
        its own causal horizon, against a per-token copy of the request's
        block-table row. That is the same selection the prefill path makes per
        query position, so the result does not depend on which path ran.

        Compression is the decode path's, widened: a request's tokens land at
        ``head`` in a buffer of ``nrm`` regions (``nrm = ceil((R - 1 + Q) / R)``),
        with a fixed per-request stride so every shape is a function of
        ``num_decodes`` and ``num_decode_tokens`` alone. Pending bookkeeping,
        including the rollback ``keep_rows``, is the general schedule's; see the
        comment there.

        ``persistent`` lands everything in buffers allocated once, so a captured
        graph's replay reads this step's values.
        """
        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        nd, ndt = md.num_decodes, md.num_decode_tokens
        q_max = max(self.spec_query_len, 1)
        nrm = -(-(region - 1 + q_max) // region)
        trash = self.pending_trash_row

        qsl = md.query_start_loc_cpu[: nd + 1].numpy().astype(np.int64, copy=False)
        seq = md.seq_lens_cpu[:nd].numpy().astype(np.int64, copy=False)
        # A non-monotonic query_start_loc would make np.repeat below raise and
        # take the whole engine down. Clamp instead, so a stale/garbage row
        # degrades into dead padding and is masked out by `alive` rather than
        # killing the run.
        qlen = np.maximum(np.diff(qsl), 0)
        if nd and int(qlen.max()) > q_max:
            raise RuntimeError(
                f"Step-4 DSA: a verify row has {int(qlen.max())} query tokens, "
                f"more than spec_query_len = {q_max}."
            )
        table = md.block_table_np
        if table is None or table.shape[0] < nd:
            table = md.block_table[:nd].to("cpu", non_blocking=False).numpy()

        if persistent:
            if nd > self.max_num_reqs or ndt > self.max_num_reqs * q_max:
                raise RuntimeError(
                    f"Step-4 DSA: verify batch of {nd} requests / {ndt} tokens "
                    f"exceeds the persistent buffers ({self.max_num_reqs} x {q_max})."
                )
            self._trim_graph_width(md)
            self._ensure_spec_buffers(md, nrm, q_max)
            layout = self._spec_layout
        else:
            layout = self._spec_fields(nd, nrm, ndt)
        total = sum(n for _, n in layout.values())
        h = np.zeros(total, dtype=np.int64)

        def field(name: str, n: int) -> np.ndarray:
            off, _ = layout[name]
            return h[off : off + n]

        if persistent:
            # Captured views can be wider than this step's logical batch.
            # Never let their inactive rows address a live pending row or
            # overwrite the first real token's staging location.
            for name in ("pending_in", "pending_out", "keep_rows"):
                field(name, layout[name][1])[:] = trash
            capacity = layout["tok_buf"][1]
            field("tok_buf", capacity)[:] = nd * nrm * region + np.arange(
                capacity, dtype=np.int64
            )

        head = field("head", nd)
        tail = field("tail", nd)
        p_in = field("pending_in", nd)
        p_out = field("pending_out", nd)
        tail_reg = field("tail_reg", nd)
        counts = field("counts", nd * nrm)
        slots = field("slots", nd * nrm)
        live = field("live", nd * nrm)
        keep_rows = field("keep_rows", nd * nrm)
        tok_buf = field("tok_buf", ndt)
        tok_req = field("tok_req", ndt)
        tok_seq = field("tok_seq", ndt)
        tok_ns = field("tok_nscore", ndt)

        p_in[:] = trash
        p_out[:] = trash
        keep_rows[:] = trash
        reqs = np.arange(nd, dtype=np.int64)
        tail_reg[:] = reqs * nrm

        # Padded rows (CUDA-graph batch padding, dummy runs) are not requests:
        # vLLM gives them qlen 0 and seq_len 0, and block 0 (NULL_BLOCK_ID) is
        # never a real page. Masked exactly as in _decode_schedule.
        past = seq - qlen
        alive = (qlen > 0) & (seq > 0) & (past >= 0)
        hd = np.where(alive, past % region, 0)
        fr = np.where(alive, past // region, 0)
        if nd:
            page_idx = np.minimum(fr // rpp, table.shape[1] - 1)
            page0 = table[reqs, page_idx].astype(np.int64, copy=False)
            alive &= page0 > 0
        hd = np.where(alive, hd, 0)

        # Per-token rows. Tokens past the last request (graph padding) go to
        # a scratch area at the end of the buffer with seq_len 0.
        ntok = int(qsl[nd]) if nd else 0
        r_of_t = np.repeat(reqs, qlen)
        i_of_t = np.arange(ntok, dtype=np.int64) - qsl[r_of_t]
        live_t = alive[r_of_t]
        pos_t = past[r_of_t] + i_of_t
        tok_req[:ntok] = r_of_t
        tok_seq[:ntok] = np.where(live_t, pos_t + 1, 0)
        tok_ns[:ntok] = np.where(live_t, pos_t // region, 0)
        tok_buf[:ntok] = r_of_t * (nrm * region) + hd[r_of_t] + i_of_t
        tok_buf[ntok:] = nd * nrm * region + np.arange(ntok, ndt, dtype=np.int64)
        widest = int(tok_ns.max()) if ndt else 0

        self.end_step()  # see the note in _general_schedule
        self.pending.begin_step()
        sched = Step4DSASchedule()
        alive_rows = np.flatnonzero(alive).tolist()
        reg_slots_of: dict[int, list[int]] = {}
        for r in alive_rows:
            first = int(fr[r])
            nreg = -(-(int(hd[r]) + int(qlen[r])) // region)
            pages = table[r]
            reg_slots_of[r] = [
                int(pages[(first + j) // rpp]) * rpp + (first + j) % rpp
                for j in range(nreg)
            ]
        # Every head read first: get() touches its row for this step, so none of
        # the acquire() calls below can recycle a row another request is about
        # to resume from (a rolled-back request reads a retired row).
        for r in alive_rows:
            if hd[r]:
                p_in[r] = self.pending.get(reg_slots_of[r][0])
        for r in alive_rows:
            q = int(qlen[r])
            hr = int(hd[r])
            nbuf = hr + q
            reg_slots = reg_slots_of[r]
            nreg = len(reg_slots)
            base = r * nrm
            for j in range(nreg):
                slots[base + j] = reg_slots[j]
                if nbuf >= (j + 1) * region:
                    counts[base + j] = region
                    live[base + j] = 1
            head[r] = hr
            if hr and nbuf >= region:
                sched.completed_slots.append(reg_slots[0])
            # Regions closed here may be reopened by a rejection; keep them
            # recoverable exactly as _general_schedule does: written back whole
            # and retired (not released) at step end.
            if 1 < q <= self.spec_query_len:
                for j in range(nreg):
                    if nbuf - j * region < region:
                        break
                    if j == 0 and hr:
                        keep_rows[base + j] = p_in[r]  # already in completed_slots
                    else:
                        keep_rows[base + j] = self.pending.acquire(reg_slots[j])
                        sched.completed_slots.append(reg_slots[j])
            t = nbuf % region
            tail[r] = t
            tail_reg[r] = base + nreg - 1
            if t:
                p_out[r] = self.pending.acquire(reg_slots[-1])

        if persistent:
            # Same pinned-ring discipline as _decode_schedule, for the same
            # write-after-read reason.
            stage = self._spec_stage
            self._spec_stage = (stage + 1) % _STAGING_DEPTH
            ev = self._spec_ev[stage] if self._spec_ev else None
            if ev is not None:
                ev.synchronize()
            host = self._spec_host[stage]
            host.numpy()[:] = h
            raw = self._spec_raw
            raw.copy_(host, non_blocking=ev is not None)
            if ev is not None:
                ev.record()
        else:
            raw = torch.from_numpy(h).to(device)

        def dev(name: str, n: int) -> torch.Tensor:
            off, _ = layout[name]
            return raw[off : off + n]

        def dev_i32(name: str, n: int) -> torch.Tensor:
            if persistent:
                backing = self._spec_i32[name]
                # A previously captured larger slice still sees the backing
                # storage after a batch shrinks or this DP becomes idle.
                backing[n:].zero_()
                out = backing[:n]
                out.copy_(dev(name, n))
                return out
            return dev(name, n).to(torch.int32)

        sched.spec = True
        sched.num_decodes = nd
        sched.num_tokens = ndt
        sched.nrm = nrm
        sched.buf_len = nd * nrm * region + ndt
        sched.starts = (
            self._spec_starts[: nd * nrm]
            if persistent
            else torch.arange(nd * nrm, dtype=torch.int32, device=device) * region
        )
        sched.counts = dev_i32("counts", nd * nrm)
        sched.slots = dev("slots", nd * nrm)
        sched.live = dev("live", nd * nrm)
        sched.keep_rows = dev("keep_rows", nd * nrm)
        sched.head = dev("head", nd)
        sched.tail = dev("tail", nd)
        sched.pending_in = dev("pending_in", nd)
        sched.pending_out = dev("pending_out", nd)
        sched.tail_reg = dev("tail_reg", nd)
        sched.tok_buf = dev("tok_buf", ndt)
        sched.tok_seq = dev_i32("tok_seq", ndt)
        sched.tok_nscore = dev_i32("tok_nscore", ndt)
        tok_req_d = dev("tok_req", ndt)
        if persistent:
            sched.tok_table = self._spec_table[:ndt]
            self._spec_table[ndt:].zero_()
            torch.index_select(md.block_table, 0, tok_req_d, out=sched.tok_table)
        else:
            sched.tok_table = md.block_table.index_select(0, tok_req_d)
        sched.req_table = md.block_table
        sched.query_start_loc = md.query_start_loc

        sched.max_nscore = widest
        if persistent:
            if widest > self.graph_regions:
                raise RuntimeError(
                    f"Step-4 DSA: a request needs {widest} region scores but the "
                    f"captured decode width is {self.graph_regions}. Raise "
                    f"VLLM_STEP4_DSA_GRAPH_REGIONS or --max-model-len; the width "
                    f"is a graph constant and cannot be grown per step."
                )
            if self.num_kv_groups * ndt > self._decode_logits.shape[0]:
                raise RuntimeError(
                    f"Step-4 DSA: {ndt} verify rows overflow the score pool "
                    f"({self._decode_logits.shape[0]} group-major rows)."
                )
            sched.graph_regions = self.graph_regions
        return sched

    def finish_layer(self) -> None:
        """Called by each DSA layer once it has consumed this step's schedule."""
        self._layers_done += 1
        if self._layers_done >= len(self._registered) and not (
            self._schedule is not None and self._schedule.builder_owned
        ):
            # A builder-owned schedule is released at the *start* of the next
            # one instead. Releasing here would be wrong the moment a step is a
            # graph replay, because then no layer calls finish_layer at all and
            # the counter this branch reads is whatever the capture left behind.
            self.end_step()

    def end_step(self) -> None:
        """Release pending rows for regions that closed."""
        if self._schedule is None:
            return
        # With speculative decoding a closed region may be reopened by a
        # rollback on the next step, so its row is only demoted, not freed.
        drop = self.pending.retire if self.spec_query_len > 1 else self.pending.release
        for slot in self._schedule.completed_slots:
            drop(slot)
        self._schedule.completed_slots.clear()


# --------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------


class Step4IndexerNorm(nn.Module):
    """Parameter holder for one indexer norm; the maths happens in the kernel.

    ``indexer_norm_rope`` fuses normalisation, the affine, and the partial
    rotation into a single pass, so nothing here has a ``forward``. It exists
    because the checkpoint stores these as ``...sparse_indexer_q_norm.weight``
    and ``...sparse_indexer_k_norm.{weight,bias}``, and a module is the only way
    to make ``named_parameters()`` produce those exact names.

    fp32, like every other norm parameter in this checkpoint. The q-norm is
    applied as ``x * (1 + w)``, so rounding a zero-centred ``w`` to bf16 would
    perturb the multiplier by far more than bf16 compute noise -- the same
    reason ``_fp32_gemma_rms_norm`` exists on the attention side.
    """

    def __init__(self, dim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))


def _group_slice_loader(first_group: int, num_groups: int, total_groups: int):
    """Weight loader for a tensor sliced along dim 0 by *provider group*.

    ``sparse_indexer_q`` and ``sparse_indexer_w`` are the only indexer tensors
    that shard, and they shard by provider group rather than by rank: at TP8
    with 4 groups, ranks 2k and 2k+1 both hold group k's rows, which is the same
    replica convention ``QKVParallelLinear`` uses for Step-4's 4 KV heads. This
    mirrors ``inference/convert.py``'s ``_slice_grouped_dim``.
    """

    def loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        chunk = loaded_weight.shape[0] // total_groups
        if chunk * total_groups != loaded_weight.shape[0]:
            raise ValueError(
                f"indexer tensor of {loaded_weight.shape[0]} rows does not split "
                f"into {total_groups} provider groups"
            )
        shard = loaded_weight[first_group * chunk : (first_group + num_groups) * chunk]
        if tuple(shard.shape) != tuple(param.data.shape):
            raise ValueError(
                f"indexer shard {tuple(shard.shape)} does not fit parameter "
                f"{tuple(param.data.shape)}"
            )
        param.data.copy_(shard)

    return loader


class Step4Indexer:
    """Scores every completed region of the context and picks the top ``topk``.

    Deliberately not an ``nn.Module``. Its parameters live directly on the
    owning ``Step4Attention`` because that is where the checkpoint puts them
    (``model.layers.N.self_attn.sparse_indexer_q.weight``); wrapping them in a
    submodule would rename every one of them and force a mapping table. The
    reference implementation makes the same call for the same reason.

    It also has to run model-side rather than inside the attention backend: it
    consumes ``hidden_states``, and ``Attention.forward`` only ever sees q/k/v.
    The result is handed to the backend through ``layer.step4_sparse_plan``.

    Per step, in this order -- the order matters, because a region that fills
    with *this* chunk's tokens must be scorable by later tokens of the same
    chunk:

    1. project and normalise/rotate the proxy q/k/z,
    2. fold the pending tail of the previous step in, compress every region
       that is now complete, buffer the new tail,
    3. score each request's queries against its own summaries,
    4. select and pack, and publish the plan.
    """

    def __init__(
        self,
        owner: nn.Module,
        *,
        geometry: Step4DSAGeometry,
        state: Step4DSAState,
        dsa_slot: int,
        num_kv_groups: int,
        hidden_size: int,
        tp_rank: int,
        tp_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: dict | None,
        max_position: int,
        params_dtype: torch.dtype,
        num_attention_heads: int,
    ) -> None:
        self.owner = owner
        self.geometry = geometry
        self.state = state
        self.slot = dsa_slot
        self.num_kv_groups = num_kv_groups
        # The state sizes its group-major score pool from this and cannot know
        # it on its own. Every DSA layer on a rank holds the same number of KV
        # groups, so the first one to be built settles it.
        if state.num_kv_groups not in (0, num_kv_groups):
            raise ValueError(
                f"Step-4 DSA: layer declares {num_kv_groups} KV groups but the "
                f"shared state was already sized for {state.num_kv_groups}."
            )
        state.num_kv_groups = num_kv_groups
        self.eps = rms_norm_eps
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.max_position = int(max_position)

        groups_total = geometry.num_provider_groups
        heads_per_group = geometry.num_indexer_heads // groups_total
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid tensor-parallel rank {tp_rank}/{tp_size}")
        if max(tp_size, groups_total) % min(tp_size, groups_total):
            raise ValueError(
                f"TP size {tp_size} and {groups_total} provider groups must divide "
                "evenly for group sharding or replication"
            )
        if tp_size <= groups_total:
            local_groups = groups_total // tp_size
            first_group = tp_rank * local_groups
        else:
            # More ranks than groups: each group is replicated across
            # tp_size // groups_total ranks, matching QKVParallelLinear's KV
            # replica rule (tp_rank // num_kv_head_replicas).
            local_groups = 1
            first_group = tp_rank // (tp_size // groups_total)
        if local_groups != num_kv_groups:
            raise ValueError(
                f"indexer resolves {local_groups} provider group(s) on this rank "
                f"but attention has {num_kv_groups} KV group(s); the two index the "
                f"same group axis and must agree"
            )
        self.heads_per_group = heads_per_group
        state.heads_per_group = heads_per_group
        self.num_heads = local_groups * heads_per_group
        # The divisor is the heads inside ONE provider group, not every head
        # resident on this rank. At TP8 those differ (4 vs 4 only because a rank
        # holds exactly one group); deriving it from local head count on a
        # different TP degree gives the wrong scale and silently changes which
        # regions get selected.
        self.weight_prescale = float(heads_per_group) ** -0.5

        proxy = geometry.proxy_dim
        owner.sparse_indexer_q = nn.Linear(
            hidden_size, self.num_heads * proxy, bias=False, dtype=params_dtype
        )
        owner.sparse_indexer_k = nn.Linear(
            hidden_size, proxy, bias=False, dtype=params_dtype
        )
        owner.sparse_indexer_z = nn.Linear(
            hidden_size, proxy, bias=False, dtype=params_dtype
        )
        # fp32 in the checkpoint and kept that way; it is rounded to the
        # activation dtype at call time instead, which is the rounding boundary
        # the deployed kernel has.
        owner.sparse_indexer_w = nn.Linear(
            hidden_size, self.num_heads, bias=False, dtype=torch.float32
        )
        owner.sparse_indexer_q_norm = Step4IndexerNorm(proxy, bias=False)
        owner.sparse_indexer_k_norm = Step4IndexerNorm(proxy, bias=True)
        # Present on every head of every layer at its initialisation value and
        # unused by the deployed kernels: the indexer scores with a weighted
        # ReLU, so there is no softmax for a scalable-softmax scale to act on.
        # Registered only so that loading the checkpoint stays exhaustive.
        owner.ssmax_s = nn.Parameter(
            torch.zeros(num_attention_heads, dtype=torch.float32),
            requires_grad=False,
        )

        sharded = _group_slice_loader(first_group, local_groups, groups_total)
        owner.sparse_indexer_q.weight.weight_loader = sharded
        owner.sparse_indexer_w.weight.weight_loader = sharded

        state.register_layer(dsa_slot)

        # The op is dispatched by name, because a Python object cannot cross a
        # custom-op boundary. Names are unique per layer per process, and vLLM
        # already relies on that for the attention layers themselves.
        self.layer_name = owner.attn.layer_name
        _INDEXERS[self.layer_name] = self
        # Mutated by the op, and the reason the op is not optimised away: see
        # step4_dsa_index. Non-persistent so it stays out of the state dict, but
        # a real buffer so that .cuda() moves it with the rest of the layer.
        owner.register_buffer(
            "step4_dsa_plan_token",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )

    # -- forward ----------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> None:
        """Score and select, via the opaque op so that ``torch.compile`` cannot
        trace into this. See ``step4_dsa_index`` for why that matters."""
        torch.ops.vllm.step4_dsa_index(
            hidden_states,
            positions,
            self.layer_name,
            # Read through the owner, not cached: .cuda() rebinds the buffer.
            self.owner.step4_dsa_plan_token,
        )

    def run(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> int:
        attn = self.owner.attn
        raw = get_forward_context().attn_metadata
        if isinstance(raw, dict):
            md = raw.get(attn.layer_name)
        elif isinstance(raw, list):
            md = raw[0].get(attn.layer_name)
        else:
            md = raw
        kv_cache = attn.kv_cache
        if not isinstance(md, Step4DSAMetadata) or kv_cache.numel() == 0:
            # Profiling or a dummy run: there is no cache to summarise and the
            # impl short-circuits on the same condition.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{attn.layer_name}: the DSA indexer is being captured into "
                    f"a CUDA graph with no attention metadata (or an unallocated "
                    f"KV cache). The recorded graph would contain no DSA work at "
                    f"all and every replay would attend to nothing, silently."
                )
            attn.step4_sparse_plan = None
            return 0

        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and md.schedule is None:
            # Everything a captured step needs must have been written by
            # Step4DSAMetadataBuilder.build(), into buffers whose addresses the
            # recorded kernels keep pointing at. Without that the schedule below
            # is built here, from tensors allocated inside the capture, and
            # every replay would re-read this one step's values forever. The
            # host round-trip in _general_schedule would fail first -- but with
            # a stream-capture error, not a reason.
            raise RuntimeError(
                f"{attn.layer_name}: CUDA graph capture reached the DSA indexer "
                f"without a builder-supplied schedule "
                f"(num_prefills={md.num_prefills}, num_decodes={md.num_decodes}, "
                f"graph_decode={self.state.graph_decode}). Only decode-only steps "
                f"are capturable; see Step4DSAMetadataBuilder.build."
            )

        state = self.state
        state.ensure_allocated(kv_cache, hidden_states.device)
        sched = state.schedule(md)
        if self.slot == state.first_slot:
            state.step_cache = {}

        num_tokens = md.num_actual_tokens
        index_q, index_k, index_z, weights = self._project(
            hidden_states[:num_tokens], positions[:num_tokens], num_tokens
        )
        self._update_summaries(sched, index_k, index_z)
        prefill_logits, decode_logits, width = self._score(md, sched, index_q, weights)

        topk = max(width, 1) if DENSE_EQUIV else self.geometry.topk
        attn.step4_sparse_plan = build_plan(
            md,
            prefill_logits,
            decode_logits,
            topk=topk,
            num_kv_groups=self.num_kv_groups,
            regions_per_page=state.regions_per_page,
            decode_seq_lens=sched.tok_seq if sched.spec else None,
            decode_table=sched.tok_table if sched.spec else None,
            step_cache=state.step_cache,
        )
        state.finish_layer()
        return width

    # -- stage 1: projection ----------------------------------------------

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        owner = self.owner
        proxy = self.geometry.proxy_dim

        q_norm_w, k_norm_w, k_norm_b, w_weight = self._cast_params(hidden_states.dtype)
        merged = self._merged_projection(hidden_states.dtype)
        if merged is not None:
            # One GEMM for q | k | z | w; the outputs are column views of it.
            nq = self.num_heads * proxy
            out = F.linear(hidden_states, merged)
            index_q = out[:, :nq]
            index_k = out[:, nq : nq + proxy]
            index_z = out[:, nq + proxy : nq + 2 * proxy]
            raw_weights = out[:, nq + 2 * proxy :]
        else:
            index_q = owner.sparse_indexer_q(hidden_states).contiguous()
            index_k = owner.sparse_indexer_k(hidden_states).contiguous()
            index_z = owner.sparse_indexer_z(hidden_states).contiguous()
            raw_weights = None

        cos, sin = self.state.rope_tables(
            theta=self.rope_theta,
            scaling=self.rope_scaling,
            max_position=self.max_position,
            device=hidden_states.device,
            dtype=index_q.dtype,
        )
        # Only index_k is normalised and rotated; index_q is normalised without
        # rotation and index_z is returned untouched -- it is a per-dimension
        # softmax logit, not a key.
        index_q, index_k, index_z = indexer_norm_rope(
            index_q,
            index_k,
            index_z,
            q_norm_w,
            k_norm_w,
            k_norm_b,
            cos,
            sin,
            positions,
            head_dim=proxy,
            num_q_heads=self.num_heads,
            num_k_heads=1,
            rotary_dim=cos.shape[1],
            eps=self.eps,
            q_norm_weight_bias=1.0,
            # Every scorer applies round_activations_e4m3 to q; doing it in the
            # norm's store makes that idempotent pass skippable (q_prerounded).
            round_q=True,
            # Row-strided z is fine: the summary kernels take its row stride.
            contiguous_z=merged is None,
        )

        # Round the fp32 weight to the activation dtype *before* the GEMM and
        # widen the result back to fp32 afterwards. Both boundaries are part of
        # the deployed semantics: doing the GEMM in fp32 gives different scores.
        # The widening and the prescale are left to the scorer (_prescaled, or
        # folded into spec_indexer_logits), so the GEMM output is returned raw.
        if raw_weights is None:
            raw_weights = F.linear(hidden_states, w_weight)
        weights = raw_weights.view(num_tokens, self.num_kv_groups, self.heads_per_group)

        return (
            index_q.view(num_tokens, self.num_kv_groups, self.heads_per_group, proxy),
            index_k.view(num_tokens, 1, proxy),
            index_z.view(num_tokens, 1, proxy),
            weights,
        )

    def _merged_projection(self, dtype: torch.dtype) -> torch.Tensor | None:
        """``[q; k; z; w]`` weights stacked into one ``[N, hidden]`` matrix, or None.

        Four decode-sized GEMMs cost ~35 us per DSA layer at M=64 and one GEMM over
        the stacked weight ~24 us. Built once, outside graph capture: q/k/z are
        re-pointed at their rows of it (no extra memory) and w is its activation-
        dtype cast, the rounding boundary _project already applies to it. The
        merged GEMM's rounding matches the separate GEMMs except for a few 1-ulp
        differences. STEP5_DSA_MERGED_PROJ=0 keeps the separate GEMMs.
        """
        merged = getattr(self, "_merged_w", None)
        if merged is not None and merged.dtype == dtype:
            return merged
        owner = self.owner
        q, k, z = (
            owner.sparse_indexer_q.weight,
            owner.sparse_indexer_k.weight,
            owner.sparse_indexer_z.weight,
        )
        if (
            not _MERGED_PROJ
            or torch.cuda.is_current_stream_capturing()
            or q.dtype != dtype
            or not (k.dtype == z.dtype == dtype)
        ):
            return None
        w = owner.sparse_indexer_w.weight
        rows = [q.shape[0], k.shape[0], z.shape[0], w.shape[0]]
        merged = torch.empty(sum(rows), q.shape[1], dtype=dtype, device=q.device)
        parts = merged.split(rows)
        for dst, src in zip(parts, (q, k, z, w)):
            dst.copy_(src.detach())
        for param, view in zip((q, k, z), parts):
            param.data = view
        self._merged_w = merged
        logger.info_once("DSA indexer: q/k/z/w projections merged into one GEMM")
        return merged

    def _cast_params(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The norm parameters and the fp32 head-weight matrix in ``dtype``.

        Cast once and kept, rather than per call: four casts per layer per step
        were four launches each, ~20 us of every DSA layer at decode. Keyed on
        the source storage so a re-bound parameter is picked up; a cast made
        during graph capture is not kept, since it lives in the graph's pool.
        """
        owner = self.owner
        sources = (
            owner.sparse_indexer_q_norm.weight,
            owner.sparse_indexer_k_norm.weight,
            owner.sparse_indexer_k_norm.bias,
            owner.sparse_indexer_w.weight,
        )
        key = (dtype,) + tuple(t.data_ptr() for t in sources)
        cached = getattr(self, "_param_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        casts = tuple(t.to(dtype) for t in sources)
        if not torch.cuda.is_current_stream_capturing():
            self._param_cache = (key, casts)
        return casts

    def _prescaled(self, weights: torch.Tensor) -> torch.Tensor:
        """Raw head weights from ``_project``, widened to fp32 and prescaled."""
        return (weights.float() * self.weight_prescale).contiguous()

    # -- stage 2: summaries -------------------------------------------------

    def _update_summaries(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """Compress every region that completed this step; re-buffer the tails.

        The compression buffer is assembled rather than compressed in place
        because a region can span a chunk boundary: its leading tokens were
        produced by an earlier forward pass and only survive in pending. The
        reference gets a contiguous per-request buffer for free; here it has to
        be built, which is the one structural difference between the two.

        The summary is a per-dimension softmax-weighted mean over the region's
        tokens with a shared shift, so it is not incrementally computable
        byte-for-byte -- holding the tokens and compressing once is what keeps
        the streaming result identical to the batched one rather than merely
        close to it.
        """
        if sched.spec:
            self._update_summaries_spec(sched, index_k, index_z)
            return
        if sched.decode:
            self._update_summaries_decode(sched, index_k, index_z)
            return
        if sched.starts is None:
            return
        state = self.state
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]
        proxy = self.geometry.proxy_dim

        buffer_key = index_k.new_zeros((sched.buf_len, 1, proxy))
        buffer_z = index_z.new_zeros((sched.buf_len, 1, proxy))
        for plan in sched.requests:
            base = plan.buf_start
            if plan.head:
                rows = slice(0, plan.head)
                buffer_key[base : base + plan.head] = pending_key[
                    plan.pending_row_in, rows
                ].unsqueeze(1)
                buffer_z[base : base + plan.head] = pending_z[
                    plan.pending_row_in, rows
                ].unsqueeze(1)
            start = base + plan.head
            tokens = slice(plan.tok_start, plan.tok_start + plan.qlen)
            buffer_key[start : start + plan.qlen] = index_k[tokens]
            buffer_z[start : start + plan.qlen] = index_z[tokens]

        _, summary_fp8 = csa_compress_regions(
            buffer_key,
            buffer_z,
            sched.starts,
            sched.counts,
            region_size=self.geometry.region_size,
        )
        _store_live_summary_rows(
            state.summary[self.slot],
            summary_fp8,
            sched.slots,
            sched.live,
        )

        for plan in sched.requests:
            for row, off in plan.keep_rows:
                pending_key[row] = buffer_key[off : off + self.geometry.region_size, 0]
                pending_z[row] = buffer_z[off : off + self.geometry.region_size, 0]
            if not plan.tail:
                continue
            end = plan.buf_start + plan.head + plan.qlen
            pending_key[plan.pending_row_out, : plan.tail] = buffer_key[
                end - plan.tail : end, 0
            ]
            pending_z[plan.pending_row_out, : plan.tail] = buffer_z[
                end - plan.tail : end, 0
            ]

    # -- stage 2b: summaries, decode-only fast path -------------------------

    def _update_summaries_decode(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """The same compression, with the request loop replaced by a batch axis.

        Every decode request contributes exactly one token, so the buffer can be
        given a fixed ``region_size`` stride per request instead of the packed
        variable stride the general path computes. Compression addresses regions
        through ``starts``/``counts``, so the padding between them is never read
        and the fixed stride is result-neutral -- but it makes every shape here a
        function of ``num_decodes`` alone, which is what capture needs.

        Requests that do not close a region still write their tail; they are
        routed to ``pending_trash_row`` so the write is unconditional and the
        host never has to branch per request.
        """
        state = self.state
        region = self.geometry.region_size
        proxy = self.geometry.proxy_dim
        num_decodes = sched.num_decodes
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]

        positions = torch.arange(region, device=index_k.device)
        # Slots [0, head) come from pending; the rest is padding until the
        # current token is scattered in at `head`.
        head_mask = (positions[None, :] < sched.head[:, None]).unsqueeze(-1)
        zero = pending_key.new_zeros(())
        buffer_key = torch.where(head_mask, pending_key[sched.pending_in], zero)
        buffer_z = torch.where(head_mask, pending_z[sched.pending_in], zero)

        at_head = sched.head.view(num_decodes, 1, 1).expand(num_decodes, 1, proxy)
        buffer_key.scatter_(
            1, at_head, index_k[:num_decodes].view(num_decodes, 1, proxy)
        )
        buffer_z.scatter_(1, at_head, index_z[:num_decodes].view(num_decodes, 1, proxy))

        _, summary_fp8 = csa_compress_regions(
            buffer_key.view(num_decodes * region, 1, proxy),
            buffer_z.view(num_decodes * region, 1, proxy),
            sched.starts,
            sched.counts,
            region_size=region,
        )
        _store_live_summary_rows(
            state.summary[self.slot],
            summary_fp8,
            sched.slots,
            sched.live,
        )

        tail_mask = (positions[None, :] < sched.tail[:, None]).unsqueeze(-1)
        rows = sched.pending_out
        pending_key[rows] = torch.where(tail_mask, buffer_key, pending_key[rows])
        pending_z[rows] = torch.where(tail_mask, buffer_z, pending_z[rows])

    # -- stage 2c: summaries, speculative verify fast path ------------------

    def _update_summaries_spec(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """The decode fast path for ``Q``-token rows.

        Each request owns ``nrm`` regions of buffer: region 0 starts with its
        ``head`` pending tokens, and its query tokens are scattered in after
        them through ``tok_buf``. Graph-padding tokens land in a scratch tail
        past the last request. Closed regions are compressed and stored; the
        trailing partial region goes back to pending, and regions a rejection
        may reopen are copied whole into their ``keep_rows`` -- the general
        path's three steps, with the request loop turned into a batch axis.
        """
        state = self.state
        region = self.geometry.region_size
        proxy = self.geometry.proxy_dim
        nd, ndt, nrm = sched.num_decodes, sched.num_tokens, sched.nrm
        span = nrm * region
        regions = nd * nrm
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]

        if _SPEC_FUSED_SUMMARIES and sched.query_start_loc is not None:
            if regions == 0:
                return
            summary = state.summary[self.slot]
            _spec_summaries_kernel[(regions,)](
                index_k,
                index_z,
                pending_key,
                pending_z,
                summary,
                sched.query_start_loc,
                sched.head,
                sched.tail,
                sched.pending_in,
                sched.pending_out,
                sched.tail_reg,
                sched.counts,
                sched.slots,
                sched.live,
                sched.keep_rows,
                ndt,
                state.pending_trash_row,
                index_k.stride(0),
                index_z.stride(0),
                pending_key.stride(0),
                pending_key.stride(1),
                summary.stride(0),
                NRM=nrm,
                REGION=region,
                PROXY_DIM=proxy,
                BLOCK_T=triton.next_power_of_2(region),
                BLOCK_D=triton.next_power_of_2(proxy),
                num_warps=4,
                **_NO_FP_FUSION,
            )
            return

        positions = torch.arange(region, device=index_k.device)
        head_mask = (positions[None, :] < sched.head[:, None]).unsqueeze(-1)
        tail_mask = (positions[None, :] < sched.tail[:, None]).unsqueeze(-1)
        zero = pending_key.new_zeros(())

        flats = []
        for pending, tokens in ((pending_key, index_k), (pending_z, index_z)):
            buffer = tokens.new_zeros((regions * region + ndt, proxy))
            per_req = buffer[: regions * region].view(nd, span, proxy)
            per_req[:, :region] = torch.where(
                head_mask, pending[sched.pending_in], zero
            )
            buffer.index_copy_(0, sched.tok_buf, tokens[:ndt, 0])
            flats.append(buffer)
        buffer_key, buffer_z = flats

        _, summary_fp8 = csa_compress_regions(
            buffer_key[: regions * region].unsqueeze(1),
            buffer_z[: regions * region].unsqueeze(1),
            sched.starts,
            sched.counts,
            region_size=region,
        )
        _store_live_summary_rows(
            state.summary[self.slot],
            summary_fp8,
            sched.slots,
            sched.live,
        )

        rows = sched.pending_out
        for pending, buffer in ((pending_key, buffer_key), (pending_z, buffer_z)):
            by_region = buffer[: regions * region].view(regions, region, proxy)
            pending[rows] = torch.where(
                tail_mask, by_region[sched.tail_reg], pending[rows]
            )
            # Rows not kept point at the trash row; their writes collide there
            # harmlessly.
            pending[sched.keep_rows] = by_region

    # -- stage 3: scoring ---------------------------------------------------

    def _score(
        self,
        md: Step4DSAMetadata,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Per-request indexer scores, in the selector's group-major row order.

        Two tensors rather than one because the two selectors disagree about
        what a row is: decode has one row per request, prefill one per token.

        The width is the largest number of complete regions any row can see, not
        the summary pool's capacity. The difference is not cosmetic -- at a 64K
        context the capacity-sized tensor for a 16K chunk would be tens of GB
        for a live corner a thousandth of that size.

        The loop is over requests because each scores against a different key
        set and the kernel takes a single shared key head. There is no longer a
        loop over the query axis: the kernel writes into ``prefill_logits`` /
        ``decode_logits`` in place, so nothing has to be chunked to bound a
        temporary. See ``INDEXER_Q_CHUNK`` for the valve that restores chunking.
        """
        state = self.state
        groups = self.num_kv_groups
        num_decodes = md.num_decodes
        num_decode_tokens = md.num_decode_tokens
        num_prefill_tokens = md.num_actual_tokens - num_decode_tokens
        device = index_q.device
        summary = state.summary[self.slot]

        if sched.spec:
            return self._score_spec(sched, index_q, weights)
        weights = self._prescaled(weights)
        if sched.decode:
            return self._score_decode(md, sched, index_q, weights)

        width_decode = max(
            (p.nscore for p in sched.requests if p.row < num_decodes), default=0
        )
        width_prefill = max(
            (p.nscore for p in sched.requests if p.row >= num_decodes), default=0
        )

        decode_logits = None
        if num_decodes > 0:
            decode_logits = torch.zeros(
                (groups * num_decodes, max(width_decode, 1)),
                dtype=torch.float32,
                device=device,
            )
        prefill_logits = None
        if md.num_prefills > 0:
            prefill_logits = torch.zeros(
                (groups * num_prefill_tokens, max(width_prefill, 1)),
                dtype=torch.float32,
                device=device,
            )
        if DENSE_EQUIV:
            # All-zero scores plus topk >= every row's region count selects
            # every visible region, which makes DSA exactly full causal
            # attention. Any output difference against the dense backend is then
            # a paging/metadata bug and nothing else.
            return prefill_logits, decode_logits, max(width_decode, width_prefill)

        for plan in sched.requests:
            if plan.nscore == 0:
                continue
            keys = summary[plan.score_slots].to(index_q.dtype)
            decode = plan.row < num_decodes
            out = decode_logits if decode else prefill_logits
            stride = num_decodes if decode else num_prefill_tokens
            base = plan.row if decode else plan.tok_start - num_decode_tokens

            for begin in range(0, plan.qlen, INDEXER_Q_CHUNK or plan.qlen):
                count = min(INDEXER_Q_CHUNK or plan.qlen, plan.qlen - begin)
                first = plan.tok_start + begin
                indexer_logits(
                    index_q[first : first + count].contiguous(),
                    weights[first : first + count].contiguous(),
                    keys,
                    out=out,
                    out_row_base=base + begin,
                    out_group_stride=stride,
                    # The pool is e4m3, so the key-side rounding is the identity --
                    # the same argument ``decode_indexer_logits`` already makes.
                    quantize_keys=False,
                    q_prerounded=True,
                )

        return prefill_logits, decode_logits, max(width_decode, width_prefill)

    def _score_decode(
        self,
        md: Step4DSAMetadata,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[None, torch.Tensor, int]:
        """One scoring launch for the whole decode batch instead of one per request.

        The general path must call ``indexer_logits`` per request because each
        one scores against a different key set and that kernel takes a single
        shared key head. Here the request axis moves into the launch grid and
        the kernel reads the summary pool in place through the block table, so
        nothing is gathered.

        Under ``graph_regions`` the score tensor is a slice of a pool allocated
        once, at a width that covers ``max_model_len`` rather than this batch's
        longest context, because a captured launch cannot resize. The slice is
        deliberately not cleared: a row is written for its own ``nscore``
        columns, the selector reads exactly ``(seq_len - 1) // region_size`` of
        them, and those are the same number -- so nothing this step did not
        write is ever read, and clearing 64K columns per row per layer would
        cost more than the scoring does.
        """
        state = self.state
        num_decodes = sched.num_decodes
        rows = self.num_kv_groups * num_decodes

        if sched.graph_regions:
            width = sched.graph_regions
            decode_logits = state._decode_logits[:rows, :width]
        else:
            width = sched.max_nscore
            decode_logits = torch.zeros(
                (rows, max(width, 1)),
                dtype=torch.float32,
                device=index_q.device,
            )
        if DENSE_EQUIV or width == 0:
            return None, decode_logits, width

        decode_indexer_logits(
            index_q[:num_decodes].contiguous(),
            weights[:num_decodes].contiguous(),
            state.summary[self.slot],
            md.block_table[:num_decodes],
            sched.nscore,
            decode_logits,
            regions_per_page=state.regions_per_page,
            q_prerounded=True,
        )
        return None, decode_logits, width

    def _score_spec(
        self,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[None, torch.Tensor, int]:
        """``_score_decode`` with one row per query token.

        Each token scores against its own request's summaries through the
        per-token block-table row, bounded by its own ``nscore`` -- which is
        why a region the step's earlier tokens just closed is visible to the
        later ones, exactly as on the prefill path.
        """
        state = self.state
        ndt = sched.num_tokens
        rows = self.num_kv_groups * ndt

        if sched.graph_regions:
            width = sched.graph_regions
            decode_logits = state._decode_logits[:rows, :width]
        else:
            width = sched.max_nscore
            decode_logits = torch.zeros(
                (rows, max(width, 1)),
                dtype=torch.float32,
                device=index_q.device,
            )
        if width == 0:
            return None, decode_logits, width

        if (
            _SPEC_INDEXER_PER_REQUEST
            and state.spec_query_len <= SPEC_INDEXER_MAX_QUERY
            and sched.req_table is not None
        ):
            spec_indexer_logits(
                index_q[:ndt].contiguous(),
                # Row-strided under the merged projection; the kernels take strides.
                weights[:ndt],
                state.summary[self.slot],
                sched.req_table,
                sched.query_start_loc,
                sched.tok_nscore,
                decode_logits,
                num_reqs=sched.num_decodes,
                regions_per_page=state.regions_per_page,
                weight_scale=self.weight_prescale,
                q_prerounded=True,
                max_query=state.spec_query_len,
            )
            return None, decode_logits, width

        decode_indexer_logits(
            index_q[:ndt].contiguous(),
            self._prescaled(weights[:ndt]),
            state.summary[self.slot],
            sched.tok_table,
            sched.tok_nscore,
            decode_logits,
            regions_per_page=state.regions_per_page,
            q_prerounded=True,
        )
        return None, decode_logits, width


# ---------------------------------------------------------------------------
# The indexer as an opaque op
# ---------------------------------------------------------------------------
#
# Calling Step4Indexer.run() directly from Step4Attention.forward does not
# survive torch.compile. The model is traced once, during the profiling run,
# where attn_metadata is None -- so Dynamo takes run()'s early-return branch,
# bakes "do nothing" into the graph, and every real step afterwards reaches the
# attention layer with no plan at all. Even without that, the body is untraceable
# in principle: it branches on CPU sequence lengths and allocates tensors whose
# shapes depend on them.
#
# Wrapping it in a custom op fixes both. The body is never traced, it runs
# eagerly at the point the op appears in the graph, and the Python attribute it
# writes (layer.step4_sparse_plan) is set at execution time rather than trace
# time. This is exactly how vLLM's own DeepSeek indexer is wired
# (model_executor/layers/sparse_attn_indexer.py:395).
#
# plan_token is what keeps the op alive. An op with no outputs and no declared
# mutations is dead code to the functionaliser; declaring the write makes the
# buffer a graph input that has to be updated, so the call is kept and stays
# ordered against the attention op that consumes its plan. The value written is
# the real one -- the number of scored regions this step -- so the buffer is
# also readable when debugging a layer that selected nothing.
#
# Being a custom op is necessary but NOT sufficient: the op must also be listed
# in CompilationConfig.splitting_ops. vLLM compiles piecewise and captures each
# piece between splitting ops into a CUDA graph, and a replayed graph re-runs
# recorded kernels, not Python -- so an unsplit indexer runs once, at capture
# time, and every subsequent step silently reuses that plan. vLLM's in-tree
# indexer is in the default list ("vllm::sparse_attn_indexer",
# config/compilation.py:750); this one is out of tree, so Step4Config.__init__
# appends it to CompilationConfig._attention_ops (see
# transformers_utils/configs/step4.py for why the registration lives there and
# not here). _require_splitting_op below is what turns a bypassed registration
# into a startup error instead of a wrong answer at the first request.


def step4_dsa_index(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    plan_token: torch.Tensor,
) -> None:
    indexer = _INDEXERS.get(layer_name)
    if indexer is None:
        raise RuntimeError(
            f"{layer_name}: no Step4Indexer registered. The op is dispatched by "
            f"layer name, so the indexer must be constructed before the first "
            f"forward pass."
        )
    plan_token.fill_(indexer.run(hidden_states, positions))


def step4_dsa_index_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    plan_token: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="step4_dsa_index",
    op_func=step4_dsa_index,
    mutates_args=["plan_token"],
    fake_impl=step4_dsa_index_fake,
    dispatch_key=current_platform.dispatch_key,
)
