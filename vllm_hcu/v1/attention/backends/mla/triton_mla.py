# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import os
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.logger import logger as _vllogger  # configured; reaches the log file
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
)

logger = init_logger(__name__)

_DSPARK_FLASH_ENV = "VLLM_HCU_K3_DSPARK_FLASH_MLA"
_FP8_E4M3 = torch.float8_e4m3fn
# The vendor fp8 kernel consumes 64-token pages (its kBlockN tile). Hybrid
# page-size alignment may enlarge the *manager* block to a whole multiple of
# that tile (Kimi-K3: 1536 = 24x64); such blocks are accepted via an expanded
# block table, anything else must stay on the stock Triton path.
_FLASH_PAGE_BLOCK = 64


def _is_flash_kv_dtype(kv_cache_dtype: str) -> bool:
    """True for KV dtypes whose byte layout the vendor fp8 kernel accepts.

    The kernel consumes e4m3-encoded bytes only, so e5m2 and nvfp4 keep
    the stock Triton path. The platform's internal remap ("fp8_ds_mla")
    is byte-compatible e4m3 storage and stays on the fast path.
    """
    return (
        kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_e5m2"
        and not kv_cache_dtype.endswith("per_token_head")
        and kv_cache_dtype != "nvfp4"
    )


# Debug-level knob for the non-causal fast path (TP8 fault hunt,
# 2026-10-27). Semantics, highest first; a fault at level L with a pass at
# L-1 localizes the trigger to the feature L adds:
#   4 full fast path: buffers + metadata calls + launches (OPT-IN ONLY)
#   3           level 2 + single-split eager launches (fix10 guard)
#   2           level 1 + ALL vendor metadata calls (builder per-step
#               get_mla_metadata_dense_fp8, eager rebuild), zero kernel
#               launches from the fast path
#   1           persistent buffers allocated, zero vendor ops, zero
#               launches -- pure VA-layout probe
#   0           fast path fully inert (allocations included)
# DEFAULT IS 0 (inert): on TP8 every gate-on startup this day crashed
# before serving one request. Re-enable with LEVEL=4 or the legacy
# VLLM_HCU_K3_DSPARK_FLASH_MLA=1 once the root cause is fixed.
_DFLASH_LEVEL_ENV = "VLLM_HCU_K3_DSPARK_FLASH_LEVEL"

# FIX13 (TP8 root cause): process-lifetime storage for flash fast-path
# metadata buffers. The engine rebuilds builders after the capture epoch
# (KV-cache sizing step); each rebuild must re-bind the SAME tensors, not
# allocate fresh ones a captured graph would never see. Keyed by
# (device, layer-names group, page, dims); values survive for the whole
# process -- a few MB per rank, deliberately immortal.
_FLASH_BUF_REGISTRY: dict = {}


def _dflash_level() -> int:
    raw = os.environ.get(_DFLASH_LEVEL_ENV, "").strip()
    if raw:
        try:
            return max(0, min(4, int(raw)))
        except ValueError:
            pass
    # Default INERT until the TP8 fault root cause is fixed: every gate-on
    # restart this day (160236-174757) died before serving one request, so
    # silence must mean safe, and enabling the fast path is always an
    # explicit act (LEVEL=4, or the legacy gate set to a truthy value).
    gate = os.environ.get(_DSPARK_FLASH_ENV)
    if gate is None:
        return 0
    return 0 if gate in ("0", "false", "False") else 4


def _flash_gate_enabled() -> bool:
    # Kill-switch for the non-causal multi-row decode fast path.
    return _dflash_level() > 0


def _dense_flash_ok() -> bool:
    try:
        from vllm_hcu.v1.attention.ops.flashmla import (
            is_flashmla_dense_supported,
        )

        return bool(is_flashmla_dense_supported()[0])
    except Exception:
        return False


class HcuTritonMLAMetadataBuilder(TritonMLAMetadataBuilder):
    """Feeds the FlashMLA fast path from persistent, capture-stable buffers.

    The fast-path kernel consumes tile-scheduler metadata computed per step.
    Under full cudagraph capture those tensors must live at stable addresses
    and be refilled eagerly, so the builder allocates every buffer once in
    ``__init__`` at the service's upper bounds and each step only writes into
    fixed-size views of them. Runtime overflow raises instead of reallocating:
    a ``data_ptr`` change after capture would leave graphs reading stale
    addresses (the TP8 root cause of 2026-10-27: vLLM rebuilt this builder
    after the capture epoch -- KV-cache sizing step -- which freed the
    epoch-1 buffers that captured DSpark graphs still referenced; the
    replayed flash node then read page ids from memory recycled into the
    KV pool). Quantities derived from Python ints (query_len, head count)
    are stashed alongside so the impl can reject a stale match.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cache_dtype = vllm_config.cache_config.cache_dtype
        page = getattr(kv_cache_spec, "block_size", None)
        group_non_causal = getattr(
            self, "non_causal_multi_token_decode", False
        )
        subpage = (
            page // _FLASH_PAGE_BLOCK
            if isinstance(page, int) and page > 0
            and page % _FLASH_PAGE_BLOCK == 0
            else None
        )
        self._hcu_subpage = subpage
        self._hcu_flash_ready = (
            group_non_causal
            and _flash_gate_enabled()
            and _dense_flash_ok()
            and _is_flash_kv_dtype(cache_dtype)
            and subpage is not None
        )
        _vllogger.debug(
            "HcuTritonMLA: non-causal group %s flash_ready=%s "
            "(spec=%s, page=%s, subpage=%s, dtype=%s, layers=%s)",
            id(self) % 10000, self._hcu_flash_ready,
            type(kv_cache_spec).__name__, page, subpage, cache_dtype,
            list(layer_names),
        )
        self._hcu_sched_buf = None
        self._hcu_nsplits_buf = None
        self._hcu_bt_buf = None
        self._hcu_bt_sub_arange = None
        if self._hcu_flash_ready:
            num_sms = num_compute_units(device)
            # HCU dense-FP8 tile metadata can exceed one row per CU (the
            # dense fp8 builder documents the same: rows grow with
            # num_q_tokens_per_head_k, e.g. 128 for 7x16 query rows on a
            # 64-CU part). Allocate 4x the SM count as a fixed upper
            # bound -- unlike the dense path we never grow at runtime,
            # because a reallocated buffer would dangle inside graphs
            # already captured around the flash kernel. Overflow still
            # raises into the graceful stock-path fallback.
            max_seqs = vllm_config.scheduler_config.max_num_seqs + 1
            # Block-table expansion buffer at the service's upper bounds:
            # [max rows, max manager blocks x subpage]. Never reallocated.
            max_cols = (
                -(-vllm_config.model_config.max_model_len // page) + 1
            )
            self._hcu_bt_cols = max_cols
            self._hcu_bt_width = max_cols * subpage
            # FIX13 (TP8 root cause): builder rebuilds on every engine
            # re-init (KV-cache sizing reconstructs workers), and each
            # rebuild used to allocate NEW buffers, freeing the old ones
            # that captured graphs still reference (use-after-free at the
            # next FULL-graph replay). Register the buffers in a
            # process-lifetime registry keyed by (device, group): every
            # rebuild of the same group Binds the SAME tensors, so a
            # captured graph's pointers stay valid for the whole service
            # lifetime. The registry holds strong references -- these
            # few MB are deliberately immortal.
            reg_key = (
                str(device), tuple(sorted(layer_names)), page,
                num_sms * 4, max_seqs, self._hcu_bt_width,
            )
            reg = _FLASH_BUF_REGISTRY.get(reg_key)
            if reg is None:
                reg = {
                    "sched": torch.zeros(
                        (num_sms * 4, 8), dtype=torch.int32,
                        device=device,
                    ),
                    "nsplits": torch.empty(
                        max_seqs, dtype=torch.int32, device=device
                    ),
                    "bt": torch.zeros(
                        (max_seqs, self._hcu_bt_width),
                        dtype=torch.int32, device=device,
                    ),
                    "arange": torch.arange(
                        subpage, dtype=torch.int32, device=device
                    ).view(1, 1, subpage),
                    "gen": 0,
                }
                _FLASH_BUF_REGISTRY[reg_key] = reg
            reg["gen"] += 1
            self._hcu_sched_buf = reg["sched"]
            self._hcu_nsplits_buf = reg["nsplits"]
            self._hcu_bt_buf = reg["bt"]
            self._hcu_bt_sub_arange = reg["arange"]
            self._hcu_buf_gen = reg["gen"]
            self._hcu_buf_reused = reg["gen"] > 1
            # Expanded-unit page count of this group's own pool slice.
            # vLLM pads table columns beyond a request's allocation with
            # out-of-pool null-block ids; every id this builder hands to
            # the flash kernel is clipped into [0, _hcu_pool_pages) so
            # even a speculative over-read lands on a real page.
            self._hcu_pool_pages = (
                int(getattr(kv_cache_spec, "num_blocks", 0) or 0) * subpage
            )
            # Faulthunt VA ledger (2026-10-27): one line per rank mapping
            # every persistent tensor address+extent, so a fault VA can be
            # placed against (or between) tensor bounds without a rerun.
            # fix13 adds reuse= : the acceptance check for the TP8 fix is
            # now "16 unique address sets, each logged twice with
            # reuse=yes" (one line per draft group per rebuild wave).
            _vllogger.info(
                "HcuTritonMLA VA-LEDGER dev=%s level=%d reuse=%s gen=%d "
                "sched=[0x%x +64 rows x8] nsplits=[0x%x +%d] "
                "bt=[0x%x +%d B (%dx%d i32)] pool_pages=%d",
                device, _dflash_level(),
                "yes" if self._hcu_buf_reused else "no",
                self._hcu_buf_gen,
                self._hcu_sched_buf.data_ptr(),
                self._hcu_nsplits_buf.data_ptr(),
                max_seqs,
                self._hcu_bt_buf.data_ptr(),
                self._hcu_bt_buf.numel() * 4,
                max_seqs, self._hcu_bt_width,
                self._hcu_pool_pages,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ):
        decode_md = super()._build_decode(
            block_table_tensor,
            seq_lens_device,
            max_seq_len,
            query_start_loc_cpu,
            query_start_loc_device,
            num_decode_tokens,
            dcp_tot_seq_lens_device,
        )
        if self._hcu_flash_ready and _dflash_level() >= 2:
            # Level 4/3 launch the kernel; level 2 exercises the vendor
            # metadata side effect only (faulthunt); the fill below raises
            # on multi-split demand, so level 2 still stops short of any
            # metadata output being consumed by a captured graph.
            try:
                self._fill_flash_sched(
                    decode_md, seq_lens_device, block_table_tensor,
                    max_seq_len, num_decode_tokens,
                )
            except Exception as exc:
                # Refresh failures must not disable the builder: graphs
                # already captured around the flash kernel keep replaying
                # from these buffers, and a stopped builder would leave
                # them reading stale page tables. New shapes re-derive
                # their scheduler only on the next successful fill.
                # warning_once takes no exc_info kwarg in this vLLM, so
                # the cause is inlined instead; str(exc) (not the exc
                # object) keeps the once-dedupe key hashable by value --
                # exception instances would log a line per step and pin
                # their tracebacks in the logger's lru_cache.
                logger.warning_once(
                    "HcuTritonMLA: FlashMLA scheduler precompute failed "
                    "this step (%s: %s); later steps retry the flash "
                    "path and keep the stock Triton path meanwhile.",
                    type(exc).__name__, str(exc),
                )
        return decode_md

    def _fill_flash_sched(
        self, decode_md, seq_lens_device: torch.Tensor,
        block_table_tensor: torch.Tensor, max_seq_len: int,
        num_decode_tokens: int,
    ) -> None:
        from vllm_hcu.v1.attention.ops.flashmla import (
            get_mla_metadata_dense_fp8,
        )

        q_heads = self.num_heads * self.dcp_world_size
        # Derive the per-step query shape from the _build_decode inputs,
        # never from metadata attributes: MLACommonDecodeMetadata carries
        # no num_decodes (it lives on the outer metadata), so any read of
        # the class surface breaks exactly under cudagraph capture. The
        # decode seq_lens row count with the num_decode_tokens argument
        # are the numbers the stock path itself feeds the kernel.
        num_decodes = int(seq_lens_device.shape[0])
        if num_decodes <= 0 or num_decode_tokens % num_decodes != 0:
            raise RuntimeError(
                f"hcu scheduler shape mismatch: {num_decode_tokens} "
                f"tokens over {num_decodes} decode rows"
            )
        query_len = num_decode_tokens // num_decodes
        subpage = self._hcu_subpage

        # Manager-block -> kernel-page block_table expansion into the
        # persistent buffer. Fixed full row width keeps rows contiguous, so
        # replayed graphs bind one stable data_ptr forever.
        #
        # Failure ordering: every check that can raise happens BEFORE any
        # buffer write. A fill that aborted mid-write would leave a full
        # graph replaying against a half-refreshed buffer (block ids of
        # one step under the tile schedule of another), so the write
        # section below is no-throw by construction.
        rows, cols = block_table_tensor.shape
        if rows > self._hcu_bt_buf.shape[0]:
            raise RuntimeError(
                f"hcu block table rows {rows} exceed flash buffer "
                f"{self._hcu_bt_buf.shape[0]}"
            )
        # Walk window: the kernel reads, per request, ceil(seq/64) leading
        # 64-token pages; this step's longest request bounds the window on
        # the CPU side (no device sync). rows may be padded beyond
        # num_decodes (capture), but rows themselves were checked above.
        demand_cols = (
            -(-max_seq_len // _FLASH_PAGE_BLOCK) + subpage - 1
        ) // subpage
        if demand_cols > 1:
            # Split-KV engagement guard (TP8 VMFault, runs 160236/163011/
            # 171312): every observed fault wave had the fast-path kernel
            # in a multi-split launch (real long-prefix decode, or the
            # profile-length FULL-capture dummy), while every green
            # launch -- DSpark capture dummies on both nodes, TP4 smoke,
            # 12 startup passes -- was single-split (<= one manager
            # block). The in-pool id clamp removed the sentinel writer,
            # yet the fault persisted, so a second over-read lives in the
            # vendor split/combine path's assumptions about our persistent
            # scheduler views. Until that is pinned, refuse multi-split
            # fills here on the CPU side (no device sync, capture-safe):
            # the step degrades to the stock kernel via the missing
            # hcu_flash_sched metadata, never to a partial buffer.
            raise RuntimeError(
                f"hcu flash fast path refused: multi-split window "
                f"(pages {demand_cols} > 1) not yet proven safe"
            )
        if demand_cols > self._hcu_bt_cols:
            raise RuntimeError(
                f"hcu walk window {demand_cols} exceeds flash buffer "
                f"width {self._hcu_bt_cols}"
            )
        if demand_cols > cols:
            raise RuntimeError(
                f"hcu walk window {demand_cols} exceeds table width {cols}"
            )

        tile, nsplits = get_mla_metadata_dense_fp8(
            seq_lens_device, query_len * q_heads, 1  # MQA shared latent
        )
        n = tile.shape[0]
        if n > self._hcu_sched_buf.shape[0]:
            raise RuntimeError(
                f"hcu scheduler rows {n} exceed buffer "
                f"{self._hcu_sched_buf.shape[0]}"
            )
        m = nsplits.shape[0]
        if m > self._hcu_nsplits_buf.shape[0]:
            raise RuntimeError(
                f"hcu nsplits rows {m} exceed buffer "
                f"{self._hcu_nsplits_buf.shape[0]}"
            )

        # Write section (no-throw): only the walked window is expanded.
        # The window is a superset of every row's own walk for a
        # uniform-length decode group, but the live table's cells past a
        # SHORT row's allocation still carry vLLM's out-of-pool null ids
        # (6551 seen against this group's 3233-block pool; the stock
        # kernels tolerate them only because each row's walk is capped by
        # its own seq len). Clip every written id into the group pool so
        # any over-read -- our kernel does read the full per-request page
        # count the schedule hands it -- lands on a real page instead of
        # an unmapped one; legal ids pass the clamp unchanged. Cells past
        # the window keep bootstrap zeros (page 0, in-pool).
        walk_cols = demand_cols
        bt3 = self._hcu_bt_buf[:rows].view(rows, self._hcu_bt_cols, subpage)
        bt3[:, :walk_cols].copy_(
            block_table_tensor[:, :walk_cols].unsqueeze(-1) * subpage
        )
        bt3[:, :walk_cols].add_(self._hcu_bt_sub_arange)
        if self._hcu_pool_pages > 0:
            bt3[:, :walk_cols].clamp_(
                max=self._hcu_pool_pages - 1
            )
        # Views of the persistent buffers keep their data_ptr across steps;
        # the captured kernel binds these addresses at capture time.
        tile_view = self._hcu_sched_buf[:n]
        splits_view = self._hcu_nsplits_buf[:m]
        tile_view.copy_(tile)
        splits_view.copy_(nsplits)
        decode_md.hcu_flash_sched = (
            (tile_view, splits_view),
            self._hcu_bt_buf,
            query_len * q_heads,
            q_heads,
            subpage,
        )


class HcuTritonMLAImpl(TritonMLAImpl):
    """TritonMLA impl with a FlashMLA fast path for non-causal draft decode.

    The stock non-causal path (DSpark draft proposal blocks) flattens every
    query row of a block into an independent decode row that re-scans the
    whole committed KV prefix: query_len rows cast query_len full-prefix
    passes (measured on the Kimi-K3 DSpark draft: 5 layers x 7 rows,
    ~1.55 TB/s effective, 41-52 ms per decode iteration at ~100k context --
    7x the unavoidable KV bytes). The FlashMLA vendor kernel takes the same
    multi-token query per request and reads the prefix once for all rows;
    ``causal=False`` with the committed-prefix ``cache_seqlens`` reproduces
    the stock non-causal semantics exactly (every row attends [0, prefix),
    never the sibling block tokens).
    """

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not attn_metadata.causal
            and _flash_gate_enabled()
            and _dflash_level() >= 3
            # FIX13 removes the eager-only exclusion: the TP8 fault was a
            # builder rebuild after capture (use-after-free of the
            # epoch-1 persistent buffers inside replayed draft graphs),
            # not a capture-context hazard. The flash node belongs in the
            # DSpark FULL graph; the registry keeps its buffers alive.
            and _is_flash_kv_dtype(self.kv_cache_dtype)
            # The vendor kernel walks 64-token pages. A manager block of
            # any whole multiple works through an expanded block table;
            # anything else is a device-side abort, so gate before launch.
            and kv_c_and_k_pe_cache.ndim == 3
            and kv_c_and_k_pe_cache.shape[1] > 0
            and kv_c_and_k_pe_cache.shape[1] % _FLASH_PAGE_BLOCK == 0
            and attn_metadata.decode is not None
        ):
            num_decodes = attn_metadata.num_decodes
            query_len = attn_metadata.num_decode_tokens // num_decodes
            if query_len > 1:
                flash_out = self._forward_noncausal_flash(
                    q, kv_c_and_k_pe_cache, attn_metadata, layer,
                    num_decodes, query_len,
                )
                if flash_out is not None:
                    return flash_out
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

    def _forward_noncausal_flash(
        self, q, kv_c_and_k_pe_cache, attn_metadata, layer,
        num_decodes: int, query_len: int,
    ):
        try:
            return self._noncausal_flash_inner(
                q, kv_c_and_k_pe_cache, attn_metadata, layer,
                num_decodes, query_len,
            )
        except Exception as exc:
            logger.warning_once(
                "HcuTritonMLA: non-causal FlashMLA fast path suspended "
                "(%s: %s); falling back to the stock Triton path.",
                type(exc).__name__, str(exc),
            )
            return None

    def _noncausal_flash_inner(
        self, q, kv_c_and_k_pe_cache, attn_metadata, layer,
        num_decodes: int, query_len: int,
    ):
        from vllm_hcu.v1.attention.ops.flashmla import (
            flash_mla_with_kvcache_fp8,
            get_mla_metadata_dense_fp8,
        )

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        decode = attn_metadata.decode
        seq_lens = decode.seq_lens
        # [B*L, H, D] -> [B, L, H, D]; matches the stock flatten's
        # request-major/token-minor row order.
        q_spec = reshape_query_for_spec_decode(q, num_decodes)
        q_heads = q_spec.shape[2]  # includes the DCP all-gather
        num_q_tokens_per_head_k = query_len * q_heads
        capturing = torch.cuda.is_current_stream_capturing()

        sched = getattr(decode, "hcu_flash_sched", None)
        tile = bt_expanded = None
        nsplits = None
        subpage = kv_c_and_k_pe_cache.shape[1] // _FLASH_PAGE_BLOCK
        if sched is not None:
            # The builder is the single source of truth: it refills the
            # persistent tile/split buffers from the live inputs of every
            # step (the same numbers the dispatched graph's shape
            # dictates), and the sched rides this group's own decode
            # metadata, so no stamped-shape or layout re-verification is
            # wanted here. Every check we tried (query shape, head count,
            # subpage) misfired exactly under synthetic capture inputs
            # while being provably constant at service lifetime elsewhere.
            (tile, nsplits), bt_expanded = sched[0], sched[1]
        if tile is None:
            if capturing:
                # Never allocate/bind fresh tensors inside capture: the
                # builder precomputes exactly to avoid this branch.
                logger.warning_once(
                    "HcuTritonMLA: capture-time fast path has no builder "
                    "scheduler; this capture replays the stock Triton path."
                )
                return None
            # faulthunt levels: level 2 = metadata side effect only, and it
            # must not reach a launch; level >=3 pays the .item() sync for
            # the single-split gate too, before any vendor call lands.
            level = _dflash_level()
            pages_needed = -(
                -int(seq_lens[:num_decodes].max().item())
                // _FLASH_PAGE_BLOCK
            )
            if level < 3 and pages_needed > subpage:
                logger.warning_once(
                    "HcuTritonMLA: non-causal flash rebuild skipped "
                    "(level=%d, pages %d > %d).",
                    level, pages_needed, subpage,
                )
                return None
            tile, nsplits = get_mla_metadata_dense_fp8(
                seq_lens, num_q_tokens_per_head_k, 1  # MQA shared latent
            )
            # Eager-only rebuild: expand the stock table onto fresh memory.
            # Expand ONLY the walked-page budget of the longest request and
            # zero-init the rest: vLLM pads table columns beyond the
            # fill with out-of-pool sentinel ids (6551 seen vs a 3233-
            # block pool); a full-width expansion would hand those to
            # the kernel, and split-KV stragglers do read them.
            bt_stock = decode.block_table[:num_decodes]
            pages_needed = -(
                -int(seq_lens[:num_decodes].max().item())
                // _FLASH_PAGE_BLOCK
            )
            if pages_needed > subpage:
                # Guard order note (faulthunt): the level<3 case with a
                # multi-split demand bailed above BEFORE the vendor call;
                # this second check only ever sees level>=3 (or a
                # single-split demand that passed the first), so the
                # multi-split launch guard keeps its fix10 meaning.
                return None
            src_cols = min(
                -(-pages_needed // subpage),
                bt_stock.shape[1],
            )
            bt_expanded = torch.zeros(
                (bt_stock.shape[0], src_cols * subpage),
                dtype=bt_stock.dtype, device=bt_stock.device,
            )
            bt3v = bt_expanded.view(
                bt_stock.shape[0], src_cols, subpage
            )
            bt3v.copy_(bt_stock[:, :src_cols].unsqueeze(-1) * subpage)
            bt3v.add_(
                torch.arange(
                    subpage, dtype=bt_stock.dtype, device=bt_stock.device
                )
            )
            # Same pool clip as the builder path: padded table cells past
            # a short request's allocation carry out-of-pool null ids.
            pool_cap = (
                kv_c_and_k_pe_cache.shape[0] * subpage - 1
            )
            bt3v.clamp_(max=pool_cap)

        # Kernel contract: [pages, page_tokens, 1, D] (kv head axis at -2)
        # over 64-token kernel pages. A stored-view cache reshapes without
        # a copy via .view; a padded/non-contiguous one raises and falls
        # back instead of silently cloning gigabytes inside a graph.
        blocks = kv_c_and_k_pe_cache.shape[0]
        k_cache = kv_c_and_k_pe_cache.view(
            blocks * subpage, _FLASH_PAGE_BLOCK, kv_c_and_k_pe_cache.shape[2]
        ).unsqueeze(-2)
        if k_cache.dtype != _FP8_E4M3:
            k_cache = k_cache.view(_FP8_E4M3)
        # gfx938 fp8 kernel consumes the pre-quantized query (same contract
        # as the plugin's dense fp8 path): static per-tensor QuantFP8 with
        # the layer's q scale, descale fused via descale_q.
        q_fp8 = (
            (q_spec.to(torch.float32) / layer._q_scale.to(torch.float32))
            .clamp(-448.0, 448.0)
            .to(_FP8_E4M3)
        )
        o, lse = flash_mla_with_kvcache_fp8(
            q=q_fp8,
            k_cache=k_cache,
            block_table=bt_expanded[:num_decodes],
            cache_seqlens=seq_lens[:num_decodes],
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=tile,
            num_splits=nsplits,
            softmax_scale=self.scale,
            causal=False,
            descale_q=layer._q_scale.reshape(1),
            descale_k=layer._k_scale.reshape(1),
            kv_cache_dtype="fp8_e4m3",
        )
        _vllogger.info_once(
            "HcuTritonMLA: DSpark non-causal decode routed through "
            "FlashMLA (query_len=%d, heads=%d).",
            query_len, q_heads,
        )
        o = reshape_attn_output_for_spec_decode(o)
        # Stock returns lse as [rows, H] in q-dtype; the dense kernel
        # returns fp32 [B, H, L] -> req-major/token-minor rows.
        lse = lse.permute(0, 2, 1).reshape(-1, q_heads).to(q.dtype)
        return o, lse


class HcuTritonMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_impl_cls() -> type["HcuTritonMLAImpl"]:
        return HcuTritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["HcuTritonMLAMetadataBuilder"]:
        return HcuTritonMLAMetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"
