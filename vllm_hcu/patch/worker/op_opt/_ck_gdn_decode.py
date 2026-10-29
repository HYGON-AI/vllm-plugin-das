# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Non-speculative Qwen GDN decode through ``aiter.ck_gdn_decode``.

CK has no state-pool index argument, so the recurrent states of the
scheduled slots are gathered into a dense ``[N, HV, V, K]`` fp32 tensor, run
through CK (``transpose_state=True`` matches the pool's ``[V, K]`` layout),
and scattered back.  vLLM treats slot ids <= 0 as padding; those rows read
and rewrite slot 0 unchanged, so the pool is never polluted.  Every
operation is device-side (no boolean indexing, no host sync).
"""

from __future__ import annotations

import torch

from ._aiter_ck_gdn import resolve_ck_gdn_decode


def ck_decode_step(
    *, q, k, v, a, b, A_log, dt_bias, scale, state_pool, state_indices,
    use_qk_l2norm, num_seqs,
):
    """One token per sequence.

    q/k ``[N, H, K]``, v ``[N, HV, V]``, a/b ``[N, HV]``; returns the output
    ``[N, HV, V]``.  ``state_pool`` is updated in place.
    """

    kernel = resolve_ck_gdn_decode()
    if kernel is None:
        raise RuntimeError(
            "GDN decode hcu-aiter-ck was requested but aiter CK GDN "
            "(aiter.ops.ck_gdn.ck_gdn_decode) is not installed"
        )
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        raise RuntimeError(
            "GDN decode hcu-aiter-ck supports only K=V=128, got "
            f"K={q.shape[-1]} V={v.shape[-1]}"
        )
    device = q.device
    idx = state_indices[:num_seqs].to(torch.int64)
    safe = torch.where(idx > 0, idx, torch.zeros_like(idx))
    state = state_pool.index_select(0, safe).to(torch.float32).contiguous()
    cu = torch.arange(num_seqs + 1, dtype=torch.int64, device=device)
    out, new_state = kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        torch.sigmoid(b.float()).contiguous(),
        a_log=A_log.float().contiguous(),
        dt_bias=dt_bias.contiguous(),
        initial_state=state,
        cu_seqlens=cu,
        scale=float(scale),
        use_g=True,
        gate_in_kernel=True,
        has_dt_bias=True,
        transpose_state=True,
        use_qk_l2norm=use_qk_l2norm,
        store_final_state=True,
        validate_varlen=False,
    )
    # Padding rows (slot <= 0) keep whatever slot 0 held.
    keep = (idx > 0).view(-1, 1, 1, 1)
    new_state = torch.where(keep, new_state, state)
    state_pool.index_copy_(0, safe, new_state.to(state_pool.dtype))
    return out
