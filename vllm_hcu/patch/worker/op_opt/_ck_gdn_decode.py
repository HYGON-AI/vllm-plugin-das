# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Non-speculative Qwen GDN decode through ``aiter.ck_gdn_decode``.

The recurrent state pool ``[slots, HV, V, K]`` fp32 and the per-sequence slot
ids go straight to CK (``initial_state_indices``; ``transpose_state=True``
matches the pool's ``[V, K]`` layout).  The kernel reads and writes the pool in
place using its ``stride(0)`` as slot stride, so there is no gather/scatter.
vLLM treats slot ids <= 0 as padding; CK starts those rows from a zero state
and does not write them back, so the pool is never polluted.  Every operation
is device-side (no host sync).
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
    if state_pool.dtype != torch.float32:
        raise RuntimeError(
            "GDN decode hcu-aiter-ck needs an fp32 recurrent state pool, got "
            f"{state_pool.dtype}"
        )
    cu = torch.arange(num_seqs + 1, dtype=torch.int64, device=q.device)
    out, _ = kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        torch.sigmoid(b.float()).contiguous(),
        a_log=A_log.float().contiguous(),
        dt_bias=dt_bias.contiguous(),
        initial_state=state_pool,
        initial_state_indices=state_indices[:num_seqs],
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
    return out
