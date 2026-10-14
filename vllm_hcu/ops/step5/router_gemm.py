# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 router logits: bf16 x [M, K] @ bf16 W [K, N] -> fp32 [M, N].

The checkpoint router weight is bf16 and the activations entering the router are
bf16, so the bf16 MFMA products are exact in fp32 and the result matches the
fp32 GEMM (``x.float() @ W.float()``) up to accumulation order. Compared with
that path this skips the x upcast, reads half the weight bytes and uses the bf16
matrix cores.

Decode-sized M uses split-K; the last split of each output tile to finish sums
the partials in split order, so the result is deterministic and no memset or
separate reduce kernel is needed. The per-tile counters reset themselves.
"""

import torch

from vllm.triton_utils import tl, triton

# (max M, (BM, BN, BK, SPLIT_K, num_warps, num_stages)), tuned on gfx938 for
# K=4096, N=352. Only the last bucket may use SPLIT_K=1.
_CONFIGS = (
    (128, (32, 32, 64, 8, 2, 2)),
    (512, (64, 64, 128, 8, 4, 1)),
    (1024, (64, 128, 32, 4, 8, 2)),
    (2048, (128, 128, 32, 4, 4, 2)),
    (None, (128, 128, 32, 1, 4, 2)),
)

_workspace: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _config(M: int):
    for max_m, cfg in _CONFIGS:
        if max_m is None or M <= max_m:
            return max_m, cfg
    raise AssertionError


@triton.jit
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    sxm,
    swk,
    som,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    pid_n = pid // num_m
    pid_m = pid % num_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per = K // SPLIT_K
    rk = pid_k * k_per + tl.arange(0, BK)
    x_p = x_ptr + rm[:, None] * sxm + rk[None, :]
    w_p = w_ptr + rk[:, None] * swk + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, k_per, BK):
        a = tl.load(x_p, mask=rm[:, None] < M, other=0.0)
        b = tl.load(w_p, mask=rn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        x_p += BK
        w_p += BK * swk
    o_p = out_ptr + rm[:, None] * som + rn[None, :]
    o_mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(o_p, acc, mask=o_mask)
    else:
        tile = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        num_tiles = tl.num_programs(0)
        tl.store(ws_ptr + (pid_k * num_tiles + pid) * BM * BN + tile, acc)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(ws_ptr + (s * num_tiles + pid) * BM * BN + tile)
            tl.store(o_p, total, mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed", scope="gpu")


def init_router_gemm_workspace(device: torch.device, N: int) -> None:
    """Allocate the split-K partials and tile counters for ``device``.

    Must run before cudagraph capture (e.g. from the module constructor): the
    counters are zero-initialized once and then reset by the kernel itself.
    """
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    if device in _workspace:
        return
    ws_elems = tiles = 0
    for max_m, (BM, BN, _, split_k, _, _) in _CONFIGS:
        if split_k > 1:
            t = triton.cdiv(max_m, BM) * triton.cdiv(N, BN)
            tiles = max(tiles, t)
            ws_elems = max(ws_elems, split_k * t * BM * BN)
    _workspace[device] = (
        torch.empty(ws_elems, device=device, dtype=torch.float32),
        torch.zeros(tiles, device=device, dtype=torch.int32),
    )


def router_gemm_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight`` in fp32, with x [M, K] bf16 and weight [K, N] bf16."""
    M, K = x.shape
    N = weight.shape[1]
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    if M == 0:
        return out
    _, (BM, BN, BK, split_k, num_warps, num_stages) = _config(M)
    assert K % (split_k * BK) == 0
    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if split_k > 1:
        if x.device not in _workspace:
            assert not torch.cuda.is_current_stream_capturing(), (
                "init_router_gemm_workspace() must run before cudagraph capture"
            )
            init_router_gemm_workspace(x.device, N)
        ws, cnt = _workspace[x.device]
        assert cnt.numel() >= num_tiles
    else:
        ws = cnt = out
    _router_gemm_kernel[(num_tiles, split_k)](
        x,
        weight,
        out,
        ws,
        cnt,
        M,
        N,
        K,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BM=BM,
        BN=BN,
        BK=BK,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# Router + shared-expert gate_up + SwigluStep in one launch, for decode-sized M.
# (max M, (BM, BN, BK, SPLIT_K, num_warps, num_stages)), tuned on gfx938 for
# K=4096, router N=352, shared intermediate 384 per rank.
_RSG_CONFIGS = ((64, (32, 64, 64, 8, 4, 2)),)
RSG_MAX_M = _RSG_CONFIGS[-1][0]

_rsg_workspace: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _router_shared_kernel(
    x_ptr,
    wr_ptr,
    wgu_ptr,
    logit_ptr,
    act_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    K,
    sxm,
    swr,
    swgu,
    slm,
    sam,
    NR: tl.constexpr,
    I: tl.constexpr,
    LIMIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Tiles [0, R_TILES) of the N grid are router logits; the rest are shared
    # gate/up column pairs. Split-K partials as in _router_gemm_kernel, two
    # BM x BN slots per tile (gate and up).
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    R_TILES: tl.constexpr = (NR + BN - 1) // BN
    pid_n = pid // num_m
    pid_m = pid % num_m
    rm = pid_m * BM + tl.arange(0, BM)
    k_per = K // SPLIT_K
    rk = pid_k * k_per + tl.arange(0, BK)
    x_p = x_ptr + rm[:, None] * sxm + rk[None, :]
    num_tiles = tl.num_programs(0)
    tile = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    base = ws_ptr + (pid_k * num_tiles + pid) * 2 * BM * BN
    if pid_n < R_TILES:
        rn = pid_n * BN + tl.arange(0, BN)
        w_p = wr_ptr + rk[:, None] * swr + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, k_per, BK):
            a = tl.load(x_p, mask=rm[:, None] < M, other=0.0)
            b = tl.load(w_p, mask=rn[None, :] < NR, other=0.0)
            acc += tl.dot(a, b)
            x_p += BK
            w_p += BK * swr
        tl.store(base + tile, acc)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(ws_ptr + (s * num_tiles + pid) * 2 * BM * BN + tile)
            o_mask = (rm[:, None] < M) & (rn[None, :] < NR)
            tl.store(logit_ptr + rm[:, None] * slm + rn[None, :], total, mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed", scope="gpu")
    else:
        rn = (pid_n - R_TILES) * BN + tl.arange(0, BN)
        n_mask = rn[None, :] < I
        g_p = wgu_ptr + rk[:, None] * swgu + rn[None, :]
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, k_per, BK):
            a = tl.load(x_p, mask=rm[:, None] < M, other=0.0)
            g = tl.load(g_p, mask=n_mask, other=0.0)
            u = tl.load(g_p + I, mask=n_mask, other=0.0)
            acc_g += tl.dot(a, g)
            acc_u += tl.dot(a, u)
            x_p += BK
            g_p += BK * swgu
        tl.store(base + tile, acc_g)
        tl.store(base + BM * BN + tile, acc_u)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:
            tg = tl.zeros((BM, BN), dtype=tl.float32)
            tu = tl.zeros((BM, BN), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                b2 = ws_ptr + (s * num_tiles + pid) * 2 * BM * BN
                tg += tl.load(b2 + tile)
                tu += tl.load(b2 + BM * BN + tile)
            # Round to bf16 first: the unfused path materializes gate_up in bf16.
            gate = tg.to(tl.bfloat16).to(tl.float32)
            up = tu.to(tl.bfloat16).to(tl.float32)
            y = tl.minimum(tl.sigmoid(gate) * gate, LIMIT) * tl.minimum(
                tl.maximum(up, -LIMIT), LIMIT
            )
            o_mask = (rm[:, None] < M) & n_mask
            tl.store(
                act_ptr + rm[:, None] * sam + rn[None, :],
                y.to(tl.bfloat16),
                mask=o_mask,
            )
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed", scope="gpu")


def _rsg_config(M: int):
    for max_m, cfg in _RSG_CONFIGS:
        if M <= max_m:
            return cfg
    raise AssertionError(f"M={M} > RSG_MAX_M={RSG_MAX_M}")


def make_router_shared_workspace(
    device: torch.device, NR: int, I: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one router/shared split-K workspace.

    Model layers use separate workspaces because CUDA graphs for different
    layers or target/draft models may execute concurrently.  Sharing the tile
    counters across those graphs can make one launch reduce another launch's
    partials.
    """
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    ws_elems = tiles = 0
    for max_m, (BM, BN, _, split_k, _, _) in _RSG_CONFIGS:
        t = triton.cdiv(max_m, BM) * (triton.cdiv(NR, BN) + triton.cdiv(I, BN))
        tiles = max(tiles, t)
        ws_elems = max(ws_elems, split_k * t * 2 * BM * BN)
    return (
        torch.empty(ws_elems, device=device, dtype=torch.float32),
        torch.zeros(tiles, device=device, dtype=torch.int32),
    )


def init_router_shared_workspace(device: torch.device, NR: int, I: int) -> None:
    """Initialize the compatibility workspace for direct callers."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    if device not in _rsg_workspace:
        _rsg_workspace[device] = make_router_shared_workspace(device, NR, I)


def router_shared_swiglustep(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    limit: float,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(``x @ w_router`` in fp32, ``swiglustep(x @ w_gate_up)`` in bf16).

    x [M, K] bf16 with M <= RSG_MAX_M, w_router [K, NR] bf16, w_gate_up [K, 2I]
    bf16 (gate columns first). Matches router_gemm_bf16_fp32 plus the bf16
    gate_up GEMM and SwigluStepAndMul up to accumulation order.
    """
    M, K = x.shape
    NR = w_router.shape[1]
    I = w_gate_up.shape[1] // 2
    logits = torch.empty(M, NR, device=x.device, dtype=torch.float32)
    act = torch.empty(M, I, device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return logits, act
    BM, BN, BK, split_k, num_warps, num_stages = _rsg_config(M)
    assert K % (split_k * BK) == 0 and split_k > 1
    if workspace is None:
        if x.device not in _rsg_workspace:
            assert not torch.cuda.is_current_stream_capturing(), (
                "init_router_shared_workspace() must run before cudagraph capture"
            )
            init_router_shared_workspace(x.device, NR, I)
        workspace = _rsg_workspace[x.device]
    ws, cnt = workspace
    num_tiles = triton.cdiv(M, BM) * (triton.cdiv(NR, BN) + triton.cdiv(I, BN))
    assert cnt.numel() >= num_tiles
    assert ws.numel() >= split_k * num_tiles * 2 * BM * BN
    _router_shared_kernel[(num_tiles, split_k)](
        x,
        w_router,
        w_gate_up,
        logits,
        act,
        ws,
        cnt,
        M,
        K,
        x.stride(0),
        w_router.stride(0),
        w_gate_up.stride(0),
        logits.stride(0),
        act.stride(0),
        NR=NR,
        I=I,
        LIMIT=limit,
        BM=BM,
        BN=BN,
        BK=BK,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits, act
