# Step5 fused per-token FP8 quantization + DP IPC all-gather (opt-in)

## Scope

The Step5 decode MoE dispatch runs `token_fp8_quant` (one Triton kernel) immediately before the
DP IPC all-gather (another Triton kernel). With `VLLM_HCU_UP02_FUSED_DISPATCH=1`, the
quantization is folded into the all-gather prologue: each all-gather program owns whole activation
rows (`row % NPROG == pid`), applies the same per-token quantization math (same BK block, same
fp32 max/scale/clamp/cast), writes the fp8 row and fp32 scale to the local output and pushes the
same bytes into every peer's IPC slot. The topk weights/ids keep the production element-chunk
push/copy, so their bytes and layout are unchanged.

Default OFF. With the gate unset the runtime behavior is exactly the certified one. The deferral
is enabled only when a world-2 DP IPC quant all-gather exists; other layouts (including the
TP1/DP8 prefill contract) keep the certified local quant. Any capacity/shape mismatch falls back
to a communicator-local quant plus the standard gather (identical outputs).

## Correctness

- Fused output is byte-identical to `token_fp8_quant + IpcAllGather`: gathered fp8 activation
  bytes, fp32 scale, weights and ids all `torch.equal`, on both ranks, for repeated eager calls
  and 200 replayed graph launches (M = 32/64/100).
- The communicator helper returns bitwise-equal outputs for a marked dispatch; the large-M
  capacity path declines the fused branch and returns the quantized tensors for the standard
  gather.
- Model level: two vLLM servers, same session and certified Step5 config, 16 x 127,986-token
  prompts, greedy 64 tokens: both produce `outputs_sha256 85bc95cccc321744…`, 16/16 requests
  byte-identical.
- The generated kernel keeps the existing CTA ordering barrier between the quant stores and the
  aliased int32 push reads; the production flag/slot/counter protocol is copied verbatim.
- Fused path confirmed active at runtime (not inferred): a profiled server run on the frozen
  target base (16 × 128K, 6 s steady-state window, 119 decode steps) shows
  `_ipc_quant_gather_prod_kernel` 10,545 calls (~88/step), with `_token_fp8_quant_kernel` and
  `_ipc_all_gather_kernel` at 0 calls — the standalone quant and plain all-gather are replaced,
  so there is no silent fallback.

## Local performance (2 ranks, production dispatch order, 200 graph replays)

| M per rank | quant + all-gather (baseline) | fused | delta |
|---|---:|---:|---:|
| 32 | 20.76 us | 16.91 us | +3.85 us |
| 64 | 23.90 us | 17.69 us | +6.21 us |
| 100 | 31.84 us | 24.30 us | +7.54 us |

## E2E decode performance

The R3 result below is measured on the frozen target base with the DSA resident-selector
change **not applied**: it is the **UP-02 on DSASEL-OFF** configuration. It must not be
presented as the increment on the formal DSASEL-ON baseline.

Target-base validation (frozen upstream base `5f181a36…`, matched pair, order BA, symmetric env):
**R3 (128K ctx, BS50, MTP3): +1.71 % (Welch t=9.92, df=4.00, p=0.00058)**. ITL reduced from
88.87 ms (baseline) to 87.38 ms (candidate). All formal rounds (3/3 per leg) audit-clean, 50/50
complete requests, 0 preemptions.

Historical sessions (Step5 lineage, reported as directional context):

| workload | result | notes |
|---|---|---|
| R1 (128K BS16) | +1.43 % (Welch p=0.18); latest session clean rounds -0.40 % | **not confirmed**; reported as inconclusive |
| R2 (128K BS32) | +2.02 % (p=0.033, earlier session); +0.59 % (p=0.40, latest session) | session-dependent; no single-number claim |
| R3 (128K BS50) | +1.60 % (Welch p=0.005341; seed-paired p=0.007792) | robust to Welch vs paired |
| R4 (64K BS32) | +1.99 % (pooled 6-vs-6 Welch p=0.001990; seed-paired p=0.012568; BA p=0.052961; AB p=0.066873) | positive in both orders; p depends on the analysis unit; one anomalous A0 pair was excluded post-hoc and replaced |

Prefill non-regression: dedicated TP1/DP8 128K contract 23032.2 ± 4.7 TPS vs certified 23050.6
(-0.08 %, inside the ±2 % non-regression gate). An earlier build deferred the quant at world 8
as well and regressed prefill by 3.13 %; the regression was traced to the deferred branch (not
the fused kernel) and fixed by the world-2-only deferral in this change.

The DSASEL + fused-dispatch stack result is a separate optimization combination and is not
reported here as the fused-dispatch gain.

### UP-02 on the formal DSASEL-ON baseline

The standalone DSASEL-ON measurements available at delivery time are:

| workload | throughput delta | Welch p | conclusion |
|---|---:|---:|---|
| R2 (128K BS32) | +1.06 % | 0.24 | not significant |
| R1 (128K BS16) | +0.79 % | ≈0.54 | not significant |

Neither result establishes a DSASEL-ON gain; this change claims no statistically significant
increment on that baseline. A powered DSASEL-ON A/B remains a GPU-window action, not a
correctness claim.

## Compatibility

The patch targets `v0.28.1-dev` (runtime validation used the frozen upstream base SHA
`5f181a36c0e5b9749…`; the delivery branch was later synced to `061d8bd` and no functional
overlap exists between that base delta and the six files changed here). Runtime validation was
performed on both the Step5 development lineage and the frozen target base:

- **Target-base model equivalence**: two vLLM servers, same session/config, 16 × 128K prompts,
  greedy 64 tokens; both legs produced `outputs_sha256 85bc95cccc321744…`, 16/16
  byte-identical.
- **Target-base decode E2E** (R3 = 128K ctx, total BS50, MTP3; matched pair, order BA,
  symmetric env, 2 warmups + 3 formal rounds per leg): A0 (frozen base) 1576.86 ± 3.30 TPS,
  B1 (candidate) 1603.82 ± 3.36 TPS → +1.71 % (Welch t=9.92, df=4.00, p=0.00058).
- **Prefill non-regression** (TP1/DP8 contract, 128K BS16): A0 23290.78 ± 3.56 TPS,
  B1 23287.36 ± 11.50 TPS → −0.015 % (gate ±2 %).
