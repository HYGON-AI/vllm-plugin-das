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

## Local performance (2 ranks, production dispatch order, 200 graph replays)

| M per rank | quant + all-gather (baseline) | fused | delta |
|---|---:|---:|---:|
| 32 | 20.76 us | 16.91 us | +3.85 us |
| 64 | 23.90 us | 17.69 us | +6.21 us |
| 100 | 31.84 us | 24.30 us | +7.54 us |

## Paired decode E2E (historical sessions; decode-window TPS)

| workload | result | notes |
|---|---|---|
| R1 (128K BS16) | +1.43 % (Welch p=0.18); latest session clean rounds -0.40 % | **not confirmed**; reported as inconclusive |
| R2 (128K BS32) | +2.02 % (p=0.033, earlier session); +0.59 % (p=0.40, latest session) | session-dependent; no single-number claim |
| R3 (128K BS50) | +1.60 % (Welch p=0.005341; seed-paired p=0.007792) | robust to Welch vs paired |
| R4 (64K BS32) | +1.99 % (pooled 6-vs-6 Welch p=0.001990; seed-paired p=0.012568; BA p=0.052961; AB p=0.066873) | positive in both orders; p depends on the analysis unit; one anomalous A0 pair was excluded post-hoc and replaced |

Prefill: dedicated TP1/DP8 128K contract 23032.2 +- 4.7 TPS vs certified 23050.6 (-0.08 %, inside
the +-2 % non-regression gate). An earlier build deferred the quant at world 8 as well and
regressed prefill by 3.13 %; the regression was traced to the deferred branch (not the fused
kernel) and fixed by the world-2-only deferral in this change.

The DSASEL + fused-dispatch stack result is a separate optimization combination and is not
reported here as the fused-dispatch gain.

## Compatibility

The patch targets `v0.28.1-dev`. Runtime validation was performed on the Step5 branch lineage;
the target-base runtime validation (model equivalence + paired decode on the frozen target SHA)
is pending and must be completed before this change is marked ready for review.
