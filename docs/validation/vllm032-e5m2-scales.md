# E5M2 mixed-dtype attention scales on HCU vLLM 0.32

MR321 writes non-unit-scale E5M2 KV bytes correctly, but the installed vendor
mixed-BF16-query/E5M2-KV path produces attention as if both KV descales were one.
This affects paged prefill and single-token decode. Making expanded descales
contiguous or additionally supplying unit query descales did not repair decode.

The backend now applies K scale to a separate query buffer and V scale to the
native output buffer, using two small GPU operations. It then passes no KV
descales to native attention, so a later vendor implementation cannot apply
them twice. Query and output multiplication use FP32 intermediates and round
to the query/output dtype. Per-sequence scale selection uses the real ragged
query offsets and supports scalar or `[batch, KV heads]` scale strides.

The original input query, physical KV pages, native writer, native paged
attention and native prefill gather remain in use. The caller's output buffer
is updated in place. No CPU scale reads, extra KV traversal, new flag or global
attention replacement is introduced. Other cache dtypes retain their path.
E5M2 remains an explicit cache-dtype choice; this change does not advertise
backend support for per-head calibration or batch invariance.

Remove this adaptation when a native mixed-dtype paged API applies descales
and passes the same non-unit physical-cache, ragged-batch and changed-input
Graph checks. Do not infer capability from an exact wheel string.

## Validation boundaries

Evidence root for this run: `/data/models/vllm032-upgrade-20261031`. Target
vLLM is SourceFind `a198423b7be0ccf9c9ac3fb01c02ac03ab145e93`, Adapter PR17
`35f8f0569df4ffbed045bf0dc18974eaafd5ab1a`, and the base Plugin is MR321
`153e4d9eb88a82f3c59a6fd5e87d790c59675104`. Hardware is gfx936, DTK26.10,
Torch2.11. `e5m2-scale-final-attestation.json` binds source and wheel content.

- Final physical prefill matrix: **56/56**, HND/NHD, BF16 controls and E5M2
  non-unit scales, prefixes3072/8192, q11/12/13/74/75/76/77. It covers high,
  shuffled physical pages, nonzero storage offset and page padding. Graph
  replay changes Q/K/V, scale, slots and page table at the same addresses.
- Final direct-decode matrix: **12/12**, BF16, E5M2 unit/non-unit scales,
  both layouts and prefixes3072/8192. Dispatch still reaches native paged
  attention, without contiguous gather. Both frozen harness audits pass.
- Final HCU suite: **36 passed**, including sixteen BF16/FP16, HND/NHD,
  two-sequence prefill/decode regression cases with three changed-input
  Graph waves, preserved query and output alias, and prefill LSE return.
  Scales include the original range and values multiplied by0.001, covering
  a smaller calibration scale range without changing the acceptance tolerance.
- The preceding candidate's portable suite: **140 passed**. The final
  computational change only widens the GPU row index to 64 bits; final
  physical/HCU tests above execute that change.

Original failed matrices, unit-scale controls and ABI diagnostics are retained.
The independent FP32 reference is not the vendor gather output or an unscaled
reference. Q/output rounding is accepted only within the recorded numerical
tolerances; this is not an assertion of bitwise equality.

Performance and model accuracy are recorded independently. Final synthetic
Graph measurements use Hq32/Hkv8/D128, HND, batch1/8, prefixes3072/8192,
q1/12/77, unit scales for a valid native control, five warmups and three
alternating rounds of100 replays. Prefill adds0.43–1.30% latency; decode adds
6.64–7.09us (4.81–25.61%, depending on the baseline shape). These are kernel
measurements, not model throughput; the correctness repair has a measurable
cost. See `benchmark-e5m2-scale-final`.

A prior approach
that scaled gathered KV passed numerics but added11–21% prefill latency in the
measured shapes; its artifacts remain separate. The two-row-operation approach
avoids that extra KV pass. Final Graph timings and HumanEval32 results must be
read with their exact protocol and checkpoint.

Qwen3-8B HumanEval/0–31 uses the frozen evaluation protocol (temperature0,
seed0, max2048 output tokens, thinking disabled), the standard output head,
and the original FLASH_ATTN/HND/Graph/prefix settings. The final candidate's
two complete runs score raw19/normalized30 and raw20/normalized31. MR321's
two runs score raw20/normalized31 and raw19/normalized30. Both 30-score runs
fail tasks19 and26; both 31-score runs fail only26. Raw and formatting-only
normalized results are reported separately, and completions are scored in
the namespace/chroot sandbox. All four runs remain in
`humaneval32-summary.json`; the better repeat does not replace the first run.

The final candidate's original 72-request prefix probe still fails cold/warm
exact-token equality. Similar failure exists in the MR321 BF16 and E5M2
controls. These repetitions establish the observed score range, not numerical
equivalence or completion of the model-accuracy gate. E5M2 remains opt-in and
the overall upgrade remains unaccepted. See
`probe-qwen3-e5m2-scale-final-prefix3072` and the two
`humaneval32-qwen3-e5m2-scale-final*` runs.
