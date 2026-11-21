# RMSNorm ownership on vLLM 0.33

## Current candidate: Core owns compiled execution

The optional HCU class delegates compiled calls to the complete official
`RMSNorm.forward_cuda` path. Core's IR lowering and default precision policy own
fusion across residual chains. Eligible eager calls retain LightOp and its
existing ordinary/mutating custom-op ABIs. This adds no model-specific route
and does not disable Graph, prefix caching or compiler precision behavior.

The compiled provider contract is tested with genuine Core and OOT layers,
BF16/FP16, widths 128/2048/4096, dynamic batches 17/33, ordinary calls, two
successive residual additions and three changed-input Graph replays per batch.
Both sides use Core's `VllmBackend` and default arithmetic. Generated Triton
source and selected IR providers must match. Equal launch configurations must
be bitwise equal; independently autotuned configurations use the existing
official dtype tolerance, with residuals and input preservation still exact.

This distinction is supported by a separate Core-vs-Core control: independent
compilations of identical source select different launch geometries and can
differ by one low-precision ULP. At 4096x1024 BF16, Core and OOT both select
reduction blocks 2048 or4096; same-configuration pairs match bitwise. The original
failed exact comparisons and controls remain preserved. No precision setting
was changed. Ordinary `torch.compile` without the serving IR passes is a
different pipeline and is not the compiled serving oracle.

Selected validation on gfx936: 83 portable/cache cases and30 native/Graph cases
pass without skips. The same frozen Core0.33/Adapter combination runs original
HumanEval0–31 on Qwen3-1.7B BF16/TP1 with default FULL_AND_PIECEWISE Graphs,
prefix caching, Triton attention, seed0 and thinking disabled. All32 choices
and raw sample bytes match the no-Plugin baseline: raw2/32, normalized28/32,
4285 completion tokens,32 normal stops, no per-task losses. Profiling records
125 actual compiled RMS GPU events; it does not establish compiled vendor RMS
coverage. All10 checkpoint assets match original/pre/post dual reads,33312
physical installed files pass RECORD checks, and the owned API/engine exit.

Strict instruction-format probes fail on this small model and its baseline;
one concurrent candidate answer also differs. These are retained open gates,
not claimed as functional acceptance. The current compiled-owner candidate
still needs8B BF16/INT8, broader models, lifecycle and required remote CI.
The partial synthetic performance experiment stopped at a cross-configuration
bitwise comparison and is not a completed serving-performance result.
No gfx836/gfx938 or full-upgrade acceptance is claimed.

Runtime provenance: Core source `98a09850d56b4ab99a8c9bc4b28d411b72ed5147`,
Adapter source `60930a785fd1b0a550e292f23ed78ded5fb65400`,
Plugin wheel SHA256
`df4ff349106f516137aecb46714d158ddfce16bae0b4c40aa0a4896c66e77907`.
This is a combined candidate; other source changes in the cohort are not
transferred to this PR by the scoped RMS tests. Recheck the exact final merged
head and its compatible Adapter/CI prerequisites.

Reproduce the new composition test in an installed, ABI-matched0.33 stack:

```bash
HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter HCU_VLLM033_RMSNORM_LIVE=1 \
  python -m pytest tests/vllm033/test_rms_norm_hcu.py \
  tests/vllm033/test_rms_norm_composed_hcu.py -q
```

Complete requests, responses, scores, controls and seals are retained in
`publication/mr-r33/qwen3-17b-rms-upstream-pair.json` and the associated evidence
under `/data/models/gfx836-v033-20261115`. The following results belong to the
older staged vendor candidate and do not qualify the current compiled route.

## Historical staged vendor experiment

## Current 0.33.0 acceptance and epsilon follow-up

The exact Core0.33.0 candidate98a09850/Adapter60930a7/Plugin415+409+410
Qwen3-1.7B run is **not accepted**: RMS raw1/32 versus baseline2/32,
normalized28/32 in both, losing HumanEval/10 and gaining5. Strict output-format
probes and clean lifecycle remain unaccepted. Both owned API/engine identities
exited; original nanobind warnings remain. The results below from the earlier
Core0.33.1 development stack are historical diagnostics, not release0.33.0
qualification. Local bound receipt: `publication/mr-r29/qwen3-17b-rms-pair.json`.

An independent small-input diagnostic found another contract mismatch:
`rmsnorm_forward_autograd` narrows epsilon to the activation dtype. At
epsilon1e-6, constant BF16 input0.0006561279296875 produces0.55078125 instead
of the independently computed0.546875; FP16 input0.0006084442138671875 gives
0.51708984375 instead of0.52001953125. Constant rows remove reduction-order
ambiguity. The public `lightop.norm.rms_norm` inference entry retains FP32
epsilon. The Plugin now uses it with an explicit fresh output to preserve the
ordinary operation's non-mutating ABI, unit weights and intermediate rounding.
The fused provider is unchanged. Activation validates the new provider's actual
four-positional-argument ABI before registering factors or the OOT class.

Eight new live Graph cases cover BF16/FP16, epsilon1e-6/1e-5, widths128/2048,
and changed positive/negative constant inputs with an independent FP64 scalar
oracle. The previously installed candidate fails all eight. On actual gfx936,
the replacement installed wheel passes the complete18-case RMS file, zero
skips. Extension/operator/cache portable coverage closes81distinct cases:
the initial run passes80 and fails one because VLLM_TARGET_ROOT was absent;
the complete four-case cache file passes with the documented explicit target
(three overlap earlier passes). The original failure remains retained.
All32 Plugin Python files match source/wheel/installation;33,312 physical
package files pass raw hashed RECORD checks. The complete selected native
integration also passes150cases,11portable deselections,zero skips; runner
exit0/JUnit verified. Three original Torch finalizer ValueErrors remain, so
clean lifecycle is not accepted. The fresh same-protocol Qwen3-1.7B RMS run still scores raw1/32 and
normalized28/32 (4480completiontokens,32stop), with the same HumanEval/10 loss
and5 gain. Both owned identities exited; original/pre/post checkpoint reads
match and post-run RECORD verifies33312physical files. Epsilon repair does
not resolve model acceptance. Receipt:
`publication/mr-r30/qwen3-17b-rms-epsilon-pair.json`.

The source-level intermediate-cast formula below describes **eager native IR**.
The actual official033 baseline's generated Inductor kernels omit that cast
with default `emulate_precision_casts=False`. Installed official eager versus
compiled native probes (BF16/FP16,ordinary/fused,17/33rows,width2048) reproduce
the difference. Direct weighted public vendor RMS is bit-exact with compiled
BF16 native in all four probes; staged unit weights followed by BF16 multiply
is bit-exact with eager BF16 instead. FP16 reduction differences remain.
This means eager-formula agreement alone is insufficient to explain or accept
the compiled serving path. No compiler configuration or production default
was changed during this diagnosis. Evidence:
`evidence/rms-compiled-native-r30/result.json`.

## Intermediate rounding and historical diagnostic evidence

The target native IR casts the FP32 normalized value to the weight dtype
before multiplying the weight. LightOp's weighted ordinary and fused RMSNorm
round only the final weighted result. With BF16 inputs this can change around
one quarter of output elements while passing a loose FP32-formula comparison.
INT8 activation quantization can expose these differences downstream.

The optional Plugin path now uses LightOp to normalize with unit weights,
then multiplies the original weight in the input dtype. The fused path retains
its in-place x/residual ABI and stores the rounded FP32 residual sum. All
eligibility and rollback conditions still use the existing public OOT entry.

Regression coverage uses an independent intermediate-cast formula at widths
128 and 4096, BF16/FP16, ordinary/fused calls, and changed-input Graph replay.
The original installed wheel fails the new regression. The corrected isolated
wheel passes all 10 live RMS tests, 28 selected portable/cache tests and the
558-test non-DeepSeek portable suite (92 deselected, 18 warnings). These
are gfx936 results; gfx836 and gfx938 have not been validated in this run.

For live tests, enable `HCU_VLLM033_RMSNORM_LIVE=1` in the prepared isolated
HCU environment and run:

```bash
python -m pytest tests/vllm033/test_rms_norm_hcu.py -q
python -m pytest tests/vllm033/test_rms_norm.py tests/vllm033/test_rms_compile_cache.py -q
```

Set `VLLM_TARGET_ROOT` to the installed target site-packages for the actual AOT
cache test. Retain installed distribution/RECORD ownership checks and separate
compile caches for old and corrected wheels when evaluating models.

The initial complete Qwen3-8B BF16 run uses the same frozen HumanEval/0–31
requests, default FULL_AND_PIECEWISE Graphs and prefix caching as its baseline.
Both score 20/32 raw and 31/32 after the recorded syntax-format normalization,
with the same failed semantic task /26. All 32 finish normally, the repeated
prefix probe hits 3936 tokens, and two four-request concurrent waves pass.
This does not claim byte-identical generation or 32/32 semantic accuracy.

Unit-weight normalization and a separate multiplication add kernel work.
Synthetic same-card Graph timing shows roughly 2–3x component cost at the
larger tested shapes; it is not an end-to-end performance result.
The unrelated original AITER/finalizer and service-shutdown findings remain
open; this change makes no full-lifecycle claim.

The corrected Qwen3-8B Channel-INT8 run also scores 20/32 raw and 31/32
normalized, matching the official CT/Triton baseline's pass set. The original
weighted LightOp path scores 30/32 normalized in two complete cold runs and
adds a semantic failure at /24. That additional failure disappears with this
rounding correction. Both corrected BF16 and INT8 runs retain default Graphs,
prefix caching, 3936-token probe hits and two concurrent waves. INT8 text is
not byte-identical to its baseline.

The repeated sequential serving comparison uses the same physical gfx936 GPU3,
Qwen3-8B BF16/TP1, default Graphs, prefix caching, max model length 4096,
max sequences 16 and max batched tokens 1024. Sixteen requests warm the
endpoint; each measured request emits exactly 128 tokens with ignore_eos,
temperature/seed 0 and thinking disabled. Each concurrency has three measured
waves using identical request bodies on both arms. Median output tokens/s:

| Concurrency | Original weighted LightOp | Corrected rounding | Change |
| --- | ---: | ---: | ---: |
| 1 | 51.54 | 50.76 | -1.5% |
| 4 | 271.62 | 258.90 | -4.7% |
| 16 | 904.84 | 866.24 | -4.3% |

Median timing pairs below are original / corrected, in milliseconds:

| Concurrency | TTFT | Time per output token | Request latency |
| --- | ---: | ---: | ---: |
| 1 | 45.98 / 45.29 | 19.19 / 19.50 | 2482.80 / 2521.31 |
| 4 | 80.59 / 78.72 | 14.18 / 14.93 | 1882.18 / 1975.00 |
| 16 | 130.66 / 127.76 | 16.68 / 17.51 | 2250.87 / 2352.41 |

Other owned serving endpoints were idle and no model compilation ran during
these measured waves. Clocks/power were not locked. An earlier repetition
with competing activity on other GPUs also measured lower throughput. This
scoped correctness repair has a measured serving cost and makes no speedup
claim; the optimization remains explicitly opt-in. Wider model, INT8 serving
performance and architecture acceptance remain separate gates.

Artifact provenance for this scoped run:

- Core: `0014f6fce6de03125c02aa1541344ecba54ef114`, compared with official
  `4ba787ec9959e566d3b587a796dff07673324da5` native IR.
- Adapter: `21cf8f7b8697159be6efdec1961f71ab575008f1`; original serving
  dependencies retained in both isolated environments.
- Plugin base: `2cdc04242cee17fb24f7e758e62cb5c5ea0a3222`.
- Corrected wheel SHA256:
  `bfdd4d1b0a2c3730a537bd286604defdebe7d4530c2cdb18e410deef79c4a55c`.
  All 32 installed Plugin Python files match the candidate source; parent and
  spawned-process origins and installed RECORD hashes pass.
- HumanEval revision: `6d43fb980f9fee3c892a914eda09951f772ad10d`, original
  tasks 0–31. Generated code is scored as nobody in a mount/network/PID
  namespace with a read-only chroot. Raw and format-normalized scores remain
  separate; no precision-loss exception is applied.

Local logs, XML, request/response bodies, isolated scores, timing waves and
source/RECORD manifests are retained under
`/data/models/gfx836-v033-20261115/evidence/`. This is an RMS-specific result;
it does not certify FlashAttention, other model families or the full upgrade.
