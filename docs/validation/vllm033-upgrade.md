# v0.33 Plugin migration

Target: official vLLM `v0.33.0`, commit
`4ba787ec9959e566d3b587a796dff07673324da5`.
The release branch starts at merged Plugin commit
`153e4d9eb88a82f3c59a6fd5e87d790c59675104` (MR321 included).
Unmerged PR323/324/326 are reviewed migration inputs; their v0.32 results remain
historical. Adapter stays on its shared main line and admits both releases.

## Ownership and changes

Core owns official model execution, scheduling, metadata and native operator
implementations. DTK build compatibility belongs in the OpenDAS build delta.
Adapter owns real hardware admission, the sole platform/general entry points,
extension discovery, required capability checks and process health.
This package contributes optional providers through Adapter's extension API.

The Flash implementation now derives from the public `AttentionImpl` rather
than the official `FlashAttentionImpl`. Official v0.33 has none of the old
fork's three private cache/kernel hooks. The provider reuses the official
metadata builder and public registry, owns dense cache aliases and the vendor
FA2 call, and delegates ordinary cache writes to the official native operator.
E5M2 retains its writer and non-unit-scale adapter. Native fusion capability
defaults remain false because their cache ABI differs from this provider.

The implementation rejects context parallelism, sliding windows, sinks,
non-decoder and dynamic/non-causal metadata, multimodal PrefixLM, R-SWA and
fused output quantization. This rejection does not remove those requirements
from the upgrade ledger; they need their own supported backend and validation.

v0.33 splits AITER experts by quantization format. The Plugin INT8 expert
advertises exactly channel-weight/token-activation INT8 instead of inheriting
other formats from the common parent. Default backend priority is preserved.
RMS policy still participates in official normal/AOT cache keys, and provider
ABI checks precede operator publication.

## Current evidence and remaining gates

Portable suite after the release-documentation and Graph-version review:
**225 passed, 42 skipped**. A targeted suite
covering the actual v0.33 public ABC, forward metadata/output padding,
unsupported configuration, cache views and real target AOT cache keys:
**64 passed**; after tightening head-size admission, the full Flash CPU file
passed **62 tests**. Commands use `VLLM_TARGET_ROOT` for the new source and hide GPUs;
these are CPU contracts, not native/device or model acceptance.
After adding explicit rejection cases for the previous 0.32 release line,
the focused version-admission suite passed **15 tests** (36 deselected).
Those two rejection cases were added after the full portable batch above.

All three trial wheels were built and installed normally into an isolated
environment. Parent and real spawn child loaded the new Core's four native
extensions with matching module origins and no GPU initialization. The actual
installed v0.33 public ROCm selector selected the Plugin backend through
its registered owner. Normal, clear, foreign, fork and spawn CPU cases passed;
an additional configuration case rejected unsupported head/page sizes, DCP,
MLA, sparse and sinks through the actual official validation (zero compute).

The current MR production sources were rebuilt into wheels and installed in
fresh environments. Core was built for gfx936 with DTK 26.10 / Torch 2.11 /
Python 3.12; parent and real spawn loaded its four native extensions from that
environment. Source/wheel Python bytes matched (Core 2,991 files, Adapter 203,
Plugin 13). Later changes to tests and documentation do not replace the build
commit identity or create new numerical claims.

## gfx936 device checks (2026-11-03)

| Installed-wheel test | Result | Scope |
| --- | --- | --- |
| `test_flash_attention_hcu.py` + `test_e5m2_scales_hcu.py` | 36 passed | BF16/FP16, both layouts, E5M2 non-unit scales, changed-input Graph |
| `test_rms_norm_hcu.py` | 2 passed | ordinary/fused RMSNorm, fallback, compiled changed-input replay |
| `test_operator_graph_hcu.py` | 1 passed | optimized/official operator selection and compiled numerical comparison |
| `test_aiter_int8_hcu.py` | 3 passed | native AITER W8A8 M=1/16/128, Triton comparison, changed-input Graph |

The first operator Graph attempt reached both providers but its test rejected
the valid `0.33.0+das...` build suffix. The revised test checks epoch zero and
the 0.33 release line, preserving the existing path, provider and numerical
assertions; its independent rerun passed. No numerical threshold was changed.
These device checks cover the listed shapes and gfx936, not gfx92a or a model
family. Model accuracy, prefix/MTP, communication and performance are separate
acceptance gates; the Qwen2 observations below do not close the model matrix.

## Qwen2 model observations (2026-11-03)

Candidate summary, checkpoint/tool hashes and wheel build identities:
[`qwen2-gfx936-candidate.json`](../../validation/v0.33.0/qwen2-gfx936-candidate.json).
This is an unresolved candidate record, not a new compatibility lock.

The installed-wheel runs use the same `Qwen2-0.5B-Instruct` checkpoint,
BF16 weights, TP1, seed 0, temperature 0, HumanEval/0–31 and 2,048 generated
tokens per sample. Prefix caching and the default FULL_AND_PIECEWISE Graph
remain enabled. Each arm was independently restarted with fresh caches and
repeated using identical HTTP request bodies; endpoints differ.

| Backend / KV | raw, both runs | normalized, both runs | Prefix hit delta per run |
| --- | --- | --- | --- |
| Official Triton / BF16, no Plugin | 0/32 | 16/32 | 1,184 tokens |
| Plugin Flash / BF16 | 0/32 | 16/32 | 1,152 tokens |
| Plugin Flash / E5M2 | 0/32 | 13/32 | 1,152 tokens |

Raw outputs include code formatting; normalized scoring is reported separately.
Both scorings execute only inside the existing isolated evaluator, uid 65534,
with networking and GPU access blocked. Each arm's repeated raw sample file
is byte-identical. BF16's equal total masks a changed task set: Plugin newly
fails /4 and newly passes /29 versus official Triton. E5M2 versus Plugin BF16
newly fails /0,/8,/16,/25,/31 and newly passes /5,/26. Its lower score is
reproducible and unaccepted; no accuracy tolerance was introduced. Control and
Plugin BF16 each have one length-limited completion per run; E5M2 has none.

An additional official Triton E5M2/page64 control fails before generation:
upstream initializes query quantization, then Attention.forward asserts that
the cache format is E4M3/NVFP4. This is a startup failure with no accuracy
score. No Core runtime workaround was added, and this failed control does not
establish the cause of the Plugin E5M2 score loss.

All six completed runs pass arithmetic and three repeated long-prefix output
probes, record positive cache hits and exit zero. Shutdown still reports one
resource_tracker semaphore warning. Owned services were stopped and all eight
cards returned to approximately 2 MiB. GPU5/6 were never used. The original
diagnostic-endpoint 404 failures are retained as zero-inference tool failures.
**Quality remains OPEN and last_good=false; the Plugin PR remains a draft.**

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  VLLM_TARGET_ROOT=/path/to/official-v033 \
  PYTHONPATH=/path/to/adapter:/path/to/plugin/adapter_extension/src \
  .venv/bin/python -m pytest -q tests/vllm033
```

The listed wheel/provenance, selector and operator checks have passed. Broader
loader/model integration and model-accuracy gates remain open. Additional
Qwen3 INT8/TP2, GLM MLA/EP/MTP, official communication and GDN observations
are recorded separately in [the additional feature batch](vllm033-feature-batch.md);
its exact scope and remaining gates apply. No v0.32 score or wheel
hash is relabeled as v0.33. The old compatibility lock under `validation/v0.32.0`
is retained unchanged; the new combination is not last-good.
