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
family. HumanEval32, model prefix/MTP, communication and performance are still
separate acceptance gates.

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  VLLM_TARGET_ROOT=/path/to/official-v033 \
  PYTHONPATH=/path/to/adapter:/path/to/plugin/adapter_extension/src \
  .venv/bin/python -m pytest -q tests/vllm033
```

The listed wheel/provenance, selector and operator checks have passed. Full
loader/model integration, model accuracy, prefix/MTP, communication and
performance remain separate gates. No v0.32 score or wheel
hash is relabeled as v0.33. The old compatibility lock under `validation/v0.32.0`
is retained unchanged; the new combination is not last-good.
