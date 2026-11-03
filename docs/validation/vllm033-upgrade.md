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

Portable suite: **223 passed, 42 skipped**. A subsequent targeted suite
covering the actual v0.33 public ABC, forward metadata/output padding,
unsupported configuration, cache views and real target AOT cache keys:
**64 passed**; after tightening head-size admission, the full Flash CPU file
passed **62 tests**. Commands use `VLLM_TARGET_ROOT` for the new source and hide GPUs;
these are CPU contracts, not native/device or model acceptance.

All three trial wheels were built and installed normally into an isolated
environment. Parent and real spawn child loaded the new Core's four native
extensions with matching module origins and no GPU initialization. The actual
installed v0.33 public ROCm selector also selected the Plugin backend through
its registered owner in the CPU-only normal case (zero provider compute).

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  VLLM_TARGET_ROOT=/path/to/official-v033 \
  PYTHONPATH=/path/to/adapter:/path/to/plugin/adapter_extension/src \
  .venv/bin/python -m pytest -q tests/vllm033
```

Normal wheel installation/provenance, real selector and loader integration,
BF16/E5M2 numerical and changed-input Graph tests, model accuracy, prefix/MTP,
communication and performance remain separate gates. No v0.32 score or wheel
hash is relabeled as v0.33. The old compatibility lock under `validation/v0.32.0`
is retained unchanged; the new combination is not last-good.
