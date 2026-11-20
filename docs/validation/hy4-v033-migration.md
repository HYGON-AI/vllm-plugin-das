# HY4 v0.25.1 → v0.33.0

## Scope and ownership

Target checkpoint: `hygon/Hy4-preview-Channel-INT4-w4a8`. Preserve PCP16 + EP16
with offline R16 placement on P and DP16 + EP16 with natural MTP3 on D.
The reference is the W4A8 1P1D section of cookbook commit
`522735c83ca1248fd69d5277aec2f2b6c4855c4f`, not its separate FP8-v2 decode benchmark.
Uniform routing and synthetic MTP are diagnostic controls, not accuracy gates.

Source inputs inspected on 2026-11-20:

| Repository | Revision |
| --- | --- |
| OpenDAS/vllm v0.25.1 | `7b108ad1a51b217e9abec0ddc047978405481bae` |
| Plugin v0.25.1 reference tree | `20e14f7d81d0fc7c63543dd875a0515d0a63e53b` |
| OpenDAS/vllm v0.33.0 | `c440c089fe9621c5b03e1618cf86a13e95de8bf1` |
| Plugin v0.33.0 | `8631a126c254a355269a866fb3a9a9ecdbd3cb32` |
| Adapter main | `af8b0ea8ca383dd929b15255e22e69e860bab173` |

The source reference is not a newly reproduced 0.25.1 deployment baseline.
Branch names are moving inputs; freeze the tested revisions again before device runs.

The ownership rule comes from Adapter `skills/hcu-adapter-dev/SKILL.md`,
`docs/governance.md` §§0, 3, 7, `docs/frameworks/vllm033.md`, and this repository's
`skills/upgrading-vllm-hcu-upstream-first/SKILL.md`:

- Core: official model, scheduler, cache allocation, loader, runner and public registries.
- Adapter: restore existing reference-platform behavior on HCU; own runtime ABI,
  platform/general entry points, extension discovery, common lifecycle and failure latch.
- Plugin: additional formats, algorithms and optional providers absent from Core.
  Provider implementation belongs here; common admission does not.

Each changed repository gets its own Draft MR. Do not create an empty Core/Adapter
MR or copy an existing mechanism just to obtain three MRs.

## Migration ledger

“Reuse candidate” below means source availability, not device equivalence or acceptance.

| ID | 0.25.1 feature / control | Owner and 0.33 disposition | Status / required evidence |
| --- | --- | --- | --- |
| H01 | HY4 backbone, iHC, routing and SwiGLU clamp | Reuse Core `vllm/models/hy_v4/{amd,nvidia}` | Source inspected; HCU checkpoint/accuracy pending |
| H02 | MTP3 and inherited quantization | Reuse Core `hy_v4/nvidia/mtp.py`; preserve the checkpoint's actual MTP dtype | Source inspected; HCU loader/drafted/accepted tokens pending |
| H03 | PCP16 metadata, indexer and cache ownership | Reuse Core `v1/attention/backends/mla/{indexer,flashmla_sparse}.py` | Requires HY4 HCU dispatch/stride/empty-chunk and distributed tests |
| H04 | Learnable attention sink | Reuse Core wrappers; Adapter restores DTK call ABI through the public backend registry | Component prefill/packed decode/changed-input Graph checks pass on gfx938; distributed/model accuracy pending; DCP excluded |
| H05 | Public `fp8_e4m3` vs physical `fp8_ds_mla` | Core owns geometry; Adapter restores HCU ABI and provider eligibility | Packed cache, scales, RoPE and actual writer pending |
| H06 | CT W4A8 routed experts, weight/scale reload | Distinct Plugin `hcu_hy4_ct_w4a8` config; reuse existing HIPC method and Core loader/runner | HY4 schema/delegation CPU checks added; actual load, native calls and layerwise reload remain pending |
| H07 | Offline R16 EPLB, nearest dispatch and recording | Plugin mechanism, reusing `static_eplb*`; Core owns router and loader APIs | Existing runtime is owned-model/PCP-specific; HY4 backbone/MTP mapping and redundant slot sizes pending |
| H08 | `VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH` | Plugin optimization with correct communicator ownership | Recheck target modular DeepEP buffers before carrying legacy scratch |
| H09 | `VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING` | First assess Core sparse prefill batching; Plugin only for remaining algorithm | Not yet proven equivalent on HCU; retain as pending |
| H10 | `VLLM_HCU_ENABLE_TRITON_CP_GATHER` | First reuse Core CP gather; Adapter for ABI-only gaps | Packed page strides and changed-input tests pending |
| H11 | `VLLM_HCU_HYV4_MOE_CHUNK_SIZE` | Core scheduling/runner first; Plugin only for missing chunk mechanism | Shared experts, actual per-rank capacity and padding pending |
| H12 | `VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE` | First inspect Core shared-expert overlap; Plugin only for absent mechanism | Stream/event ownership and Graph tests pending |
| H13 | LightOp MQA logits and quantization substitutions | Adapter when restoring existing operator behavior through vendor routing | Match real ABI, grouping/scale dtype, masks and fallback; do not copy entire indexer |
| H14 | Fused RMS quant and HY4 attention-input overlap | Adapter for equivalent native operator routing; Plugin for additional fusion/overlap | Classify subfeatures separately; numerical and stream contracts pending |
| H15 | ByteLevel prompt-tail priming | Optional Plugin frontend algorithm; retain Core construction/output/stop handling | Implemented; 63 CPU tests pass with import-isolated real target functions |
| H16 | P synchronous scheduling and four renderer threads | Reuse Core configuration | Validate target CLI/config and preserve fair load; not a new provider |
| H17 | Prefix caching and request workload | Reuse Core cache manager and benchmark | 146 then 2920, same seed and service lifetime; prefix-hit counters required |
| H18 | Mooncake transport and proxy | Reuse Core connector; retain cookbook proxy protocol | HCU RDMA/device-affinity and first-token timings pending |
| H19 | PCP profiling memory budget | Core profiling first; Plugin only if rank-local accounting still differs | Do not infer available cache from a different model's provider |

## ByteLevel provider

The separate `das_hcu_hy4_frontend` Adapter extension imports neither vLLM nor
Torch during discovery. It uses the existing
`VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1` opt-in; unset/0 leaves the original
constructor object untouched. Default-off activation does not import Core.

Core has no public detokenizer-provider registry at this target, so this provider
wraps only `FastIncrementalDetokenizer.__init__` after Adapter admission. It calls
the original constructor, then replaces only a genuine native ByteLevel
DecodeStream's prompt priming context. It does not replace `DecodeStream`, request
objects, stop handling or generated token bookkeeping. Native/shim streams with a
different contract are preserved.

Tail selection tries 32–35 and 64–67 tokens, at most eight decodes and 396 input
tokens. Invalid/empty/replacement-character tails preserve the complete prompt.
This covers three-byte UTF-8 boundaries missed by doubling-only scans.

The Plugin exposes a read-only owner/policy health check and signals published
failure with the public `ExtensionCommitError`. Adapter's existing
`ProcessLocalLifecycle` owns serialization, report/PID binding and persistent
failure. No duplicate platform/version/runtime guard or custom lifecycle is added.

Run portable checks with Python 3.12, pytest, tokenizers and packaging installed:

```bash
HCU_ADAPTER_HOST=none VLLM_TARGET_ROOT=/path/to/vllm-v033 \
  PYTHONPATH=/path/to/adapter-das:/path/to/plugin/adapter_extension/src \
  python -m pytest -q tests/vllm033/test_hy4_detokenizer_cpu.py
```

Result: **63 passed**, tokenizers 0.23.3. Forty-eight cases execute the actual Core
detokenizer class/function bodies with imports isolated; this is not a full Core
installation test. Other cases cover native streams, bounded fallback, foreign
streams, idempotence and the real Adapter extension/health protocol. No HCU
kernel, model accuracy or serving performance claim is made.

The pure-Python Plugin wheel was built and installed in a separate Python 3.12
environment. Actual entry-point discovery/default-off activation passed in the
parent and a real spawn child without importing Torch or vLLM. This probe uses
Adapter's source protocol; it does not certify an installed Adapter/Core/native stack.

## Acceptance order

### HY4 CT admission increment

The paired Adapter adds `set_additional_config_provider(name, factory)` to its
existing CT registration transaction. It retains the legacy DSV4.1 name and
never overwrites official CT/FP8 entries. The Plugin supplies
`HcuHY4CTW4A8Config` as a lazy provider, restricted to `model_type=hy_v4` or Core's
native `hy_v4_mtp` draft type, packed
static channel INT4 weights and dynamic symmetric token INT8 activations.
Other models, explicit quantizer choices and different formats are not claimed.

The HY4 config delegates scheme parsing and execution to the existing packed
method, including INT8 dense layers and positive SwiGLU clamp. No DSV4.1 router,
model class or weights are substituted. The shared schema predicate was factored
into a framework-free helper without changing its DSV4.1 contract.

The additional `das_hcu_hy4_ct_w4a8` extension checks the Adapter-owned public
registration; it does not own a second registry transaction or common admission.
Older Adapter revisions retain DSV4.1 binding, but the HY4 capability explicitly
refuses the missing additional-provider seam. For an HY4 validation deployment,
require `HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hy4-ct-w4a8`; keep the existing
CT native/stock policy and kill switch.

Installed-stack probing found two startup gaps not covered by the initial CPU
doubles: the HY4 capability ran before CT publication (priority 106 vs 105), and
Core renames the draft model type to `hy_v4_mtp` while propagating the target's
quantizer. The capability now runs at priority 104, and both exact HY4 types
retain the same packed scheme checks. Other families/quantizers stay excluded.
Both failures were reproduced before fixing; new tests cover actual Adapter
dispatch ordering and draft admission. The installed Core/Adapter/Plugin CPU
suite then passed **106 tests**. Configuration probing also passed for PCP16 and
natural MTP3; no model weights or generated outputs were involved.

Adapter's paired increment restores D576 sparse FlashMLA with Core's original
sink/padding/metadata/PCP methods, changing only vendor-call bindings. Explicit
`--attention-backend FLASHMLA_SPARSE` selects it. BF16 prefill and packed-FP8
decode/changed-input Graph component tests passed on actual gfx938; see Adapter
MR153 for the scope and key-only DTK LSE contract. This does not establish
whole-model accuracy, offline R16 readiness or performance equivalence.

Portable source checks: **93 passed, 1 deselected** across HY4 frontend, HY4 CT
policy/delegation and discovery. The deselected existing full-provider-contract
test was attempted and failed at its unavailable `vllm` import;
it is not a passing integration test. No Core/Torch native stack is installed in
this portable environment. This increment does not yet provide the HY4-specific
offline-R16 model binding or establish native HIPC/DeepEP ABI acceptance.

### Device order

1. Finish source ownership/disposition and affected CPU contracts.
2. Install ABI-matched Core + Adapter + Plugin wheels; verify parent/spawn module origins.
3. Run the full 164-task HumanEval with natural routing/MTP, matching 0.25.1
   checkpoint bytes, template, seed, sampling, timeout and token budget. Preserve raw
   outputs, original score, failures and truncation; do not silently normalize scores.
4. Only after accuracy passes, run the matched 146/2920 workload. Record client
   TTFT percentiles, TPOT/ITL, completed RPS, output/total throughput, MTP statistics,
   P/D engine timings, cache hits and expert balance.
5. Compare repeated matched baseline/candidate runs. Pending CPU or device rows
   cannot be called accepted; performance parity remains open until measured.

Current stage is source migration and CPU checks only, as requested. No machines
were reserved and no service/container installation was changed in this stage.
