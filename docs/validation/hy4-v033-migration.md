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
| H07 | Offline R16 EPLB, nearest dispatch and recording | Plugin mechanism, reusing `static_eplb*`; Core owns router and loader APIs | HY4 backbone/MTP deployment binding implemented for eager PCP16; CPU/config checks below. HY4 recording and full-model/native placement remain pending |
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
| H18 | Mooncake transport and proxy | Plugin selects one canonical PCP sender through Core's external connector seam; native scheduler/cache/RDMA and completion aggregator stay in Core | Ten installed CPU contracts pass; actual RDMA/PD accuracy and timings pending |
| H19 | PCP profiling memory budget | Adapter runner factory bounds the local dummy profile using Core's decode-row bound; real scheduling budget unchanged | PCP16 full-model startup passed; whole-model Graph and throughput pending |

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

Earlier portable source checks: **93 passed, 1 deselected** across HY4 frontend, HY4 CT
policy/delegation and discovery. The deselected existing full-provider-contract
test was attempted and failed at its unavailable `vllm` import;
it is not a passing integration test. No Core/Torch native stack is installed in
this portable environment. These earlier checks did not provide the HY4-specific
offline-R16 model binding or establish native HIPC/DeepEP ABI acceptance.

### HY4 offline placement increment

The optional `das_hcu_hy4_static` extension publishes owned backbone and MTP
classes through the public Core ModelRegistry. Adapter owns admission, lifecycle
and persistent failure. Core model globals, forward methods, checkpoint mapping,
runner and compile constructor wrapper remain intact; private constructor
bindings select owned MoE children. Request
`HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hy4-ct-w4a8,hy4-static-eplb` explicitly.

The initial scope is eager PCP16/TP1/DP1/PP1/DCP1, DeepEP HT and DeepGEMM, with
`FLASHMLA_SPARSE`, LoRA off and Core dynamic EPLB off. The sidecar allocates 272
physical experts before parameter creation (256 logical +16 redundant), then
reuses the existing named loader fan-out and nearest static router. Unsupported
topologies, routing simulation and HY4 calibration fail before allocation.
The existing V41 scope and strict proposal reader are retained.

The separate HY4 reader accepts the original cookbook version-2 map with
`HYV4ForCausalLM` and `HYV4MTP` keys. This artifact has no recorded checkpoint
name or calibration metadata; do not infer either from compatibility. The
tested map SHA256 is
`9b99af7c6aa4824cfeca3e008365cf8a8bdaff071f458ddec9039f62b568a94b`.
The actual checkpoint declares 78 backbone layers (77 MoE) and one MTP layer
at index 78. Both map entries and physical counts are checked before loading;
file changes between preflight and loader binding are rejected.

Core's native MTP proposer does not inherit the target's attention backend.
Include `attention_backend` in the speculative configuration as well:

```bash
--attention-backend FLASHMLA_SPARSE \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}' \
--additional-config '{"hcu_static_eplb":{"expert_map_path":"r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}}'
```

The native HY4 MTP has no `get_top_tokens` protocol in this Core revision.
Do not request `use_local_argmax_reduction`; TP1 has no vocabulary communication
to remove. This retains natural MTP acceptance. MTP precision follows Core's checkpoint
configuration rather than forcing a new draft quantization format.

Validation covers map/schema failures, unchanged compile wrappers/Core globals,
pre-allocation physical capacity, public registry ownership, startup failure
latching, backbone/MTP loader order, CT weight/scale/shape fan-out and positional
or keyword `param` replacement replay. Distributed membership, expert allocation
and device packing in these CPU cases are leaf doubles. Replacement-argument
replay is not a full post-packing layerwise reload test; the existing packed
loader deliberately rejects unsupported reload after conversion.

The installed-stack affected CPU suite passed **218 tests**, including the
existing V41 static suite, complete extension discovery and wheel boundary.
This replaces overlapping earlier counts; it is not an additional 218 native
kernel or model-accuracy checks. Independent source review found no confirmed
correctness defect in this initial scope.

Installed configuration probing selects both owned model classes and validates
the complete original 77+1-row map. It does not allocate full weights or execute
MoE kernels. Full-model HIPC/DeepEP loading, HumanEval and performance remain
required before this scope can be accepted.

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

The initial stage used source/CPU checks. Later authorized component validation
built an isolated Python3.12/DTK2610 stack in a dedicated POD1_CN2 container;
Core source remains unchanged. Both nodes became occupied by other workloads
before full-model validation. No other containers or GPU jobs were stopped.

### November 21 startup findings

The ordinary PCP16 startup now passes backbone and MTP checkpoint loading after
two corrections. Static native mappings emit only logical expert IDs because
the owned loader already fans each logical key into its R16 physical replicas.
The original dynamic mapping repeated those keys with physical IDs256–271.
Physical allocation and routing still use272 experts,17 local slots.

The checkpoint's indexer WK is symmetric channel INT8 with an independent
FP32 scale. Core's BF16 WK/weights_proj fusion initially recognized FP8 only.
The owned loader buffers INT8 WK and reuses Core's pairing, scaled dequantization
and shard loading; BF16/FP8 retain the native path. Native loader loops, compile
wrapper and module globals remain unchanged. Asymmetric WK is rejected.
Explicit `hy4-ct-w4a8` publishes lazy CT model subclasses for ordinary D; static
P includes these loaders and retains sole registry ownership.

The Adapter also reuses Core's decode-row bound for persistent sparse conversion
tables. At98304 prefill tokens/S256/MTP3, each table has1025 rows instead of98305;
larger prefills retain Core's temporary conversion fallback. This removed the
construction-time OOM in the ordinary startup without lowering the batch budget.

The final installed affected CPU suite passed229 tests, including WK admission,
model registry and wheel ownership. This replaces the overlapping219/18-test
results. Adapter subsequently restored the gfx938 HY4 indexer ABI and bounded
rank-local PCP profiling. Its ragged decode adaptation packs Q and the matching
weights through Core's existing pack/unpack kernels. Full-model eager startup
then passed with the paired Adapter commit
`4c0b74ce1f211520390d0f1583d7193577be9331`.

The frozen 164-task EvalScope1.9.1 run uses temperature0, seed0, no_think,
max2048, stream and concurrency16. The unchanged checker runs in isolated CPU
containers. On the current gfx938/DTK2610 platform:

| Diagnostic deployment | Passed | Syntax failures |
| --- | --- | --- |
| P PCP16/EP16, offline R16, natural MTP3 | 151/164 | 6 |
| Same P, MTP disabled | 158/164 | 0 |
| D DP16/EP16, natural MTP3 | 157/164 | 0 |

The P no-MTP run isolates the interaction; it is not the target deployment or
an acceptance workaround. A mixed-batch diagnostic confirms API text exactly
matches decoding the returned generated token IDs, including malformed code.
The fault is already in generated IDs; its computation/lifecycle cause is still
under investigation. The historical 0.25.1 score158/164 came from a different
hardware/software combination and is not a matched current-platform control.
PD accuracy and formal146/2920 performance have not passed.

### PCP Mooncake ownership

Select the optional connector on both sides through the existing Core seam:

```json
{"kv_connector":"MooncakeConnector","kv_connector_module_path":"hcu_vllm_das.hy4_pcp_mooncake","kv_role":"kv_producer"}
```

Use `kv_consumer` for D. Retain the original connector name so Core's
configuration compatibility checks still recognize Mooncake; the external
module path selects this provider. PCP>1 is restricted to an HY4 text producer
with TP1/PP1/DP1/DCP1. PCP1 consumers retain the native worker behavior, including
DP16. Core gathers complete PCP MLA/indexer cache replicas; only PCP rank0 binds
the native sender listener/bootstrap coordinate. Other ranks report completion
after finished block metadata arrives. Core's world-size completion aggregator
still waits for the real sender's acknowledgement before freeing KV. Aborted
requests retain Core's immediate-free path and emit no replica-only completion.

The Plugin privately binds Core's constructor to an owned worker subclass.
It changes neither Core globals nor its connector registry, scheduler, cache
registration, transfer implementation or proxy protocol. Leaving this connector
unselected retains native behavior. The paired Adapter supplies the generic
`infra.native_namespace.bind_native` helper; no wheel build stamp is required.

Ten installed CPU tests cover public external-module selection, unchanged Core
bindings, canonical listener ownership, PCP1 consumer delegation, native
metadata readiness/cancellation, prevalidation cleanup and delayed completion
aggregation and the original connector identifier's compatibility predicates.
The complete affected installed CPU run passed 246 tests, including these ten
cases; this supersedes overlapping earlier counts. No RDMA engine or
GPU transfer is instantiated in these tests.
Actual PCP16→DP16 transport, cancellation under load and full PD HumanEval
remain required before this connector can be accepted.

### PCP MTP decisions

A subsequent read-only PCP16 trace found different draft IDs at proposal4:
rank0 `[4239,1246,926]`, rank6 `[4239,1246,25]`. The next verification accepted
three tokens on rank0 and four on rank6 for the same request. Per-rank request
history then diverged. This establishes a distributed state-consistency defect;
it does not establish why the initial floating-point sampling decisions differed.

The static HY4 provider now registers an existing Adapter post-model-load
callback before weight allocation. It binds only the loaded native MTP
instance's `sample_draft`: execute the original sampler, then use Core's PCP
rank0 broadcast. Synchronization happens at every draw, before the next draft
forward writes KV. Synchronizing only the completed proposal would be too late.
Core classes, globals, proposal loops, target verification and state updates
remain unchanged. This is neither synthetic acceptance nor uniform MoE routing.

The scope is eager MRV2/TP1/PP1/DP1/DCP1, PCP>1, native MTP and greedy draft
sampling. Unsupported workers are refused before allocation. Probabilistic
drafts, adaptive verification and watermarking are excluded; PCP1 and no-MTP
instances retain their original methods. Owner/process/group drift latches an
error rather than continuing with a partially synchronized instance.

The updated installed affected CPU suite passed245 tests, including12 native
sampling and shared load-seam cases. In a separate sixteen-request experiment,
seven raw draft draws still differed, but no completed proposal or target
sample/count disagreement was captured within the bounded trace. All sixteen
responses parsed as Python. The experiment used batch budget32768 to avoid an
intermittent HSA allocation failure in the98304 startup profile; it is not a
best-configuration throughput measurement. Full accuracy with the packaged
provider remains pending.

The current-platform0.25.1 D control completed158/164, with failures103,115,116,
132,145,160. Core7b108a and Plugin7a5a4af use the same DTK2610/Torch2.11/FlashMLA
as the candidate, on the same D node and read-only model mount. The metadata
probe hashes config, checkpoint index and tokenizer; it does not recompute all
checkpoint shard hashes. This is a single-D control, not the full historical PD
stack or a performance-parity result.

## PD startup follow-up (2026-11-21)

Packaged Plugin d7e5cff with Adapter4c0b74c completed the full single-P
HumanEval164 run:158 passed, no syntax failures or API errors, all responses
stopped normally. This used batch32768; the earlier151 result used98304, so
neither the score difference nor throughput can be attributed to one patch.

The first real-PD startup exposed an earlier bootstrap branch in Core's worker
constructor: its TP/PP/DP predicate started a server on all16 PCP ranks. The
owned constructor now privately binds that predicate to also require PCP rank0;
all native configuration checks and transport code remain unchanged. Constructor
failure cleans the resources already created without invoking Core shutdown on
missing fields or masking the original exception. The full native constructor
replay reproduced16 launches before the fix and one after it; external bootstrap
and non-PCP consumer behavior are also covered.

The D startup failed while Dynamo traced the Adapter indexer's lazy-init lock.
The paired Adapter fix publishes the opaque operator before compilation, with
health checks inside its execution boundary. Source-snapshot CPU checks passed202
tests; gfx938 tests passed32/64 heads, including actual indexer compilation and
changed-input Graph replay. These are component checks, not a full-model Graph
or RDMA acceptance result. Both MRs remain Draft pending PD accuracy/performance.
