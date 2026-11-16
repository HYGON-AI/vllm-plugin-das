# HCU V41 eager PCP attention source integration

Status: failure modes fixed before production implementation. This stage owns attention only; Engram PCP sharding remains a separate stage. No GPU, distributed-process, Graph, installation or full-model accuracy claim.

## Failure modes and fixed replay contract

1. Merely removing AMD `_select_dsv4_attn_cls` rejection leaves both attention and SWA backends advertising no PCP support. Publish owned backend subclasses through an owned model, without modifying Core registries except the public ModelRegistry entry.
2. AMD fast forward omits the common PCP global latent gather and can issue collectives on auxiliary streams in incompatible order. PCP must execute common preparation serially and preserve the order SWA KV gather -> CR2 predecessor row gather -> latent gather -> source main/index K stores -> local scoring/attention.
3. Rank-major gathered values, cache positions and slots must use the same padded local extent. Dual front/back fragments must retain absolute causal positions; do not sort values separately or use a contiguous rank slice as logical token order.
4. CR2 groups crossing fragments need the native predecessor linking/ring writes. Keep native compressor metadata/forward, including empty fragments and newest-open-row persistence. Do not reintroduce the old three-fragment manager.
5. Source main cache/index K writes must happen once per source with global rows, consumers must share cache ownership, and local query/output/candidate rows must remain local. Native indexer metadata already supports PCP/DCP1; reuse it.
6. Existing HCU reference indexer explicitly admits PCP1 only. A separate owned PCP indexer must consume the same SHUFFLE cache with local native metadata; never enable the original OOT by falsifying parallel config.
7. Native AMD forward_mqa rotates the entire padded output. Zero local padding and return early for zero actual queries; only actual local rows reach attention/inverse RoPE. Inverse RoPE belongs to AMD forward_mqa, not output projection.
8. A private constructor namespace may replace native class names only. Preserve source bytecode/defaults/closure and refuse source drift. No process-global Core method/class replacement, temporary config mutation, hardware spoofing or copied parameter lists.
9. Keep flag-off/PCP1 native, foreign/lost registry owners rejected, frozen policy checked, publication idempotent. Discovery must not import Torch/vLLM.
10. Admission must reject unsupported semantics before allocation: non-eager, DCP/PP/DP combinations lacking the native PCP contract, speculative decode (native CR2 ring restriction), monokernel, MXFP4/sparse logits, KV connector/replay, and Engram layout until its dedicated PCP adapter exists. Do not use a parallel-width whitelist.

## Pre-implementation replay

The replay is written before production code. It will import production modules freshly and record failures while those modules are absent. Fixed scenarios cover public registration and private constructors; admission/non-PCP delegation; 200-seed x eight-rank row/slot/causal mapping; native common preparation and CR2 predecessor linking with CPU kernel/collective doubles; source-owned K/cache reuse; actual local FP8 indexer scoring; local output/padding/inverse-RoPE ownership. Provider GPU kernels and transport are explicit doubles. A JSON report records Core/plugin paths, revisions, source hashes and CUDA initialization state.

## Source seams already inspected

- Core `amd/model.py:152-174,182-217,494-658,1320-1347`: hard rejection and constructor class names.
- Core `amd/vl_model.py:165-211`: outer language-model constructor.
- Core `amd/rocm.py:526-550,683-842,964-1095`: backends, fast forward, inverse-RoPE owner.
- Core `attention.py:859-959,1105-1122,1476-1508`: serial PCP preparation, SWA gather and source K store.
- Core `sparse_mla.py:55-94,213-253`: local query metadata/global cache slots and positions.
- Core `compressor.py:153-203,322-416`: predecessor/ring and source insertion.
- Core `v1/attention/backends/mla/indexer.py:1415-1597`: local Q, rank-major compressed slots, DCP1 prefill chunk metadata.

## Implementation and ownership

The opt-in extension `das_hcu_v41_pcp` owns only the public `DeepseekV41ForCausalLM` ModelRegistry entry when `VLLM_HCU_DSV41_PCP_ATTENTION=1`. Default is off. The policy is frozen at activation and included in the compile factor `VLLM_HCU_DAS_V41_PCP_ATTENTION`. Changed policy, foreign/lost owners, partial publication and constructor source drift fail closed. PCP1 constructs the native outer model and language model.

Three reviewed constructor functions execute unchanged bytecode/defaults/closures in private namespace copies. Only the VL language-model name, backbone decoder name and decoder attention selector differ. Core globals and methods are never rebound. Native model loading, MoE, mHC, model-state/PCPManager and multimodal processor inheritance remain native. No Adapter source or old `vllm_hcu` code is changed.

| Path | Owner / behavior |
|---|---|
| SWA KV + global positions/slots | Native V41 common preparation and metadata builder; rank-major gather |
| CR2 predecessor + ring | Native compressor metadata and kernels; no replacement |
| Main/index K cache publication | Native source-owned writers with gathered latent/positions |
| Query/top-k/candidate buffers | Local rows; own FP8 reference scorer, separate from PCP1 OOT |
| Attention / inverse RoPE / output projection | AMD native; owned wrapper trims actual rows and clears padding |
| Final hidden-state gather and restore | Native PCPManager, outside attention |

Both owned backend subclasses advertise PCP and retain the native AMD builders. The owned attention uses common `forward` and disables auxiliary streams, avoiding AMD's local-latent fast chain. Only BF16-style Tensor output allocation is used. Empty ranks participate in earlier cache/compressor collectives and then skip attention; zero-token decode placeholders are also skipped. Inverse RoPE still runs once, inside AMD forward_mqa.

## Replay evidence

The production module was absent in `before/report.json`: 0/7. The fixed seven scenarios then passed 7/7 in `candidate-fixed-fixture/report.json`; CUDA was never initialized. The first candidate was 6/7 because the CPU SWA metadata double omitted `block_size`; that field was repaired without changing assertions or adding scenarios. The config double also gained the required `use_v2_model_runner=True`. The existing packaging entry dictionary was synchronized before production implementation; no unit-test scenarios/assertions were added after implementation.

- Native CR2 predecessor linking: 200 iterations x eight rank layouts, two symmetric fragments per rank. These iterations cycle five fragment extents; they are not 200 independent randomized ring workloads. Closed full batches are covered; arbitrary open-ring persistence remains native and is not newly numerically proven here.
- Native common preparation and source K scheduling: 200 iterations x eight ranks, cycling four fragment extents. Actual common SWA gather/insert scheduling, local query preparation and native index K producer execute; transport and cache kernels are explicit doubles. All gathered payloads/positions/slots must match rank-major order, and local output rows must match local absolute positions.
- Production FP8 scoring: 200 independently seeded SHUFFLE caches x eight local causal query bounds. Real CPU Torch decode logits compare exactly with an independent logical-cache reference (`atol=rtol=0`); cache bytes remain unchanged and padding remains `-1`.
- Constructor namespace identity, signatures/defaults/closures and public registry idempotence/owner loss are checked. The replay does not construct/forward a complete V41 model or execute GPU metadata kernels.
- Actual local output/tail clearing checks cover zero, one and three active rows, with an explicit native attention boundary double. AMD attention/inverse-RoPE GPU numerics are not measured.

Evidence root: `tests/vllm033/.cache/v41-pcp-attention/` (ignored local artifacts). The runner records actual Core source paths and distribution metadata; `source-manifest.json` also hashes the model constructor bridge. Execute `bash tests/vllm033/run_v41_pcp_attention_cpu.sh OUTPUT` in the existing CPU-capable container with `VLLM_TEST_PYTHON=/usr/local/bin/python3.12`.

## Known limits

This record describes attention integration at commit `16e15b1`, where nonempty `engram_layer_ids` were rejected before allocation. The subsequent private TP×PCP Engram adapter is documented in `v41-pcp-engram.md`; it enables those layouts only with its supported private host-storage contract. Removing Engram layer IDs is never a valid full-model workaround. Native MRV2 admission rejects `supports_multimodal_inputs=True` at startup, even if a request contains no images. Text-only mode requires zero limits for every supported modality (for V41, `--limit-mm-per-prompt '{"image":0}'`) with `enable_mm_embeds=False`; a checkpoint may retain its unused native vision tower. This follows native `config/model.py:925-970` and `gpu/pcp_manager.py:172-173`, without changing capability flags.

Requires MRV2, eager, DCP1/PP1/DP1, source-owned FP8 index K, CR1/2, no speculative decoding, LoRA/EPLB/DBO/monokernel/KV connector/SWA bounded replay. PCP width is not whitelisted. The existing PCP1 reference flag is not required; its separate OOT class is unchanged. The HCU reference scorer uses host metadata and loops, so this is a correctness path with no performance claim.

Core and Adapter remain read-only. GPU provider/ABI, real eight-process transport and rank order, GPU metadata/cache writers, Graph, end-to-end checkpoint loading, full-model accuracy and performance are all unverified.

## Verification and results

Static parsing passed for all three production modules and the fixed replay. CPU source integration: 7/7. Existing full regression: 312 passed, 66 skipped, 14 warnings in 182.23s (`regression.log` / `regression.xml`), exit code 0. Constructor-specific independent review found no remaining construction blocker: native zero-argument super closures, private lambda globals, LLM model_cls, processor inheritance and PCP1 delegation are preserved. A separate static hash-mismatch finding was rejected after host AST line slices and container inspect.getsource matched all three production hashes and both full-file hashes; see `hash-review.json`. The reported static hash used dedented source, while production hashes raw inspect.getsource; for example dedenting the decoder changes 919c9809 to the reported 5853f2b9. No production change was required. Final independent execution review found no remaining concrete blocker in common-forward selection, serial preparation, actual-local-row slicing, padding clearing, empty-rank handling or inverse-RoPE ownership. Accuracy: not measured. Performance: not measured.
