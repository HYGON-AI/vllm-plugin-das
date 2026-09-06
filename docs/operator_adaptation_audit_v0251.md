# LightOp/AITER Operator Adaptation Audit for vLLM 0.25.1

Date: 2026-09-07

## Purpose

This document maps HCU operator families used by `HYGON-AI/sglang-das` to the
current vLLM HCU plugin. It is a decision record, not a claim that every SGLang
operator belongs in vLLM. Operators are accepted only after an independent
live-HCU numerical test and a same-shape performance comparison.

The audit baseline is:

- vLLM HCU plugin branch baseline `6925ca37`;
- vLLM source `/models/zb/vllm_025/vllm` at `7b108ad1`;
- `sglang-das` main at `0457572`;
- installed LightOp distribution `0.6.0+das.dtk2604`; and
- installed AITER
  `0.1.5+das185.dtk2604.torch2110.2608180853.g40a705`.

## Dispositions

| Disposition | Meaning |
|---|---|
| `already-covered` | The plugin already owns an equivalent route and no duplicate is needed. |
| `adapt-candidate` | Installed ABI and a vLLM seam exist; numerical/performance screening is required. |
| `dependency-unavailable` | The installed package does not expose the required callable ABI. |
| `no-vllm-seam` | The operator depends on SGLang-only state/layout or vLLM has no equivalent boundary. |
| `accuracy-rejected` | Live-HCU output does not meet the independent reference contract. |
| `performance-rejected` | Correct output was observed but target-shape speedup was below 5%. |
| `performance-evidence-insufficient` | The measured path omitted a mandatory cost or did not match the consumer semantics. |
| `accepted` | Accuracy, route, fallback and performance gates passed. |

The production-source inventory contains 603 records: 183 AITER imports, 243
AITER calls, two dynamic AITER references, 71 LightOp imports and 103 LightOp
calls. The corresponding all-Python inventory contains 667 records. The raw
manifests are `/tmp/sglang-das-lightop-aiter-production-0457572.tsv` and
`/tmp/sglang-das-lightop-aiter-all-python-0457572.tsv`.

## LightOp audit

| Family / public symbols | SGLang use | Current plugin mapping | Initial disposition | Reason / next evidence |
|---|---|---|---|---|
| Biased sigmoid MoE gate: `moe_fused_gate` | Grouped expert routing | `router_runtime.py` and `ops/fuse_moe_gate.py` | `already-covered` | Plugin has capability-aware scoring/renormalization fallback and master/leaf gates. |
| DeepSeek V4 gate: `moe_fused_gate_sqrtsoftplus` | Non-hash DeepSeek V4 routing | `sqrtsoftplus_routing.py` through the exact fused-top-k-bias patch | `accepted` | Live HCU accuracy passed 75/75. All 40 benchmark shapes passed the 5% gate. Hash-table routing stays official. |
| MoE alignment: `moe_align_block_size_out` | Triton/Marlin preparation | `patch_moe_align_block_size.py` | `already-covered` | Exact patch and leaf switch already exist. When the pinned vLLM `_moe_C` symbol is absent, the fixed-size Torch fallback is graph-capture tested on live HCU. |
| EP permutation: `ep_scatter`, `ep_gather` | DeepEP high-throughput dispatch/combine | `deep_gemm_utils.py` | `already-covered` | Plugin has LightOp routes plus Triton fallback. |
| EP m-index construction: `ep_build_m_indices` | Builds padded token-to-expert rows | `deep_gemm_utils.ep_scatter` already fills `m_indices` and inverse permutation together | `already-covered` | The only equivalent vLLM DeepEP stage needs both results. Its existing LightOp `ep_scatter` performs both writes in one launch; adding the standalone builder would duplicate the m-index work. The same function retains two Triton kernels when the route is disabled. |
| W16A16 Marlin MoE: `get_moe_cuda_marlin_config_w16a16`, `moe_gemm_marlin_w16a16`, `moe_sum` | BF16 Qwen/HCU MoE | Exact unquantized oracle adapter with AITER/Triton pre-pack fallback | `accepted` | Opt-in only for the measured Qwen3.6 shape and `max_num_tokens<=16`; larger batches remain on AITER/Triton because the layouts cannot be mixed safely. |
| Quantized Marlin MoE: `fused_experts_impl_fp8_marlin`, `fused_experts_impl_int8_marlin` | W8A8/FP8 expert execution | `compressed_tensors_moe_marlin.py` | `already-covered` | Plugin owns LightOp quantized Marlin routes and alignment compatibility. |
| W8A8/FP8 MoE GEMMs: `m_grouped_w8a8_gemm_*` and fused activation/quant variants | DeepEP and channel-quant MoE | `deep_gemm_moe.py`, `batched_deep_gemm_moe.py`, `dpsk_v4_deep_gemm_moe.py` | `already-covered` | Plugin has contiguous/masked and DeepSeek V4 routes. |
| W4A8 helpers and Marlin repack | SlimQuant expert execution | Plugin SlimQuant/DeepGEMM and AITER W4A8 paths | `already-covered` | Existing plugin code owns the vLLM quantization lifecycle; importing LightOp private `_lmslim_native` is not allowed. |
| Activation: `silu_and_mul_opt`, `fuse_silu_and_mul` | Dense MLP and W16A16 MoE | `ops/silu_and_mul.py` and the accepted W16 expert runtime | `already-covered` | Dense LightOp routing already existed; the W16 backend now uses the output-buffer form inside its validated fused pipeline. |
| Fused activation+quant: `lm_fuse_silu_mul_quant`, `fuse_silu_mul_fp8_quant`, EP variants | Quantized linear/MoE handoff | `ops/fuse_silu_mul_quant.py` and DeepGEMM experts | `already-covered` | Existing plugin routes cover general and EP channel-quant use. |
| RMSNorm: `rmsnorm_forward_autograd`, `fused_add_rms_norm`, Gemma RMSNorm | Transformer normalization | `ops/rms_norm.py`, `ops/gemma_rms_norm.py` | `already-covered` | Master/leaf switches and vLLM OOT registration already exist. |
| Gated RMSNorm: `lightop.norm.layer_norm_fwd_1pass_opt` | Qwen GDN output normalization and gating | Registered HCU custom op plus a worker-owned exact Qwen captured-class binding | `accepted` | Strict BF16 contiguous width-128/256 SiLU route only. All other dtypes, layouts, groups, activations, unmeasured row counts, and missing categorized exports retain vLLM Triton. |
| Generic MoE top-k: `lightop.moe.topk_softmax` | Softmax top-k routing | No production route | `dependency-unavailable` | The installed binary requires UInt64 IDs at one ABI layer but the following PyTorch layer requires Long. Int32/int64 abort in native code and uint64 raises, so no safe callable dtype exists. |
| EPLB postprocess: `topk_ids_postprocess` | SGLang logical-to-physical expert mapping | vLLM `BaseRouter` owns mapping and load recording | `no-vllm-seam` | The three-argument LightOp ABI cannot replace vLLM replica selection, atomic load recording, ubatch and EPLB lifecycle. |
| `fused_rms_norm_contiguous` | MiniMax Q/K RMSNorm | Existing RMSNorm/MiniMax TP implementations | `performance-evidence-insufficient` | Initial screening accidentally used Gemma `(1+weight)` semantics and excluded the mandatory derived-weight cost. It is not accepted as MiniMax evidence or added as a route. |
| Fused RMS+dynamic quant: `lm_faster_rmsquant`, `rms_norm_per_token_fp8_quant` | Linear/communicator handoff | `ops/fuse_rms_norm_quant.py` uses categorized `rms_norm_dynamic_per_token_quant`; AITER fused variants also exist | `already-covered` | Installed `lm_faster_rmsquant` is an allocation wrapper around the private LMSlim dynamic-per-token operator with the same input/residual/update/scale contract. The plugin deliberately uses the public categorized LightOp wrapper and already has live INT8 accuracy plus AITER/vLLM INT8/FP8 fallbacks. |
| Per-token quant: `per_token_quant_int8`, `per_token_quant_fp8`, `per_token_group_quant_fp8` | Dense and MoE inputs | `int8_runtime.py`, `lightop_fp8_runtime.py`, AITER group quant paths | `already-covered` | Plugin has dynamic INT8/FP8 and per-group coverage. |
| Dense W8A8 GEMM: `hipblaslt_w8a8_gemm`, channelwise GEMM | Compressed-tensor linear | `int8_runtime.py`, `runtime_compat/scaled_mm.py` | `already-covered` | Existing code validates required LightOp symbols and scale semantics. |
| Sparse MLA logits/top-k: `mqa_logits`, `paged_mqa_logits`, `top_k_per_row_*` | DSA/DeepSeek V4 indexer | `rocm_aiter_mla_sparse.py` | `already-covered` | Plugin already selects LightOp and AITER/Triton alternatives by availability and mode. |
| DeepSeek V4 cache/attention fusion | Compressor, cache insert, sparse attention | DeepSeek V4 attention/model runtime and DSpark paths | `already-covered` | Plugin has HCU-owned DSv4 attention and cache routes. Exact SGLang scheduler kernels are not duplicated. |
| Generic RMS+RoPE+KV-store: `rms_rotary_embedding_fuse_with_kv_store` | Hunyuan, Qwen and Bailing model forwards | No generic plugin hook | `no-vllm-seam` | SGLang passes its flat token-pool buffers and `out_cache_loc`; vLLM 0.25.1 attention owns paged KV tensors and block-slot mapping at a different boundary. There is no exact canonical adapter seam, and model-name forward replacement is disallowed. |
| Split QKV+RMS+RoPE+KV-store quant | Qwen2 model forward | Plugin GLM/DSv4-specific fused routes | `no-vllm-seam` | The SGLang call combines its projection split with token-pool cache mutation. vLLM exposes neither the same buffer layout nor the same atomic mutation contract. Existing GLM/DSv4 routes remain model-contract-specific. |
| `fused_metadata_kernel_general` | SGLang attention metadata construction | vLLM scheduler metadata differs | `no-vllm-seam` | SGLang request/scheduler metadata is not ABI-compatible with vLLM 0.25.1. |
| `ds_cat` | SGLang attention concatenation | FlashMLA decode exact seam in `lightop_concat_runtime.py` | `accepted` | The former manual helper was replaced at the one exact vLLM call site. Master/leaf/legacy gates, strict BF16 `512+64` layout checks, a registered custom op/fake, dependency fallback and native `torch.cat` fallback now own the route. |
| Sampling namespace | SGLang top-k/top-p sampling | `ops/topk_topp_sample.py` | `already-covered` | Plugin already owns a LightOp sampling route with fallback. |
| LightOp custom all-reduce primitives | SGLang communicator | Plugin owns its HCU communicator module | `already-covered` | Treat communicator lifecycle as plugin infrastructure, not a new operator migration. |

## AITER audit

| Family / public symbols | SGLang use | Current plugin mapping | Initial disposition | Reason / next evidence |
|---|---|---|---|---|
| `silu_and_mul` / Triton MoE activation | Dense and MoE activation | Current dense route is LightOp; diagnostic AITER accuracy test exists | `performance-rejected` | Installed two-argument AITER ABI passed BF16 accuracy but was slower than both the current plugin LightOp route and vLLM native across every screened shape. |
| `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add`, fused RMS quant | Normalization and quant handoff | `aiter_ops.py` and LightOp norm routes | `already-covered` | vLLM AITER replacement already exposes fused dynamic quant and residual variants. |
| `layernorm2d_fwd` | Generic LayerNorm | vLLM `LayerNorm` | `accuracy-rejected` | vLLM uses BF16 activation with FP32 weight/bias; installed AITER produced NaN/Inf or extreme values for all 80 screened cases and only worked when parameter dtype matched activation dtype. |
| `per_token_quant_hip`, per-group quant, dynamic quant | FP8/INT8 inputs | `aiter_ops.py`, `aiter_runtime.py`, LightOp quant routes | `already-covered` | Current plugin selects by quantization contract and installed ABI. |
| `gemm_a8w8_*`, preshuffle GEMM, blockscale GEMM | Quantized dense layers | `aiter_runtime.py`, `aiter_ops.py` | `already-covered` | Includes tuned config discovery and Triton alternatives. |
| `aiter_moe`, `fused_moe`, W16/W8/W4 solutions | MoE execution | Unified AITER dispatch and quantized runtimes | `already-covered` | Plugin owns selection, layout generation, shuffle, scale conversion, caching and fail-closed execution. |
| Weight/scale shuffle APIs | AITER MoE preparation | Unified AITER dispatch | `already-covered` | Existing tests cover native/ASM layout consistency and no-solution fallback. |
| `topk_softmax`, biased/grouped top-k | MoE routing | `aiter_ops.py`, `aiter_runtime.py`, official router integration | `already-covered` | Plugin handles requested index dtype and routing semantics. |
| `topk_gating` | New SGLang MoE routing | Symbol absent in installed AITER | `dependency-unavailable` | Revisit only with a dependency upgrade in the same reviewed change. |
| `greedy_sample` | Sampler fast path | Symbol absent in installed AITER | `dependency-unavailable` | Existing vLLM/LightOp sampling remains authoritative. |
| `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | Qwen multimodal attention | Categorized implementation is installed, but the SGLang root export is absent and vLLM owns a different paged-cache mutation contract | `no-vllm-seam` | Do not bypass vLLM's cache-slot, quantization, and graph-capture ownership with a model-local SGLang mutation. |
| Flash attention, paged attention, unified attention | Prefill/decode attention | Plugin attention backend and AITER replacement module | `already-covered` | Existing selection includes classic, varlen, unified and custom modes. |
| MLA prefill/decode and sparse MQA Triton kernels | DeepSeek MLA/DSA | Plugin MLA and sparse attention runtimes | `already-covered` | Plugin already provides version-tolerant AITER module-path fallbacks. |
| Cache reshape/indexer quant and RoPE kernels | Attention cache update | `fa_utils.py`, AITER runtime, DSv4 paths | `already-covered` | Additional SGLang-only cache index layouts require a canonical vLLM seam. |
| GDN and causal-conv update | Linear attention | Plugin GDN patch and AITER replacement | `already-covered` | Qwen3.5/3.6 linear-attention support already routes these operators. |
| `fused_recurrent_gated_delta_rule_packed_decode` | Qwen GDN decode | Exact vLLM FLA seam | `performance-rejected` | Positive 1-based indices passed 12/12 FP32 comparisons, but vLLM treats state index 0 as invalid while AITER mutates slot 0. The required index remap made all 12 shapes slower; no route was restored. |
| `gemm_afp4wfp4_pre_quant` and batched variant | MXFP4 dense/MLA GEMM | Quark/MXFP4 seams | `dependency-unavailable` | gfx938 maps to BW200B but the package has no BW200B prequant configs. Explicit MI350 configs fail Triton lowering with `Unsupported DotScaleOp`; neither kernel executed. |
| MHC pre/post/fused post-pre | DeepSeek V4 MHC | Plugin MHC runtime and TileLang/AITER routes | `already-covered` | Existing environment switches retain the current fallback chain. |
| FLA chunk kernels | FLA models | Exact HCU patch adapters | `already-covered` | Both chunk-delta-h and chunk-o adapters are master-gated. |
| Fused QK/RMSNorm/group quant variants | DeepSeek attention | Plugin `aiter_ops.py` probes old/new ABI | `already-covered` | No duplicate import path is needed. |
| Tuned BF16 `tgemm` | Router and attention projections | vLLM's exact unquantized GEMM seam currently selects `torch.nn.functional.linear` on gfx938 | `performance-rejected` | All 36 representative BF16 shapes were correct, but none passed the 5% gate; the installed dispatcher selected its default solution and only added overhead. |
| `batched_gemm_bf16` for DeepSeek V4 WO_A | Decode grouped projection | Plugin has the exact `torch.einsum("tgd,grd->tgr")` seam | `performance-rejected` | The SGLang import path is absent, but the installed replacement path was accuracy-screened with an explicit BW200B tile. All 28 TP/decode shapes were 2.04x--5.24x slower after required layout conversions. |
| `fused_add_rmsnorm_pad`, `fused_clamp_act_mul`, fused MLA cache kernel | GPT-OSS/DeepSeek specialized forwards | Existing plugin normalization/clamp/cache boundaries differ | `dependency-unavailable` | These SGLang-imported AITER modules are absent from the pinned runtime. No environment switch is registered for an unavailable dependency. Clamp SwiGLU itself remains covered by the existing vLLM `_C` and LightOp MoE paths. |
| Custom all-reduce | Distributed communication | Plugin whole-module communicator replacement | `already-covered` | Lifecycle and topology behavior are already HCU-owned. |
| AITER Conv2D for multimodal tower | Kimi vision path | No generic vLLM HCU vision hook in this scope | `no-vllm-seam` | This audit targets general LLM serving operators, not multimodal model rewrites. |

## Triton alternatives

The accepted route never removes the existing alternative:

- sqrt-softplus routing falls back to vLLM's official ROCm/Triton/custom-op
  implementation, including every hash-routing layer;
- unquantized BF16 MoE falls back before packing to the current AITER or Triton
  oracle selection;
- dense SiLU keeps the vLLM implementation when the master or leaf switch is
  disabled;
- EP scatter/gather/m-index work retains the plugin's existing Triton kernels;
- fused norm/quant retains current vLLM/AITER paths when LightOp is ineligible;
- Qwen gated RMSNorm retains vLLM's canonical Triton implementation for every
  disabled or ineligible call;
  and
- sparse attention retains the existing AITER Triton kernels where LightOp is
  unavailable.

## Validation record

Results are added only after commands complete on an idle HCU. The benchmark
tool is `tools/benchmark_sglang_operator_candidates.py`; test and report paths
will be recorded here and summarized in
`docs/sglang_operator_adaptation_validation.md`.

| Candidate | Accuracy | Benchmark | Route/fallback | Final disposition |
|---|---|---|---|---|
| LightOp sqrt-softplus gate | 75/75 passed | 40/40 shapes passed; 6.83% min, 22.61% median, 44.92% max | Master/leaf enabled route; hash and ineligible inputs use official vLLM | `accepted` |
| LightOp W16A16 Marlin MoE | 6/6 passed, including Qwen3.6 expert shape | Accepted M=1/2/4/8/16: 6.28% min vs fastest fallback; larger M rejected | Opt-in master/leaf route; shape/range/config failures delegate before packing | `accepted` |
| LightOp Qwen gated RMSNorm | 40/40 BF16 width-128/256 shapes passed against FP32 and vLLM; 14 live routed cases plus one FP32 fallback case | Every screened shape passed; width-128 median speedup 74.62%--81.06%; width-256 observed speedup 77.45%--85.39% | Master/leaf route with exact dependency/device/dtype/layout/group/activation/row guards; Qwen-only captured-class binding; vLLM Triton fallback | `accepted` |
| LightOp generic top-k | No executable output-index dtype in installed ABI | Not timed after ABI gate failed | Existing vLLM/AITER/Torch routing | `dependency-unavailable` |
| AITER LayerNorm2D | 0/80 for vLLM's BF16-input/FP32-parameter contract | Not timed after accuracy gate failed | Existing vLLM LayerNorm | `accuracy-rejected` |
| AITER fused recurrent packed decode | 12/12 positive-index cases passed; index 0 violates vLLM state ABI | ABI-safe route passed 0/12 performance shapes | Existing vLLM FLA Triton kernel | `performance-rejected` |
| AITER MXFP4 prequant GEMMs | Kernel never compiled on gfx938 | Not timed | Existing Quark/Triton paths | `dependency-unavailable` |
| AITER SiLU-and-multiply | 3/3 existing live-HCU cases passed; benchmark outputs within `rtol=0.02,atol=0.05` | 20/20 shapes slower than current LightOp by 40.58%--96.17% | Current LightOp and vLLM native routes unchanged | `performance-rejected` |
| LightOp EP m-indices | Existing combined `ep_scatter` tests cover the produced indices | Standalone timing not applicable because it duplicates work | Master/leaf LightOp `ep_scatter`; two-stage Triton fallback | `already-covered` |
| LightOp fused RMS/quant aliases | Existing live INT8 RMS+quant cases 2/2 passed | Alias adds no new kernel route | Public categorized LightOp; AITER/vLLM fallback | `already-covered` |
| Generic LightOp KV-store fusion | ABI/layout comparison rejected before kernel execution | No identical vLLM seam to benchmark | Existing vLLM attention/cache implementations | `no-vllm-seam` |
| LightOp MLA decode `ds_cat` | 4/4 live-HCU cases passed exactly, including production strides | 33/33 shapes passed; 8.84% min, 47.03% median, 77.46% max | Master/leaf route; legacy opt-out; ineligible/unavailable uses `torch.cat` | `accepted` |
| AITER tuned BF16 GEMM | 36/36 benchmark shapes matched (`max_abs=0`) | 0/36 passed; -27.51% min, -0.12% median, 0.46% max | Existing vLLM `torch.nn.functional.linear` | `performance-rejected` |
| AITER DeepSeek V4 batched BF16 GEMM | 6/6 live-HCU cases passed against FP32 | 0/28 passed; complete path was 2.04x--5.24x slower | Existing BF16 einsum | `performance-rejected` |

### LightOp sqrt-softplus evidence

- Device: BW1100, `gfx938:sramecc+:xnack-`.
- Runtime: PyTorch 2.11.0, vLLM 0.25.1, LightOp 0.6.0.
- Accuracy seed: 20260906.
- Accuracy shapes: expert counts 256/384, top-k 6/8/16, token counts
  1/33/128, renormalization on/off, routed scale 1.0/1.5, plus constant
  -80/0/80 inputs.
- Accuracy result: 75 passed. Expert IDs matched exactly after order
  normalization; weights matched the independent FP32 reference with
  `rtol=1e-5`, `atol=1e-6`; all outputs were finite and inputs unchanged.
- The first run exposed that LightOp's
  `apply_routed_scaling_factor_on_output=False` defers the scale. A direct
  two-value experiment confirmed that `True` is required to match vLLM's
  router-weight contract. The final 1.5-scale matrix passed after this single
  semantic correction; tolerances were not relaxed.
- Benchmark: 50 warmups, 200 iterations, 7 repeats per shape. Expert counts
  256/384, top-k 6/8, token counts 1/2/4/8/16/32/64/128/256/1024.
- All 40 shapes exceeded 5% after including vLLM's three upstream output
  allocations in the candidate path. Overall speedup was 6.83% minimum,
  22.61% median and 44.92% maximum. The DeepSeek V4 E=256/top-k=6 family was
  6.83% minimum, 12.08% median and 43.53% maximum.
- Raw local report: `/tmp/vllm-hcu-sqrtsoftplus-benchmark.json`.

### LightOp W16A16 Marlin MoE evidence

- Device/runtime: BW1100 (`gfx938:sramecc+:xnack-`), PyTorch 2.11.0,
  vLLM 0.25.1, LightOp 0.6.0 and the installed HCU AITER build.
- Accuracy seed: 20260906. Six live-HCU cases passed: M=1/7/33/128 with
  eight experts, an exact Qwen3.6 expert shape
  `E=256,K=2048,N=512,top-k=8,M=4`, and a zero-input case. Results matched
  an independent FP32 per-token/per-expert PyTorch reference at
  `rtol=0.03`, `atol=0.03`; outputs were finite and sampled canonical source
  weights were unchanged after packing.
- The real plugin lifecycle was also exercised in import-coordinator order:
  the HCU oracle selected `HCU_LIGHTOP_W16A16`, installed packed parameter
  shapes `[256,128,16384]` and `[256,32,32768]`, constructed a vLLM modular
  `FusedMoEKernel`, and executed through vLLM's workspace manager.
- Qwen3.6 model startup exposed that its model module can capture
  `routed_experts.UnquantizedFusedMoEMethod` before the package-level factory
  wrapper runs. The final patch validates and replaces that exact captured
  symbol through the worker patch lifecycle; a regression test covers both
  package and captured bindings. The real model log then showed the LightOp
  packing marker and no Triton unquantized-MoE selection.
- Benchmark: 20 warmups, 100 iterations and 7 repeats for every M in
  1/2/4/8/16/32/64/128/256 at the Qwen3.6 shape. The report compares the same
  LightOp result with both vLLM Triton and the plugin's shuffled ASM AITER
  route and checks BF16 output agreement before timing.
- For M=1/2/4/8/16, LightOp was 31.08%--68.22% faster than Triton and
  6.28%--71.35% faster than AITER. At M=32/64/128 the AITER advantage reduced
  the LightOp gain below 5%; at M=256 LightOp was 7.51% slower than AITER.
  Since these backends require incompatible permanent weight layouts, the
  accepted oracle profile is deliberately limited to
  `E=256,K=2048,N=512,top-k=8,max_num_tokens<=16`. Every other configuration
  delegates to AITER/Triton before parameter replacement.
- Raw local report: `/tmp/vllm-hcu-w16a16-benchmark.json`.

### LightOp Qwen gated RMSNorm evidence

- The accepted seam is `RMSNormGated.forward_cuda`, before vLLM makes
  non-contiguous inputs contiguous. The Qwen GDN module imports that class by
  value, so an exact worker import callback first validates the canonical
  captured class and then rebinds only that local symbol to the HCU subclass
  before model construction. Other RMSNormGated consumers remain canonical.
  The implementation imports only the categorized
  `lightop.norm.layer_norm_fwd_1pass_opt` callable.
- The route requires BF16 contiguous `x`, `z`, and weight; two-dimensional
  `[M,128]` or `[M,256]` inputs; no bias; effective group size equal to the
  last dimension; gate after RMSNorm; SiLU; and one of the 20 measured row
  counts. The master switch and
  `VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED` must both be enabled.
- Across both verified widths and the Qwen3.6 token-head geometries, all 40
  shapes passed an independent FP32 reference at `rtol=0.02,atol=0.05`;
  inputs and parameters were unchanged. Width 256 was bit-identical to vLLM
  on 19/20 shapes and had maximum absolute difference 0.03125 at M=1024.
- The width-128 stable runs used 20 warmups, 100 iterations and seven paired
  alternating repeats, including allocations, device context, pybind dispatch
  and launch. Every case passed the 5% gate. The Qwen3.6-35B prefill geometry
  M=512 improved from 65.930 us to 12.887 us median (80.434%). The width-256
  confirmation used the same warmup/iteration/repeat counts but sequential
  baseline/candidate timing and persisted aggregate medians rather than raw
  samples; its 77.45%--85.39% range is reported as observational evidence.
- Raw reports:
  `/tmp/vllm-hcu-norm-candidates-hcu5.json`,
  `/tmp/lightop-layer-norm-fwd-qwen-exact-hcu4.json`, and
  `/tmp/lightop-layer-norm-fwd-qwen-token-heads-hcu4.json`, and
  `/tmp/lightop-layer-norm-fwd-qwen-width256-hcu3.json`.

### Rejected symbol-level candidates

- AITER recurrent decode: `/tmp/fused_recurrent_aiter_hcu4_0457572.json`.
  AITER treats only negative state indices as invalid, while vLLM also treats
  zero as `NULL_BLOCK_ID`. A safe `where(index > 0, index, -1)` remap made the
  candidate 2.90%--28.25% slower and no shape passed the gate.
- LightOp top-k:
  `/tmp/vllm-hcu-lightop-topk-softmax-screening.json`. The binary's UInt64
  assertion conflicts with its downstream Long requirement; eligible calls
  can terminate the process and therefore cannot be wrapped with fallback.
- AITER LayerNorm:
  `/tmp/vllm-hcu-norm-candidates-hcu5.json`. Mixed BF16 activation and FP32
  affine parameters produced non-finite/extreme output in all 80 cases.
- AITER MXFP4 prequant:
  `/tmp/vllm-hcu-aiter-mxfp4-hcu7.json`. Default BW200B configs are absent;
  explicit MI350 configs fail at `DotScaleOp` lowering on gfx938.

### AITER SiLU screening evidence

- Device/runtime: the same BW1100/PyTorch/vLLM environment; installed AITER
  exposes `(output, input)` while the newer SGLang source can also pass a
  clamp limit. The installed ABI was tested without inventing an unsupported
  argument.
- Existing independent accuracy cases passed for token counts 1/16/128. The
  benchmark additionally checked hidden sizes 512/2048 and token counts
  1/2/4/8/16/32/64/128/256/1024 before every timing.
- With 50 warmups, 200 iterations and 7 repeats, AITER was 40.58%--96.17%
  slower than the plugin's current categorized LightOp kernel and
  5.20%--26.18% slower than vLLM native. No production switch or dormant route
  was added. Raw local report:
  `/tmp/vllm-hcu-aiter-silu-benchmark.json`.

### LightOp MLA decode concat evidence

- Four live-HCU cases matched `torch.cat` bit-for-bit and preserved both
  inputs. They cover token/head pairs `(1,8)`, `(33,16)`, `(768,32)` and the
  non-contiguous FlashMLA strides used by the production decode path.
- The stable benchmark used those production strides, head counts 8/16/32,
  token counts 1/2/4/8/16/32/64/128/256/512/768, 50 warmups, 500 iterations
  and 7 repeats. All 33 shapes passed the 5% gate: 8.84% minimum, 47.03%
  median and 77.46% maximum speedup over allocating `torch.cat`.
- The accepted route is limited to BF16 three-dimensional `512+64` MLA query
  parts, matching device/leading dimensions, unit last stride, and fewer than
  1024 rows. `VLLM_HCU_USE_CUSTOM_OPS=0`, the new leaf switch, the legacy
  opt-out, unsupported inputs, or a missing categorized LightOp symbol use
  `torch.cat`; an eligible kernel failure is not hidden.
- Raw local report: `/tmp/vllm-hcu-mla-decode-cat-benchmark.json`.

### AITER BF16 GEMM screening evidence

- `tgemm.mm` was exercised through the installed public dispatcher at input
  and output dimensions `4096x4096`, `4096x1024`, `2048x4096`, and
  `2048x512`, with token counts 1/2/4/8/16/32/64/128/256. All 36 results
  matched the BF16 `torch.nn.functional.linear` baseline exactly.
- With 20 warmups, 100 iterations and 7 repeats, no shape passed the 5%
  performance gate. Speedup ranged from -27.51% to 0.46%, with a -0.12%
  median. The gfx938 AITER dispatcher reported its default solution, so
  enabling the generic vLLM tgemm route would only add dispatch overhead.
- Raw local tgemm report: `/tmp/vllm-hcu-aiter-tgemm-benchmark.json`.

### AITER DeepSeek-V4 WO_A batched GEMM screening evidence

- The SGLang path
  `aiter.ops.triton.gemm.batched.batched_gemm_bf16` remains absent, while the
  current AITER 0.1.5 package provides
  `aiter.ops.triton.batched_gemm_bf16.batched_gemm_bf16`. Its default lookup
  has no `BW200B-BATCHED_GEMM-A16W16.json`, so the audit used an explicit
  conservative tile rather than treating importability as support.
- Six live-HCU accuracy cases passed against an independent FP32 reference at
  the production `K=4096,N=1024` dimensions, covering `G=8/4/2/1` and decode
  token counts 1/8/16/64. The test declared `rtol=0.02,atol=0.05` before
  execution; inputs and weights were unchanged.
- The repeated benchmark covered all 28 combinations of `G=8/4/2/1` and
  `M=1/2/4/8/16/32/64`, including input transpose/contiguous, output
  allocation, kernel execution, and output transpose/contiguous. Zero shapes
  passed the 5% gate. Candidate latency was 2.04x--5.24x the existing einsum
  latency (median 4.02x), so no production route or dormant switch is added.
  Screening used the predeclared `rtol=0.02,atol=0.5`; the largest absolute
  difference from the independent FP32 result was 0.9863 and remained within
  the combined tolerance for its output magnitude.
- Exact benchmark command:

  ```bash
  HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
    PYTHONPATH=/models/zb/vllm_025/vllm:. \
    python3 tools/benchmark_sglang_operator_candidates.py \
      aiter-batched-gemm-bf16 --warmup 20 --iterations 100 --repeats 7 \
      --json-output /tmp/vllm-hcu-aiter-batched-gemm-bf16.json
  ```

- Raw local report:
  `/tmp/vllm-hcu-aiter-batched-gemm-bf16.json`.

## Source-import coverage notes

The inventory was generated from every Python `lightop*` and `aiter*` import
under the referenced SGLang checkout, then grouped by execution contract. In
addition to the rows above:

- FP4/MXFP4, W4A8, W8A8/FP8 GEMMs, preshuffle/layout helpers, MoE solutions,
  per-token/group quantization and fused activation-quant imports map to the
  plugin's existing quantization and unified AITER/LightOp MoE runtimes.
- Flash/paged/unified attention, MLA prefill/decode, sparse MQA logits,
  indexer-cache quantization, RoPE, GDN/FLA, MHC and custom all-reduce imports
  map to existing HCU-owned attention, linear-attention and communicator
  replacements, each retaining its current fallback.
- Sampling and most top-k imports map to existing HCU sampler/router paths.
  LightOp `topk_softmax` is separately dependency-rejected, and
  `topk_ids_postprocess` has no vLLM EPLB ownership seam. The installed AITER
  build lacks `topk_gating` and `greedy_sample`. The categorized multimodal QK
  callable is installed, but its root export is absent and its cache mutation
  contract still has no vLLM seam.
- AITER GroupNorm/Conv2D and the separate SGLang diffusion Triton kernels
  belong to SGLang's diffusion/VAE runtime, for which this vLLM LLM-serving
  plugin has no module or lifecycle seam. They are classified `no-vllm-seam`,
  not silently omitted.
- SGLang scheduler metadata and flat token-pool cache mutations are framework
  contracts rather than interchangeable device kernels. Their operators are
  rejected at the adapter boundary even when a symbol happens to import.
