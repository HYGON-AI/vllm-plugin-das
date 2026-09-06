# LightOp/AITER Operator Adaptation Audit for vLLM 0.25.1

Date: 2026-09-06

## Purpose

This document maps HCU operator families used by `HYGON-AI/sglang-das` to the
current vLLM HCU plugin. It is a decision record, not a claim that every SGLang
operator belongs in vLLM. Operators are accepted only after an independent
live-HCU numerical test and a same-shape performance comparison.

The audit baseline is:

- vLLM HCU plugin `origin/v0.25.1` at `88f8a07`;
- vLLM source `/models/zb/vllm_025/vllm`;
- `sglang-das` main at `0457572`;
- installed LightOp 0.6.0; and
- the AITER package installed with the current HCU runner image.

## Dispositions

| Disposition | Meaning |
|---|---|
| `already-covered` | The plugin already owns an equivalent route and no duplicate is needed. |
| `adapt-candidate` | Installed ABI and a vLLM seam exist; numerical/performance screening is required. |
| `dependency-unavailable` | The installed package does not expose the required callable ABI. |
| `no-vllm-seam` | The operator depends on SGLang-only state/layout or vLLM has no equivalent boundary. |
| `accuracy-rejected` | Live-HCU output does not meet the independent reference contract. |
| `performance-rejected` | Correct output was observed but target-shape speedup was below 5%. |
| `accepted` | Accuracy, route, fallback and performance gates passed. |

No row is marked `accepted` in the initial inventory.

## LightOp audit

| Family / public symbols | SGLang use | Current plugin mapping | Initial disposition | Reason / next evidence |
|---|---|---|---|---|
| Biased sigmoid MoE gate: `moe_fused_gate` | Grouped expert routing | `router_runtime.py` and `ops/fuse_moe_gate.py` | `already-covered` | Plugin has capability-aware scoring/renormalization fallback and master/leaf gates. |
| DeepSeek V4 gate: `moe_fused_gate_sqrtsoftplus` | Non-hash DeepSeek V4 routing | `sqrtsoftplus_routing.py` through the exact fused-top-k-bias patch | `accepted` | Live HCU accuracy passed 75/75. All 40 benchmark shapes passed the 5% gate. Hash-table routing stays official. |
| MoE alignment: `moe_align_block_size_out` | Triton/Marlin preparation | `patch_moe_align_block_size.py` | `already-covered` | Exact patch, leaf switch and native fallback already exist. |
| EP permutation: `ep_scatter`, `ep_gather` | DeepEP high-throughput dispatch/combine | `deep_gemm_utils.py` | `already-covered` | Plugin has LightOp routes plus Triton fallback. |
| EP m-index construction: `ep_build_m_indices` | Builds padded token-to-expert rows | `deep_gemm_utils.ep_scatter` already fills `m_indices` and inverse permutation together | `already-covered` | The only equivalent vLLM DeepEP stage needs both results. Its existing LightOp `ep_scatter` performs both writes in one launch; adding the standalone builder would duplicate the m-index work. The same function retains two Triton kernels when the route is disabled. |
| W16A16 Marlin MoE: `get_moe_cuda_marlin_config_w16a16`, `moe_gemm_marlin_w16a16`, `moe_sum` | BF16 Qwen/HCU MoE | Exact unquantized oracle adapter with AITER/Triton pre-pack fallback | `accepted` | Opt-in only for the measured Qwen3.6 shape and `max_num_tokens<=16`; larger batches remain on AITER/Triton because the layouts cannot be mixed safely. |
| Quantized Marlin MoE: `fused_experts_impl_fp8_marlin`, `fused_experts_impl_int8_marlin` | W8A8/FP8 expert execution | `compressed_tensors_moe_marlin.py` | `already-covered` | Plugin owns LightOp quantized Marlin routes and alignment compatibility. |
| W8A8/FP8 MoE GEMMs: `m_grouped_w8a8_gemm_*` and fused activation/quant variants | DeepEP and channel-quant MoE | `deep_gemm_moe.py`, `batched_deep_gemm_moe.py`, `dpsk_v4_deep_gemm_moe.py` | `already-covered` | Plugin has contiguous/masked and DeepSeek V4 routes. |
| W4A8 helpers and Marlin repack | SlimQuant expert execution | Plugin SlimQuant/DeepGEMM and AITER W4A8 paths | `already-covered` | Existing plugin code owns the vLLM quantization lifecycle; importing LightOp private `_lmslim_native` is not allowed. |
| Activation: `silu_and_mul_opt`, `fuse_silu_and_mul` | Dense MLP and W16A16 MoE | `ops/silu_and_mul.py` and the accepted W16 expert runtime | `already-covered` | Dense LightOp routing already existed; the W16 backend now uses the output-buffer form inside its validated fused pipeline. |
| Fused activation+quant: `lm_fuse_silu_mul_quant`, `fuse_silu_mul_fp8_quant`, EP variants | Quantized linear/MoE handoff | `ops/fuse_silu_mul_quant.py` and DeepGEMM experts | `already-covered` | Existing plugin routes cover general and EP channel-quant use. |
| RMSNorm: `rmsnorm_forward_autograd`, `fused_add_rms_norm`, Gemma RMSNorm | Transformer normalization | `ops/rms_norm.py`, `ops/gemma_rms_norm.py` | `already-covered` | Master/leaf switches and vLLM OOT registration already exist. |
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
| `per_token_quant_hip`, per-group quant, dynamic quant | FP8/INT8 inputs | `aiter_ops.py`, `aiter_runtime.py`, LightOp quant routes | `already-covered` | Current plugin selects by quantization contract and installed ABI. |
| `gemm_a8w8_*`, preshuffle GEMM, blockscale GEMM | Quantized dense layers | `aiter_runtime.py`, `aiter_ops.py` | `already-covered` | Includes tuned config discovery and Triton alternatives. |
| `aiter_moe`, `fused_moe`, W16/W8/W4 solutions | MoE execution | Unified AITER dispatch and quantized runtimes | `already-covered` | Plugin owns selection, layout generation, shuffle, scale conversion, caching and fail-closed execution. |
| Weight/scale shuffle APIs | AITER MoE preparation | Unified AITER dispatch | `already-covered` | Existing tests cover native/ASM layout consistency and no-solution fallback. |
| `topk_softmax`, biased/grouped top-k | MoE routing | `aiter_ops.py`, `aiter_runtime.py`, official router integration | `already-covered` | Plugin handles requested index dtype and routing semantics. |
| `topk_gating` | New SGLang MoE routing | Symbol absent in installed AITER | `dependency-unavailable` | Revisit only with a dependency upgrade in the same reviewed change. |
| `greedy_sample` | Sampler fast path | Symbol absent in installed AITER | `dependency-unavailable` | Existing vLLM/LightOp sampling remains authoritative. |
| `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | Qwen multimodal attention | Symbol absent in installed AITER | `dependency-unavailable` | No dormant environment route is added. |
| Flash attention, paged attention, unified attention | Prefill/decode attention | Plugin attention backend and AITER replacement module | `already-covered` | Existing selection includes classic, varlen, unified and custom modes. |
| MLA prefill/decode and sparse MQA Triton kernels | DeepSeek MLA/DSA | Plugin MLA and sparse attention runtimes | `already-covered` | Plugin already provides version-tolerant AITER module-path fallbacks. |
| Cache reshape/indexer quant and RoPE kernels | Attention cache update | `fa_utils.py`, AITER runtime, DSv4 paths | `already-covered` | Additional SGLang-only cache index layouts require a canonical vLLM seam. |
| GDN and causal-conv update | Linear attention | Plugin GDN patch and AITER replacement | `already-covered` | Qwen3.5/3.6 linear-attention support already routes these operators. |
| MHC pre/post/fused post-pre | DeepSeek V4 MHC | Plugin MHC runtime and TileLang/AITER routes | `already-covered` | Existing environment switches retain the current fallback chain. |
| FLA chunk kernels | FLA models | Exact HCU patch adapters | `already-covered` | Both chunk-delta-h and chunk-o adapters are master-gated. |
| Fused QK/RMSNorm/group quant variants | DeepSeek attention | Plugin `aiter_ops.py` probes old/new ABI | `already-covered` | No duplicate import path is needed. |
| Tuned BF16 `tgemm` | Router and attention projections | vLLM's exact unquantized GEMM seam currently selects `torch.nn.functional.linear` on gfx938 | `performance-rejected` | All 36 representative BF16 shapes were correct, but none passed the 5% gate; the installed dispatcher selected its default solution and only added overhead. |
| `batched_gemm_bf16` for DeepSeek V4 WO_A | Decode grouped projection | Plugin has the exact `torch.einsum("tgd,grd->tgr")` seam | `dependency-unavailable` | The SGLang source module `aiter.ops.triton.gemm.batched.batched_gemm_bf16` is absent from the installed AITER build. The existing plugin path remains authoritative. |
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
| AITER SiLU-and-multiply | 3/3 existing live-HCU cases passed; benchmark outputs within `rtol=0.02,atol=0.05` | 20/20 shapes slower than current LightOp by 40.58%--96.17% | Current LightOp and vLLM native routes unchanged | `performance-rejected` |
| LightOp EP m-indices | Existing combined `ep_scatter` tests cover the produced indices | Standalone timing not applicable because it duplicates work | Master/leaf LightOp `ep_scatter`; two-stage Triton fallback | `already-covered` |
| LightOp fused RMS/quant aliases | Existing live INT8 RMS+quant cases 2/2 passed | Alias adds no new kernel route | Public categorized LightOp; AITER/vLLM fallback | `already-covered` |
| Generic LightOp KV-store fusion | ABI/layout comparison rejected before kernel execution | No identical vLLM seam to benchmark | Existing vLLM attention/cache implementations | `no-vllm-seam` |
| LightOp MLA decode `ds_cat` | 4/4 live-HCU cases passed exactly, including production strides | 33/33 shapes passed; 8.84% min, 47.03% median, 77.46% max | Master/leaf route; legacy opt-out; ineligible/unavailable uses `torch.cat` | `accepted` |
| AITER tuned BF16 GEMM | 36/36 benchmark shapes matched (`max_abs=0`) | 0/36 passed; -27.51% min, -0.12% median, 0.46% max | Existing vLLM `torch.nn.functional.linear` | `performance-rejected` |
| AITER DeepSeek V4 batched BF16 GEMM | Import probe failed | Not runnable | Existing BF16 einsum | `dependency-unavailable` |

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
- The more specific DeepSeek V4 SGLang route could not be screened because
  `aiter.ops.triton.gemm.batched.batched_gemm_bf16` is not installed. Raw
  local tgemm report: `/tmp/vllm-hcu-aiter-tgemm-benchmark.json`.

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
- Sampling and top-k imports map to the existing HCU sampler/router paths;
  the installed AITER build lacks the newer `topk_gating`, `greedy_sample`,
  and fused multimodal QK symbol.
- AITER GroupNorm/Conv2D and the separate SGLang diffusion Triton kernels
  belong to SGLang's diffusion/VAE runtime, for which this vLLM LLM-serving
  plugin has no module or lifecycle seam. They are classified `no-vllm-seam`,
  not silently omitted.
- SGLang scheduler metadata and flat token-pool cache mutations are framework
  contracts rather than interchangeable device kernels. Their operators are
  rejected at the adapter boundary even when a symbol happens to import.
