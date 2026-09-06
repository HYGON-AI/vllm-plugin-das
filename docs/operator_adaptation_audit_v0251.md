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
| EP m-index construction: `ep_build_m_indices` | Builds padded token-to-expert rows | No standalone plugin call | `adapt-candidate` | Installed symbol exists. Must first prove a vLLM DeepEP stage exposes identical sorted-token/alignment inputs. |
| W16A16 Marlin MoE: `get_moe_cuda_marlin_config_w16a16`, `moe_gemm_marlin_w16a16`, `moe_sum` | BF16 Qwen/HCU MoE | Exact unquantized oracle adapter with AITER/Triton pre-pack fallback | `accepted` | Opt-in only for the measured Qwen3.6 shape and `max_num_tokens<=16`; larger batches remain on AITER/Triton because the layouts cannot be mixed safely. |
| Quantized Marlin MoE: `fused_experts_impl_fp8_marlin`, `fused_experts_impl_int8_marlin` | W8A8/FP8 expert execution | `compressed_tensors_moe_marlin.py` | `already-covered` | Plugin owns LightOp quantized Marlin routes and alignment compatibility. |
| W8A8/FP8 MoE GEMMs: `m_grouped_w8a8_gemm_*` and fused activation/quant variants | DeepEP and channel-quant MoE | `deep_gemm_moe.py`, `batched_deep_gemm_moe.py`, `dpsk_v4_deep_gemm_moe.py` | `already-covered` | Plugin has contiguous/masked and DeepSeek V4 routes. |
| W4A8 helpers and Marlin repack | SlimQuant expert execution | Plugin SlimQuant/DeepGEMM and AITER W4A8 paths | `already-covered` | Existing plugin code owns the vLLM quantization lifecycle; importing LightOp private `_lmslim_native` is not allowed. |
| Activation: `silu_and_mul_opt`, `fuse_silu_and_mul` | Dense MLP and W16A16 MoE | `ops/silu_and_mul.py` and the accepted W16 expert runtime | `already-covered` | Dense LightOp routing already existed; the W16 backend now uses the output-buffer form inside its validated fused pipeline. |
| Fused activation+quant: `lm_fuse_silu_mul_quant`, `fuse_silu_mul_fp8_quant`, EP variants | Quantized linear/MoE handoff | `ops/fuse_silu_mul_quant.py` and DeepGEMM experts | `already-covered` | Existing plugin routes cover general and EP channel-quant use. |
| RMSNorm: `rmsnorm_forward_autograd`, `fused_add_rms_norm`, Gemma RMSNorm | Transformer normalization | `ops/rms_norm.py`, `ops/gemma_rms_norm.py` | `already-covered` | Master/leaf switches and vLLM OOT registration already exist. |
| Fused RMS+dynamic quant: `lm_faster_rmsquant`, `rms_norm_per_token_fp8_quant` | Linear/communicator handoff | `ops/fuse_rms_norm_quant.py` uses `rms_norm_dynamic_per_token_quant`; AITER fused variants also exist | `adapt-candidate` | Compare exact residual mutation, scale layout and dtype contracts. Classify as already covered if only an ABI alias. |
| Per-token quant: `per_token_quant_int8`, `per_token_quant_fp8`, `per_token_group_quant_fp8` | Dense and MoE inputs | `int8_runtime.py`, `lightop_fp8_runtime.py`, AITER group quant paths | `already-covered` | Plugin has dynamic INT8/FP8 and per-group coverage. |
| Dense W8A8 GEMM: `hipblaslt_w8a8_gemm`, channelwise GEMM | Compressed-tensor linear | `int8_runtime.py`, `runtime_compat/scaled_mm.py` | `already-covered` | Existing code validates required LightOp symbols and scale semantics. |
| Sparse MLA logits/top-k: `mqa_logits`, `paged_mqa_logits`, `top_k_per_row_*` | DSA/DeepSeek V4 indexer | `rocm_aiter_mla_sparse.py` | `already-covered` | Plugin already selects LightOp and AITER/Triton alternatives by availability and mode. |
| DeepSeek V4 cache/attention fusion | Compressor, cache insert, sparse attention | DeepSeek V4 attention/model runtime and DSpark paths | `already-covered` | Plugin has HCU-owned DSv4 attention and cache routes. Exact SGLang scheduler kernels are not duplicated. |
| Generic RMS+RoPE+KV-store: `rms_rotary_embedding_fuse_with_kv_store` | Hunyuan, Qwen and Bailing model forwards | No generic plugin hook | `adapt-candidate` | Installed ABI must be compared with vLLM paged-cache layout and mutation contract. Model-name patching is not acceptable. |
| Split QKV+RMS+RoPE+KV-store quant | Qwen2 model forward | Plugin GLM/DSv4-specific fused routes | `adapt-candidate` | Accept only if a canonical vLLM attention seam represents the same layout; otherwise `no-vllm-seam`. |
| `fused_metadata_kernel_general` | SGLang attention metadata construction | vLLM scheduler metadata differs | `no-vllm-seam` | SGLang request/scheduler metadata is not ABI-compatible with vLLM 0.25.1. |
| `ds_cat` | SGLang attention concatenation | Only non-production plugin diagnostic references it | `adapt-candidate` | Compare against vLLM concatenate call sites; do not patch global `torch.cat`. |
| Sampling namespace | SGLang top-k/top-p sampling | `ops/topk_topp_sample.py` | `already-covered` | Plugin already owns a LightOp sampling route with fallback. |
| LightOp custom all-reduce primitives | SGLang communicator | Plugin owns its HCU communicator module | `already-covered` | Treat communicator lifecycle as plugin infrastructure, not a new operator migration. |

## AITER audit

| Family / public symbols | SGLang use | Current plugin mapping | Initial disposition | Reason / next evidence |
|---|---|---|---|---|
| `silu_and_mul` / Triton MoE activation | Dense and MoE activation | Current dense route is LightOp; diagnostic AITER accuracy test exists | `adapt-candidate` | Benchmark actual model shapes and clamp semantics against current selected route. |
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
| Tuned BF16/batched GEMM | Router and attention projections | Some model-specific uses remain standard vLLM linear calls | `adapt-candidate` | Screen only at an exact vLLM linear/attention seam; do not patch global matmul. |
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
| AITER SiLU-and-multiply | Historical diagnostic only | Not run | Current route is LightOp | `adapt-candidate` |
| LightOp EP m-indices | Not run | Not run | Seam review pending | `adapt-candidate` |
| LightOp fused RMS/quant aliases | Not run | Not run | Semantic comparison pending | `adapt-candidate` |
| Generic LightOp KV-store fusion | Not run | Not run | Seam review pending | `adapt-candidate` |

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
