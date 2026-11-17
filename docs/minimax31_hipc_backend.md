# MiniMax M3.1 DeepGEMM HIPC routed backend

The routed local-expert core now has an executable DeepGEMM HIPC path. Select it in the derived model quantization configuration with hcu_minimax_routed_backend=deepgemm_hipc_v1. The MiniMax fixed-count DeepEP runner is required; shared experts and outer routed scaling retain their existing owners. A config without the field retains Bolt. HCU_MINIMAX_ROUTED_BACKEND, when supplied, must agree with the configuration; an environment-only layout switch is rejected.

## Weight and runtime contract

The new signed-high-nibble-first-hipc-split-v1 layout starts from raw checkpoint split gate/up weights. The loader calls DeepGEMM pack_w4a8_moe_hipc_weight once for w13 and w2 before publication. Raw FP32 per-channel scales remain unchanged: HIPC supplies the factor 16 internally. The fresh native packer requests the loader device context. IPC restoration rebuilds Python metadata without repacking, moving or changing weight storage.

Use a new producer/config identity. The task prepare_wheel_config.py accepts --routed-backend deepgemm_hipc_v1 and binds the backend in the effective config hash. It additionally fingerprints this HIPC module, the installed DeepGEMM packer/GEMM Python sources and the exact Python 3.12 native extension. Never consume an old interleaved Bolt cache under the HIPC configuration.

The full route path sorts local IDs, pads each expert to 128 rows, copies activations/weights with explicit 64-bit pointer products and records the inverse route mapping. It then performs input INT8 quantization, two DeepGEMM m_grouped_w4a8_gemm_nt_contiguous_hipc calls with MODE1006, and native BF16-rounded OAI/INT8. Processing uses 32768-row chunks. Route weights multiply FP32 activation scales before GEMM2's BF16 store; received-token output uses the existing sequential FP32 top-k sum. Input hidden states and producer weights are not mutated.

## Necessary numerical checks

The first full-path test failed on real expert input seed632: relative L2 0.00459482. The original failure and intermediate tensors are retained. Stage diagnosis found four input INT8 values different by one due to x*(127/absmax) versus the stock x*reciprocal(absmax/127) FP32 order. Given actual quantized inputs and scales, both HIPC GEMMs were BF16 bit-exact against independent CPU integer/FP64 calculations, including route weight timing. OAI had one adjacent-integer disagreement at a bounded FP32 half-integer, with identical scales.

The product input quantizer now preserves reciprocal-after-scale semantics. Direct q/scale checks are bit-exact against the independent CPU implementation. With the original relative L2 <=0.0031 and cosine >=0.9999967 gates unchanged:

| Input | Relative L2 | Cosine |
|---|---:|---:|
| Synthetic H512/I256 | 0 | 1 |
| Real layer3 experts0/1, seed631, H6144/I3072 | 0.000564952 | 0.999999840 |
| Real layer3 experts0/1, seed632, H6144/I3072 | 0.001437634 | 0.999998969 |

All cases preserved the original hidden tensor; zero and invalid routes returned zero. Peak allocated memory for these bounded single-card checks was 302399488 bytes. These are module correctness results, not whole-model quality or target performance claims.

A saved full product-path trace confirms two actual DeepGEMM WASP BM128/BN512/BK64 kernels for the current image's MODE1006, along with pad/quant/native OAI/restore. The historical SGLang Marlin template name is not the current image's kernel name. The initial trace parser expected the historical name; the saved trace was reclassified offline without another GPU run.

Evidence: task_dir/support_minmax_31_pcp8_ep8_perf/prototypes/sol/hipc_v1/{correctness_result.json,stage_seed632.json,product_kernel_result.json,loader_ipc_cpu_result.json,config_cpu_result.json}. CPU regression: python tests/model_executor/test_minimax31_hipc_contracts.py.

## Remaining validation

Full-model startup, functional/quality comparison, five target samples and an eight-rank trace with this new backend are pending resource availability. No model service or preload producer was started during the bounded module tests. The earlier MSA/DeepEP/Bolt v5 performance remains the measured whole-model result; do not attribute it to this HIPC candidate.

Subsequent full-model measurements and runtime-only tiled pack/restore are recorded in [minimax31_tiled_routes.md](minimax31_tiled_routes.md). The earlier pending-validation paragraph describes the initial HIPC module cutoff; the later report distinguishes completed checks from unresolved quality attribution and the unmet SGLang target.
