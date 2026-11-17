# MiniMax W8A8 hipBLASLt candidate

This model-specific candidate replaces only the scaled INT8 matrix multiplication inside MiniMax W8A8 Linear schemes. The high-precision router and the routed W4A8 DeepGEMM backend retain their existing paths. It is not a global framework operator replacement.

## Configuration and lifetime

quantization_config.hcu_minimax_w8a8_backend defaults to hipblaslt_v1; triton is an explicit comparison configuration. hcu_minimax_w8a8_fallback defaults to error; setting triton explicitly permits a logged fallback for unsupported input dtype/layout/alignment. There is no environment override; see [deployment configuration](minimax31_configuration.md).

The task configuration builder accepts --w8a8-backend hipblaslt_v1 --w8a8-fallback error. The backend is part of the effective configuration identity, and the runtime contract fingerprints the actual LightOP wrapper/native extension. Generate a new derived identity rather than relabeling a live cache.

Permanent weight processing remains in process_weights_after_loading on fresh load. The stock W8A8 processor transposes the original contiguous NK INT8 weight into KN stride(1,K). IPC restore skips that processing, validates the same parameters and recreates only the kernel Python object. At apply time, w.t() is a contiguous NK view with the same data_ptr; there is no persistent weight allocation, repack or scale permutation.

## Actual native interface

The candidate calls LightOP hipblaslt_w8a8_channelwise_gemm, which directly calls torch.ops.lmslim.hipblaslt_gemm_channelwise and the LightOP native hipBLASLt entry. It deliberately avoids the *_kme Python wrapper whose gfx938 branch performs FP32 torch.matmul.

Input quantization remains the exact stock vLLM ops.scaled_int8_quant dynamic symmetric entry. The native call receives contiguous INT8 A[M,K], an NK view of the existing INT8 weight, FP32 token scales[M,1], FP32 channel scales[N,1], BF16 output and optional BF16 bias[N]. The native singleton-batch result [1,M,N] is normalized only by a zero-copy view. Bias epilogue order and numerical correctness still require the blocked device retest below.

The explicit scope is BF16 HIP input, dynamic symmetric per-token activation/per-channel weight quantization, K/N divisible by16, exact KN storage and no INT32 dimension/address overflow. Unsupported cases reject by default or use the explicitly configured/logged Triton fallback. Native failures are not silently converted to another algorithm.

## Verification and current limit

Four CPU contracts cover configuration selection without changing routed experts, IPC weight/scale pointer preservation, the exact singleton-batch output view guard, and explicit unsupported-input fallback/error behavior. They can run with python tests/model_executor/test_minimax31_w8a8_contracts.py and do not initialize a GPU.

The first bounded device attempt reached the native channelwise API but was stopped by the product return-shape guard because the vendor retained the batch=1 dimension. The guard was corrected from [M,N] to the exact accepted [M,N] or [1,M,N] contiguous forms. The original log is retained.

The identical necessary correctness retest after this ABI fix was initially rejected by automatic approval review, which cited the earlier user instruction to stop testing. It did not run then. After the user explicitly reopened the 22-node alignment work, the same test ran successfully: synthetic M17/N512/K256 without bias and with bias before BF16 store, plus real checkpoint layer3 o_proj M3/N6144/K8192, were all BF16 bit-exact against independent CPU INT64 accumulation and FP32 scale/bias arithmetic. Original hidden input and the weight storage remained unchanged; peak allocated GPU memory was 50608640 bytes.

The captured native trace contains the unchanged vLLM dynamic_scaled_int8_quant_kernel and Cijk_Alik_Bljk_I8BS_MT16x16x64_TT2_2_WG8_8_1_MFWGS64_WGM1_bias_channelwise. It shows actual hipBLASLt INT8 execution, with no FP32 matmul fallback. These are necessary module checks, not target performance or full-model quality acceptance.

Artifacts: task_dir/support_minmax_31_pcp8_ep8_perf/prototypes/sol/w8a8_hipblaslt_v1/{failed_batch_dimension.log,cpu_tests.log,test_product_w8a8.py,correctness_result.json,native_kernel_trace.json,native_kernel_hit.json}. The test covers independent CPU integer accumulation with actual quantized inputs/scales, bias before BF16 store, and a real checkpoint o_proj. It launches neither a producer nor a full model.
