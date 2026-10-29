// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <rocm_version.h>
// PyTorch gates the managed BLASLt handle declaration on ROCM_VERSION, while
// extension builds may only receive the toolkit's component version macros.
#if defined(USE_ROCM) && !defined(ROCM_VERSION)
#define ROCM_VERSION (ROCM_VERSION_MAJOR * 10000 + ROCM_VERSION_MINOR * 100 + ROCM_VERSION_PATCH)
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hipblaslt/hipblaslt.h>

#include <map>
#include <memory>
#include <tuple>

namespace {
void check(hipblasStatus_t status) {
  TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
              "HY4 HIPBLASLt FP8 BMM failed with status ", int(status));
}

// Thread-local plans keep heuristic searches out of steady-state execution.
// Plans own only descriptors/algorithms, borrowing PyTorch's managed handle.
// No descriptor is shared across host threads. Device and all layout
// dimensions participate in the cache key.
struct Plan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulHeuristicResult_t result{};

  ~Plan() {
    if (preference) hipblasLtMatmulPreferenceDestroy(preference);
    if (d) hipblasLtMatrixLayoutDestroy(d);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (desc) hipblasLtMatmulDescDestroy(desc);
  }

  void set_scales(const void* weight_scale, const void* x_scale) {
    check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &weight_scale, sizeof(weight_scale)));
    check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &x_scale, sizeof(x_scale)));
  }

  void initialize(hipblasLtHandle_t handle, int64_t tokens, int32_t heads,
                  int64_t latent, int64_t value, hipDataType dtype,
                  const void* weight_scale, const void* x_scale) {
    check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    // Scale-pointer presence participates in heuristic filtering (UseScaleAB).
    // Select an algorithm for the actual scalar-scaled problem, never for a
    // descriptor with null scales that is modified only after selection.
    set_scales(weight_scale, x_scale);
    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    check(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    // Column-major GEMM: W[h]^T [V,K] @ X[h] [K,T] -> Out[h] [V,T].
    // X and Out interleave heads within each token, so their leading
    // dimensions span a whole token and their batch offsets span one head.
    check(hipblasLtMatrixLayoutCreate(&a, dtype, latent, value, latent));
    check(hipblasLtMatrixLayoutCreate(&b, dtype, latent, tokens, heads * latent));
    check(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, value, tokens, heads * value));
    const int64_t stride_a = value * latent;
    const int64_t stride_b = latent;
    const int64_t stride_d = value;
    for (auto layout : {a, b, d}) {
      check(hipblasLtMatrixLayoutSetAttribute(
          layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &heads, sizeof(heads)));
    }
    check(hipblasLtMatrixLayoutSetAttribute(
        a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
    check(hipblasLtMatrixLayoutSetAttribute(
        b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
    check(hipblasLtMatrixLayoutSetAttribute(
        d, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));

    check(hipblasLtMatmulPreferenceCreate(&preference));
    const size_t workspace_limit = 4 * 1024 * 1024;
    check(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit)));
    int count = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
        handle, desc, a, b, d, d, preference, 1, &result, &count));
    TORCH_CHECK(count > 0 && result.state == HIPBLAS_STATUS_SUCCESS,
                "No HIPBLASLt FP8 BMM algorithm for tokens=", tokens,
                ", heads=", heads, ", latent=", latent, ", value=", value);
  }
};

using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int>;
}  // namespace

void hipblaslt_fp8_bmm(torch::Tensor const& x, torch::Tensor const& weight,
                      torch::Tensor const& x_scale,
                      torch::Tensor const& weight_scale, torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda() && weight.device() == x.device() &&
              out.device() == x.device() && x_scale.device() == x.device() &&
              weight_scale.device() == x.device(), "Expected tensors on the same HCU");
  TORCH_CHECK(x.dim() == 3 && weight.dim() == 3 && out.dim() == 3,
              "Expected X[T,H,K], W[H,V,K], Out[T,H,V]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(),
              "HIPBLASLt FP8 BMM requires contiguous token-major X/Out and W[H,V,K]");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "FP8 input dtypes must match");
  hipDataType dtype;
  if (x.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    dtype = HIP_R_8F_E4M3;
  } else {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                "Expected E4M3 or E4M3 FNUZ inputs");
    dtype = HIP_R_8F_E4M3_FNUZ;
  }
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "Expected BF16 output");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && x_scale.numel() == 1 &&
              weight_scale.scalar_type() == at::kFloat && weight_scale.numel() == 1,
              "HIPBLASLt FP8 BMM requires scalar FP32 dequantization scales");
  const int64_t tokens = x.size(0), heads = x.size(1), latent = x.size(2);
  const int64_t value = weight.size(1);
  TORCH_CHECK(heads > 0 && heads <= INT32_MAX && latent > 0 && value > 0 &&
              weight.size(0) == heads && weight.size(2) == latent &&
              out.size(0) == tokens && out.size(1) == heads && out.size(2) == value,
              "Incompatible HY4 FP8 BMM dimensions");
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  static thread_local std::map<Key, std::unique_ptr<Plan>> plans;
  const Key key{x.get_device(), tokens, heads, latent, value, int(dtype)};
  auto it = plans.find(key);
  if (it == plans.end()) {
    // Warm up each token shape before CUDA graph capture. Do not run host
    // heuristic searches or handle initialization inside capture.
    hipStreamCaptureStatus capture_status;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(hipStreamIsCapturing(stream, &capture_status) == hipSuccess,
                "Could not query HCU stream capture status");
    TORCH_CHECK(capture_status == hipStreamCaptureStatusNone,
                "Warm up HY4 HIPBLASLt FP8 BMM before graph capture");
    auto plan = std::make_unique<Plan>();
    plan->initialize(at::cuda::getCurrentCUDABlasLtHandle(), tokens,
                     static_cast<int32_t>(heads), latent, value, dtype,
                     weight_scale.data_ptr(), x_scale.data_ptr());
    it = plans.emplace(key, std::move(plan)).first;
  }
  auto& plan = *it->second;
  plan.set_scales(weight_scale.data_ptr(), x_scale.data_ptr());
  // PyTorch's allocator handles temporary workspace lifetime during capture
  // and on the current stream, without synchronizing or allocating raw HIP memory.
  auto workspace = torch::empty(
      {static_cast<int64_t>(plan.result.workspaceSize)}, x.options().dtype(at::kByte));
  const float alpha = 1.0f, beta = 0.0f;
  check(hipblasLtMatmul(
      at::cuda::getCurrentCUDABlasLtHandle(), plan.desc, &alpha, weight.data_ptr(), plan.a,
      x.data_ptr(), plan.b, &beta, out.data_ptr(), plan.d,
      out.data_ptr(), plan.d, &plan.result.algo, workspace.data_ptr(),
      plan.result.workspaceSize, at::cuda::getCurrentCUDAStream().stream()));
}
