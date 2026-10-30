// SPDX-License-Identifier: Apache-2.0
// Host-only hipBLASLt implementation to compile into the existing hcu_ops.
// This template can be named .cpp in CUDAExtension; it defines no GPU kernel.
#include <torch/all.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void check_blas(hipblasStatus_t code, const char* operation) {
  TORCH_CHECK(code == HIPBLAS_STATUS_SUCCESS, operation,
              " returned hipBLASLt status ", static_cast<int>(code));
}

void check_hip(hipError_t code, const char* operation) {
  TORCH_CHECK(code == hipSuccess, operation, ": ", hipGetErrorString(code));
}

struct Key {
  int device;
  int64_t m, n, k, algorithm;
  bool operator==(const Key& other) const {
    return device == other.device && m == other.m && n == other.n
        && k == other.k && algorithm == other.algorithm;
  }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t value = std::hash<int>{}(key.device);
    for (int64_t x : {key.m, key.n, key.k, key.algorithm})
      value ^= std::hash<int64_t>{}(x) + 0x9e3779b9 + (value << 6) + (value >> 2);
    return value;
  }
};

struct Plan {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t algorithm{};
  std::mutex mutex;

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    // Never throw while unwinding a failed initialization.
    if (preference) hipblasLtMatmulPreferenceDestroy(preference);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (d) hipblasLtMatrixLayoutDestroy(d);
    if (desc) hipblasLtMatmulDescDestroy(desc);
    if (handle) hipblasLtDestroy(handle);
  }

  void initialize(const Key& key, const at::Tensor& xs, const at::Tensor& ws) {
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, key.device), "hipGetDeviceProperties");
    std::string architecture(properties.gcnArchName);
    architecture = architecture.substr(0, architecture.find(':'));
    TORCH_CHECK(architecture == "gfx938", "Step5 tuned FP8 recipe requires gfx938, got ", architecture);
    check_blas(hipblasLtCreate(&handle), "hipblasLtCreate");
    check_blas(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "create matmul descriptor");
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "set opA");
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "set opB");
    hipblasLtMatmulMatrixScale_t vector_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &vector_mode, sizeof(vector_mode)), "set A outer-vector mode");
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &vector_mode, sizeof(vector_mode)), "set B outer-vector mode");
    void* ws_ptr = ws.data_ptr(); void* xs_ptr = xs.data_ptr();
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr, sizeof(ws_ptr)), "set initial A scale");
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs_ptr, sizeof(xs_ptr)), "set initial B scale");
    // Row-major w[N,K], x[M,K], out[M,N] are column-major A[K,N],
    // B[K,M], D[N,M]. opA=T, opB=N gives the unchanged x @ w.T math.
    check_blas(hipblasLtMatrixLayoutCreate(&a, HIP_R_8F_E4M3, key.k, key.n, key.k), "create A layout");
    check_blas(hipblasLtMatrixLayoutCreate(&b, HIP_R_8F_E4M3, key.k, key.m, key.k), "create B layout");
    check_blas(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, key.n, key.m, key.n), "create D layout");
    check_blas(hipblasLtMatmulPreferenceCreate(&preference), "create preference");
    const size_t candidate_workspace_limit = 64ULL * 1024 * 1024;
    check_blas(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &candidate_workspace_limit,
        sizeof(candidate_workspace_limit)), "set heuristic workspace limit");
    std::vector<hipblasLtMatmulHeuristicResult_t> results(128);
    int count = 0;
    check_blas(hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, d, d,
        preference, results.size(), results.data(), &count), "query tuned FP8 candidates");
    int matches = 0;
    for (int i = 0; i < count; ++i) {
      auto& result = results[i];
      if (result.state == HIPBLAS_STATUS_SUCCESS
          && hipblaslt_ext::getIndexFromAlgo(result.algo) == key.algorithm) {
        TORCH_CHECK(result.workspaceSize == 0,
            "Step5 tuned FP8 recipe requires zero workspace; algorithm ",
            key.algorithm, " requested ", result.workspaceSize);
        algorithm = result.algo;
        ++matches;
      }
    }
    TORCH_CHECK(matches == 1, "STEP5_DENSE_UNSUPPORTED_RECIPE: M=", key.m,
        " N=", key.n, " K=", key.k, " algorithm=", key.algorithm,
        " matching candidates=", matches);
  }

  void enqueue(const at::Tensor& x, const at::Tensor& xs,
               const at::Tensor& w, const at::Tensor& ws,
               const at::Tensor& out, hipStream_t stream) {
    // hipBLASLt consumes these descriptor attributes while enqueueing. Protect
    // updates and enqueue together across the two DBO host threads.
    std::lock_guard<std::mutex> lock(mutex);
    void* ws_ptr = ws.data_ptr(); void* xs_ptr = xs.data_ptr();
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr, sizeof(ws_ptr)), "set A scale");
    check_blas(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs_ptr, sizeof(xs_ptr)), "set B scale");
    const float alpha = 1.f, beta = 0.f;
    check_blas(hipblasLtMatmul(handle, desc, &alpha, w.data_ptr(), a,
        x.data_ptr(), b, &beta, out.data_ptr(), d, out.data_ptr(), d,
        &algorithm, nullptr, 0, stream), "Step5 tuned FP8 matmul");
  }
};

struct Cache {
  std::mutex mutex;
  std::unordered_map<Key, std::shared_ptr<Plan>, KeyHash> plans;
};

Cache& cache() {
  // Process-lifetime plans avoid HIP handle destruction after the
  // framework has already torn down its device context at process exit.
  static Cache* value = new Cache;
  return *value;
}

void validate(const at::Tensor& x, const at::Tensor& xs,
              const at::Tensor& w, const at::Tensor& ws) {
  TORCH_CHECK(x.is_cuda() && xs.is_cuda() && w.is_cuda() && ws.is_cuda(),
              "Step5 FP8 dense inputs must be device tensors");
  TORCH_CHECK(x.device() == xs.device() && x.device() == w.device()
              && x.device() == ws.device(), "Step5 FP8 dense device mismatch");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && xs.dim() == 2 && ws.dim() == 2,
              "Step5 FP8 dense inputs must be matrices");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fn
              && w.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "Step5 FP8 dense requires e4m3fn x/weight");
  TORCH_CHECK(xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat,
              "Step5 FP8 dense requires FP32 token/channel scales");
  TORCH_CHECK(x.is_contiguous() && xs.is_contiguous()
              && w.is_contiguous() && ws.is_contiguous(),
              "Step5 FP8 dense inputs must be contiguous");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && w.size(0) > 0
              && x.size(1) == w.size(1), "Step5 FP8 dense shape mismatch");
  TORCH_CHECK(xs.size(0) == x.size(0) && xs.size(1) == 1
              && ws.size(0) == w.size(0) && ws.size(1) == 1,
              "Step5 FP8 dense scale shape mismatch");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max()
              && x.size(1) <= std::numeric_limits<int>::max()
              && w.size(0) <= std::numeric_limits<int>::max(),
              "Step5 FP8 dense dimensions exceed supported range");
}

}  // namespace

at::Tensor step5_fp8_dense_gemm(const at::Tensor& x, const at::Tensor& xs,
                              const at::Tensor& w, const at::Tensor& ws,
                              int64_t algorithm_id, bool allow_initialize) {
  validate(x, xs, w, ws);
  TORCH_CHECK(algorithm_id >= 0, "Step5 FP8 dense algorithm ID must be nonnegative");
  const c10::cuda::CUDAGuard guard(x.device());
  const hipStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const Key key{static_cast<int>(x.get_device()), x.size(0), w.size(0), x.size(1), algorithm_id};
  auto out = at::empty({key.m, key.n}, x.options().dtype(at::kBFloat16));
  auto& state = cache(); std::shared_ptr<Plan> plan;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    auto existing = state.plans.find(key);
    if (existing != state.plans.end()) {
      plan = existing->second;
    } else {
      TORCH_CHECK(allow_initialize,
          "STEP5_DENSE_COLD_PLAN: prewarm during the unsplit startup profile; "
          "initialization is forbidden on DBO/graph hot paths. M=", key.m,
          " N=", key.n, " K=", key.k, " algorithm=", key.algorithm);
      hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
      check_hip(hipStreamIsCapturing(stream, &capture_status), "hipStreamIsCapturing");
      TORCH_CHECK(capture_status == hipStreamCaptureStatusNone,
                  "Step5 FP8 dense plan initialization is forbidden during graph capture");
      plan = std::make_shared<Plan>();
      plan->initialize(key, xs, ws);
      // Publish only after the first enqueue has loaded the chosen device
      // code. Other host threads cannot observe an un-warmed plan.
      plan->enqueue(x, xs, w, ws, out, stream);
      state.plans.emplace(key, plan);
      return out;
    }
  }
  plan->enqueue(x, xs, w, ws, out, stream);
  return out;
}

std::string step5_fp8_dense_backend_library() {
  Dl_info info{};
  TORCH_CHECK(dladdr(reinterpret_cast<void*>(&hipblasLtMatmul), &info) != 0
              && info.dli_fname != nullptr, "Cannot identify loaded hipBLASLt");
  return std::string(info.dli_fname);
}
