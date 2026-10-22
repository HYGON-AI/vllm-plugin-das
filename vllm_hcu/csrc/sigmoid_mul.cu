// SPDX-License-Identifier: Apache-2.0

#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ops.h"

namespace {

// Return-type overloading is not allowed in C++, so the dtype conversions
// live in explicit specializations instead of free-function overloads.
template <typename T>
struct SigmoidMulCvt;

template <>
struct SigmoidMulCvt<__half> {
  static inline __device__ float to_float(__half value) {
    return __half2float(value);
  }
  static inline __device__ __half from_float(float value) {
    return __float2half(value);
  }
};

template <>
struct SigmoidMulCvt<__hip_bfloat16> {
  static inline __device__ float to_float(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  static inline __device__ __hip_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }
};

template <typename T>
__global__ void sigmoid_mul_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ value,
    T* __restrict__ output,
    int columns) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }

  // Sigmoid runs in fp32 and rounds to the storage dtype before the
  // multiply, mirroring torch.sigmoid(gate) * value bit-for-bit.
  const T gate_sigmoid = SigmoidMulCvt<T>::from_float(
      1.0f / (1.0f + expf(-SigmoidMulCvt<T>::to_float(gate[row]))));
  output[row * columns + column] = SigmoidMulCvt<T>::from_float(
      SigmoidMulCvt<T>::to_float(gate_sigmoid) *
      SigmoidMulCvt<T>::to_float(value[row * columns + column]));
}

}  // namespace

template <typename T>
static void launch_sigmoid_mul(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    torch::Tensor& output) {
  const int columns = static_cast<int>(output.size(-1));
  const int rows = static_cast<int>(output.numel() / columns);
  dim3 block(256, 1, 1);
  dim3 grid((columns + block.x - 1) / block.x, rows, 1);
  sigmoid_mul_kernel<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(gate.data_ptr()),
      reinterpret_cast<const T*>(value.data_ptr()),
      reinterpret_cast<T*>(output.data_ptr()),
      columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor sigmoid_mul_hcu(torch::Tensor gate, torch::Tensor value) {
  TORCH_CHECK(gate.is_cuda() && value.is_cuda(), "sigmoid_mul requires HCU tensors");
  TORCH_CHECK(
      (gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16),
      "sigmoid_mul requires fp16 or bf16 tensors");
  TORCH_CHECK(gate.scalar_type() == value.scalar_type(),
              "sigmoid_mul gate and value dtypes must match");
  TORCH_CHECK(gate.is_contiguous() && value.is_contiguous(),
              "sigmoid_mul inputs must be contiguous");
  TORCH_CHECK(gate.dim() == value.dim(), "sigmoid_mul input ranks must match");
  TORCH_CHECK(gate.numel() == value.size(0),
              "sigmoid_mul gate rows do not match value rows");
  TORCH_CHECK(value.size(-1) % 2 == 0,
              "sigmoid_mul value columns must be even");

  auto output = torch::empty_like(value);
  if (output.numel() == 0) {
    return output;
  }

  if (gate.scalar_type() == at::kHalf) {
    launch_sigmoid_mul<__half>(gate, value, output);
  } else {
    launch_sigmoid_mul<__hip_bfloat16>(gate, value, output);
  }
  return output;
}
