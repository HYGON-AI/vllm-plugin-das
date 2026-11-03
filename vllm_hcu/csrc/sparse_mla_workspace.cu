// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co.,
// Ltd.

#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

namespace {

__global__ void gather_fp8_pages(const uint8_t *__restrict__ src,
                                 __hip_bfloat16 *__restrict__ dst,
                                 const int32_t *__restrict__ descriptors,
                                 const int blocks_per_page,
                                 const int64_t src_block_stride,
                                 const int64_t src_entry_stride,
                                 const int64_t dst_entry_stride) {
  const int64_t page = blockIdx.x / blocks_per_page;
  const int token =
      ((blockIdx.x % blocks_per_page) * blockDim.x + threadIdx.x) >> 5;
  if (token >= descriptors[page * 3 + 1])
    return;
  const int lane = threadIdx.x & 31;
  const int64_t physical = descriptors[page * 3];
  const int64_t start = descriptors[page * 3 + 2];
  const uint8_t *source =
      src + physical * src_block_stride + token * src_entry_stride;
  __hip_bfloat16 *target = dst + (start + token) * dst_entry_stride;
  const int4 packed = reinterpret_cast<const int4 *>(source)[lane];
  const float scale = reinterpret_cast<const float *>(source + 512)[lane >> 3];
  alignas(16) __hip_bfloat16 converted[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    __hip_fp8_e4m3 value;
    value.__x = reinterpret_cast<const uint8_t *>(&packed)[i];
    converted[i] = __float2bfloat16(static_cast<float>(value) * scale);
  }
  int4 *output = reinterpret_cast<int4 *>(target) + lane * 2;
  output[0] = reinterpret_cast<const int4 *>(converted)[0];
  output[1] = reinterpret_cast<const int4 *>(converted)[1];
  reinterpret_cast<int *>(target + 512)[lane] =
      reinterpret_cast<const int *>(source + 528)[lane];
}

} // namespace

void cp_gather_and_upconvert_fp8_kv_cache_pages(
    const torch::Tensor &src, const torch::Tensor &dst,
    const torch::Tensor &descriptors) {
  TORCH_CHECK(src.dim() == 3 && src.size(2) == 656,
              "src must have shape [num_blocks, block_size, 656]");
  TORCH_CHECK(src.scalar_type() == at::kByte ||
                  src.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "src must contain packed E4M3 KV entries");
  TORCH_CHECK(dst.dim() == 2 && dst.size(1) == 576 &&
                  dst.scalar_type() == at::kBFloat16,
              "dst must be BF16 [num_tokens, 576]");
  TORCH_CHECK(descriptors.dim() == 2 && descriptors.size(1) == 3 &&
                  descriptors.scalar_type() == at::kInt &&
                  descriptors.is_contiguous(),
              "descriptors must be contiguous int32 [num_pages, 3]");
  TORCH_CHECK(src.is_cuda() && src.device() == dst.device() &&
                  src.device() == descriptors.device(),
              "All tensors must be on the same GPU");
  TORCH_CHECK(src.stride(2) == 1 && src.stride(1) % 16 == 0 &&
                  src.stride(0) % 16 == 0 && dst.stride(1) == 1 &&
                  dst.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0,
              "Cache and workspace rows must support aligned vector access");
  if (descriptors.size(0) == 0 || dst.size(0) == 0)
    return;
  TORCH_CHECK(src.size(1) > 0, "block_size must be positive");
  const int64_t tiles = (src.size(1) + 7) / 8;
  const int64_t grid = descriptors.size(0) * tiles;
  TORCH_CHECK(tiles <= INT_MAX && grid <= INT_MAX, "Too many page tiles");
  const c10::cuda::CUDAGuard guard(src.device());
  gather_fp8_pages<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t *>(src.data_ptr()),
      reinterpret_cast<__hip_bfloat16 *>(dst.data_ptr()),
      descriptors.data_ptr<int32_t>(), static_cast<int>(tiles), src.stride(0),
      src.stride(1), dst.stride(0));
}
