/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * DeepSeek-V4 inverse RoPE kernel for the WO_A BF16 path.
 *
 * The execution model is adapted from SGLang's DeepSeek-V4 rope.cuh:
 * one warp handles one token/head row and each lane handles one BF16 complex
 * pair in the 64-dim RoPE tail. This implementation is self-contained and
 * consumes vLLM's native cos_sin_cache layout [cos..., sin...].
 */

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/cuda.h>

namespace {

constexpr int kRopeDim = 64;
constexpr int kHalfRope = 32;
constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(x) << 16;
  return v.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_rne(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  uint32_t const lsb = (v.u >> 16) & 1;
  uint32_t const rounding_bias = 0x7fff + lsb;
  return static_cast<uint16_t>((v.u + rounding_bias) >> 16);
}

template <typename PosT>
__global__ __launch_bounds__(kBlockSize, 16) void deepseek_v4_inv_rope_kernel(
    void* __restrict__ rope, const PosT* __restrict__ positions,
    const float* __restrict__ cos_sin_cache, int64_t rope_stride_token,
    int64_t rope_stride_head, int64_t cache_stride_pos, int32_t num_tokens,
    int32_t num_heads) {
  int const warp_id = threadIdx.x / kWarpSize;
  int const lane_id = threadIdx.x % kWarpSize;
  int const global_warp_id = blockIdx.x * kWarpsPerBlock + warp_id;
  int const token = global_warp_id / num_heads;
  int const head = global_warp_id - token * num_heads;
  if (token >= num_tokens) {
    return;
  }

  int64_t const pos = static_cast<int64_t>(positions[token]);
  char* base_bytes = static_cast<char*>(rope) +
                     token * rope_stride_token *
                         static_cast<int64_t>(sizeof(uint16_t)) +
                     head * rope_stride_head *
                         static_cast<int64_t>(sizeof(uint16_t));
  uint16_t* base = reinterpret_cast<uint16_t*>(base_bytes);
  float const* cache = cos_sin_cache + pos * cache_stride_pos;

  uint16_t const even_b = base[lane_id * 2];
  uint16_t const odd_b = base[lane_id * 2 + 1];
  float const even = bf16_to_float(even_b);
  float const odd = bf16_to_float(odd_b);
  float const c = cache[lane_id];
  float const s = cache[kHalfRope + lane_id];

  // Inverse GPT-J RoPE: (a + bi) * (c - di).
  float const out_even = even * c + odd * s;
  float const out_odd = odd * c - even * s;
  base[lane_id * 2] = float_to_bf16_rne(out_even);
  base[lane_id * 2 + 1] = float_to_bf16_rne(out_odd);
}

template <typename PosT>
void launch_deepseek_v4_inv_rope(torch::Tensor& rope,
                                 torch::Tensor const& positions,
                                 torch::Tensor const& cos_sin_cache) {
  int32_t const num_tokens = static_cast<int32_t>(rope.size(0));
  int32_t const num_heads = static_cast<int32_t>(rope.size(1));
  if (num_tokens == 0 || num_heads == 0) {
    return;
  }
  int const total_warps = num_tokens * num_heads;
  int const num_blocks =
      (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto stream = at::cuda::getCurrentCUDAStream();

  deepseek_v4_inv_rope_kernel<PosT><<<num_blocks, kBlockSize, 0, stream>>>(
      rope.data_ptr(), positions.data_ptr<PosT>(),
      cos_sin_cache.data_ptr<float>(), rope.stride(0), rope.stride(1),
      cos_sin_cache.stride(0), num_tokens, num_heads);
}

}  // namespace

void deepseek_v4_inv_rope(torch::Tensor& rope, torch::Tensor const& positions,
                          torch::Tensor const& cos_sin_cache) {
  TORCH_CHECK(rope.is_cuda(), "rope must be CUDA/ROCm");
  TORCH_CHECK(positions.is_cuda(), "positions must be CUDA/ROCm");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be CUDA/ROCm");
  TORCH_CHECK(rope.dtype() == torch::kBFloat16, "rope must be bf16");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(rope.dim() == 3 && rope.size(2) == kRopeDim,
              "rope must be [num_tokens, num_heads, 64]");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim,
              "cos_sin_cache must be [max_pos, 64]");

  at::cuda::OptionalCUDAGuard device_guard(device_of(rope));
  if (positions.dtype() == torch::kInt64) {
    launch_deepseek_v4_inv_rope<int64_t>(rope, positions, cos_sin_cache);
  } else if (positions.dtype() == torch::kInt32) {
    launch_deepseek_v4_inv_rope<int32_t>(rope, positions, cos_sin_cache);
  } else {
    TORCH_CHECK(false, "positions must be int32 or int64");
  }
}
