# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small-M BF16xBF16 -> FP32 NT GEMM on HCU MMAC.

Ported from sglang-das ``deepseek_v4/linear_bf16_fp32.cuh``. The DeepSeek V4
compressor projections run with M <= 64 tokens per rank during decode, where
hipBLASLt picks a 256x256 tile and takes ~135us regardless of N. These kernels
cover M in [1, 64], N in {256, 512, 1024, 2048} and K % 128 == 0.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

SUPPORTED_N = frozenset((256, 512, 1024, 2048))
_MMAC_ARCHES = ("gfx928", "gfx936", "gfx938")
MAX_M = 64
_BF16_BYTES = 2
# Must match the byte range encoded in make_buffer_resource().
_BUFFER_RESOURCE_RANGE_BYTES = 0x80000000

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <cstdint>

typedef __hip_bfloat16 bf16_t;
typedef _Float16 hcu_half4_t __attribute__((__vector_size__(8)));
typedef short hcu_bf16_mfma_t __attribute__((__vector_size__(8)));
typedef float hcu_float4_t __attribute__((__vector_size__(16)));
typedef std::uint32_t hcu_uint32x4_t __attribute__((ext_vector_type(4)));

struct HcuBf16x8 {
  hcu_half4_t data[2];
};

__device__ __forceinline__ void hcu_mmac_bf16(
    const hcu_half4_t& reg_a, const hcu_half4_t& reg_b, hcu_float4_t& reg_c) {
#if defined(__gfx936__) || defined(__gfx928__)
  reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_a),
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_b), reg_c);
#elif defined(__gfx938__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_a),
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_b), reg_c, false, false);
#elif defined(__HIP_DEVICE_COMPILE__)
  // Torch builds every arch in PYTORCH_ROCM_ARCH; supported() never routes
  // devices without MMAC here.
  __builtin_trap();
#endif
}

__device__ __forceinline__ hcu_uint32x4_t make_buffer_resource(
    const std::uint32_t* ptr) {
  hcu_uint32x4_t resource = {};
  const std::uint64_t address = reinterpret_cast<std::uint64_t>(ptr);
  resource[0] =
      __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(address));
  resource[1] = __builtin_amdgcn_readfirstlane(
      static_cast<std::uint32_t>(address >> 32));
  resource[2] = 0x80000000u;
  resource[3] = 0x00020000u;
  return resource;
}

template <typename T>
__device__ __forceinline__ void async_load8(
    T* lds_base, std::int32_t lds_offset, hcu_uint32x4_t resource,
    std::int32_t gmem_offset) {
  auto* destination =
      (__attribute__((address_space(3))) int*)(lds_base + lds_offset);
  __builtin_hcu_raw_buffer_load_lds(
      resource, destination, 16, gmem_offset * sizeof(T), 0, 0, 0);
}

// One 16-row x kNPerBlock-col tile per block; waves split K and reduce in LDS.
template <int kWarps, int kNPerBlock>
__global__ void gemm_nt_bf16_fp32(
    const bf16_t* __restrict__ a, const bf16_t* __restrict__ b,
    float* __restrict__ out, std::int32_t m, std::int32_t n, std::int32_t k) {
  constexpr std::int32_t kWaveSize = 64;
  const std::int32_t block_x = blockIdx.x;
  const std::int32_t block_y = blockIdx.y;
  const std::int32_t thread = threadIdx.x;
  const std::int32_t wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const std::int32_t lane = thread % kWaveSize;
  const std::int32_t row_in_tile = lane % 16;
  const std::int32_t row_group = lane / 16;
  const std::int32_t row = block_y * 16 + row_in_tile;
  const std::int32_t col = block_x * kNPerBlock + row_in_tile;
  const std::int32_t k_offset = wave * 16 * 2 + row_group * 4 * 2;

  HcuBf16x8 a_vec = {};
  HcuBf16x8 b_vec;
  hcu_float4_t accumulator = {0, 0, 0, 0};
  const bf16_t* a_row = row < m ? a + row * k + k_offset : a;
  const bf16_t* b_row = b + col * k + k_offset;

  for (std::int32_t i = 0; i + 16 * 2 * wave < k; i += 16 * 2 * kWarps) {
    if (row < m) {
      a_vec = *reinterpret_cast<const HcuBf16x8*>(a_row + i);
    }
    b_vec = *reinterpret_cast<const HcuBf16x8*>(b_row + i);
    hcu_mmac_bf16(a_vec.data[0], b_vec.data[0], accumulator);
    hcu_mmac_bf16(a_vec.data[1], b_vec.data[1], accumulator);
  }

  extern __shared__ hcu_float4_t reduction[];
#pragma unroll
  for (std::int32_t groups = kWarps; groups > 1; groups /= 2) {
    const std::int32_t midpoint = groups / 2;
    if (row < m && wave >= midpoint && wave < groups) {
      reduction[(wave - midpoint) * 4 * 16 + row_in_tile * 4 + row_group] =
          accumulator;
    }
    __syncthreads();
    if (row < m && wave < midpoint) {
      const hcu_float4_t partial =
          reduction[wave * 4 * 16 + row_in_tile * 4 + row_group];
#pragma unroll
      for (std::int32_t i = 0; i < 4; ++i) {
        accumulator[i] += partial[i];
      }
    }
    __syncthreads();
  }

  if (row < m && wave == 0) {
    float* output = out + block_x * kNPerBlock + row * n + row_group;
#pragma unroll
    for (std::int32_t i = 0; i < 4; ++i) {
      output[i * 4] = accumulator[i];
    }
  }
}

// The current DTK toolchain emits an empty gfx938 code object when the split-K raw
// LDS load path is present in this JIT translation unit. Keep it out of device
// compilation until it can be built as an independent extension.
#define VLLM_HCU_ENABLE_MMAC_SPLITK 0
#if VLLM_HCU_ENABLE_MMAC_SPLITK
// Split-K variant with double-buffered LDS staging; partial sums are
// accumulated with atomicAdd, so the output must be zero-initialized.
template <int kWarps, int kNPerBlock, int kStage, int kSplit>
__global__ void gemm_nt_bf16_fp32_splitk(
    const bf16_t* __restrict__ a, const bf16_t* __restrict__ b,
    float* __restrict__ out, std::int32_t m, std::int32_t n, std::int32_t k) {
  constexpr std::int32_t kWaveSize = 64;
  constexpr std::int32_t kNPerWave = kNPerBlock / kWarps;
  const std::int32_t block_x = blockIdx.x;
  const std::int32_t block_y = blockIdx.y;
  const std::int32_t split = blockIdx.z;
  const std::int32_t thread = threadIdx.x;
  const std::int32_t wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const std::int32_t lane = thread % kWaveSize;
  const std::int32_t row_in_tile = lane % 16;
  const std::int32_t row_group = lane / 16;
  const std::int32_t row = block_y * 16 + row_in_tile;
  const std::int32_t col_wave = block_x * kNPerBlock + wave * kNPerWave;
  const std::int32_t k_offset = row_group * 8;
  const std::int32_t k_per_split = k / kSplit;
  const std::int32_t k_start = split * k_per_split;
  const std::int32_t stages = k_per_split / kStage;

  __shared__ bf16_t a_shared[2][16][kStage];
  __shared__ bf16_t b_shared[2][kNPerBlock][kStage];
  const auto a_resource =
      make_buffer_resource(reinterpret_cast<const std::uint32_t*>(a));
  const auto b_resource =
      make_buffer_resource(reinterpret_cast<const std::uint32_t*>(b));
  const std::int32_t b_row_start = block_x * kNPerBlock;
  hcu_float4_t accumulator = {0, 0, 0, 0};

  auto load_stage = [&](std::int32_t k_block, std::int32_t buffer) {
    if (thread < 128) {
      const std::int32_t row_index = thread >> 3;
      const std::int32_t vector = thread & 7;
      const std::int32_t a_row = block_y * 16 + row_index;
      auto* destination = &a_shared[buffer][row_index][vector * 8];
      if (a_row < m) {
        async_load8(&a_shared[buffer][0][0], row_index * kStage + vector * 8,
                    a_resource, a_row * k + k_block + vector * 8);
      } else {
        *reinterpret_cast<HcuBf16x8*>(destination) = {};
      }
    }
#pragma unroll
    for (std::int32_t j = 0; j < 2; ++j) {
      const std::int32_t index = thread + j * 256;
      const std::int32_t row_index = index >> 3;
      const std::int32_t vector = index & 7;
      async_load8(&b_shared[buffer][0][0], row_index * kStage + vector * 8,
                  b_resource,
                  (b_row_start + row_index) * k + k_block + vector * 8);
    }
  };

  auto compute_stage = [&](std::int32_t buffer) {
#pragma unroll
    for (std::int32_t i = 0; i < kStage; i += 32) {
      const auto a_vec = *reinterpret_cast<const HcuBf16x8*>(
          &a_shared[buffer][row_in_tile][k_offset + i]);
      const auto b_vec = *reinterpret_cast<const HcuBf16x8*>(
          &b_shared[buffer][wave * kNPerWave + row_in_tile][k_offset + i]);
      hcu_mmac_bf16(a_vec.data[0], b_vec.data[0], accumulator);
      hcu_mmac_bf16(a_vec.data[1], b_vec.data[1], accumulator);
    }
  };

  load_stage(k_start, 0);
  __builtin_amdgcn_s_waitcnt(0xF70);
  __builtin_amdgcn_s_barrier();

  for (std::int32_t stage = 1; stage < stages; ++stage) {
    const std::int32_t next = stage & 1;
    load_stage(k_start + stage * kStage, next);
    compute_stage(next ^ 1);
    __builtin_amdgcn_s_waitcnt(0xF70);
    __builtin_amdgcn_s_barrier();
  }
  compute_stage((stages - 1) & 1);

  if (row < m) {
    float* output = out + row * n + col_wave + row_group;
#pragma unroll
    for (std::int32_t i = 0; i < 4; ++i) {
      atomicAdd(&output[i * 4], accumulator[i]);
    }
  }
}
static bool uses_splitk(int64_t m, int64_t n, int64_t k) {
  return k % 512 == 0 && (n == 1024 || (n == 2048 && m >= 8));
}
#endif

torch::Tensor linear_bf16_fp32_mmac(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "x and w must be 2-D");
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device(), "x and w must be on the same device");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                  w.scalar_type() == torch::kBFloat16,
              "x and w must be BF16");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "x and w must be contiguous");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k, "x and w have incompatible K dimensions");
  TORCH_CHECK(m > 0 && m <= 64, "M must be in [1, 64], got ", m);
  TORCH_CHECK(n == 256 || n == 512 || n == 1024 || n == 2048,
              "unsupported N ", n);
  TORCH_CHECK(k > 0 && k % 128 == 0, "K must be a multiple of 128, got ", k);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
#if VLLM_HCU_ENABLE_MMAC_SPLITK
  const bool splitk = uses_splitk(m, n, k);
#else
  constexpr bool splitk = false;
#endif
  auto out = splitk
      ? torch::zeros({m, n}, x.options().dtype(torch::kFloat32))
      : torch::empty({m, n}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* a = reinterpret_cast<const bf16_t*>(x.data_ptr());
  const auto* b = reinterpret_cast<const bf16_t*>(w.data_ptr());
  auto* o = out.data_ptr<float>();
  const int mi = static_cast<int>(m), ni = static_cast<int>(n),
            ki = static_cast<int>(k);
  const unsigned blocks_y = static_cast<unsigned>((m + 15) / 16);
#if VLLM_HCU_ENABLE_MMAC_SPLITK
  if (splitk) {
    const dim3 grid(static_cast<unsigned>(n / 64), blocks_y, n == 1024 ? 8 : 4);
    if (n == 1024) {
      hipLaunchKernelGGL((gemm_nt_bf16_fp32_splitk<4, 64, 64, 8>), grid,
                         dim3(256), 0, stream, a, b, o, mi, ni, ki);
    } else {
      hipLaunchKernelGGL((gemm_nt_bf16_fp32_splitk<4, 64, 64, 4>), grid,
                         dim3(256), 0, stream, a, b, o, mi, ni, ki);
    }
  } else
#endif
  {
    const dim3 grid(static_cast<unsigned>(n / 16), blocks_y);
    hipLaunchKernelGGL((gemm_nt_bf16_fp32<4, 16>), grid, dim3(256),
                       2 * 16 * 16 * sizeof(float), stream, a, b, o, mi, ni,
                       ki);
  }
  C10_HIP_CHECK(hipGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bf16_fp32_mmac", &linear_bf16_fp32_mmac,
        "HCU MMAC small-M BF16xBF16 -> FP32 NT GEMM");
}
"""


@functools.lru_cache(maxsize=1)
def _jit_module() -> Any:
    import torch.utils.cpp_extension

    return torch.utils.cpp_extension.load_inline(
        name="vllm_hcu_linear_bf16_fp32_mmac",
        cpp_sources="",
        cuda_sources=_SOURCE,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def _device_has_mmac(device_index: int) -> bool:
    arch = torch.cuda.get_device_properties(device_index).gcnArchName
    return arch.split(":")[0] in _MMAC_ARCHES


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether ``x @ weight.T`` fits the MMAC kernel contract."""
    if x.dim() != 2 or weight.dim() != 2 or not x.is_cuda:
        return False
    if not _device_has_mmac(x.device.index or 0):
        return False
    m, k = x.shape
    n, weight_k = weight.shape
    return (
        0 < m <= MAX_M
        and n in SUPPORTED_N
        and k == weight_k
        and k > 0
        and k % 128 == 0
        and m * k * _BF16_BYTES <= _BUFFER_RESOURCE_RANGE_BYTES
        and n * k * _BF16_BYTES <= _BUFFER_RESOURCE_RANGE_BYTES
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def linear_bf16_fp32_mmac(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T`` for BF16 [M, K] x [N, K] -> FP32 [M, N]."""
    if not supported(x, weight):
        raise ValueError("MMAC BF16-to-FP32 GEMM does not support these tensors")
    return _jit_module().linear_bf16_fp32_mmac(x, weight)
