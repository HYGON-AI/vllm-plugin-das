// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;

// 1. 基础定义 (必须在最前面，供后续所有模板使用)
enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  kInt8 = 3
};

// Define custom BF16 vector data types.
struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};

struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};

// Define custom FP32 vector data types.
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
  __shfl_xor(var, lane_mask, width)

// 2. 补全量化转换函数 (Device 端)
namespace fp8 {

  // KV-CACHE int8
static inline __device__ float fp8_to_float(uint8_t input) {
  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));
  return c10::detail::fp32_from_bits(result);
}

// float -> fp8
static inline __device__ uint8_t float_to_fp8_e4m3(float f) {
  constexpr uint32_t fp8_finite_max = UINT32_C(1086) << 20;
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits > fp32_inf) {
    result = 0x7f;
  } else if (f_bits > fp8_finite_max) {
    // Match the OCP E4M3 SATFINITE conversion used by upstream vLLM:
    // all finite overflow (and infinity) saturates, while NaNs remain NaNs.
    result = 0x7e;
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      f_bits =
        c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits) + c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint8_t mant_odd = (f_bits >> 20) & 1;
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

static inline __device__ uint8_t float_to_fp8_e5m2(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
               + c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint32_t mant_odd = (f_bits >> 21) & 1;
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

static inline __device__ float fp8e5m2_to_fp32(const uint8_t& input) {
  union uf16 {
    uint16_t as_bits;
    _Float16 as_value;
  };
  uf16 u16;
  u16.as_bits = (uint16_t)input << 8;
  return (float)u16.as_value;
}

inline __device__ float half_to_float(uint16_t h) {
  float f;
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
  return f;
}

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
  return tmp.u16[0];
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale, Fp8KVCacheDataType kv_type) {
  return x;
}

using __nv_bfloat16 = __hip_bfloat16;

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a, float scale, Fp8KVCacheDataType kv_type) {
  if (kv_type == Fp8KVCacheDataType::kFp8E5M2) {
    return __float2bfloat16(fp8e5m2_to_fp32(a) * scale);
  }

  return __float2bfloat16(fp8_to_float(a) * scale);
  // fp8_type f8;
  // f8.__x = a;
  // return __float2bfloat16(static_cast<float>(f8) * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a,
                                                float scale, Fp8KVCacheDataType kv_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale, kv_type);
  res.y =
      scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), scale, kv_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t
scaled_vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a, float scale, Fp8KVCacheDataType kv_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale, kv_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, kv_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t
scaled_vec_conversion<bf16_8_t, uint2>(const uint2& a, float scale, Fp8KVCacheDataType kv_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, kv_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, kv_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, float scale, Fp8KVCacheDataType kv_type) {
    if (kv_type == Fp8KVCacheDataType::kFp8E5M2) {
      return fp8e5m2_to_fp32(a) * scale;
    }
    return fp8_to_float(a) * scale;
  // fp8_type f8;
  // f8.__x = a;
  // return static_cast<float>(f8) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2
scaled_vec_conversion<float2, uint16_t>(const uint16_t& a, float scale, Fp8KVCacheDataType kv_type) {
    float2 f2r;
    f2r.x = scaled_vec_conversion<float, uint8_t>((uint8_t)a, scale, kv_type);
    f2r.y = scaled_vec_conversion<float, uint8_t>((uint8_t)(a >> 8U), scale, kv_type);
    return f2r;
    // [[maybe_unused]] 
  // fp8x2_type f8x2;
  // f8x2.__x = a;
  // return static_cast<float2>(f8x2) * scale;
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_
scaled_vec_conversion<Float4_, uint32_t>(const uint32_t& a, const float scale, Fp8KVCacheDataType kv_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, kv_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale, kv_type);
  return res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4
scaled_vec_conversion<float4, uint32_t>(const uint32_t& a, float scale, Fp8KVCacheDataType kv_type) {
  Float4_ res = scaled_vec_conversion<Float4_, uint32_t>(a, scale, kv_type);
  return {res.x.x, res.x.y, res.y.x, res.y.y};
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_
scaled_vec_conversion<Float8_, uint2>(const uint2& a, float scale, Fp8KVCacheDataType kv_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, kv_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, kv_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint8_t>(const uint8_t& a, float scale, Fp8KVCacheDataType kv_type) {
  if (kv_type == Fp8KVCacheDataType::kFp8E5M2) {
    return float_to_half(fp8e5m2_to_fp32(a) * scale);
  }
  float res = fp8_to_float(a) * scale;
  return float_to_half(res);
  // __half_raw res;
  // res.data = scaled_vec_conversion<float, uint8_t>(a, scale);
  // return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } res;
  res.u16[0] = scaled_vec_conversion<uint16_t, uint8_t>((uint8_t)a, scale, kv_type);
  res.u16[1] = scaled_vec_conversion<uint16_t, uint8_t>((uint8_t)(a >> 8U), scale, kv_type);
  return res.u32;
  // [[maybe_unused]] __half2_raw h2r =
  //     __hip_cvt_fp8x2_to_halfraw2(a, fp8_type::__default_interpret);
  // union {
  //   __half2_raw h2r;
  //   uint32_t ui32;
  // } tmp;
  // tmp.h2r = __hip_cvt_fp8x2_to_halfraw2(a, fp8_type::__default_interpret);
  // tmp.h2r.x.data *= scale;
  // tmp.h2r.y.data *= scale;
  // return tmp.ui32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, uint32_t>(const uint32_t& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, kv_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), scale, kv_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2& a,
                                                                float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, kv_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, kv_type);
  return tmp.u64x2;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, uint16_t>(const uint16_t& a, float scale, Fp8KVCacheDataType kv_type) {
  float res_f = half_to_float(a) / scale;
  if (kv_type == Fp8KVCacheDataType::kFp8E4M3) {
    return float_to_fp8_e4m3(res_f);
  } else {
    return float_to_fp8_e5m2(res_f);
  }
}

// halfx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, uint32_t>(const uint32_t& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  union {
    uint32_t ui32;
    half2 h2r;
  } tmp_a;
  tmp_a.ui32 = a;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, uint16_t>(tmp_a.h2r.data[0], scale, kv_type);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, uint16_t>(tmp_a.h2r.data[1], scale, kv_type);
  return tmp.ui16;
  // union {
  //   uint32_t ui32;
  //   __half2_raw h2r;
  // } tmp;
  // tmp.ui32 = a;
  // tmp.h2r.x.data /= scale;
  // tmp.h2r.y.data /= scale;
  // return __hip_cvt_halfraw2_to_fp8x2(tmp.h2r, fp8_type::__default_saturation,
  //                                    fp8_type::__default_interpret);
}

// half2x2 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, uint2>(const uint2& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, uint32_t>(a.x, scale, kv_type);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, uint32_t>(a.y, scale, kv_type);
  return tmp.ui32;
}

// half2x4 -> fp8x8
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint4>(const uint4& a,
                                                                float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint2 ui2[2];
    uint4 ui4;
  } tmp;
  tmp.ui4 = a;
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[0], scale, kv_type);
  res.y = scaled_vec_conversion<uint32_t, uint2>(tmp.ui2[1], scale, kv_type);
  return res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __hip_bfloat16>(
    const __nv_bfloat16& a, float scale, Fp8KVCacheDataType kv_type) {
      float res_f = (static_cast<float>(a)) / scale;
      if (kv_type == Fp8KVCacheDataType::kFp8E4M3) {
        return float_to_fp8_e4m3(res_f);
      } else {
        return float_to_fp8_e5m2(res_f);
      }
}

// bf16x2 -> fp8x2
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, __nv_bfloat162>(
    const __nv_bfloat162& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.x, scale, kv_type);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, __nv_bfloat16>(a.y, scale, kv_type);
  return tmp.ui16;
}

// bf16x4 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, bf16_4_t>(const bf16_4_t& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.x, scale, kv_type);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, __nv_bfloat162>(a.y, scale, kv_type);
  return tmp.ui32;
}

// bf16x8 -> fp8x8
template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, bf16_8_t>(const bf16_8_t& a, float scale, Fp8KVCacheDataType kv_type) {
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, bf16_4_t>({a.x, a.y}, scale, kv_type);
  res.y = scaled_vec_conversion<uint32_t, bf16_4_t>({a.z, a.w}, scale, kv_type);
  return res;
}

// float -> fp8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion<uint8_t, float>(const float& a, float scale, Fp8KVCacheDataType kv_type) {
  if (kv_type == Fp8KVCacheDataType::kFp8E4M3) {
    return float_to_fp8_e4m3(a / scale);
  } else {
    return float_to_fp8_e5m2(a / scale);
  }
  // return __hip_cvt_float_to_fp8(a / scale, fp8_type::__default_saturation,
  //                               fp8_type::__default_interpret);
}

// floatx2 -> fp8x2
template <>
__inline__ __device__ uint16_t
scaled_vec_conversion<uint16_t, float2>(const float2& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint8_t ui8[2];
    uint16_t ui16;
  } tmp;
  tmp.ui8[0] = scaled_vec_conversion<uint8_t, float>(a.x, scale, kv_type);
  tmp.ui8[1] = scaled_vec_conversion<uint8_t, float>(a.y, scale, kv_type);
  return tmp.ui16;
  // return __hip_cvt_float2_to_fp8x2(a / scale, fp8_type::__default_saturation,
  //                                  fp8_type::__default_interpret);
}

// floatx4 -> fp8x4
template <>
__inline__ __device__ uint32_t
scaled_vec_conversion<uint32_t, float4>(const float4& a, float scale, Fp8KVCacheDataType kv_type) {
  union {
    uint16_t ui16[2];
    uint32_t ui32;
  } tmp;
  tmp.ui16[0] = scaled_vec_conversion<uint16_t, float2>({a.x, a.y}, scale, kv_type);
  tmp.ui16[1] = scaled_vec_conversion<uint16_t, float2>({a.z, a.w}, scale, kv_type);
  return tmp.ui32;
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ Tout scaled_convert(const Tin& val, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3 || kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(val, scale, kv_dt);
  }
  assert(false);
  return {};  // Squash missing return statement warning
}

}

namespace int8 {
template <typename cache_t, typename scalar_t>
__device__ __forceinline__ cache_t scaled_vec_conversion_int8(scalar_t val, float scale) {
    float scaled_val = static_cast<float>(val) / scale;
    if (scaled_val > 127.0f) scaled_val = 127.0f;
    if (scaled_val < -128.0f) scaled_val = -128.0f;
    return static_cast<cache_t>(static_cast<int8_t>(scaled_val));
}
}

// 3. 核函数实现
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel_hcu(
    const scalar_t* __restrict__ key, // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache, // [num_blocks, num_heads, block_size, head_size]  target layout
    cache_t* __restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size]  
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, 
    const int head_size, const int block_size, int x,    
    const float* k_scale, const float* v_scale) {
  
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx  = i / head_size;   
    const int head_offset = i % head_size; 
    // ---------- calculate target index ----------
    // K: [num_blocks, num_heads, block_size, head_size]
    const int64_t tgt_key_idx = 
      block_idx * num_heads * block_size * head_size +
      head_idx * block_size * head_size + block_offset * head_size +
      head_offset;

    const int64_t tgt_value_idx = 
      block_idx  * num_heads * head_size * block_size +
      head_idx * head_size * block_size + head_offset * block_size +
      block_offset;

    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
      key_cache[tgt_key_idx] =
          int8::scaled_vec_conversion_int8<cache_t, scalar_t>(tgt_key, 
                                                              *k_scale);
      value_cache[tgt_value_idx] =
          int8::scaled_vec_conversion_int8<cache_t, scalar_t>(tgt_value, 
                                                              *v_scale);
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel_hcu(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // logical [blocks, pages, heads, dim]
    cache_t* __restrict__ value_cache,   // logical [blocks, pages, heads, dim]
    const int64_t* __restrict__ slot_mapping,
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const int kv_scale_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int num_elements = num_heads * head_size;

  for (int element = threadIdx.x; element < num_elements;
       element += blockDim.x) {
    const int head_idx = element / head_size;
    const int head_offset = element % head_size;
    const int64_t cache_idx = block_idx * block_stride +
                              block_offset * page_stride +
                              head_idx * head_stride + head_offset;
    const scalar_t key_value = key[token_idx * key_stride + element];
    const scalar_t value_value = value[token_idx * value_stride + element];

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[cache_idx] = key_value;
      value_cache[cache_idx] = value_value;
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
      key_cache[cache_idx] =
          int8::scaled_vec_conversion_int8<cache_t, scalar_t>(
              key_value, k_scale[head_idx * kv_scale_stride]);
      value_cache[cache_idx] =
          int8::scaled_vec_conversion_int8<cache_t, scalar_t>(
              value_value, v_scale[head_idx * kv_scale_stride]);
    } else {
      key_cache[cache_idx] = fp8::scaled_convert<cache_t, scalar_t, kv_dt>(
          key_value, k_scale[head_idx * kv_scale_stride]);
      value_cache[cache_idx] = fp8::scaled_convert<cache_t, scalar_t, kv_dt>(
          value_value, v_scale[head_idx * kv_scale_stride]);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = max_abs / 448.f;
  tile_scale = fmaxf(tile_scale, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

// 4. 定义调用宏 (放在函数定义之前)
#define CALL_RESHAPE_AND_CACHE_HCU(KV_T, CACHE_T, KV_DTYPE)               \
  reshape_and_cache_kernel_hcu<KV_T, CACHE_T, KV_DTYPE>                   \
      <<<grid, block, 0, stream>>>(                                        \
          reinterpret_cast<KV_T*>(key.data_ptr()),                         \
          reinterpret_cast<KV_T*>(value.data_ptr()),                       \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,      \
          num_heads, head_size, block_size, 1,                             \
          reinterpret_cast<const float*>(k_scale.data_ptr()),              \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

#define CALL_RESHAPE_AND_CACHE_FLASH_HCU(KV_T, CACHE_T, KV_DTYPE)          \
  reshape_and_cache_flash_kernel_hcu<KV_T, CACHE_T, KV_DTYPE>              \
      <<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<KV_T*>(key.data_ptr()),                          \
          reinterpret_cast<KV_T*>(value.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                 \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),               \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,      \
          head_stride, key_stride, value_stride, num_heads, head_size,      \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()),   \
          reinterpret_cast<const float*>(v_scale.data_ptr()),               \
          kv_scale_stride);

#define CALL_CONCAT_AND_CACHE_MLA_HCU(KV_T, CACHE_T, KV_DTYPE)          \
  concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                  \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA_HCU(KV_T, CACHE_T, KV_DTYPE)       \
  concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>               \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// 5. 定义分发宏 (优化了 ROCm 下 BF16 的映射)
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, Fp8KVCacheDataType::kAuto);                           \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__hip_bfloat16, __hip_bfloat16, Fp8KVCacheDataType::kAuto);         \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type: ", SRC_DTYPE);             \
      }                                                                        \
    } else if (KV_DTYPE == "int8") {                                           \
      if (SRC_DTYPE == at::ScalarType::Half) {                                 \
        FN(uint16_t, int8_t, Fp8KVCacheDataType::kInt8);                       \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__hip_bfloat16, int8_t, Fp8KVCacheDataType::kInt8);                 \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported int8 src type");                       \
      }                                                                        \
    } else if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3" || KV_DTYPE == "fp8_ds_mla") { \
      if (SRC_DTYPE == at::ScalarType::Half) {                                 \
        FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                   \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__hip_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);             \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported fp8 src type");                        \
      }                                                                        \
    } else if (KV_DTYPE == "fp8_e5m2") {                                       \
        if (SRC_DTYPE == at::ScalarType::Half) {                               \
          FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E5M2);                 \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__hip_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E5M2);            \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
    } else {                                                                   \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
    }

// 6. Host 端函数
void reshape_and_cache_hcu(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype, 
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must be [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "cache tensor shape mismatch");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
              key_cache.size(1) == value_cache.size(1) &&
              key_cache.size(2) == value_cache.size(3) &&
              key_cache.size(3) == value_cache.size(2),
              "key/value cache dimension mismatch"); 
  int num_tokens = slot_mapping.size(0);
  int num_heads  = key.size(1);
  int head_size  = key.size(2);
  int block_size = key_cache.size(2);
  int key_stride   = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  const at::OptionalDeviceGuard device_guard(device_of(key));
  hipStream_t stream = at::hip::getCurrentHIPStream();

  // 使用 scalar_type() 替代 dtype()
  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
      CALL_RESHAPE_AND_CACHE_HCU);
}

void reshape_and_cache_flash_hcu(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // logical [blocks, pages, heads, dim]
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must be [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "cache must be a logical [blocks, pages, heads, dim] view");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "key and value cache must have the same shape");
  TORCH_CHECK(slot_mapping.dim() == 1 &&
                  slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be a one-dimensional int64 tensor");

  const int num_tokens = slot_mapping.size(0);
  const int num_heads = key.size(1);
  const int head_size = key.size(2);
  const int block_size = key_cache.size(1);
  TORCH_CHECK(key_cache.size(2) == num_heads &&
                  key_cache.size(3) == head_size,
              "cache head dimensions must match key/value");
  TORCH_CHECK(key_cache.strides() == value_cache.strides(),
              "key and value cache must have the same strides");
  TORCH_CHECK(k_scale.sizes() == v_scale.sizes(),
              "k_scale and v_scale must have the same shape");
  TORCH_CHECK(k_scale.numel() == 1 || k_scale.numel() == num_heads,
              "k_scale and v_scale must contain one or num_heads values");

  const int64_t key_stride = key.stride(0);
  const int64_t value_stride = value.stride(0);
  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride = key_cache.stride(1);
  const int64_t head_stride = key_cache.stride(2);
  const int kv_scale_stride = k_scale.numel() > 1 ? 1 : 0;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 256));
  const at::OptionalDeviceGuard device_guard(device_of(key));
  hipStream_t stream = at::hip::getCurrentHIPStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
      CALL_RESHAPE_AND_CACHE_FLASH_HCU);
}

void concat_and_cache_mla_hcu(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.itemsize(),
                "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    TORCH_CHECK(kv_c.itemsize() == 2,
                "kv_c.itemsize() must be 2 for fp8_ds_mla");
    TORCH_CHECK(k_pe.itemsize() == 2,
                "k_pe.itemsize() must be 2 for fp8_ds_mla");
  } else {
    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::OptionalDeviceGuard device_guard(device_of(kv_c));
  // HCU PyTorch exposes the HIP-backed stream through the CUDA-compatible
  // namespace, matching the rest of vLLM's ROCm kernels.
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA_HCU);
  } else {
    dim3 grid(num_tokens);
    // gfx938 HCU kernels are compiled with a 256-thread launch bound.  The
    // kernel loops over the feature dimension with a blockDim.x stride, so a
    // 256-thread block still covers kv_lora_rank=512 without changing output.
    dim3 block(std::min(kv_lora_rank, 256));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA_HCU);
  }
}

// 7. PyBind 绑定
// PYBIND11_MODULE(hcu_ops, m) {
//     m.def("reshape_and_cache", &reshape_and_cache_hcu, "HCU reshape_and_cache kernel");
// }
