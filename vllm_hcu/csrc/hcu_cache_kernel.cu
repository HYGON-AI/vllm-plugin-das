#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// 1. 基础定义 (必须在最前面，供后续所有模板使用)
enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  kInt8 = 3
};

// 2. 补全量化转换函数 (Device 端)
namespace fp8 {

// float -> fp8
static inline __device__ uint8_t float_to_fp8_e4m3(float f) {
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = 0x7f;
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

inline __device__ float half_to_float(uint16_t h) {
  float f;
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
  return f;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale, Fp8KVCacheDataType kv_type) {
  return x;
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

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ Tout scaled_convert(const Tin& val, const float scale) {
  return scaled_vec_conversion<Tout, Tin>(val, scale, kv_dt);
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
    } else if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                 \
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

// 7. PyBind 绑定
// PYBIND11_MODULE(hcu_ops, m) {
//     m.def("reshape_and_cache", &reshape_and_cache_hcu, "HCU reshape_and_cache kernel");
// }