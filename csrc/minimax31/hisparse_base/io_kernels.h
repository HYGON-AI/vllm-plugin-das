#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#define USE_ROCM 1
namespace sglang {
constexpr int WARP_SIZE=64;
using TransferVec4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

__device__ __forceinline__ void transfer_item_warp(
    int32_t lane_id, const void* __restrict__ src_addr, void* __restrict__ dst_addr, int64_t item_size_bytes) {
  const auto src = static_cast<const char*>(src_addr);
  auto dst = static_cast<char*>(dst_addr);

  // Wide path: one 128-bit dwordx4 per lane instead of two 64-bit dwordx2,
  // which halves the load/store instruction count and the number of serialized
  // round trips per item. The source is usually pinned host DRAM, so those
  // round trips are what the copy is actually paying for.
  //
  // Gated on item_size_bytes >= WARP_SIZE * 16, and this gate is load-bearing:
  // a 16B-per-lane step only covers item_size_bytes/16 lanes, so below that
  // threshold the wide path idles part of the wavefront while needing the same
  // number of iterations as the 8B path. Measured on MI355X (gfx950, wave64)
  // with 512B items, ungated widening was 14-22% SLOWER because 512/16 = 32
  // lanes work instead of 512/8 = 64. At 1024B (= WARP_SIZE * 16) the whole
  // wavefront stays busy and the item moves in a single instruction per lane.
  //
  // The load is non-temporal because these items are streamed in once; there
  // is no reuse to preserve and they should not evict resident lines from
  // L2/MALL. The store is deliberately left cached: its destination is the
  // device buffer that the attention kernel reads immediately afterwards.
  int64_t byte_pos = 0;
  const bool aligned_16b = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 0xF) == 0;
  const bool wide_fills_wave = item_size_bytes >= static_cast<int64_t>(WARP_SIZE) * 16;
  if (aligned_16b && wide_fills_wave) {
    constexpr int64_t kVecBytes = static_cast<int64_t>(sizeof(TransferVec4));
    const int64_t vec_count = item_size_bytes / kVecBytes;
    const auto src_vec = reinterpret_cast<const TransferVec4*>(src);
    auto dst_vec = reinterpret_cast<TransferVec4*>(dst);
    for (int64_t i = lane_id; i < vec_count; i += WARP_SIZE) {
      dst_vec[i] = __builtin_nontemporal_load(&src_vec[i]);
    }
    byte_pos = vec_count * kVecBytes;
  }

  // 64-bit path: covers the unaligned case in full, and the <= 8-byte
  // remainder the wide path leaves behind.
  const int64_t word_count = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const int64_t word_start = byte_pos / static_cast<int64_t>(sizeof(uint64_t));
  const auto src_words = reinterpret_cast<const uint64_t*>(src);
  auto dst_words = reinterpret_cast<uint64_t*>(dst);
  for (int64_t i = word_start + lane_id; i < word_count; i += WARP_SIZE) {
    dst_words[i] = src_words[i];
  }

  const int64_t tail_start = word_count * static_cast<int64_t>(sizeof(uint64_t));
  for (int64_t i = tail_start + lane_id; i < item_size_bytes; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}

template <bool IsMLA, bool IsDsv4Layout>
__device__ __forceinline__ void copy_miss_item(int32_t lane_id,
 const void* host_cache_k,const void* host_cache_v,void* device_buffer_k,void* device_buffer_v,
 int64_t src_loc,int64_t dst_loc,int64_t item_size_bytes) {
 static_assert(!IsMLA && !IsDsv4Layout,"MiniMax MHA only");
 const auto src_k=static_cast<const char*>(host_cache_k)+src_loc*item_size_bytes;
 auto dst_k=static_cast<char*>(device_buffer_k)+dst_loc*item_size_bytes;
 transfer_item_warp(lane_id,src_k,dst_k,item_size_bytes);
 const auto src_v=static_cast<const char*>(host_cache_v)+src_loc*item_size_bytes;
 auto dst_v=static_cast<char*>(device_buffer_v)+dst_loc*item_size_bytes;
 transfer_item_warp(lane_id,src_v,dst_v,item_size_bytes);
}
template <int BLOCK_SIZE>
__global__ void commit_mha_all_layers_kernel(
    const int64_t* scratch, const int32_t* host_locs, const int32_t* accepted,
    const uint64_t* source_k, const uint64_t* source_v,
    char* host_k, char* host_v, int batch, int width, int blocks_per_layer,
    int64_t host_layer_stride, int64_t item_size_bytes) {
  constexpr int warps_per_block = BLOCK_SIZE / WARP_SIZE;
  const int layer = blockIdx.x / blocks_per_layer;
  const int item = (blockIdx.x % blocks_per_layer) * warps_per_block + threadIdx.x / WARP_SIZE;
  if (item >= batch * width || item % width >= accepted[item / width]) return;
  const int64_t dst = host_locs[item];
  if (dst < 0) return;
  copy_miss_item<false, false>(
      threadIdx.x % WARP_SIZE, reinterpret_cast<const void*>(source_k[layer]),
      reinterpret_cast<const void*>(source_v[layer]), host_k + layer * host_layer_stride,
      host_v + layer * host_layer_stride, scratch[item], dst, item_size_bytes);
}

template <int BLOCK_SIZE, int BLOCKS_PER_REQUEST, int HEADS_PER_REQUEST = 1>
__global__ void swap_mha_pages_tiled_kernel(
    const int64_t* src_locs, const int32_t* dst_locs, const int32_t* counts,
    const int32_t* real, const uint4* host_k, const uint4* host_v,
    uint4* device_k, uint4* device_v, int64_t stride) {
  constexpr int vectors_per_page = 65536 / sizeof(uint4);
  constexpr int vectors_per_head = 128 / sizeof(uint4);
  const int plan_row = blockIdx.x / BLOCKS_PER_REQUEST;
  if (plan_row / HEADS_PER_REQUEST >= real[0]) return;
  const int head = plan_row % HEADS_PER_REQUEST;
  const int count = counts[plan_row];
  for (int item = blockIdx.x % BLOCKS_PER_REQUEST; item < count; item += BLOCKS_PER_REQUEST) {
    const int64_t offset = plan_row * stride + item;
    const int64_t src = src_locs[offset] * vectors_per_page;
    const int64_t dst = int64_t(dst_locs[offset]) * vectors_per_page;
    #pragma unroll 4
    for (int i = threadIdx.x; i < vectors_per_page / HEADS_PER_REQUEST; i += BLOCK_SIZE) {
      const int lane = HEADS_PER_REQUEST == 1 ? i :
          (i / vectors_per_head * HEADS_PER_REQUEST + head) * vectors_per_head + i % vectors_per_head;
      const uint4 k = host_k[src + lane];
      const uint4 v = host_v[src + lane];
      device_k[dst + lane] = k;
      device_v[dst + lane] = v;
    }
  }
}


}
