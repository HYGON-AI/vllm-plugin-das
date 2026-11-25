#pragma once
#include <hip/hip_runtime.h>
#include <hipcub/block/block_scan.hpp>
#include <stdint.h>
namespace sglang {
namespace minimax_verify {

constexpr int Page = 128;
constexpr int Width = 8;
constexpr int Heads = 4;
constexpr int Top = 16;

template <bool Headwise>
__global__ void plan_spans_kernel(
    const int64_t* reqs, const int* real, const int* topk, const int* ends,
    const int64_t* host, const int64_t* device, int64_t* tags, int* hot,
    int* page_slots, int64_t* miss_src, int* miss_dst, int* miss_count,
    int64_t* verify_src, int* verify_dst, int* verify_count,
    int req_slots, int host_stride, int device_stride, int map_stride,
    int head_stride, int query_stride, int choice_stride, int buffer_size) {
  constexpr int HeadsPerRow = Headwise ? 1 : Heads;
  constexpr int PlansPerRequest = Headwise ? Heads : 1;
  constexpr int Count = Width * HeadsPerRow * Top;
  constexpr int Threads = Headwise ? 128 : 256;
  constexpr int Items = Count / Threads;
  constexpr int MaxCacheSlots = Headwise ? 256 : Count;
  constexpr int CacheItems = MaxCacheSlots / Threads;
  const int cache_slots = buffer_size / Page;
  using Scan = hipcub::BlockScan<int, Threads>;
  const int plan_row = blockIdx.x;
  const int row = plan_row / PlansPerRequest;
  const int head_base = Headwise ? plan_row % Heads : 0;
  const int tid = threadIdx.x;
  const int64_t request = reqs[row];
  if (row >= *real || request < 0 || request >= req_slots ||
      device[request * device_stride] <= 0) {
    for (int j = tid; j < Count; j += Threads) {
      const int selection = Headwise ? j / Top * Heads * Top + head_base * Top + j % Top : j;
      reinterpret_cast<int4*>(hot)[row * Width * Heads * Top + selection] =
          make_int4(-1, 0, 0, 0);
    }
    if (tid == 0) {
      miss_count[plan_row] = 0;
      if constexpr (!Headwise) verify_count[row] = 0;
    }
    return;
  }

  extern __shared__ int by_page[];
  int* owner = by_page;
  int* slot_by_page = by_page + map_stride;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int occupied[MaxCacheSlots];
  __shared__ int free_slots[MaxCacheSlots];
  const int last = ends[row * Width + Width - 1];
  const int prefix = last - Width;
  const auto* device_row = device + request * device_stride;
  const auto* host_row = host + request * host_stride;
  const int64_t cache_row = request * PlansPerRequest + head_base;
  auto* tags_row = tags + cache_row * cache_slots;
  auto* slot_row = page_slots + cache_row * map_stride;

  for (int i = tid; i < map_stride; i += Threads) owner[i] = Count;
  for (int i = tid; i < cache_slots; i += Threads) occupied[i] = 0;
  if (tid == 0) {
    slot_by_page[prefix / Page] = -1;
    slot_by_page[(last - 1) / Page] = -1;
  }
  __syncthreads();

  int pages[Items];
  for (int i = 0; i < Items; ++i) {
    const int j = tid * Items + i;
    const int q = j / (HeadsPerRow * Top);
    const int h = head_base + j / Top % HeadsPerRow;
    const int logical = topk[h * head_stride + (row * Width + q) * query_stride + (j % Top) * choice_stride];
    const bool valid = logical >= 0 && logical < map_stride &&
                       logical * Page < last && logical * Page < host_stride;
    pages[i] = valid ? logical : -1;
    if (valid) atomicMin(owner + logical, j);
  }
  __syncthreads();

  bool first[Items];
  int prior[Items];
  int needs_slot[Items];
  for (int i = 0; i < Items; ++i) {
    const int j = tid * Items + i;
    const int page = pages[i];
    first[i] = page >= 0 && owner[page] == j;
    prior[i] = first[i] ? slot_row[page] : -1;
    if (prior[i] < 0 || prior[i] >= cache_slots || tags_row[prior[i]] != page) prior[i] = -1;
    if (prior[i] >= 0) occupied[prior[i]] = 1;
    needs_slot[i] = first[i] && prior[i] < 0;
  }
  __syncthreads();

  int free[CacheItems];
  int free_at[CacheItems];
  if constexpr (Headwise) {
    int empty[CacheItems];
    int empty_at[CacheItems];
    for (int i = 0; i < CacheItems; ++i) {
      const int slot = tid * CacheItems + i;
      const bool available = slot < cache_slots && occupied[slot] == 0;
      empty[i] = available && tags_row[slot] < 0;
      free[i] = available && !empty[i];
    }
    int empty_count;
    Scan(scan).ExclusiveSum(empty, empty_at, empty_count);
    __syncthreads();
    Scan(scan).ExclusiveSum(free, free_at);
    for (int i = 0; i < CacheItems; ++i) {
      const int slot = tid * CacheItems + i;
      if (empty[i]) free_slots[empty_at[i]] = slot;
      if (free[i]) free_slots[empty_count + free_at[i]] = slot;
    }
  } else {
    for (int i = 0; i < CacheItems; ++i)
      free[i] = occupied[tid * CacheItems + i] == 0;
    Scan(scan).ExclusiveSum(free, free_at);
    for (int i = 0; i < CacheItems; ++i)
      if (free[i]) free_slots[free_at[i]] = tid * CacheItems + i;
  }
  __syncthreads();
  int need_at[Items];
  Scan(scan).ExclusiveSum(needs_slot, need_at);
  __syncthreads();

  int slot[Items];
  int miss[Items];
  int64_t source[Items];
  int destination[Items];
  bool reusable[Items];
  for (int i = 0; i < Items; ++i) {
    const int page = pages[i];
    slot[i] = prior[i] >= 0 ? prior[i] : needs_slot[i] ? free_slots[need_at[i]] : -1;
    source[i] = first[i] && page * Page < prefix ? host_row[page * Page] : -1;
    destination[i] = first[i] ? device_row[slot[i] * Page] / Page : 0;
    reusable[i] = first[i] && page < prefix / Page && destination[i] > 0 && source[i] >= 0;
    miss[i] = first[i] && source[i] >= 0 && destination[i] > 0 &&
              !(reusable[i] && tags_row[slot[i]] == page);
  }
  __syncthreads();

  int miss_at[Items];
  int total_misses;
  Scan(scan).ExclusiveSum(miss, miss_at, total_misses);
  for (int i = 0; i < Items; ++i) {
    if (!first[i]) continue;
    const int page = pages[i];
    slot_by_page[page] = destination[i];
    slot_row[page] = slot[i];
    tags_row[slot[i]] = reusable[i] ? page : -1;
    if (miss[i]) {
      miss_src[plan_row * Count + miss_at[i]] = source[i] / Page;
      miss_dst[plan_row * Count + miss_at[i]] = destination[i];
    }
  }
  __syncthreads();

  for (int i = 0; i < Items; ++i) {
    const int page = pages[i];
    const int out = page < 0 ? 0 : slot_by_page[page];
    const int j = tid * Items + i;
    const int selection = j / (HeadsPerRow * Top) * Heads * Top +
                          (head_base + j / Top % HeadsPerRow) * Top + j % Top;
    int4 span = make_int4(-1, 0, 0, 0);
    if (out > 0) {
      const int begin = page * Page;
      const int query = j / (HeadsPerRow * Top);
      const int end = ends[row * Width + query];
      const int history = min(Page, max(0, min(prefix, end) - begin));
      const int scratch = static_cast<int>(device_row[buffer_size]);
      const int total = scratch > 0 ? min(Page, max(0, end - begin)) : history;
      const int base = history > 0 && host_row[begin] >= 0 ? out * Page : -1;
      span = make_int4(base, history, scratch + begin - prefix, total);
    }
    reinterpret_cast<int4*>(hot)[row * Width * Heads * Top + selection] = span;
  }
  if (tid == 0) {
    miss_count[plan_row] = total_misses;
    // Head-wise slots only serve history. Verify tokens use the scratch page
    // through direct indices, since a full-row copy would overwrite other heads.
    if constexpr (Headwise) return;
    int count = 0;
    for (int i = 0; i < Width; ++i) {
      const int pos = prefix + i;
      const int dst = slot_by_page[pos / Page];
      const int64_t src = device_row[buffer_size + i];
      if (dst <= 0 || src <= 0) continue;
      verify_src[row * Width + count] = src;
      verify_dst[row * Width + count] = dst * Page + pos % Page;
      ++count;
    }
    verify_count[row] = count;
  }
}


} }
