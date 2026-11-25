// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "msa128_stage1_common.cuh"
#include "msa128_stage1_launch.h"
#include "msa128_stage2_topk_plain_hip.cuh"
#include "verify8_msa.h"

void launch_verify8_stage1_decode_fp8_gfx938(
    const uint8_t*, const uint8_t*, float*, const uint32_t*, const int*, const int*,
    const int*, const float*, const float*, int, int, int, int, int, int, int,
    hipStream_t, int);

namespace {
constexpr int kQueriesPerRequest = 8;
constexpr int kHeads = 4;
constexpr int kTop = 16;
constexpr uint32_t kInvalidPage = 0x7fffffffu;
static_assert(kFp8DecodeBlockM == kQueriesPerRequest * kHeads);

struct ScoreRow {
    const float* pages;
    int stride;
    int end_page;

    __device__ __forceinline__ float operator[](int page) const {
        // Stage1 only writes visible scores. Match the baseline -inf fill
        // here, and force the last causal page as in force_local_pages().
        if (page > end_page) return -INFINITY;
        if (page == end_page) return 1.0e30f;
        return pages[static_cast<int64_t>(page) * stride];
    }
};

__device__ __forceinline__ uint32_t sort_page_id(uint32_t key) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int width = 2; width <= 32; width *= 2) {
        #pragma unroll
        for (int distance = width / 2; distance > 0; distance /= 2) {
            const auto peer = static_cast<uint32_t>(
                __shfl_xor(static_cast<int>(key), distance, 32));
            const bool lower = ((lane & width) == 0) == ((lane & distance) == 0);
            key = lower ? min(key, peer) : max(key, peer);
        }
    }
    return key;
}

template<int Layout>
__global__ __launch_bounds__(256, 2) void verify8_top16(
    const float* scores, const int* ends, int* selected,
    int queries, int padded_pages, int pages) {
    const int head = blockIdx.x / queries;
    const int query = blockIdx.x % queries;
    const int end = min(max(ends[query], 0), pages * kPageSize);
    const int end_page = end == 0 ? -1 : (end - 1) / kPageSize;
    int64_t row_base = static_cast<int64_t>(head) * padded_pages * queries + query;
    int stride = queries;
    if constexpr (Layout == 2) {
        row_base = (static_cast<int64_t>(head) * queries + query) * padded_pages;
        stride = 1;
    } else if constexpr (Layout == 3) {
        const auto quartet = static_cast<int64_t>(head) * (queries / 4) + query / 4;
        row_base = quartet * padded_pages * 4 + query % 4;
        stride = 4;
    }
    const ScoreRow row{scores + row_base, stride, end_page};
    extern __shared__ char scratch[];
    __shared__ float values[kTop];
    __shared__ int ids[kTop];
    // Keep the vendor's full-K partition, merge and tie behaviour. Shortening
    // the input to end_page + 1 changes which equal-score pages win.
    msa128_stage2::topk::BlockTopkSort<64, true, float, int> sort(kTop, -INFINITY, scratch);
    sort(row, values, ids, 0, padded_pages);
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t key = kInvalidPage;
        if (threadIdx.x < kTop && ids[threadIdx.x] >= 0 && ids[threadIdx.x] < pages) {
            key = ids[threadIdx.x];
        }
        key = sort_page_id(key);
        if (threadIdx.x < kTop) {
            selected[(query * kHeads + head) * kTop + threadIdx.x] =
                key == kInvalidPage ? -1 : static_cast<int>(key);
        }
    }
}

void check_tensor(const at::Tensor& tensor, const c10::Device& device,
                  at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype &&
                tensor.is_contiguous(), "Invalid Verify8 ", name, " tensor");
}

void require_launch(const char* operation) {
    const auto status = hipGetLastError();
    TORCH_CHECK(status == hipSuccess, operation, ": ", hipGetErrorString(status));
}
}  // namespace

Verify8PageSelector::Verify8PageSelector(at::Tensor ends, at::Tensor page_ids,
                                       at::Tensor scale, int64_t pages_per_request, int mode, int query_width)
    : ends_(std::move(ends)), page_ids_(std::move(page_ids)), scale_(std::move(scale)),
      mode_(mode) {
    TORCH_CHECK(mode_ >= 0 && mode_ <= 5, "Unknown Verify8 scanner mode");
    const auto device = ends_.device();
    TORCH_CHECK(ends_.is_cuda(), "Verify8 workspace requires a GPU");
    c10::cuda::CUDAGuard guard(device);
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties && std::string(properties->gcnArchName).rfind("gfx938", 0) == 0,
                "Verify8 workspace requires gfx938");
    check_tensor(ends_, device, at::kInt, "causal ends");
    check_tensor(page_ids_, device, at::kInt, "page IDs");
    check_tensor(scale_, device, at::kFloat, "scale");
    TORCH_CHECK(query_width == 1 || query_width == 2 || query_width == 4 || query_width == 8,
                "Expected 1, 2, 4 or 8 queries per request");
    TORCH_CHECK(ends_.dim() == 1 && ends_.numel() >= query_width &&
                ends_.numel() <= query_width * 32 && ends_.numel() % query_width == 0,
                "Expected 1-32 requests");
    TORCH_CHECK(mode_ != 3 || ends_.numel() % 4 == 0,
                "The four-query layout requires a multiple of four live queries");
    TORCH_CHECK(pages_per_request >= 1 && pages_per_request <= 1161,
                "Verify8 page capacity exceeds the qualified layout");
    pages_ = static_cast<int>(pages_per_request);
    queries_ = ends_.numel();
    batch_ = queries_ / query_width;
    TORCH_CHECK(page_ids_.dim() == 1 && page_ids_.numel() == batch_ * pages_ &&
                scale_.numel() == 1, "Invalid Verify8 page table or scale size");
    padded_pages_ = ((pages_ + kPageSize - 1) / kPageSize) * kPageSize;
    const int page_group_size = mode_ == 4 ? 8 : (mode_ == 5 ? 32 : kFp8DecodePages);
    const int groups = (pages_ + page_group_size - 1) / page_group_size;
    work_ = batch_ * groups;

    // Width, batch capacity and page-table shape are invariant across layers
    // and decode steps. Stage1 still reads live per-query ends and page IDs.
    std::vector<int> metadata(batch_ * kMetaStride, 0);
    std::vector<int> work_prefix(batch_ + 1);
    for (int request = 0; request < batch_; ++request) {
        auto* row = metadata.data() + request * kMetaStride;
        row[kMetaQOff] = request * query_width;
        row[kMetaQLen] = query_width;
        row[kMetaKvLen] = pages_ * kPageSize;
        row[kMetaPageOff] = request * pages_;
        row[kMetaNPages] = pages_;
        row[kMetaNQTiles] = 1;
        row[kMetaNPageGroups] = groups;
        row[kMetaPagesPerGroup] = page_group_size;
        work_prefix[request] = request * groups;
    }
    work_prefix[batch_] = work_;
    const auto cpu = at::TensorOptions().dtype(at::kInt);
    metadata_ = at::tensor(metadata, cpu).to(device);
    work_prefix_ = at::tensor(work_prefix, cpu).to(device);
    scores_ = at::empty({kHeads, padded_pages_, queries_}, ends_.options().dtype(at::kFloat));
    selected_ = at::empty({queries_, kHeads, kTop}, ends_.options());
}

at::Tensor Verify8PageSelector::select(const at::Tensor& query, const at::Tensor& cache) {
    const auto device = ends_.device();
    check_tensor(query, device, at::kFloat8_e4m3fn, "query");
    check_tensor(cache, device, at::kFloat8_e4m3fn, "index cache");
    TORCH_CHECK(query.sizes() == at::IntArrayRef({queries_, kHeads, kHeadDim}) &&
                reinterpret_cast<uintptr_t>(query.data_ptr()) % 128 == 0,
                "Verify8 requires aligned [Q, 4, 128] FP8 queries");
    TORCH_CHECK(cache.dim() == 4 && cache.size(0) >= batch_ * pages_ &&
                cache.size(1) == 1 && cache.size(2) == kPageSize &&
                cache.size(3) == kHeadDim, "Invalid Verify8 paged index cache");
    c10::cuda::CUDAGuard guard(device);
    const auto stream = reinterpret_cast<hipStream_t>(at::cuda::getCurrentCUDAStream().stream());
    launch_verify8_stage1_decode_fp8_gfx938(
        static_cast<const uint8_t*>(query.data_ptr()), static_cast<const uint8_t*>(cache.data_ptr()),
        scores_.data_ptr<float>(), reinterpret_cast<const uint32_t*>(ends_.data_ptr<int>()),
        page_ids_.data_ptr<int>(), metadata_.data_ptr<int>(), work_prefix_.data_ptr<int>(),
        scale_.data_ptr<float>(), scale_.data_ptr<float>(), batch_, work_, kHeads, 1,
        kHeadDim, queries_, padded_pages_, stream, mode_);
    require_launch("Verify8 FP8 page-max launch failed");
    return select_scores_impl(scores_, mode_);
}

at::Tensor Verify8PageSelector::select_scores(const at::Tensor& scores) {
    return select_scores_impl(scores, 0);
}

at::Tensor Verify8PageSelector::select_scores_impl(const at::Tensor& scores, int score_layout) {
    const auto device = ends_.device();
    check_tensor(scores, device, at::kFloat, "page scores");
    TORCH_CHECK(scores.sizes() == scores_.sizes(), "Verify8 page score shape changed");
    c10::cuda::CUDAGuard guard(device);
    const auto stream = reinterpret_cast<hipStream_t>(at::cuda::getCurrentCUDAStream().stream());
    using namespace msa128_stage2::topk;
    const int shared_size = calc_lds_size_for_block_wide(kNumWaves, kTop);
    auto kernel = verify8_top16<0>;
    if (score_layout == 2) kernel = verify8_top16<2>;
    if (score_layout == 3) kernel = verify8_top16<3>;
    hipLaunchKernelGGL(kernel, dim3(kHeads * queries_), dim3(256), shared_size, stream,
                       scores.data_ptr<float>(), ends_.data_ptr<int>(), selected_.data_ptr<int>(),
                       queries_, padded_pages_, pages_);
    require_launch("Verify8 fused Top16 launch failed");
    return selected_;
}
