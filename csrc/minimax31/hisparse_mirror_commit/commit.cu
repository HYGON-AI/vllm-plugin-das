#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

__global__ void commit_with_device_tail(
        const int64_t* scratch, const int32_t* host_locs, const int32_t* accepted,
        const int32_t* prefix, const uint64_t* device_k, const uint64_t* device_v,
        uint64_t* host_k, uint64_t* host_v, int32_t* tails, int requests,
        int ctas_per_layer, int64_t host_words_per_layer) {
    const int layer = blockIdx.x / ctas_per_layer;
    const int item = (blockIdx.x % ctas_per_layer) * 4 + threadIdx.x / 64;
    const int request = item / 8;
    const int step = item % 8;
    if (request >= requests || step >= accepted[request]) return;
    const int lane = threadIdx.x % 64;
    const int head = lane / 16;
    const int host_pos = host_locs[item];
    if (host_pos < 0) return;
    const int64_t src = scratch[item] * 64 + lane;
    auto* gpu_k = reinterpret_cast<uint64_t*>(device_k[layer]);
    auto* gpu_v = reinterpret_cast<uint64_t*>(device_v[layer]);
    const uint64_t key = gpu_k[src];
    const uint64_t value = gpu_v[src];
    const int64_t dst = layer * host_words_per_layer + int64_t(host_pos) * 64 + lane;
    host_k[dst] = key;
    host_v[dst] = value;

    const int position = prefix[request] + step;
    auto* tail = tails + ((layer * requests + request) * 4 + head) * 3;
    const int page = tail[0];
    const int slot = tail[1];
    if (slot > 0 && position / 128 == page) {
        const int64_t target = (int64_t(slot) * 128 + position % 128) * 64 + lane;
        gpu_k[target] = key;
        gpu_v[target] = value;
        if (step == 0 && lane % 16 == 0) {
            tail[2] = min(128, prefix[request] % 128 + accepted[request]);
        }
    }
}

static void* mapped(const at::Tensor& input) {
    if (input.is_cuda()) return input.data_ptr();
    void* mapping = nullptr;
    const auto status = hipHostGetDevicePointer(&mapping, input.data_ptr(), 0);
    TORCH_CHECK(status == hipSuccess, "mapped host history is required");
    return mapping;
}

void commit_mirror(const at::Tensor& scratch, const at::Tensor& host_locs,
                   const at::Tensor& accepted, const at::Tensor& prefix,
                   const at::Tensor& device_k, const at::Tensor& device_v,
                   const at::Tensor& host_k, const at::Tensor& host_v,
                   const at::Tensor& tails) {
    c10::cuda::CUDAGuard guard(scratch.device());
    const int requests = scratch.size(0);
    const int layers = host_k.size(0);
    const int ctas = requests * 2;
    TORCH_CHECK(scratch.is_cuda() && scratch.sizes() == at::IntArrayRef({requests, 8}) &&
                host_k.sizes() == host_v.sizes() && host_k.element_size() == 1 &&
                host_k.dim() == 4 && host_k.size(2) == 4 && host_k.size(3) == 128 &&
                tails.is_contiguous() && tails.sizes() == at::IntArrayRef({layers,requests*4,3}) &&
                device_k.numel() == layers && device_v.numel() == layers, "invalid tail commit layout");
    auto stream = at::cuda::getCurrentCUDAStream();
    hipLaunchKernelGGL(commit_with_device_tail, dim3(ctas * layers), dim3(256), 0, stream,
        scratch.data_ptr<int64_t>(), host_locs.data_ptr<int32_t>(), accepted.data_ptr<int32_t>(),
        prefix.data_ptr<int32_t>(), reinterpret_cast<const uint64_t*>(device_k.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(device_v.data_ptr<int64_t>()),
        static_cast<uint64_t*>(mapped(host_k)), static_cast<uint64_t*>(mapped(host_v)),
        tails.data_ptr<int32_t>(), requests, ctas, host_k.stride(0) / sizeof(uint64_t));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
