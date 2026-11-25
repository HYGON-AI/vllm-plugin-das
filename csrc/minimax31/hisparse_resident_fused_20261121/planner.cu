#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include "plan_delta.h"

static void* mapped(const at::Tensor& input) {
  if (input.is_cuda()) return input.data_ptr();
  void* device = nullptr;
  const auto status = hipHostGetDevicePointer(&device,input.data_ptr(),0);
  TORCH_CHECK(status == hipSuccess, "host history must be device-mapped");
  return device;
}

void plan_delta(const at::Tensor& reqs,const at::Tensor& real,const at::Tensor& topk,
 const at::Tensor& ends,const at::Tensor& host,const at::Tensor& device,const at::Tensor& tags,
 const at::Tensor& hot,const at::Tensor& slots,const at::Tensor& ms,const at::Tensor& md,
 const at::Tensor& mc,const at::Tensor& vs,const at::Tensor& vd,const at::Tensor& vc,
 const at::Tensor& ranges,const at::Tensor& tails,
 const at::Tensor& hk, const at::Tensor& hv, const at::Tensor& dk, const at::Tensor& dv,
 int resident_base, int resident_first, int resident_count) {
 c10::cuda::CUDAGuard guard(reqs.device());
 TORCH_CHECK(reqs.is_cuda() && reqs.scalar_type()==at::kLong && reqs.is_contiguous(),"reqs int64 CUDA contiguous");
 TORCH_CHECK(topk.scalar_type()==at::kInt && topk.dim()==3 && topk.size(0)==4 && topk.size(1)==reqs.numel()*8 && topk.size(2)==16,"TopK [4,B*8,16]");
 TORCH_CHECK(host.dim()==2 && device.dim()==2 && slots.dim()==2 && slots.size(1)<=2048,"unsupported page-map extent");
 TORCH_CHECK(device.size(1)-128==30720,"headwise replay requires 240 pages/head");
 TORCH_CHECK(hot.is_contiguous() && hot.sizes() == at::IntArrayRef({reqs.numel(), 8, 4, 64}),
             "expected contiguous [B,8,4,64] page spans");
 TORCH_CHECK(ranges.sizes() == md.sizes() && ranges.is_contiguous() &&
             tails.sizes() == at::IntArrayRef({reqs.numel()*4,3}) && tails.is_contiguous(),
             "incorrect delta-copy ranges or tail tracking");
 TORCH_CHECK(resident_count > 0 && resident_first >= 0 &&
             resident_first + resident_count <= host.size(1) / 128 &&
             resident_base >= (128 + host.size(0) * device.size(1)) / 128,
             "invalid reserved generated-page interval");
 TORCH_CHECK(hk.sizes() == hv.sizes() && dk.sizes() == dv.sizes() &&
             hk.is_contiguous() && hv.is_contiguous() && dk.is_contiguous() && dv.is_contiguous() &&
             hk.element_size() == 1 && hv.element_size() == 1 &&
             dk.element_size() == 1 && dv.element_size() == 1 &&
             hk.dim() == 3 && hk.size(1) == 4 && hk.size(2) == 128 &&
             dk.dim() == 3 && dk.size(1) == 4 && dk.size(2) == 128 &&
             dk.device() == reqs.device() && dv.device() == reqs.device() &&
             dk.size(0) >= (resident_base + host.size(0) * resident_count) * 128,
             "incompatible HiSparse host and device page storage");
 auto stream=at::cuda::getCurrentCUDAStream();
 hipLaunchKernelGGL((sglang::minimax_verify::plan_delta_spans_kernel<true>),dim3(reqs.numel()*4),dim3(128),
  2*slots.stride(0)*sizeof(int),stream,
  reqs.data_ptr<int64_t>(),real.data_ptr<int>(),topk.data_ptr<int>(),ends.data_ptr<int>(),
  host.data_ptr<int64_t>(),device.data_ptr<int64_t>(),tags.data_ptr<int64_t>(),hot.data_ptr<int>(),
  slots.data_ptr<int>(),ms.data_ptr<int64_t>(),md.data_ptr<int>(),mc.data_ptr<int>(),
  vs.data_ptr<int64_t>(),vd.data_ptr<int>(),vc.data_ptr<int>(),
  ranges.data_ptr<int>(),tails.data_ptr<int>(),
  (int)host.size(0),(int)host.stride(0),(int)device.stride(0),(int)slots.stride(0),
  (int)topk.stride(0),(int)topk.stride(1),(int)topk.stride(2),(int)device.size(1)-128, resident_base, resident_first, resident_count,
  static_cast<const uint4*>(mapped(hk)), static_cast<const uint4*>(mapped(hv)),
  static_cast<uint4*>(dk.data_ptr()), static_cast<uint4*>(dv.data_ptr()));
 TORCH_CHECK(hipGetLastError()==hipSuccess,"headwise planner launch failed");
}

