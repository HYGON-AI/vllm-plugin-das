#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include "plan_kernel.h"
#include "io_kernels.h"

static void* accessible(const at::Tensor& t) {
 if(t.is_cuda())return t.data_ptr();
 TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(),"host pool must be contiguous CPU storage");
 void* ptr=nullptr;auto rc=hipHostGetDevicePointer(&ptr,t.data_ptr(),0);
 TORCH_CHECK(rc==hipSuccess,"host pool must be hipHostRegisterMapped: ",hipGetErrorString(rc));
 return ptr;
}

void plan_headwise(const at::Tensor& reqs,const at::Tensor& real,const at::Tensor& topk,
 const at::Tensor& ends,const at::Tensor& host,const at::Tensor& device,const at::Tensor& tags,
 const at::Tensor& hot,const at::Tensor& slots,const at::Tensor& ms,const at::Tensor& md,
 const at::Tensor& mc,const at::Tensor& vs,const at::Tensor& vd,const at::Tensor& vc) {
 c10::cuda::CUDAGuard guard(reqs.device());
 TORCH_CHECK(reqs.is_cuda() && reqs.scalar_type()==at::kLong && reqs.is_contiguous(),"reqs int64 CUDA contiguous");
 TORCH_CHECK(topk.scalar_type()==at::kInt && topk.dim()==3 && topk.size(0)==4 && topk.size(1)==reqs.numel()*8 && topk.size(2)==16,"TopK [4,B*8,16]");
 TORCH_CHECK(host.dim()==2 && device.dim()==2 && slots.dim()==2 && slots.size(1)<=2048,"unsupported page-map extent");
 TORCH_CHECK(device.size(1)-128==30720,"headwise replay requires 240 pages/head");
 auto stream=at::cuda::getCurrentCUDAStream();
 hipLaunchKernelGGL((sglang::minimax_verify::plan_kernel<true>),dim3(reqs.numel()*4),dim3(128),
  2*slots.stride(0)*sizeof(int),stream,
  reqs.data_ptr<int64_t>(),real.data_ptr<int>(),topk.data_ptr<int>(),ends.data_ptr<int>(),
  host.data_ptr<int64_t>(),device.data_ptr<int64_t>(),tags.data_ptr<int64_t>(),hot.data_ptr<int>(),
  slots.data_ptr<int>(),ms.data_ptr<int64_t>(),md.data_ptr<int>(),mc.data_ptr<int>(),
  vs.data_ptr<int64_t>(),vd.data_ptr<int>(),vc.data_ptr<int>(),
  (int)host.size(0),(int)host.stride(0),(int)device.stride(0),(int)slots.stride(0),
  (int)topk.stride(0),(int)topk.stride(1),(int)topk.stride(2),(int)device.size(1)-128);
 TORCH_CHECK(hipGetLastError()==hipSuccess,"headwise planner launch failed");
}

void swap_headwise(const at::Tensor& src,const at::Tensor& dst,const at::Tensor& counts,
 const at::Tensor& real,const at::Tensor& hk,const at::Tensor& hv,const at::Tensor& dk,const at::Tensor& dv) {
 c10::cuda::CUDAGuard guard(src.device());
 TORCH_CHECK(src.is_cuda() && src.scalar_type()==at::kLong && src.dim()==2 && src.size(0)%4==0,"source [B*4,128] CUDA int64");
 TORCH_CHECK(dst.scalar_type()==at::kInt && src.strides()==dst.strides(),"plan strides differ");
 TORCH_CHECK(hk.element_size()==1 && hv.element_size()==1 && dk.element_size()==1 && dv.element_size()==1,"FP8 byte storage only");
 auto stream=at::cuda::getCurrentCUDAStream();
 hipLaunchKernelGGL((sglang::swap_mha_pages_tiled_kernel<256,4,4>),dim3(src.size(0)*4),dim3(256),0,stream,
  src.data_ptr<int64_t>(),dst.data_ptr<int>(),counts.data_ptr<int>(),real.data_ptr<int>(),
  (const uint4*)accessible(hk),(const uint4*)accessible(hv),(uint4*)dk.data_ptr(),(uint4*)dv.data_ptr(),src.stride(0));
 TORCH_CHECK(hipGetLastError()==hipSuccess,"headwise swap launch failed");
}

void commit_all(const at::Tensor& scratch,const at::Tensor& host_locs,const at::Tensor& accepted,
 const at::Tensor& sk,const at::Tensor& sv,const at::Tensor& hk,const at::Tensor& hv) {
 c10::cuda::CUDAGuard guard(scratch.device());
 TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type()==at::kLong && scratch.dim()==2 && scratch.size(1)==8,"scratch [B,8] CUDA int64");
 TORCH_CHECK(hk.element_size()==1 && hv.element_size()==1 && hk.dim()==4 && hk.size(2)==4 && hk.size(3)==128,"host layers FP8 [L,N,4,128]");
 TORCH_CHECK(sk.scalar_type()==at::kLong && sv.scalar_type()==at::kLong && sk.numel()==hk.size(0),"one source pointer per layer");
 const int batch=scratch.size(0),width=8,blocks=(batch*width+3)/4;
 auto stream=at::cuda::getCurrentCUDAStream();
 hipLaunchKernelGGL((sglang::commit_mha_all_layers_kernel<256>),dim3(sk.numel()*blocks),dim3(256),0,stream,
  scratch.data_ptr<int64_t>(),host_locs.data_ptr<int>(),accepted.data_ptr<int>(),
  (const uint64_t*)sk.data_ptr<int64_t>(),(const uint64_t*)sv.data_ptr<int64_t>(),
  (char*)accessible(hk),(char*)accessible(hv),batch,width,blocks,hk.stride(0),int64_t(512));
 TORCH_CHECK(hipGetLastError()==hipSuccess,"accepted KV commit launch failed");
}
