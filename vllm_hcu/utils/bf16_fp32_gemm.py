# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""BF16xBF16 -> FP32 GEMM helpers."""

from __future__ import annotations

import functools
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_HIPBLASLT_DISABLED = False


@functools.lru_cache(maxsize=1)
def _jit_hipblaslt_bf16_fp32() -> Any:
    import torch.utils.cpp_extension

    source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <hipblaslt/hipblaslt.h>
    #include <unordered_map>

    void check_blas(hipblasStatus_t status) {
        TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipblasLt setup failed, status=", int(status));
    }

    torch::Tensor linear_bf16_fp32(torch::Tensor X, torch::Tensor W) {
        TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(), "BF16 FP32 GEMM requires one GPU device");
        TORCH_CHECK(X.dim() == 2 && W.dim() == 2 && X.size(1) == W.size(1), "invalid NT GEMM shape");
        TORCH_CHECK(X.is_contiguous() && W.is_contiguous(), "NT GEMM requires contiguous operands");
        TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16, "NT GEMM requires BF16 inputs");
        c10::cuda::CUDAGuard device_guard(X.device());
        int batch = X.size(0);
        int in_features = X.size(1);
        int out_features = W.size(0);

        auto Y = torch::empty(
            {batch, out_features},
            torch::dtype(torch::kFloat32).device(X.device()));

        static thread_local std::unordered_map<int, hipblasLtHandle_t> handles;
        auto& handle = handles[X.get_device()];
        if (handle == nullptr) {
            TORCH_CHECK(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS, "hipblasLtCreate failed");
        }

        struct Descriptors {
            hipblasLtMatmulDesc_t matmul = nullptr;
            hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
            ~Descriptors() {
                if (matmul) hipblasLtMatmulDescDestroy(matmul);
                if (a) hipblasLtMatrixLayoutDestroy(a);
                if (b) hipblasLtMatrixLayoutDestroy(b);
                if (c) hipblasLtMatrixLayoutDestroy(c);
            }
        } descriptors;
        auto& matmul_desc = descriptors.matmul;
        check_blas(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        int transA = HIPBLAS_OP_T;
        int transB = HIPBLAS_OP_N;
        check_blas(hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        check_blas(hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

        auto& layoutA = descriptors.a;
        auto& layoutB = descriptors.b;
        auto& layoutC = descriptors.c;
        check_blas(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, in_features, out_features, in_features));
        check_blas(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, in_features, batch, in_features));
        check_blas(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, out_features, batch, out_features));

        float alpha = 1.0f;
        float beta = 0.0f;
        hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        auto status = hipblasLtMatmul(
            handle,
            matmul_desc,
            &alpha,
            W.data_ptr(), layoutA,
            X.data_ptr(), layoutB,
            &beta,
            Y.data_ptr(), layoutC,
            Y.data_ptr(), layoutC,
            nullptr,
            nullptr,
            0,
            stream
        );

        TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipblasLtMatmul failed, status=", int(status));
        return Y;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("linear_bf16_fp32", &linear_bf16_fp32,
              "hipblasLt BF16xBF16 -> FP32 linear");
    }
    """
    return torch.utils.cpp_extension.load_inline(
        name="vllm_linear_bf16_fp32_hipblaslt",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lhipblaslt"],
        verbose=False,
    )


def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T`` with BF16 inputs and FP32 output.

    On HCU/ROCm, this mirrors SGLang's fast ``hipblasLtMatmul`` path when
    ``VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT`` is enabled. Other cases fall back
    to PyTorch's native matmul.
    """
    global _HIPBLASLT_DISABLED

    # HCU VLLM_USE_NN stores non-quantized linear weights as [in_features,
    # out_features]. The upstream helper assumes [out_features, in_features].
    # Detect the NN layout here so callers can stay aligned with dpsk code.
    if x.dim() == 2 and weight.dim() == 2 and weight.shape[0] == x.shape[-1]:
        return torch.mm(x, weight, out_dtype=torch.float32)

    fallback_reason = None
    if _HIPBLASLT_DISABLED:
        fallback_reason = "previous hipblasLt failure"
    elif os.environ.get("VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT", "1").lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        fallback_reason = "VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT disabled"
    elif torch.version.hip is None:
        fallback_reason = "torch.version.hip is None"
    elif x.dtype != torch.bfloat16:
        fallback_reason = f"x dtype is {x.dtype}"
    elif weight.dtype != torch.bfloat16:
        fallback_reason = f"weight dtype is {weight.dtype}"
    elif not x.is_cuda:
        fallback_reason = "x is not CUDA/ROCm"
    elif not weight.is_cuda:
        fallback_reason = "weight is not CUDA/ROCm"
    elif x.dim() != 2:
        fallback_reason = f"x dim is {x.dim()}"
    elif weight.dim() != 2:
        fallback_reason = f"weight dim is {weight.dim()}"

    if fallback_reason is not None:
        return torch.mm(x, weight.T, out_dtype=torch.float32)

    try:
        return _jit_hipblaslt_bf16_fp32().linear_bf16_fp32(x, weight)
    except Exception as e:
        _HIPBLASLT_DISABLED = True
        logger.warning_once(
            "Failed to use hipblasLt BF16->FP32 GEMM; falling back to torch.mm: %s",
            e,
        )
        return torch.mm(x, weight.T, out_dtype=torch.float32)
