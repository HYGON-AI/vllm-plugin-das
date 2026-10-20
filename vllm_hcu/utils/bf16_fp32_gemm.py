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
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_HIPBLASLT_DISABLED = False


@functools.lru_cache(maxsize=1)
def _jit_hipblaslt_bf16_fp32() -> Any:
    import torch.utils.cpp_extension

    source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <hipblaslt/hipblaslt.h>

    #define CHECK_HIPBLASLT(expr) do { \
        hipblasStatus_t status = (expr); \
        TORCH_CHECK( \
            status == HIPBLAS_STATUS_SUCCESS, \
            #expr, " failed with hipBLASLt status ", static_cast<int>(status)); \
    } while (false)

    struct MatmulDescGuard {
        hipblasLtMatmulDesc_t value = nullptr;
        ~MatmulDescGuard() {
            if (value != nullptr) {
                hipblasLtMatmulDescDestroy(value);
            }
        }
    };

    struct MatrixLayoutGuard {
        hipblasLtMatrixLayout_t value = nullptr;
        ~MatrixLayoutGuard() {
            if (value != nullptr) {
                hipblasLtMatrixLayoutDestroy(value);
            }
        }
    };

    torch::Tensor linear_bf16_fp32(torch::Tensor X, torch::Tensor W) {
        TORCH_CHECK(X.dim() == 2, "X must be a 2-D tensor");
        TORCH_CHECK(W.dim() == 2, "W must be a 2-D tensor");
        TORCH_CHECK(X.scalar_type() == torch::kBFloat16, "X must be BF16");
        TORCH_CHECK(W.scalar_type() == torch::kBFloat16, "W must be BF16");
        TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
        TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
        TORCH_CHECK(X.size(1) == W.size(1), "X and W have incompatible K dimensions");

        int batch = X.size(0);
        int in_features = X.size(1);
        int out_features = W.size(0);

        auto Y = torch::empty(
            {batch, out_features},
            torch::dtype(torch::kFloat32).device(X.device()));

        static thread_local hipblasLtHandle_t handle = nullptr;
        if (handle == nullptr) {
            CHECK_HIPBLASLT(hipblasLtCreate(&handle));
        }

        MatmulDescGuard matmul_desc;
        CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(
            &matmul_desc.value, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        int transA = HIPBLAS_OP_T;
        int transB = HIPBLAS_OP_N;
        CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            matmul_desc.value, HIPBLASLT_MATMUL_DESC_TRANSA, &transA,
            sizeof(transA)));
        CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            matmul_desc.value, HIPBLASLT_MATMUL_DESC_TRANSB, &transB,
            sizeof(transB)));

        MatrixLayoutGuard layoutA;
        MatrixLayoutGuard layoutB;
        MatrixLayoutGuard layoutC;
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(
            &layoutA.value, HIP_R_16BF, in_features, out_features,
            in_features));
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(
            &layoutB.value, HIP_R_16BF, in_features, batch, in_features));
        CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(
            &layoutC.value, HIP_R_32F, out_features, batch, out_features));

        float alpha = 1.0f;
        float beta = 0.0f;
        hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        CHECK_HIPBLASLT(hipblasLtMatmul(
            handle,
            matmul_desc.value,
            &alpha,
            W.data_ptr(), layoutA.value,
            X.data_ptr(), layoutB.value,
            &beta,
            Y.data_ptr(), layoutC.value,
            Y.data_ptr(), layoutC.value,
            nullptr,
            nullptr,
            0,
            stream));

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


def _linear_bf16_fp32_op(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Run the eager BF16-to-FP32 GEMM implementation."""
    return linear_bf16_fp32(x, weight)


def _linear_bf16_fp32_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Describe the output shape without tracing into the backend."""
    output_features = (
        weight.shape[1]
        if weight.ndim == 2 and x.ndim == 2 and weight.shape[0] == x.shape[-1]
        else weight.shape[0]
    )
    return torch.empty(
        (x.shape[0], output_features),
        dtype=torch.float32,
        device=x.device,
    )


direct_register_custom_op(
    op_name="hcu_linear_bf16_fp32",
    op_func=_linear_bf16_fp32_op,
    mutates_args=[],
    fake_impl=_linear_bf16_fp32_fake,
)
