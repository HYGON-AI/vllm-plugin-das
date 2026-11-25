# SPDX-License-Identifier: Apache-2.0
"""Qualified VERIFY-only INT8 multiplication over unchanged KN weight storage."""
from functools import lru_cache

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel


@lru_cache(maxsize=None)
def supports_w8a8_fusion(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0] == "gfx938"


class VerifyTransposeInt8Kernel(TritonInt8ScaledMMLinearKernel):
    """Compute (W_NK @ Q_MK.T).T for the four qualified M128 shapes."""

    def __init__(self, config, layer_param_names, *, row_major=False):
        super().__init__(config, layer_param_names)
        self._row_major = row_major

    def supports_shape(self, layer, shape, dtype, device, bias=None):
        w, ws, ins, izp, azp = self._get_layer_params(layer)
        if (len(shape) < 2 or dtype != torch.bfloat16 or device.type != "cuda"
                or w.ndim != 2 or w.shape[1] != 6144
                or w.shape[0] not in (3072, 6144, 8192, 12288)
                or torch.Size(shape).numel() != 128 * w.shape[0]
                or shape[-1] != w.shape[0]
                or any(value is not None for value in (bias, ins, izp, azp))):
            return False
        k, n = w.shape
        if (w.dtype != torch.int8 or w.stride() != (1, k)
                or ws.dtype != torch.float32 or ws.shape != (n, 1)
                or not ws.is_contiguous() or ws.device != device or w.device != device):
            raise ValueError("VERIFY transposed INT8 requires unchanged KN channelwise storage")
        return True

    def supports_input(self, layer, x, bias=None):
        return isinstance(x, torch.Tensor) and self.supports_shape(
            layer, x.shape, x.dtype, x.device, bias,
        )

    def apply_weights(self, layer, x, bias=None):
        if not self.supports_input(layer, x, bias):
            return super().apply_weights(layer, x, bias)
        q, scales = self._quantize(x)
        output = self.apply_prequantized(
            layer, q, scales, x.dtype, row_major=self._row_major and supports_w8a8_fusion(x.device),
        )
        return output.contiguous().view(*x.shape[:-1], output.shape[-1])

    def apply_weights_strided(self, layer, x):
        q, scales = self._quantize(x)
        return self.apply_prequantized(layer, q, scales, x.dtype)

    @staticmethod
    def _quantize(x):
        q, scales, zp = ops.scaled_int8_quant(
            x.reshape(128, x.shape[-1]).contiguous(), None, None, symmetric=True,
        )
        if zp is not None:
            raise RuntimeError("unexpected dynamic symmetric INT8 quantizer zero point")
        return q, scales

    def apply_prequantized(self, layer, q, scales, dtype, *, row_major=False):
        w, ws, _, _, _ = self._get_layer_params(layer)
        k, n = w.shape
        m = 128
        if (q.shape != (m, k) or q.dtype != torch.int8 or not q.is_contiguous()
                or scales.shape != (m, 1) or scales.dtype != torch.float32
                or not scales.is_contiguous() or scales.device != q.device or q.device != w.device):
            raise RuntimeError("unexpected dynamic symmetric INT8 quantizer ABI")
        if row_major and k in (3072, 6144):
            from .minimax31_w8a8_direct import scaled_mm_row_major

            return scaled_mm_row_major(q, w, scales, ws)

        from lightop._lmslim_native.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm

        # Keep the library's assembly GEMM and its native output for larger K.
        ok, out = hipblaslt_w8a8_channelwise_gemm(w.t(), q, ws, scales, n, m, k, "NT", dtype)
        if (not ok or out.shape not in ((n, m), (1, n, m)) or not out.is_contiguous()
                or out.dtype != dtype or out.device != q.device):
            raise RuntimeError("unexpected VERIFY transposed Lt return ABI")
        return out.view(n, m).t()
