# vllm.model_executor.kernels.linear.scaled_mm.pytorch.
# ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm
from math import prod

def patch_fp8_scaled_mm():
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        TorchFP8ScaledMMLinearKernel,
    )
    from vllm_hcu.platforms import envs as henvs

    if getattr(ChannelWiseTorchFP8ScaledMMLinearKernel, "_hcu_fp8_patch_applied", False):
        return

    original_get_output_padding = TorchFP8ScaledMMLinearKernel.get_output_padding
    original_apply_scaled_mm = ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm

    def new_get_output_padding(self):
        return None

    TorchFP8ScaledMMLinearKernel.get_output_padding = new_get_output_padding
    TorchFP8ScaledMMLinearKernel._hcu_rocm_no_output_padding = True

    def new_apply_scaled_mm(
        self,
        *,
        A,
        B,
        As,
        Bs,
        out_dtype,
        bias,
        output_shape,
    ):
        if not henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:
            return original_apply_scaled_mm(
                self,
                A=A,
                B=B,
                As=As,
                Bs=Bs,
                out_dtype=out_dtype,
                bias=bias,
                output_shape=output_shape,
            )

        from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm

        m = A.shape[0]
        k = A.shape[1]
        n = B.shape[0]
        result_shape = [*output_shape[:-1], n]
        num_output_rows = prod(result_shape[:-1]) if len(result_shape) > 1 else result_shape[0]

        _, output = hipblaslt_w8a8_channelwise_gemm(
            a=A,
            b=B,
            scale_a=As,
            scale_b=Bs,
            m=m,
            n=n,
            k=k,
            transpose_flag="NT",
            out_dtype=out_dtype,
            bias=bias,
        )

        output = output.reshape(-1, n)
        output = output.narrow(0, 0, num_output_rows)
        return output.view(*result_shape)

    ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm = new_apply_scaled_mm
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_fp8_patch_applied = True
    ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_original_get_output_padding = (
        original_get_output_padding
    )
