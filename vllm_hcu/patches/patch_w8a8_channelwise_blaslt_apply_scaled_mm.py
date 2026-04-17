#vllm.model_executor.kernels.linear.scaled_mm.pytorch.ChannelWiseTorchFP8ScaledMMLinearKernel apply_scaled_mm
def patch_fp8_scaled_mm():
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel
    )

    def new_apply_scaled_mm(self, A, B, As, Bs, out_dtype, bias, output_shape):
        from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm

        m = A.shape[0]
        k = A.shape[1]
        n = B.shape[0]

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

        return output.view(m, n)

    # 打 patch
    ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm = new_apply_scaled_mm