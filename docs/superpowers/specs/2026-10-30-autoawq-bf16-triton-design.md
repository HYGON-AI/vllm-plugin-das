# AutoAWQ BF16 Triton Compatibility Design

## Intent

Verify that HCU `auto_awq` remains usable with both FP16 and BF16 activations.
The production DTK Triton 3.6 frontend deliberately accepts BF16 and FP16 dot
output types, so the small-batch AutoAWQ GEMM must retain the existing kernel
while a real-device regression protects that DTK contract.

## Constraints

- Cover `M=1`, `M=255`, and `M=256` with `K=N=128`, group size 128.
- Exercise the installed production Triton kernel rather than replacing it
  with a stub or a plugin-owned copy.
- Preserve FP16 behavior and the `M>=256` dequantize-plus-matmul path.
- Keep LightOp disabled in the BF16 fallback regression so the standard
  AutoAWQ route is exercised.
- Do not alter GPTQ, MoE WNA16, or unrelated quantization dispatch.

## Approaches

1. Keep the DTK kernel and add real gfx936/gfx938 CI coverage. This is the
   selected approach because production `semantic.py` accepts BF16/FP16 dot
   output and real gfx936 execution succeeds at every token boundary.
2. Patch the GEMM to accumulate in FP32. This was rejected because it would
   change a working DTK kernel's numerical and performance behavior to solve a
   restriction observed only in a different Triton distribution.
3. Reject BF16 or force it through dequantize-plus-matmul. This was rejected
   because it removes or slows a dtype path already supported by production.

## Verification

Compile and execute the real production DTK Triton kernels on HCU for BF16 at
the three M boundaries. Cover both LightOp disabled and an otherwise eligible
LightOp shape that falls back solely because its activation dtype is BF16, then
run the focused platform/AutoAWQ suites and CI selector tests.
