# HY4 HIPBLASLt FP8 value projection

Build the plugin against DTK with HIPBLASLt development headers and library,
then set `VLLM_HCU_HY4_FP8_V_UP_PROJ=1` before starting vLLM. The default is `0`.
The master switch `VLLM_HCU_USE_CUSTOM_OPS` must also be enabled (its default).
Setting it to `0` disables this path even when the FP8 sub-switch is `1`,
before FP8 dtype checks, extension imports, or weight quantization. The
original projection remains selected for both BF16 and FP16 activations.
This option requires BF16 activations and disabled AITER FP4/FP8 MLA BMM flags.
It applies only to HY4's `_v_up_proj`; query projection and other models keep
their existing paths, including DCP replicated-query weight preparation.

The projection quantizes its value weights once after loading to contiguous
`[heads, value, latent]` E4M3 FP8, using the platform's FN/FNUZ encoding. Each
forward dynamically quantizes the attention result with a tensor-wide FP32
scale. A HIPBLASLt strided batched GEMM accumulates in FP32 and writes BF16
directly into the caller's `[tokens, heads, value]` output buffer. Both FP8
operands use scalar dequantization scales shared by all heads. BF16 weights
remain available to the attention backend.

BLASLt plans cache descriptors and algorithms per host thread, device, dtype
and shape. They reuse PyTorch's managed BLASLt handle rather than creating a
handle (and its internal device buffers) for every token count. PyTorch's
handle pool owns the handle lifetime; shape plans do not destroy it. Both
non-null scalar FP8 scale pointers are installed before heuristic selection
and refreshed on each call. Warm up each token shape before CUDA graph
capture; a first call during capture raises an
error. Each plan's workspace is limited to 4 MiB. Allocation uses PyTorch's
stream-aware allocator. A missing BLASLt algorithm or unsupported format
raises an error instead of silently switching precision.

Portable tests cover weight layout, projection orchestration, output-buffer
aliasing and padding, empty batches, and configuration checks:

```bash
python -m pytest tests/accuracy/test_hyv4_fp8_v_up_proj.py -m 'not hcu' -q
```

On an HCU with the rebuilt extension, run native accuracy and graph replay
tests, including changed scale allocations, modified device scales during
replay, and driver-visible memory growth over 64 additional token counts:

```bash
python -m pytest tests/accuracy/test_hyv4_fp8_v_up_proj.py -m hcu -q
```

Run the memory-growth check on an otherwise idle HCU. It clears PyTorch's
unused allocator cache before measuring driver free memory, so raw BLASLt
allocations are included. The 128 MiB tolerance permits lazy library setup
while detecting the previous per-shape handle growth. The 4 MiB workspace
limit applies only to GEMM workspace, not the managed handle's internal buffers.

FP8 introduces rounding relative to the BF16 projection. Real-model accuracy,
end-to-end latency, and native HCU graph replay must be validated before making
this path the default. The implementation makes no measured performance claim.
