# GPTQ group-wise W4A16 AITER MoE

`--moe-backend aiter` previously mapped GPTQ experts to the WNA16 Triton
oracle but only redirected channel-wise compressed-tensors INT4 execution
through AITER. Group-wise GPTQ still executed vLLM's WNA16 expert kernels.

The grouped route now calls the shared AITER selector and executor using the
loaded canonical uint8 weights and their actual group size. It preserves those
weights for batch-dependent solution selection and vLLM fallback. Explicit
`triton`, asymmetric weights, expert parallelism, LoRA, router weights on input,
and unsupported activations retain the existing path. AITER no-solution falls
back; ABI/kernel errors propagate.

## Observed backend

On gfx936 with AITER 0.1.8, BF16, E=256, K=2048, intermediate=512,
top-k=8, group=128, the selector chooses **Triton** for M=1,4,32,64,256,2048,8192.
Explicit MOE_C and ASM probes report no solution for these shapes. This AITER
build dispatches `aiter.moe.aiter_moe` to
`boltops.fused_moe.triton.fused_moe.fused_experts_impl`. It uses untuned default
configs because the matching tuning files are absent; this is functional
integration, with no throughput improvement claim.

The runtime logs `HCU grouped W4A16 MoE executes AITER solution=triton` after
an actual call. The oracle can still report `TRITON` because the integration
retains vLLM's modular WNA16 prepare/finalize and replaces expert execution.

## Checks

- GPU independent unpack/dequantize + PyTorch MoE reference: M=1,4,32,256;
  4 passed, maximum absolute error 0.00023540.
- Grouped routing/unsupported semantics: 11 passed.
- Existing MoE/quantization runtime patch suites: 425 passed, 2 failed.
  Both failures are in SlimQuant W4A8 BoltOps public-signature checks; the
  installed BoltOps API differs from that separately audited contract.
- `python setup.py bdist_wheel`: passed on the prepared HCU stack.

Run the GPU accuracy check in its own process:

```bash
HIP_VISIBLE_DEVICES=4 python -m pytest -s -q \
  tests/accuracy/test_grouped_w4a16_moe_accuracy.py
```

## Serving

```bash
MODEL_PATH=/path/to/QGPT_Flash5.2_gptq_int4 \
  bash examples/serve_gptq_aiter.sh
```

The example retains MTP=3, FP8 E5M2 KV, V2 runner, TP=1, HND, and fine-grained
Mamba prefix caching. It adds `--skip-mm-profiling` for the text validation
profile because the checkpoint's default video-encoder profiling is expensive.
Image/video inference is outside this validation scope. Configure multimodal
memory separately before using the profile for image/video requests.
