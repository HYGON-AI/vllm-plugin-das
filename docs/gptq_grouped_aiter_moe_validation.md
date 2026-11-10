# GPTQ group-wise W4A16 AITER MoE

`--moe-backend aiter` now routes symmetric group-wise GPTQ W4A16 experts
through the shared AITER selector and executor. TP=2 QGPT shapes select the
AITER MOE_C kernel, which is backed by the HIPC C interface in the installed
AITER wheel.

The grouped route now calls the shared AITER selector and executor using the
loaded canonical uint8 weights and their actual group size. It preserves those
weights for batch-dependent solution selection and vLLM fallback. Explicit
`triton`, weights with explicit zero points, expert parallelism, LoRA, router
weights on input, clamped SwiGLU, non-default SiLU parameters, and unsupported
activations retain the existing path. Explicit zero points fall back even when
their packed values are `0x88`; the adapter only synthesizes implicit symmetric
zero points for MOE_C.
AITER no-solution falls back; ABI/kernel errors propagate.

## Observed backend

On gfx936 with AITER 0.1.8, BF16, E=256, K=2048, TP=2 local
`N1=512,N2=2048,K=2048,top-k=8,group=128` selects **MOE_C** for M=1,4,32.
The returned config uses `MODE=11001301` or `11001303`, `BLOCK_SIZE_M=16`,
`need_shuffle=False`, and `need_shuffle_scale=True`. TP=1 keeps the Triton
selection for the corresponding shape. The grouped runtime passes GPTQ's
packed symmetric qzeros to MOE_C; Triton receives the canonical scale path.

The immutable gfx936 CI image has an older AITER scale shuffler whose MOE_C
config does not expose `block_size` and assumes group 32. It may still select
MOE_C for the TP=2 group-128 shape, but its scale shuffle cannot handle the
two groups in the second GEMM. The adapter falls back to WNA16 on that image.
The hardware tests check this fallback without skipping, and still execute
MOE_C and compare against the dequantized reference when the installed AITER
supports group 128.

## Weight updates and captured scales

MOE_C may require shuffled scale copies. The grouped route caches those copies
on a scale owner that survives kernel rebuilds. The worker snapshots that owner
at `start_weight_update`; both model-runner reload paths snapshot it before
writing weights. They refresh its scale tensors in place from the new source
scales before graph replay and rebind cache keys when source parameters change.
Clearing a cache would leave captured HIP graph addresses stale.
Checkpoint reload can restore the original layer Parameters after building a
new MoE kernel. The refresh path binds the new experts' scale descriptors to
those final layer Parameters before updating captured copies, so a subsequent
kernel-format update reads the current values.

The reload regression covers source `copy_()` and `.data.copy_()` updates,
rebuilding the quant config during transfer, repeated parameter replacement
with eager forwards, and direct replay without an intervening eager forward.
A small HIP graph test checks both update APIs against the
captured shuffled scale addresses on HCU; it does not measure full-model
reload accuracy or serving performance.

## Checks

- GPU independent unpack/dequantize + PyTorch MoE reference: M=1,4,32,256;
  4 passed, maximum absolute error 0.00023540.
- Grouped routing/unsupported semantics: 11 passed.
- TP=2 group-128 MOE_C/HIPC execution: M=1,4 passed; output shape and BF16
  finiteness checks passed. The production adapter also passed an independent
  unpack/dequantized MoE reference at M=1,4 on gfx936; maximum absolute error
  was 0.00024158. All four HCU test files are registered for gfx936 CI.
- Existing MoE/quantization runtime patch suites: 425 passed, 2 failed.
  Both failures are in SlimQuant W4A8 BoltOps public-signature checks; the
  installed BoltOps API differs from that separately audited contract.
- `python setup.py bdist_wheel`: passed on the prepared HCU stack.
- Full QGPT checkpoint on one gfx936, V2 runner, MTP=3, FP8 E5M2 KV,
  HND, prefix caching and fine-grained Mamba cache: service started with
  `--skip-mm-profiling`; piecewise and full HIP graph capture completed.
  Three chat smoke requests and ten repeated-prefix requests (including
  eight concurrent requests) returned HTTP 200. All arithmetic requests
  returned `42`. Prometheus reported 8,192 prefix-cache hit tokens and
  38 accepted speculative tokens out of 108 draft tokens during this small
  smoke workload. These counters demonstrate execution, not model accuracy
  or throughput benchmarks.

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
