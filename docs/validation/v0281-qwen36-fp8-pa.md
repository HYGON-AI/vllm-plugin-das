# vLLM 0.28.1 Qwen3.6 FP8 paged-attention validation

Date: 2026-11-28. Hardware: one BW1100 / gfx938 HCU (147440 MiB).

## Scope and failures reproduced

The unmodified container fails Qwen3.6-27B startup with
`RuntimeError: PA tensors must be contiguous`.

The initial PR 553 commit (`3e470d75`) fixes single-token decode but still
fails during MTP initialization. Extending the contiguous adaptation to
uniform multi-token verification exposes another vendor restriction:
`PA K/V cache exceeds kernel indexing limits for this kernel path` when
virtual pages span the full allocated cache.

A contiguous-control experiment also reproduces the D256/KV4 E4M3
non-unit-descale numerical failure documented in PR 551. The control and
remapped PA outputs are byte-identical: page remapping is not the cause.

## Follow-up implementation

- Mirror the native 64-token PA admission for both decode and uniform MTP.
- Keep the zero-copy virtual page remapping for qualified native PA shapes.
- Fold scalar K descales into the small query tensor and V descales into
  the output. Dequantize FP8 queries to BF16 and clear native descales to
  prevent ignored scales or future double application.
- For D256/KV4 E4M3 and oversized/non-integral virtual page spans, use Core's
  stride-aware unified-attention kernel with FP8-per-tensor KV quantization.
  Preserve physical cache pages, scalar descales, output aliases, and device
  metadata; do not copy or gather the whole allocated KV cache.
- Preserve the existing B16 strided-prefix and softcap fallbacks from the
  updated `v0.28.1-dev` base.

This is a correctness/compatibility fallback, not a throughput optimization
claim. It follows PR 551's geometry qualification and small-tensor scale
adaptation, but uses the Core stride-aware kernel instead of a gathered-KV
composition on this 0.28.1 runtime.

## Runtime

- Python 3.12.12; Torch 2.11.0; HIP 7.2.26365.
- vLLM `0.28.1+dtk2610.torch2110.2610191943.g29071f`.
- Plugin wheel baseline `0.28.1rc1.dev491+dtk2610.torch2110.2611272111.gf78980`.
- FlashAttention `2.8.4+dtk2610.torch2110.2611272153.gece162`.
- The changed Python attention file is overlaid onto the existing wheel;
  no native extension or model-weight rebuild is involved.

## Reproduction command

Set `HIP_VISIBLE_DEVICES` to an idle physical HCU and use the local model path.
The validation server binds loopback on port 18553; the model/cache/parallelism
parameters below are unchanged from the reported failure.

```bash
export HIP_VISIBLE_DEVICES=1
export VLLM_KV_CACHE_LAYOUT=HND
vllm serve /model/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --host 127.0.0.1 --port 18553
```

## Regression commands

```bash
VLLM_PLUGINS=__disabled__ python -m pytest \
  tests/runtime_patch/test_flash_attention_and_pp.py -q

VLLM_PLUGINS=hcu HIP_VISIBLE_DEVICES=1 python -m pytest \
  tests/kernels/test_flash_pa_contiguous.py -q
```

The hardware tests use random nonzero fused HND pages, shuffled page IDs,
non-unit Q/K/V descales, BF16 and FP8 queries, noncontiguous output, and an
independent FP32 attention oracle. Graph replay changes queries, lengths and
page IDs after capture. D128 tests exercise native decode/MTP; D256 tests
exercise decode, MTP and prefill. Oracle tolerance is atol=0.003, rtol=0.04.

## Validation limits

- This is a smoke/regression qualification, not a model-accuracy benchmark.
- Text requests are checked; image/video requests are not qualified here.
- Configured maximum length and batch size are retained, but 262144-token
  requests and sustained 64-way concurrency are not stress-tested.
- No claims are made for DCP, attention-probability/LSE output, dropout,
  per-sequence/per-head non-scalar descales, or other HCU architectures.
## Recorded results

- CPU runtime contracts: **329 passed, 14 warnings** (9.00 s).
- Real HCU numerical / changed-input Graph replay: **10 passed** (26.60 s).
- A separate 16-case native/control experiment covers D128/D256, Q lengths
  1/4, BF16/FP8 queries and unit/non-unit scalar descales. Both dense-control
  equivalence and FP32-reference checks pass after scalar adaptation.
- Full model startup and CUDA Graph capture complete with the original
  TP1, HND, E4M3 KV, MTP3, prefix/mamba-cache and batch-token settings.
- **15 HTTP chat requests pass**: basic arithmetic, 256-token decode, three
  repeated long-prefix requests, eight concurrent shared-prefix requests,
  reasoning parsing and automatic function/tool-call parsing.
- Final metrics record 741 draft tokens, 669 accepted tokens and 23040
  prefix-cache hits. These are smoke-test counters, not benchmark results.
- Validated Python attention-file SHA256:
  `0900a475b8c9ad45dfdeb02b29b0193fca8a0d3f905522739c3d3ac062880f99`.

The initial 256-token reasoning budget truncated the model before its final
answer; the completed parser check uses a 1024-token budget. This was a test
budget issue, not an engine failure. A D128 long-prefill probe with deliberately
noncontiguous output still hits the unchanged vendor varlen rejection; that
non-PA geometry is not claimed by this qualification.