# DeepSeek-V4 ROCm FlashMLA sparse decode

## Selection and scope

Set `VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE=1` before starting vLLM. The default is `0`, which retains `DeepseekV4ROCMAiterMLAAttention._forward_decode` and its ROCm sparse kernel. The switch changes decode only; prefill, weights, and request handling remain on their existing paths. An unsupported cache layout or absent FlashMLA extension raises an error rather than selecting another kernel silently.

The five-layer test model's `compress_ratios` begin `[0, 0, 4, 128, 4]`: layers 0–1 use SWA-only, layers 2/4 use C4A, and layer 3 uses C128A. The model has a per-head attention sink. The actual ROCm model class is `vllm.models.deepseek_v4.amd.rocm.DeepseekV4ROCMAiterMLAAttention`; the plugin's separate `deepseek_v4_attention.py` class is not instantiated by `amd/model.py`. The installed `flash_mla` version checked on 2026-09-22 was `1.2.0+dtk2604.torch2110.2609011016.g9e15d1`. SGLang's DeepSeek-V4 HIP radix backend calls `hip_flash_mla.flash_mla_with_kvcache_entrypoint`, whose default `kernel` backend calls `sgl_kernel.flash_mla.flash_mla_with_kvcache`.

FlashMLA receives one fresh tile scheduler object per present layer type and decode step. The C4A path maps local top-k to dense global KV slots using vLLM's existing mapper; C128A already has dense global slots. When the switch is enabled, the ROCm metadata subclasses now reuse their base builders directly and skip the AITER-only ragged SWA and C128A conversions. The disabled-switch path still uses the original subclass builds. Both caches retain the existing 584-byte `fp8_ds_mla` UE8M0 layout. Query and output width are 512. The SWA and compressed caches, lengths, sink, and softmax scale are passed unchanged.

## Checks performed

```bash
cd /home/zhoujie6/vllm-plugin-das-bw1000
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/runtime_patch/test_rocm_flashmla_decode_adapter.py
HIP_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python tests/accuracy/check_dsv4_flashmla_decode_gpu.py
```

The seven CPU contract cases passed (SWA-only/C4A/C128A, batch 1/8), including disabled-switch fallback, the builder inits releasing the AITER-only ragged buffers when the switch is on, and the guard rejecting an unavailable FlashMLA extension. The contract test also asserts that a 32-local-head query (the TP2 shape) raises rather than reaching a missing kernel instantiation: on 2026-09-22 an on-device probe found the installed FP8 sparse decode kernel supports local query head counts ≤16, 64, and 128, and fails on 32, so the decode path guards that set explicitly.

GPU tensor comparison used real vLLM UE8M0 quantization, batch 1/8, and lengths 17/93/6000. The largest absolute difference from `rocm_sparse_attn_decode` was 0.0078125, below the script's 0.01 bound. Length 6000 makes the C128A compressed side nonempty (46 of 64 slots), so the previously-untested nonempty-compressed C128A path is now covered on device and passes. CUDA Graph capture and three replays succeeded for all three layer types at batch 8: same-values, changed-lengths (fill lengths 31 / extra 9, new query written into the captured buffers), and grown-lengths (128 / 46); the largest replay difference was 0.00390625. The changed-length replay passing confirms the planner kernel, when captured inside the graph, re-reads the `topk_length` device buffers on each replay rather than freezing the plan seen at capture — the correctness condition for FULL_DECODE_ONLY. This requires the sched captured in the graph to be fresh (`have_initialized=False`); the test warms up lazy HIP state with a throwaway sched first, matching vLLM's one-`FlashMLASchedMeta`-per-step build. The five-layer FlashMLA server started on one GPU and completed a 16-token request. This is a smoke test, not a complete request-output comparison.

## Full load and performance

The existing five-layer launch script is `/home/zhoujie6/bench_vllm_sglang/trace_vllm_first5.sh`; set the switch above for FlashMLA and `0` for the baseline, and use `/home/zhoujie6/bench_vllm_sglang/trace_vllm_first5_bench.sh` for the same 64 × (4096 input, 512 output) workload. Its `bench/throughput.log` is the unprofiled measurement. Run each configuration more than once when all eight GPUs are free. The existing historical AITER baseline was 2455.54 output tokens/s; it is not a new A/B measurement.

**Performance has not been measured for the new path.** The eight-GPU FlashMLA launch failed during worker initialization because GPU 0 had only 99.68 GiB free, below the launcher's 122.39 GiB requirement. The subsequent one-GPU AITER request comparison failed for the same reason (10.44 GiB free on GPU 1). A later read-only check showed about 90% memory use on every GPU. No other process was stopped. A full request-output comparison, repeated unprofiled A/B runs, and a decode trace confirming the new kernel, per-rank batches, and saved ragged metadata time remain required. A later GPU check showed roughly 49–50% memory use on all eight devices, which still leaves less than the launcher's required 122.39 GiB free per card. Communication time and the uneven vLLM rank batch distribution in the prior trace can limit end-to-end gains even if attention improves.
