# HYV4 Flash on vLLM 0.33 (candidate)

The plugin registers `HYV4FlashForCausalLM` and
`HYV4FlashVLForConditionalGeneration`. Model registration and static import
closure are not evidence that a checkpoint can be served on HCU.

## Repository boundary and dependency

- Reuse the vLLM/HYV4 loading and projection paths where their ABI matches.
- Keep Flash IndexPool/SWA selection, cache writes and sinks in this plugin.
- On HCU, require `hcu_adapter.bridges.vllm_paged_mqa.get_paged_mqa_provider`
  from the companion adapter-das change. Older adapters fail with an actionable
  dependency error; do not silently fall through to a CUDA DeepGEMM provider.
- On CUDA, continue using vLLM's own DeepGEMM wrappers. No framework source
  files or global DeepGEMM bindings are rewritten.

## IndexPool contract

HCU uses E4M3FN query values `[rows, 1, heads, 128]`; query scales are folded
into FP32 head weights. The page64 uint8 cache is physically laid out as
`[64*128 key bytes | 64 FP32 scales]` per page, NOT interleaved token records
and NOT the SGLang preshuffled layout. The adapter returns FP32 logits and
uses each query's block-table row independently, including prefill chunks.
The bridge currently supports `next_n=1` only. It is a Torch correctness
candidate, not a performance-qualified native kernel.

The FlashMLA path runs indexer and attention in a single eager break on the
same stream. It must not call ROCm-AITER's private logical-topk event hook.
The current runtime explicitly rejects speculative decoding.

## Validation gates

CPU-only seam tests:

```sh
python -m pytest --noconftest tests/models/hy_v4/test_flash_runtime_contracts.py
```

These test provider selection, the v0.33 three-argument MX helper ABI and
FlashMLA orchestration. They do NOT import the complete model or execute
HCU kernels. Before this change is ready to merge, run the companion
adapter's numerical tests on HCU, changing-input graph replay, real vLLM
model import/config/cache setup, checkpoint loading, multi-request short
and long prefill/decode, sink accuracy and text/VL end-to-end comparisons.
Verify the installed wheels and selected backend; report hardware, DTK,
PyTorch and vLLM versions with results. PP/TP/DCP and quantized cache
configurations need their own qualification. Do not infer those from CPU
contracts. FlashMLA/sink provider ABI remains a device validation gate.

## Recorded focused HCU validation (2026-11-17)

On BW1100/gfx938, Python 3.12.12, Torch 2.13.0, HIP 7.2.26365 and
vLLM 0.33.0+dtk2610.torch2130.2611101131.g7364da:

```sh
python -m pytest --noconftest tests/models/hy_v4/test_flash_runtime_contracts.py tests/models/hy_v4/test_registration.py tests/models/hy_v4/test_flash_indexpool_provider.py
```

**20 passed** (14 upstream Torch deprecation warnings). The integration test
uses the actual Flash FWHT quantizer and Triton page writer, then scores the
produced shared/offset cache through adapter-das against independent FP32
math, including Q-scale folding and inactive writes. The companion adapter's
focused suite and existing CPU ops reported **76 passed**, including changing
input Graph replay. Text/VL/config module import smoke also passed; it emitted
an AITER HIP-runtime discovery warning, so AITER JIT availability is not claimed.

Source overlays did not replace installed packages. These results do not
establish real checkpoint loading, full model graph execution, text/VL
end-to-end accuracy, distributed serving, wheel installation or throughput.
