# Parallel integration

This layer contains real HCU model tests for parallel execution feature axes.

Current TP+EP coverage:

- `qwen3.5/Qwen3.5-35B-A3B` with TP4/EP4 and TP2/EP2 across the unquantized
  MoE backend paths.
- `vllm-w8a8-models/DeepSeek-R1-0528-Channel-INT8` with TP8 + expert
  parallelism across the INT8 MoE backend paths.

The Qwen3.5-35B-A3B matrix intentionally fails on backend/runtime errors so a
single run reports which path is broken:

- `aiter-auto-shuffle`: AITER automatically selects ASM, HIP C++, Triton, or
  CK and may prepare the selected weight layout.
- `aiter-auto-nonshuffle`: the same AITER automatic selection with weight
  shuffle disabled.
- `triton`: vLLM Triton unquantized MoE path.

The DeepSeek-R1-0528-Channel-INT8 matrix also fails on backend/runtime errors:

- `auto`: vLLM/HCU default backend selection.
- `target-triton`: explicit vLLM Triton INT8 MoE path.
- `dpsk-deep-gemm`: explicit HCU DeepSeek DeepGEMM MoE path.

DeepSeek-R1-0528-Channel-INT8 TP+EP requires visible `gfx938` HCU devices and
is skipped on `gfx936` because the model does not fit reliably there.

Override model paths with:

- `VLLM_HCU_QWEN35_35B_A3B_MODEL`
- `VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_MODEL`

Override DeepSeek tensor-parallel size with:

- `VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_TP`

Override vLLM memory reservation with:

- `VLLM_HCU_QWEN35_35B_A3B_GPU_MEMORY_UTILIZATION` (common Qwen override)
- `VLLM_HCU_QWEN35_35B_A3B_TP2_GPU_MEMORY_UTILIZATION` (TP2 override,
  default: `0.4`)
- `VLLM_HCU_QWEN35_35B_A3B_TP4_GPU_MEMORY_UTILIZATION` (TP4 override,
  default: `0.4`)
- `VLLM_HCU_DEEPSEEK_R1_CHANNEL_INT8_GPU_MEMORY_UTILIZATION` (default: `0.6`)

The TP-specific Qwen setting takes precedence over the common Qwen setting.
On a 144 GiB gfx938 device, the Qwen defaults reserve about 57.6 GiB per
rank and leave enough room for this short smoke test's model profiling and KV
cache. `0.15` leaves no room for KV cache in TP2.

Run only this coverage with:

```bash
python tools/run_patch_tests.py --suite model -- -k tp_ep
```

Run only the Qwen3.5-35B-A3B TP+EP path matrix with:

```bash
python tools/run_patch_tests.py --suite model -- -k qwen35_35b_a3b_tp_ep_smoke
```

The Qwen3.5-35B-A3B path matrix writes separate logs under
`/tmp/vllm-hcu-integration/logs/`, for example:

- `*_Qwen3.5-35B-A3B_tp-ep-smoke-tp4-ep4-aiter-auto-shuffle.log`
- `*_Qwen3.5-35B-A3B_tp-ep-smoke-tp2-ep2-aiter-auto-nonshuffle.log`
- `*_Qwen3.5-35B-A3B_tp-ep-smoke-tp2-ep2-triton.log`

The DeepSeek-R1-0528-Channel-INT8 path matrix also writes separate logs, for
example:

- `*_DeepSeek-R1-0528-Channel-INT8_tp-ep-smoke-auto.log`
- `*_DeepSeek-R1-0528-Channel-INT8_tp-ep-smoke-target-triton.log`
- `*_DeepSeek-R1-0528-Channel-INT8_tp-ep-smoke-dpsk-deep-gemm.log`
