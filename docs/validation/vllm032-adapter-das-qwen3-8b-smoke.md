# vLLM 0.32 + adapter-das Qwen3-8B service smoke

Validation date: 2026-10-28

## Candidate

- vLLM: `0.32.0+rocm723`
- adapter-das distribution: `sgl-kernel 0.1.0`
- DAS extension: `hcu-vllm-das-plugin 0.2.0.dev0`
- model: `/model/Qwen3-8B`
- hardware: one `gfx936` card

The installed integration verifier resolved the active platform to
`hcu_adapter.bridges.vllm.HcuPlatform` and the `das_hcu` extension to active
before this service run.

## Service command

The shared node has an older system vLLM ahead of ordinary virtual-environment
paths, so the candidate site-packages directory was placed first explicitly.
This is not necessary on a clean target environment.

```bash
PYTHONPATH=/dev/shm/adapter-vllm032-contract-20261028/lib/python3.12/site-packages \
VLLM_PLUGINS=hcu_adapter \
HCU_ADAPTER_HOST=vllm \
VLLM_HCU_USE_CUSTOM_OPS=1 \
VLLM_HCU_USE_CUSTOM_RMS_NORM=1 \
VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1 \
HIP_VISIBLE_DEVICES=0 HSA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/dev/shm/hcu-vllm032-work/cache \
TMPDIR=/dev/shm/hcu-vllm032-work/tmp \
/dev/shm/adapter-vllm032-contract-20261028/bin/python \
  -m vllm.entrypoints.cli.main serve /model/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --served-model-name Qwen3-8B \
  --attention-backend TRITON_ATTN \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.35 \
  --enable-prefix-caching \
  --max-num-seqs 1
```

The command deliberately omits `--enforce-eager`. EngineCore reported:

- `enforce_eager=False`;
- compilation mode `VLLM_COMPILE`;
- CUDA Graph mode `FULL_AND_PIECEWISE`;
- successful PIECEWISE and FULL graph capture;
- TRITON_ATTN with LBNHC KV-cache layout.

## Online result

The service reached `Application startup complete`. Requests must bypass the
node HTTP proxy when using localhost, for example `curl --noproxy '*' ...`.

| Check | Result |
| --- | --- |
| `GET /health` | HTTP 200 |
| `GET /version` | HTTP 200, `0.32.0` |
| `GET /v1/models` | HTTP 200, `Qwen3-8B`, max length 512 |
| `POST /v1/chat/completions` | HTTP 200, content `42`, `finish_reason=stop` |
| response fingerprint | `vllm-0.32.0-b2902db1` |

The deterministic request disabled Qwen thinking and used temperature 0,
seed 0, and 32 output tokens.

## Teardown

The API server and EngineCore exited with status 0 after SIGINT. vLLM emitted
an `EngineDeadError` from its asynchronous output handler during the immediate
abort shutdown path; it occurred after successful inference and resource
teardown had started. No vLLM or EngineCore process remained, and all eight
cards returned to 0% utilization and the 2 MiB VRAM baseline.

This is a service smoke gate, not model parity, HumanEval-64, long-running,
concurrency, topology, or production approval evidence.
