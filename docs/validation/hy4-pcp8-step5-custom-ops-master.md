# HY4 PCP8 and custom-op master validation

Validation date: 2026-10-15 (UTC).

## Provenance and scope

- Plugin commit: `f76cfb6edb2df26ea6855ff304d6cfb79807141a`.
- Isolated plugin wheel:
  `vllm_hcu-0.28.1rc1.dev491+das.f76cfb6.dtk2604-cp310-cp310-linux_x86_64.whl`.
- Plugin wheel SHA-256:
  `3f1f991e833624f8f7c3cec9f4e76d4da9f28d2f83ecf2695fd623f7a1437cef`.
- Installed vLLM: `0.28.1+das.77acaf6.dtk2604` from
  `/usr/local/lib/python3.10/dist-packages`.
- Model: `/models/Hy4-preview-Channel-FP8-w8a8-v2`; `config.json` SHA-256
  `d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441`.
- Hardware: eight HCU devices, each reporting 147,440 MiB VRAM. Before and
  after validation every device reported 2 MiB used, 0% utilization, and no
  KFD process.

The wheel was installed with `--no-deps --target` into an ignored validation
directory. A cold import from `/tmp` confirmed both `vllm_hcu` and compiled
`vllm_hcu.hcu_ops` resolved from that isolated target, not the source checkout
or globally installed plugin.

No Step5 checkpoint was present on this host: searching model `config.json`
files found neither `model_type=step5` nor a `Step5For*` architecture. The
Step5 master-on/master-off service gate therefore remains unexecuted here; its
evidence is the 314-case CPU suite and source-level provider/fallback tests,
not a hardware claim.

## HY4 master-off fail-fast gate

The TP1/PCP8/EP8/MTP3 command was started with both
`VLLM_HCU_USE_CUSTOM_OPS=0` and
`VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD=1`. Configuration stopped before engine
worker construction or checkpoint loading with:

```text
Value error, HY V4 PCP8 requires effective linear_gate PCP sharding, but
VLLM_HCU_USE_CUSTOM_OPS=0 disables it.
```

The API process exited with status 1. No model-weight allocation appeared and
all cards remained at the idle 2 MiB value.

## HY4 master-on default and MTP3 gate

The positive run deliberately removed
`VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD`; only the master was set. The service
command was:

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_NN=1 \
  VLLM_DEEPEP_BUFFER_SIZE_MB=256 VLLM_HCU_USE_CUSTOM_OPS=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 VLLM_CACHE_ROOT=<isolated-cache> \
  PYTHONPATH=<isolated-plugin-target> \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-pcp8-ep8-ht-mtp3-master-on \
  --host 127.0.0.1 --port 8017 --trust-remote-code \
  --tensor-parallel-size 1 --prefill-context-parallel-size 8 \
  --enable-expert-parallel --moe-backend deep_gemm \
  --all2all-backend deepep_high_throughput \
  --attention-backend FLASHMLA_SPARSE --kv-cache-dtype fp8_e4m3 \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.28}' \
  --enforce-eager --enable-prefix-caching --max-model-len 4096 \
  --block-size 64 --max-num-seqs 256 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

Observed runtime evidence:

- Eight `HcuGPUModelRunnerV2` workers started as PCP ranks 0–7 and EP ranks
  0–7; `DeepEPHTAll2AllManager` and the contiguous DeepGEMM HT expert path
  were selected.
- With no leaf sharding variable, each rank loaded 116.3 GiB. This matches the
  sharded baseline and is materially below the prior replicated-gate 129.27
  GiB result.
- Peak activation was 4.13 GiB per rank. FP8 KV allocation was 3.20–3.47 GiB
  per rank and 65,536 tokens globally (16.00x maximum 4,096-token
  concurrency). The service reached `Application startup complete`.
- `/health` returned HTTP 200. A deterministic smoke request returned
  `HY4_SMOKE_OK` with `finish_reason=stop`.
- The full speculative JSON was accepted. Every worker logged local-argmax
  draft generation. During the concurrent gate, observed mean acceptance
  length was 3.21–3.24 and average draft acceptance was 73.8–74.7%, proving
  the MTP3/synthetic path was active rather than ignored.
- The master-on log contained no traceback, OOM, or request error. Shutdown
  released all card memory and left no KFD process.

## Concurrent HumanEval/0–15

All 16 deterministic requests were submitted concurrently to the same service
with temperature 0, seed 0, and up to 2,048 generated tokens. Every request
returned HTTP 200 and `finish_reason=stop`; completion lengths ranged from 64
to 240 tokens.

EvalScope 1.11.0's isolated `check_correctness` scorer reported **15/16**. The
only failure was `HumanEval/10`: the response completed normally but generated
an incorrect palindrome slice boundary. This is reported as an accuracy miss,
not rounded into a pass and not classified as a runtime failure.

Retained ignored artifacts:

- `hy4-master-on-server.log`
- `hy4-humaneval16.json`, SHA-256
  `99f74737bc375504ee290777d4dbe852f426dccb3a325f521a02489c8fb33956`
- `hy4-humaneval16-score.json`, SHA-256
  `1cb99f8fe68225e7382336798d3fe48e9919a714fe09860bb279c884d5b4f8ff`

## CPU and source-level verification

The post-review focused suite produced **314 passed, 1 deselected, 14 warnings**.
The deselected test is the pre-existing installed-vLLM alias mismatch where
`HYV4Config` preserves `deepseek_sparse_attention` but the repository test
expects `indexed_attention`; the unfiltered run was 302 passed and that single
failure. Production-boundary checking covered 388 Python files. Patch coverage
reported 162 files, 161 adapters, one helper, zero untested files, and zero
invalid contracts. `git diff --check` passed.
