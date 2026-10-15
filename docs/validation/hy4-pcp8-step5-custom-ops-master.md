# HY4 PCP8 and custom-op master validation

Validation dates: 2026-10-15 through 2026-10-16 (UTC).

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

## Full HumanEval with a 16,384-token client limit

The full 164-problem HumanEval dataset was rerun after the final review fixes
at plugin commit `3f918f4aa0996e1d91e100aa183ed828b22f451f`. The isolated wheel was
`vllm_hcu-0.28.1rc1.dev491+das.3f918f4.dtk2604-cp310-cp310-linux_x86_64.whl`
with SHA-256
`f888534324322c662e07fdd7cf0e00b1c2f4cadfcc2af1c9855075ee9d9f9c40`.

The server used the same TP1/PCP8/EP8, DeepGEMM, DeepEP high-throughput,
FP8-KV and MTP3 configuration as above, with `--max-model-len 32768`. Every
client request set `max_tokens=16384`; the larger server window is required
because vLLM validates prompt plus requested output tokens. Sixteen requests
were kept concurrently in flight. The 62,592-token KV cache advertised 1.91x
maximum concurrency for a full 32,768-token request, so excess long requests
would queue rather than all occupy their maximum reservation at once.

Generation results:

- 164/164 requests returned HTTP 200 and `finish_reason=stop`; length, abort
  and error counts were all zero.
- Generation completed in 201.811 seconds and produced 27,766 tokens. Actual
  completion lengths were 31–652 tokens (169.3 mean), so no response approached
  the 16,384-token client cap.
- EvalScope 1.11.0 `check_correctness` scored **125/164 (76.22%)**. The 39
  failures were generated-code correctness/syntax failures plus one timeout,
  not service or length failures.
- MTP3 remained active for the whole run: 8,629 draft steps accepted 19,171 of
  25,887 proposed tokens (74.05%), equivalent to a 3.22 mean acceptance length
  including the target token. Per-position accepted counts were
  8,629 / 8,629 / 1,913.
- After shutdown all eight devices returned to 2 MiB used, 0% utilization, and
  no KFD process.

Retained ignored artifacts:

- `hy4-humaneval164-max16384-c16.json`, SHA-256
  `28c9b6e2fb0348de6f5f5882b961dc156546f2bf88c912a9ec07ce9a0b1f7eba`
- `hy4-humaneval164-max16384-c16-score.json`, SHA-256
  `e0fb6ca75dcd58c73b42ad3b02d51a87aa22a64bb556be3bdb2026507e4a5884`

### Standard rejection control

The full run above used synthetic acceptance. A control run removed
`rejection_sample_method=synthetic` and `synthetic_acceptance_length=3.28`,
leaving standard MTP3 rejection enabled. The client explicitly used
`reasoning_effort=no_think`, temperature 0, `top_p=1`, `max_tokens=16384`, and
16 concurrent requests.

- 164/164 requests returned HTTP 200 with `finish_reason=stop`; all 164
  responses had zero reasoning tokens.
- Generation completed in 179.391 seconds and produced 28,210 completion
  tokens. Completion lengths were 31–494 tokens (172.01 mean).
- EvalScope 1.11.0 `check_correctness` scored **153/164 (93.29%)**. The failed
  task indices were 32, 59, 97, 103, 108, 115, 127, 130, 132, 145, and 160.
- This is 28 more passing problems than the synthetic run. Synthetic
  acceptance is therefore not used as accuracy evidence: accepting draft
  tokens independently of the target model is not lossless speculative
  decoding.

Retained ignored artifacts:

- `hy4-humaneval164-nothink-mtp3-standard-c16.json`, SHA-256
  `83ece0a91407d7f141bd5e698e38641d13ca0ad38433af2f1a7cd420f1742f0c`
- `hy4-humaneval164-nothink-mtp3-standard-c16-score.json`, SHA-256
  `339a418f32afe8659ede59a3908e90ca00ee101b1d94f62b965f2be15f201cd4`

## CPU and source-level verification

The post-review focused suite produced **314 passed, 1 deselected, 14 warnings**.
The deselected test is the pre-existing installed-vLLM alias mismatch where
`HYV4Config` preserves `deepseek_sparse_attention` but the repository test
expects `indexed_attention`; the unfiltered run was 302 passed and that single
failure. Production-boundary checking covered 388 Python files. Patch coverage
reported 162 files, 161 adapters, one helper, zero untested files, and zero
invalid contracts. `git diff --check` passed.
