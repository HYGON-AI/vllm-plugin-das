# vLLM 0.32 AITER INT8 MoE validation

This record validates the optional Plugin-owned `--moe-backend aiter` route
against official vLLM 0.32 and the Adapter-owned HCU platform. The Plugin
reuses vLLM's official channel-weight/token-activation W8A8 configuration and
calls the public vendor `aiter.moe` API; it does not contain an AITER kernel or
replace the Adapter platform bridge.

## Candidate and installed ownership

The production code under test was commit `c40a2c67658261f50506142cf5357240c903e105`.
The exact no-dependency wheel was:

```text
hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl
sha256 bba4ffba753362a118f826398bc1eaa5f208f7401d3ceb7ea471cc78aa42f240
```

It was installed with `pip install --no-deps --force-reinstall` into an
isolated environment. Installed-state verification reported schema 3 and the
following ownership:

| Component | Version | Installed evidence |
| --- | --- | --- |
| vLLM | `0.32.0+rocm723` | wheel RECORD `e0131b37486bc3a8bdf0d9930be5af38278d1f42dd9d8a65f9f82eb0010242f4` |
| Adapter (`sgl-kernel`) | `0.1.0` | sole `HcuPlatform` owner; RECORD `51f7e0da9289d0d7c55f25f8598bf3bf4fa12c1ece5b4de383f2d3c5dba7467e` |
| DAS Plugin | `0.2.0.dev0` | `das_hcu` active; RECORD `bd05e9212a1a9bf5b44d1541635c38bbc4c3bd8b95d330069d016acbf77483fb` |
| Torch | `2.11.0+das.opt1.dtk2610.2609201632.gf74cd7` | system HCU build |
| AITER | `0.1.7+dtk2610.torch2110.2610242145.gc7625a` | system vendor package |

The verifier resolved explicit INT8 selection to
`hcu_vllm_das.moe.int8_oracle.HcuAiterInt8Experts`, while the platform stayed
`hcu_adapter.bridges.vllm.HcuPlatform`. All vLLM, Adapter, and Plugin imports
came from the isolated environment's `site-packages`; Torch and AITER came
from the prepared system runtime.

The available validation device was `gfx936` (marketing name `BW`), not the
plan's requested `gfx938`. This record therefore makes no gfx938 claim.

## Kernel numerical and Graph gate

`tests/vllm032/test_aiter_int8_hcu.py` used the target checkpoint dimensions
(`E=256`, hidden size 2048, MoE intermediate size 512, top-k 8) and legal
channel INT8 scales. For M=1, 16, and 128 it compared the public AITER route
with official vLLM Triton INT8, then captured and replayed a Graph at the same
addresses after changing the input. Every shape found a native non-shuffle
AITER configuration and executed the vendor API six times or more.

```bash
env -u PYTHONPATH \
  HIP_VISIBLE_DEVICES=7 \
  HCU_ADAPTER_HOST=none \
  HCU_VLLM032_AITER_INT8_LIVE=1 \
  PYTHONPATH=/dev/shm/adapter-das-main-20261029 \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  -m pytest -q -s tests/vllm032/test_aiter_int8_hcu.py
```

| M | AITER stage-1 solution | Reference RMS | NMAE | NRMSE | Max abs error | Replay |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 10010 | 0.0080467 | 0.0087945 | 0.0091953 | 0.0003662 | changed with input; eager match passed |
| 16 | 10013 | 0.0094071 | 0.0094544 | 0.0099830 | 0.0005493 | changed with input; eager match passed |
| 128 | 10011 | 0.0093776 | 0.0095044 | 0.0100331 | 0.0007935 | changed with input; eager match passed |

Result: `3 passed` both from the source tree and after installing the exact
candidate wheel. The audited limits were NMAE and NRMSE at most 0.03 and max
absolute error at most 0.001.

## Model service gate

The checkpoint `/model/Qwen3.5-35B-A3B-W8A8` has 40 layers, 256 routed
experts, top-k 8, hidden size 2048, and MoE intermediate size 512. Its
compressed-tensors metadata declares symmetric channel INT8 weights and
dynamic per-token INT8 activations. The validated launch was:

```bash
env -u PYTHONPATH \
  HIP_VISIBLE_DEVICES=7 \
  HCU_ADAPTER_HOST=none \
  VLLM_PLUGINS=hcu_adapter \
  PYTHONNOUSERSITE=1 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/dev/shm/vllm032-aiter-int8-cache-1e354d1b-bf16kv \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/vllm serve \
  /model/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-vllm032-aiter \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --safetensors-load-strategy prefetch \
  --moe-backend aiter \
  --enable-prefix-caching \
  --kv-cache-dtype auto \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 18032
```

No attention backend was forced. The Adapter selected `TRITON_ATTN`, because
this runtime did not provide the FlashAttention version required by the
official vLLM FlashAttention route. An earlier explicit FlashAttention trial
also showed that `fp8_e5m2` KV was unsupported, and a BF16 retry reached model
load before failing the FlashAttention availability check. These were
attention-route failures, not AITER MoE failures.

The accepted process reported `enforce_eager=False`, `VLLM_COMPILE`, and
`FULL_AND_PIECEWISE`. It captured all planned PIECEWISE and FULL Graph sizes,
then logged:

```text
Using AITER Int8 MoE backend out of potential backends: ['TRITON', 'HUMMING', 'CPU'].
HCU AITER INT8 MoE selected: M=8192 E=256 N1=1024 N2=2048 K=2048 top_k=8 solution=asm config={'SOL_ID1': 13001, 'SOL_ID2': 23001, 'BLOCK_SIZE_M': 128}
HCU AITER INT8 MoE selected: M=1 E=256 N1=1024 N2=2048 K=2048 top_k=8 solution=asm config={'SOL_ID1': 10010, 'SOL_ID2': 20000, 'BLOCK_SIZE_M': 16}
Load asm moe tuned csv: .../aiter/configs/tuned_fmoe_asm_w8a8_channel.csv
[aiter] hipModuleLoad: .../aiter_meta/hsa/gfx936/w8a8/stage1/... Success
[aiter] hipModuleLoad: .../aiter_meta/hsa/gfx936/w8a8/stage2/... Success
```

The deterministic smoke request stopped normally with 22 prompt tokens and
38 completion tokens. Its response began `Interface stability matters
because...`. Across smoke and HumanEval, the service recorded native AITER
solutions for every new runtime shape and zero Plugin provider fallbacks. The
accepted service log contained no ERROR or Traceback. One unrelated upstream
ROCm GELU warning selected the non-approximate GELU implementation.

## HumanEval/0-63 accuracy

The same live non-eager AITER service generated the ordered HumanEval/0-63
prefix from OpenAI HumanEval commit
`6d43fb980f9fee3c892a914eda09951f772ad10d`. Requests used the established
complete-code prompt, temperature 0, seed 0, thinking disabled, and a 2048
token limit:

```bash
env -u PYTHONPATH \
  python3 -S /dev/shm/hcu-vllm032-work/tools/humaneval64_client.py \
  --human-eval-root /dev/shm/hcu-vllm032-work/src/human-eval \
  --output-dir ./humaneval64 \
  --base-url http://127.0.0.1:18032 \
  --model qwen35-vllm032-aiter \
  --limit 64 --max-tokens 2048 --timeout 600
```

All 64 requests returned HTTP 200 with `finish_reason=stop`. Generation used
10,737 completion tokens and took 149.81 seconds wall time (71.68 completion
tokens per accumulated request-second); this sequential correctness run is
not a serving-performance benchmark.

Generated code was not executed by the service client or directly on the
host. Official `human_eval.execution.check_correctness` ran sequentially with
a 10-second per-task timeout inside Bubblewrap 0.6.1: network, PID, IPC, UTS,
and user namespaces were unshared; all capabilities were dropped; host binds
were read-only; UID/GID were 65534; `/tmp` was tmpfs; and limits were 2 GiB
address space, 128 processes, 256 file descriptors, and zero core files.

Result: **64/64, pass@1 1.0, zero failures, zero timeouts, and zero scoring
stderr bytes**.

| Artifact | SHA256 |
| --- | --- |
| `problems.jsonl` | `3a64d077946f2c1b8707e13217e7069cc4bcefc034a6992f7ded9104c2dc59bd` |
| `responses.jsonl` | `40c4b57e3f80b82023a4ce87b7a71207e842d981e4daef7f1e52906d2f4874c3` |
| `samples.jsonl` | `2766f7e97ae69bc939caa2edaa1a4e692900cbbfa01cb45668eb5f05ef5d374a` |
| `score-results.jsonl` | `403cc14ab14a44f018c962e2ac0817b1166c7c1451f5f911ee297261f25a4ddd` |
| `generation-summary.json` | `03207206147791de86b19a3691c60c0ee9812308c2d7009598483b7cbb0ba58c` |

Node-local evidence is under
`/data/models/vllm032-aiter-int8-validation-20261029/run-c40a2c67-final`;
the matching service log is `run-c40a2c67-final-server.log` (SHA256
`5bf53ad556459520697fac6d31e9dc595b4c199b672275898774e7531f790b52`).
The result validates this gfx936, TP1, BF16-KV topology only; it does not
establish gfx938, tensor-parallel, quantized-KV, or throughput claims.
