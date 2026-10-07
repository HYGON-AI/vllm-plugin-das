# vLLM 0.28.1 gfx938 validation provenance

Recorded at `2026-10-03T18:29:55+00:00`. This record freezes the wheel pair
used to start the gfx938 model-validation campaign. A final plugin wheel is
rebuilt after any runtime fixes; this pre-validation wheel remains immutable
evidence for the initial harness and compatibility gates.

## Source topology

- Plugin repository branch: `codex/validate-v0281-gfx938-models`
- Plugin wheel source commit:
  `f189798e80b34a94b1577de19a70fab549236661`
- Rebased PR 181 stack head:
  `592b38cf9ce9f6e36ab1c846ecd2483f03e20fd9`
- PR 180 base head:
  `f24e99be620c3a0d3611a7d96079bb7be7e63de8`
- `origin/v0.28.1-dev` baseline:
  `9788dc633fb3d0a0a0a124b5886f62c07b3de765`

The plugin wheel was built from a clean tracked worktree with:

```bash
ROCM_PATH=/opt/dtk MAX_JOBS=16 python setup.py bdist_wheel
```

## Frozen wheels

| Package | Filename | SHA256 |
| --- | --- | --- |
| vLLM | `vllm-0.28.1+dtk2604.torch2110.2609171627.g77acaf-cp310-cp310-linux_x86_64.whl` | `2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b` |
| vLLM HCU plugin | `vllm_hcu-0.28.1rc1.dev491+das.f189798.dtk26041-cp310-cp310-linux_x86_64.whl` | `407b68d3a37a1148c835e3b12a10c7c59cd5d184c2f95ceb0a0484a3a67c8fd5` |

The vLLM wheel is stored at `/models/artifacts/v0281-gfx938/`. The plugin
wheel is the `dist/` artifact from the source commit above. The `dtk26041`
local-version component is the plugin build system's encoding of the installed
DTK `26.04.1` release.

The wheels were installed without dependencies into these immutable targets:

- vLLM: `/models/.installs/vllm-v0281-gfx938-g77acaf`
- plugin: `/models/.installs/vllm-plugin-v0281-gfx938-f189798`

`importlib.metadata` reported the following distributions from those exact
roots:

- `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf`
- `vllm-hcu==0.28.1rc1.dev491+das.f189798.dtk26041`

## Parent and child import roots

The import probe ran with `PYTHONNOUSERSITE=1`, the vLLM target and plugin
target at the front of `PYTHONPATH`, and path assertions in both the parent and
a fresh child interpreter. Both processes resolved the same files:

```text
vllm      /models/.installs/vllm-v0281-gfx938-g77acaf/vllm/__init__.py
vllm_hcu  /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/__init__.py
hcu_ops   /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/hcu_ops.cpython-310-x86_64-linux-gnu.so
```

The following vLLM native modules were also resolved below the isolated vLLM
root:

```text
vllm._C                         vllm/_C.abi3.so
vllm._rocm_C                    vllm/_rocm_C.abi3.so
vllm._C_stable_libtorch         vllm/_C_stable_libtorch.abi3.so
vllm._moe_C_stable_libtorch     vllm/_moe_C_stable_libtorch.abi3.so
```

## Runtime environment

- Host kernel: `Linux 6.6.92-34.1.tl4.x86_64`
- Python: `3.10.12` (`/usr/bin/python`)
- DTK release files: `26.04.1`
- HIP compiler version: `6.3.26255`
- Torch: `2.11.0+dtk2604.2609291046.ge12687`
- Hardware: 8 x `BW1100`, `gfx938:sramecc+:xnack-`, 147440 MiB VRAM per
  device
- `/usr/lib/x86_64-linux-gnu/librt.so`: absent
- `/usr/lib/x86_64-linux-gnu/librt.so.1`: present

Installed operator/provider distributions:

| Distribution | Version |
| --- | --- |
| `aiter` | `0.1.6+dtk2604.torch2110.2609300837.gf9b9c8` |
| `boltops` | `0.1.0+dtk2604.torch2110.2609301634.g137f5d` |
| `flash-attn` | `2.8.4+dtk2604.torch2110.2609241509.g624d7b` |
| `flash-mla` | `1.2.0+dtk2604.torch2110.2609291807.gc24e3f` |
| `lightop` | `0.6.0+dtk2604.torch2110.2609301833.g9d6ed8` |
| `deep-ep` | `1.1.0+dtk2604.torch2110.2609231914.g49988b` |
| `deepgemm` | `2.1.0+dtk2604.torch2110.2609301620.g9dc3b6` |

## Artifact verification

The installed-root compatibility, packaging, and lifecycle gates were run from
outside the checkout with pytest importlib mode so the source tree could not
shadow either wheel:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/models/.installs/vllm-v0281-gfx938-g77acaf:/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_TARGET_ROOT=/models/.installs/vllm-v0281-gfx938-g77acaf \
VLLM_HCU_TARGET_ROOT=/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_PLUGINS=__disabled__ \
python -m pytest --import-mode=importlib -q \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_compatibility_gate.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_setup_packaging.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_plugin_lifecycle.py
```

Result: `58 passed, 14 warnings in 107.64s`.

## Source-tree model matrix

The following results were produced from the validation worktree with the
pinned vLLM runtime above. All successful service launches used
HcuGPUModelRunnerV2 and the resolved default Graph policy; no accuracy result
was rescued with `--enforce-eager`. Exact arguments, ports, request protocol,
and owned artifact directories are defined in
`tests/models/v0281_gfx938_humaneval16.yaml`.

| Profile | Topology and feature route | HumanEval16 | Disposition |
| --- | --- | ---: | --- |
| `deepseek_v32_channel_fp8_tp8` | TP8, sparse MLA, channel FP8 W8A8, AITER, prefix, default Graph | 16/16 | Pass; native LBNHC sparse-MLA KV layout retained |
| `deepseek_v32_channel_fp8_mtp3_kvfp8_tp8` | TP8, sparse MLA, channel FP8 W8A8, AITER, MTP3, E4M3 sparse KV, prefix, default target/speculator Graphs | 16/16 | Pass; public E4M3 mapped to `fp8_ds_mla`, MTP draft acceptance observed |
| `deepseek_r1_channel_fp8_tp8` | TP8, regular MLA, channel FP8 W8A8, AITER, prefix, default Graph | 16/16 | Pass; current OpenAI response `reasoning` field accepted by the prefix probe |
| `deepseek_r1_channel_fp8_mtp3_tp8` | TP8, regular MLA, channel FP8 W8A8, AITER, MTP3, prefix, default target/speculator Graphs | 16/16 | Pass; MTP draft acceptance observed throughout the concurrent run |
| `deepseek_r1_0528_channel_int8_kvfp8_tp8` | TP8, regular MLA, Channel INT8 W8A8, AITER INT8 MoE, native E4M3 KV, prefix, default Graph | 15/16 | Diagnostic; HumanEval/11 exhausted the 8,192-token reasoning budget, while MTP3 controls scored 14/16 |
| `deepseek_v4_flash_tp8` | TP8, sparse MLA, AITER, DSpark7, prefix, default Graph | 2/16 concurrent; 1/3 serial diagnostic | Local checkpoint is the previously documented incomplete asset; DSpark smoke passed, but no full accuracy claim |
| `glm5_w8a8_tp8` | TP8, sparse MLA, AITER INT8 MoE, MTP3, prefix | 16/16 | Pass; repeated on the final shared-expert code with target/draft FULL plus PIECEWISE Graphs |
| `glm52_channel_int8_tp8` | TP8, sparse MLA, Channel INT8, AITER INT8 MoE, E4M3 sparse KV, MTP3, prefix, default target/speculator Graphs | 16/16 | Pass; public E4M3 mapped to `fp8_ds_mla`, final-window MTP draft acceptance 97.2% |
| `glm47_w8a8_mtp2_kvfp8_tp4` | TP4, FLASH_ATTN, W8A8, AITER lookup with per-shape Triton fallback, MTP2, E4M3 KV, HND/BHSD, prefix, default target/speculator Graphs | 16/16 | Pass; M=1 AITER config miss and Triton MoE fallback remained explicit |
| `glm53_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, E4M3 public KV mapped to `fp8_ds_mla`, MTP3 | 16/16 | Pass; repeated on the final shared-expert code and extended to HumanEval 32/32 |
| `glm51_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, MTP3, prefix | 16/16 | Pass; EvalScope artifact ID collapses repeated underscores |
| `hy3_channel_fp8_mtp2_kvfp8_tp8` | TP8, FLASH_ATTN, channel FP8 W8A8, AITER, MTP2, E4M3 KV, prefix | 16/16 | Pass; HND selection resolved to the physical LBHNC cache and default target/speculator Graphs |
| `hy3_channel_fp8_dp8_ep8_mtp2_kvfp8` | DP8/TP1/EP8, FLASH_ATTN, channel FP8 W8A8, DeepEP low-latency/DeepGEMM, MTP2, E4M3 KV, prefix | 16/16 | Pass; nine-request probe demonstrated rank-local prefix reuse |
| `hy4_preview_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, MTP3, prefix | 16/16 | Pass after `indexed_attention` and `reasoning_effort=no_think` fixes |
| `minimax_m25_int8_tp4` | TP4, FLASH_ATTN, HND/BHSD kernel view, AITER, prefix | 16/16 | Pass |
| `qwen2_57b_tp2` | TP2, FLASH_ATTN, HND/LBHNC kernel view, AITER W16A16 MoE, E4M3 KV, prefix, default Graph | 16/16 | Pass; current-head rerun loaded the gfx938 AITER stage1/stage2 modules and reused 2,496 prefix tokens |
| `qwen3_30b_int8_tp2` | TP2, FLASH_ATTN, HND/LBHNC kernel view, E4M3 KV, default LightOp dense W8A8, AITER INT8 MoE, prefix, default Graph | 5/16 | Current-head controls: BF16 TP2 and TP1 6/16; official Triton dense and Triton MoE 4/16; repeated 2,433-token request hit 2,432 tokens, so service features pass but accuracy is not accepted |
| `qwen3_8b_tp2` | TP2, FLASH_ATTN, native FP8 E4M3 KV, prefix | 16/16 | Pass; BF16 control also 16/16 |
| `qwen35_35b_tp2` | TP2, FLASH_ATTN, AITER BF16 MoE, E4M3 KV, MTP3, fine-grained hybrid prefix, default target/speculator Graphs | 16/16 | Pass; current-head rerun proved a 2,112-token fine-grained sibling hit and 93.43% HumanEval MTP acceptance |
| `qwen35_35b_w8a8_tp2` | TP2, FLASH_ATTN, AITER INT8 MoE, E4M3 KV, MTP3, fine-grained hybrid prefix, default target/speculator Graphs | 16/16 | Pass; current-head rerun proved a 2,112-token fine-grained sibling hit and 92.19% HumanEval MTP acceptance |
| `qwen36_27b_w8a8_tp2` | TP2, FLASH_ATTN, W8A8, MTP3, E4M3 KV, fine-grained hybrid prefix, default target/speculator Graphs | 16/16 | Pass; current-head rerun proved a 3,136-token fine-grained sibling hit and 94.39% MTP acceptance |
| `qwen38_27b_int8_tp2` | TP2, FLASH_ATTN, INT8, MTP3, E4M3 KV, fine-grained hybrid prefix, default target/speculator Graphs | 16/16 | Pass; current MR-head rerun proved a 3,136-token fine-grained sibling hit and 95.86% MTP acceptance |
| `qwen38_flash_next_fp8_tp4` | TP4, hybrid BLNHC layout, channel FP8, public E4M3 KV, MTP3, ordered AITER lookup with Triton fallback | 16/16 | Pass; current MR-head rerun proved 3,200 manager-page prefix hits and 90.62% MTP acceptance |
| `qwen38_flash_next_w4a8_tp4` | TP4, hybrid BLNHC layout, SlimQuant W4A8, public E4M3 KV, MTP3, ordered AITER lookup with Triton fallback | 16/16 | Pass; current MR-head rerun also proved 3,200 prefix-hit tokens and 88.9% MTP acceptance |

### `v0.28.1-dev` base-alignment rerun

PR #188 was retargeted to `v0.28.1-dev` after merging remote commit
`0bd777db91fa01c10f8a85910ee7057456ebf3ab`. The alignment merge is
`76e6ea7bb4e0f06f41763abcf5732d9264cf7066`; its tree
`e64f6bc2e9c055a6425c4c8127a6edf487d22eba` is byte-for-byte the same as the
pre-merge follow-up tree, so the merge established ancestry without changing
the candidate contents. The focused static gate passed `84 passed, 1 skipped`.

Four fresh cold-start HumanEval16 gates were then run on gfx938 with the
pinned `0.28.1+das.77acaf6.dtk2604` runtime:

| Profile | Runtime evidence | HumanEval16 | Report evidence |
| --- | --- | ---: | --- |
| `qwen3_8b_tp2` | TP2, HcuGPUModelRunnerV2, HND-facing/LBHNC physical E4M3 KV, 64-token page, prefix cache, default FULL plus PIECEWISE Graphs | 16/16 | 105.43 output tok/s; 2,688/5,466 prefix hit/query tokens |
| `qwen38_flash_next_fp8_tp4` | TP4, HcuGPUModelRunnerV2, native BLNHC hybrid cache, E4M3 KV, QSA, MTP3, default target/speculator FULL plus PIECEWISE Graphs; AITER lookup fell back per unsupported shape to official Triton | 16/16 | 42.03 output tok/s; 3,200/10,906 prefix hit/query tokens; 1,782/1,953 accepted/drafted MTP tokens (91.24%) |
| `qwen35_35b_w8a8_tp2` | TP2, HcuGPUModelRunnerV2, E4M3 KV, MTP3, `mamba-cache-mode=align`, 64-token fine-grained prefix controls, default target/speculator FULL plus PIECEWISE Graphs | 16/16 | 60.31 output tok/s; 2,176/10,906 prefix hit/query tokens; 1,164/1,245 accepted/drafted MTP tokens (93.49%) |
| `glm53_channel_fp8_tp8` | TP8, HcuGPUModelRunnerV2, `FLASHMLA_SPARSE`, public E4M3 resolved to `fp8_ds_mla`, LBNHC, AITER FP8 MoE, MTP3, prefix cache, default target/speculator FULL plus PIECEWISE Graphs | 16/16 | 17.24 output tok/s; 2,624/5,466 prefix hit/query tokens; 884/966 accepted/drafted MTP tokens (91.51%) |

All four pytest acceptance invocations passed. Qwen3.8 Flash-Next required
the harness's 180-second forced-cleanup fallback after the API parent exited;
the owned workers were removed and cards 0--3 returned to the 4 MiB idle
reading. This is recorded as teardown latency, not an inference failure. After
all eight cards became idle, GLM-5.3 was independently cold-started on TP8.
Its 18/18 chat requests, including two prefix probes, returned HTTP 200 and
the current run window had no ERROR, Traceback, or RuntimeError. The test
passed in 1,107.33 seconds and cleanup returned all eight cards to idle.

Fresh artifacts:

- `/tmp/vllm-hcu-evalscope/v0281-gfx938-qwen3-8b-tp2`
- `/tmp/vllm-hcu-evalscope/v0281-gfx938-qwen38-flash-next-fp8-tp4`
- `/tmp/vllm-hcu-evalscope/v0281-gfx938-qwen35-35b-w8a8-tp2`
- `/tmp/vllm-hcu-evalscope/v0281-gfx938-glm53-channel-fp8-tp8`

### Qwen2-57B-A14B current-head rerun

The `/models/Qwen2-57B-A14B-Instruct` checkpoint was rerun at plugin commit
`163f575` with the pinned vLLM 0.28.1 wheel. The accepted server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen2-57b-current \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen2-57B-A14B-Instruct \
  --served-model-name Qwen2-57B-A14B-Instruct \
  --port 10246 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen2-57b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen2-57B-A14B-Instruct \
  --api-url http://127.0.0.1:10246/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen2-57b-current-tp2-kvfp8-run1-20261007 \
  --no-timestamp
```

Both ranks constructed `HcuGPUModelRunnerV2`. The service selected
`AiterExperts`, loaded the tuned ASM shuffle CSV, and successfully loaded
gfx938 BF16 W16A16 stage1/stage2 HSA modules during warmup and inference.
FLASH_ATTN retained its 64-token page, HND resolved to physical `LBHNC`, public
`fp8_e4m3` was accepted, and the default `FULL_AND_PIECEWISE` Graph policy
captured both Graph forms. The checkpoint does not expose an MTP layer, so no
speculative configuration was requested.

Raw HumanEval Accuracy and Pass@1 both passed 16/16. The report observed 35.17
output tok/s, 374.1 ms mean TTFT, 23.7 ms mean TPOT, and 2.118 s mean latency.
Two identical 2,498-token probes returned `17`; the second reused 2,496 tokens,
exactly 39 64-token pages. All 18 chat requests returned HTTP 200, the log
contained no ERROR, Traceback, VM fault, or provider fallback, and the delayed
driver release returned every card to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen2-57b-current-tp2-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/qwen2-57b-current-tp2-kvfp8-run1-20261007`

### Qwen3-30B-A3B Channel-INT8 current-head diagnostic

The checkpoint was rerun at plugin commit `9a28f4e` with the pinned vLLM
0.28.1 wheel. The exact primary TP2/E4M3 server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM -u VLLM_CACHE_ROOT \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --served-model-name Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --port 10244 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The exact isolated HumanEval16 command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen3-30b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10244/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp2-kvfp8-run1-20261007 \
  --no-timestamp
```

All service routes used HcuGPUModelRunnerV2 and default
FULL_AND_PIECEWISE Graph capture. The controlled results were:

| Dense W8A8 | MoE | KV | Topology | HumanEval16 | Passing tasks |
| --- | --- | --- | --- | ---: | --- |
| default LightOp redirect | AITER | E4M3 | TP2 | 5/16 | 0, 2, 3, 4, 13 |
| default LightOp redirect | AITER | BF16/auto | TP2 | 6/16 | 0, 2, 3, 4, 10, 13 |
| official Triton control | AITER | BF16/auto | TP2 | 4/16 | 0, 3, 10, 13 |
| default LightOp redirect | official Triton | BF16/auto | TP2 | 4/16 | 0, 3, 10, 13 |
| default LightOp redirect | AITER | BF16/auto | TP1 diagnostic | 6/16 | 0, 2, 3, 4, 10, 13 |

The TP1 and TP2 BF16 pass sets were identical. Changing KV dtype, dense
provider, MoE provider, or tensor parallelism did not remove the repeated
invalid pseudocode, semantic errors, or reasoning loops. The checkpoint's own
README reports GSM8K rather than HumanEval, and no unquantized sibling exists
locally for a checkpoint-level comparison. This is therefore retained as an
unaccepted checkpoint/no-thinking generation outcome; the cross-provider
evidence does not justify a plugin model-specific workaround.

The primary TP2 route still passed its runtime feature gates. Both ranks used
MRV2; dense W8A8 executed the default LightOp redirect despite the upstream
`Selected TritonInt8ScaledMMLinearKernel` frontend message; AITER loaded the
ordinary and bottom gfx938 `E=128,N=384,dtype=int8_w8a8` configs plus the
channel-shuffle table; public E4M3 used a 64-token LBHNC cache; and target
prefill/decode captured PIECEWISE and FULL Graphs. Two valid 2,433-token
requests returned `17`; the second request hit 2,432 tokens, exactly 38 cache
pages. One deliberately over-context probe returned HTTP 400 before execution
and was excluded. Exact teardown returned all cards to idle.

Evidence:

- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp2-kvfp8.log`
- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp2-kvfp8-prefix-rerun.log`
- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp2-kvauto.log`
- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp2-kvauto-triton-dense.log`
- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp2-kvauto-triton-moe.log`
- `/tmp/vllm-hcu-validation/qwen3-30b-a3b-channel-int8-tp1-kvauto.log`
- `/tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp2-kvfp8-run1-20261007`
- `/tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp2-kvauto-run1-20261007`
- `/tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp2-kvauto-triton-dense-run1-20261007`
- `/tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp2-kvauto-triton-moe-run1-20261007`
- `/tmp/vllm-hcu-evalscope/qwen3-30b-a3b-channel-int8-tp1-kvauto-run1-20261007`

### Qwen3.6-27B W8A8 current-head rerun

The `/models/Qwen3.6-27B-W8A8` checkpoint was rerun at plugin commit
`2273db8` with the pinned vLLM 0.28.1 wheel. The accepted server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen36-27b-current \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3.6-27B-W8A8 \
  --served-model-name Qwen3.6-27B-W8A8 \
  --port 10245 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen36-27b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.6-27B-W8A8 \
  --api-url http://127.0.0.1:10245/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen36-27b-w8a8-current-tp2-mtp3-kvfp8-run1-20261007 \
  --no-timestamp
```

Both TP ranks constructed `HcuGPUModelRunnerV2`. FLASH_ATTN kept its 64-token
kernel page, while hybrid cache unification selected a 1,600-token manager
page and physical `LBHNC`. Public `fp8_e4m3` was accepted. The resolved default
policy was `FULL_AND_PIECEWISE`; target and MTP prefill captured PIECEWISE and
FULL Graphs, and MTP decode captured FULL Graphs. Compressed-tensors logged a
`TritonInt8ScaledMMLinearKernel` frontend object; with the plugin's custom
quantization-GEMM switch unset, the default HCU dense W8A8 wrapper remained
enabled rather than selecting the official Triton dense control.

Raw HumanEval Accuracy and Pass@1 both passed 16/16. The report observed 75.87
output tok/s, 453.5 ms mean TTFT, 10.4 ms mean TPOT, and 1.954 s mean latency.
A fresh owner/junction/sibling probe sent three 3,590-token prompts; all
returned `17`, with per-request prefix-hit deltas of 0, 1,600, and 3,136. The
third hit is 49 times the configured 64-token match unit and is not divisible
by the 1,600-token manager page, proving fine-grained reuse. Complete-session
MTP counters were 1,767/1,872 accepted/drafted tokens (94.39%). All 19 chat
requests returned HTTP 200, the log contained no ERROR, Traceback, VM fault,
or dead engine, and teardown returned all cards to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen36-27b-w8a8-current-tp2-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/qwen36-27b-w8a8-current-tp2-mtp3-kvfp8-run1-20261007`

The final shared-expert stream-safety change was followed by fresh live
regression runs for both large GLM routes. `/models/GLM-5-W8A8` used TP8,
regular BF16/auto sparse MLA KV, AITER INT8 MoE, MTP3, prefix caching, and the
default FULL_AND_PIECEWISE target/draft Graph policy. It passed raw and
normalized HumanEval 16/16; all 18 API requests, including the two prefix
probes, returned HTTP 200 with no ERROR or Traceback. The final metric window
reported 35.2% prefix hits and 93.0% draft-token acceptance. The report
observed 27.41 output tokens/s, 4.782 s mean latency, 591.7 ms TTFT, and
32.1 ms TPOT. The route allocated 400,512 KV tokens; evidence remains under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm5-w8a8-tp8`.

`/models/GLM-5.3-Channel-FP8-w8a8` used TP8, `FLASHMLA_SPARSE`, AITER FP8
MoE, MTP3, public E4M3 KV resolved to `fp8_ds_mla`, prefix caching, and the
default FULL_AND_PIECEWISE target/draft Graph policy. A fresh HumanEval16 gate
passed raw and normalized 16/16. The extended accuracy gate then passed raw
and normalized HumanEval 32/32 with 34/34 HTTP 200 responses, no ERROR or
Traceback, 737,920 KV tokens, and a final report of 25.6 output tokens/s,
3.082 s mean latency, 594.3 ms TTFT, and 32.2 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm53-channel-fp8-tp8-humaneval32-final`.

DeepSeek V4.1 was explicitly excluded at the requester's direction. The
optional current-branch HY4 DP8/TP1/EP8 rerun subsequently passed 16/16 with
MTP3, E4M3 sparse KV, DeepEP low-latency/DeepGEMM, MRV2 on all eight ranks,
and default FULL_AND_PIECEWISE target/speculator Graph capture. Its artifacts
are under `/tmp/vllm-hcu-evalscope/v0281-gfx938-hy4-dp8-ep8-mtp3`. The
generic two-request prefix probe was not accepted as DP8 evidence because its
requests landed on different ranks; no current-branch DP8 prefix-hit claim is
made.

`/models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3` was then validated on
both the required TP8 route and the optional DP8/TP1/EP8 low-latency route. The
checkpoint is regular GQA rather than MLA, so both profiles used
`FLASH_ATTN`, Model Runner V2, public E4M3 KV, MTP2, prefix caching,
`reasoning_effort=no_think`, and the default FULL_AND_PIECEWISE target and
speculator Graph policy. `VLLM_KV_CACHE_LAYOUT=HND` selected the HND/BHSD-facing
route and resolved to the runtime's physical LBHNC cache layout.

The TP8/AITER run passed raw and independently normalized HumanEval 16/16,
with 18/18 HTTP 200 responses and no ERROR or Traceback. It allocated
4,860,672 KV tokens, reported a final 35.1% prefix-hit rate and 90.0% draft
acceptance, and observed 31.23 output tokens/s, 3.024 s mean latency, 906.9 ms
TTFT, and 21.91 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-hy3-channel-fp8-mtp2-kvfp8-tp8`.

The DP8/TP1/EP8 profile selected DeepEP low-latency and DeepGEMM, started one
Model Runner V2 engine per device, captured FULL and PIECEWISE target/draft
Graphs, and passed raw and normalized HumanEval 16/16. Its successful request
window contained 25/25 HTTP 200 responses and no runtime ERROR or Traceback;
owned SIGTERM cleanup subsequently emitted one expected cancellation-side
`EngineDeadError`. A two-request probe initially returned zero because the
requests landed on different DP ranks. The harness now supports a configurable
prefix-probe request count, and this profile sends DP-size-plus-one (nine)
identical requests; the repeated rank recorded 2,752 hit tokens. The report
observed 32.73 output tokens/s, 2.368 s mean latency, 273.3 ms TTFT, and
27.55 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-hy3-channel-fp8-dp8-ep8-mtp2-kvfp8`.

The subsequently added `/models/DeepSeek-R1-Channel-FP8-w8a8` checkpoint was
validated twice on the same eight-card host. Both the plain TP8 route and the
TP8+MTP3 route passed raw and independently normalized HumanEval16 at 16/16,
with all 16 generated programs executed successfully. The plain run recorded
10.33 output tokens/s and 95.18 ms mean TPOT. The MTP3 run recorded 24.30
output tokens/s, 41.42 ms mean TPOT, and live draft acceptance rates of about
52-65%; the two runs emitted different token counts, so this is observational
rather than a controlled performance benchmark. The MTP3 server resolved
`DeepSeekMTPModel`, compiled a separate `eagle_head`, warmed the three-token
rejection sampler, captured default FULL_AND_PIECEWISE graphs, allocated
635,200 KV tokens with LBNHC layout, and loaded AITER channel-shuffle stage1
and stage2 kernels on all eight ranks. Artifacts are under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-r1-channel-fp8-{tp8,mtp3-tp8}`.

The first plain TP8 attempt exposed a harness compatibility issue rather than
a model failure: vLLM 0.28.1 returned reasoning-only probe output in the
current `message.reasoning` field while the probe recognized only `content`
and the deprecated `reasoning_content` alias. The probe now checks those three
fields in current-protocol order, with regression coverage. After that narrow
fix, the unchanged service route passed the prefix probe and HumanEval16.

The later `/models/DeepSeek-R1-0528-Channel-INT8` checkpoint was validated at
TP8 with HcuGPUModelRunnerV2, regular `FLASHMLA`, AITER INT8 MoE, native
`fp8_e4m3` KV, prefix caching, LBNHC MLA cache layout, and the default
FULL_AND_PIECEWISE Graph policy. The recommended diagnostic server command is:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /models/DeepSeek-R1-0528-Channel-INT8 \
  --trust-remote-code --tensor-parallel-size 8 \
  --attention-backend FLASHMLA --moe-backend aiter \
  --reasoning-parser deepseek_r1 --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching --max-model-len 32768 \
  --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --served-model-name DeepSeek-R1-0528-Channel-INT8 --port 10226
```

The matching HumanEval16 client command is:

```bash
env -i \
  HOME=/tmp/vllm-hcu-eval-home-deepseek-r1-0528 \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-R1-0528-Channel-INT8 \
  --api-url http://127.0.0.1:10226/v1 --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":8192}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-r1-0528-channel-int8-kvfp8-e4m3-8k-tp8 \
  --no-timestamp
```

The no-MTP run produced 16 predictions/reviews, 18/18 HTTP 200 responses,
36.7% prefix-cache hits, no runtime ERROR or Traceback, and raw plus normalized
HumanEval 15/16 (`0.9375`). `HumanEval/11` spent all 8,192 output tokens in
reasoning and emitted no final answer. Its report observed 11.89 output
tokens/s, 144.14 s mean latency, 572.72 ms mean TTFT, and 83.86 ms mean TPOT.
The checkpoint chat template exposes no supported no-thinking switch.

Two MTP3 controls retained the same E4M3 KV and default target/speculator
Graphs. With 4,096 output tokens they scored 14/16; raising the budget to
8,192 remained 14/16 because HumanEval/1 and /11 still truncated. The 8K MTP
run observed 28.44 output tokens/s and live draft acceptance, but its lower
accuracy makes it unsuitable as the default profile. A first attempt using
the requested physical name `fp8_ds_mla` was rejected before weight loading by
the regular FLASHMLA capability contract (`kv_cache_dtype not supported`), so
the runnable public dtype is `fp8_e4m3`; no global validation was weakened.
Evidence is under the three owned
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-r1-0528-channel-int8-*`
directories.

The later `/models/DeepSeek-V3.2-channel-fp8` checkpoint was validated through
both a plain TP8 profile and a TP8+MTP3+E4M3-KV profile. Both passed raw and
independently normalized HumanEval16 at 16/16, with all generated programs
executed successfully. The plain run observed 9.31 output tokens/s and 104.13
ms mean TPOT. The combined feature run observed 22.32 output tokens/s, 38.46
ms mean TPOT, mean MTP acceptance length 3.17, and 72.5% draft-token
acceptance in its final metric window. Generation lengths differed, so the
throughput numbers are route evidence rather than a controlled benchmark.

Both routes used HcuGPUModelRunnerV2, `FLASHMLA_SPARSE`, channel-wise FP8
dense linear, tuned AITER channel-shuffle MoE, prefix caching, native LBNHC KV
layout, and the default FULL_AND_PIECEWISE Graph policy. The combined profile
mapped public `fp8_e4m3` to sparse-MLA `fp8_ds_mla`, allocated 981,376 KV
tokens, compiled the `eagle_head`, warmed the three-token rejection sampler,
and captured both target and speculator PIECEWISE/FULL graphs. Its checkpoint
keeps `model_type=deepseek_v3`, so vLLM reports `DeepSeekMTPModel`; the plugin's
shared HCU MTP implementation identifies V3.2 sparse attention from
`index_topk`. Evidence directories are
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-v32-channel-fp8-tp8` and
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-v32-channel-fp8-mtp3-kvfp8-tp8`.

The first plain attempt used a 4,096-token server context together with an
EvalScope `max_tokens=4096` output budget, leaving no room for input tokens and
producing HTTP 400 before inference. The accepted profiles use
`--max-model-len 8192` with the same 4,096-token output budget; this was a
request-contract correction, not a model or kernel fix.

The subsequently added `/models/GLM-5.2-Channel-INT8-w8a8` checkpoint is
704.33 GiB across 282 shards and was validated at TP8. Its combined acceptance
profile used HcuGPUModelRunnerV2, `FLASHMLA_SPARSE`, `--reasoning-parser
glm45`, AITER INT8 MoE, MTP3, public E4M3 KV mapped to `fp8_ds_mla`, prefix
caching, native LBNHC layout, and default FULL_AND_PIECEWISE target/speculator
Graphs. Raw EvalScope and the independent normalizer both passed HumanEval16
at 16/16, with all generated programs executed successfully.

The run selected the official `TritonInt8ScaledMMLinearKernel` for dense
compressed-tensors W8A8 and AITER for MoE; this is an expected separation of
dense and MoE providers, not a fallback of the requested MoE route. All ranks
loaded the gfx938 `E=256,N=256` ordinary and bottom-layer AITER configs plus
the channel-shuffle tuned table. The final metric window reported mean MTP
acceptance length 3.92 and 97.2% draft-token acceptance. The report observed
29.19 output tokens/s, 4.581 s mean latency, 595.19 ms mean TTFT, and 29.81 ms
mean TPOT. The feature route allocated 744,896 KV tokens; evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm52-channel-int8-tp8`.

The subsequently added `/models/GLM-4.7-W8A8` checkpoint is 334.08 GiB and
was validated at TP4. Its accepted profile used HcuGPUModelRunnerV2,
`FLASH_ATTN`, an HND selection exposed by the runtime as the physical LBHNC
cache view, MTP2, native `fp8_e4m3` KV, prefix caching, and the unmodified
default FULL_AND_PIECEWISE target/speculator Graph policy. Target and draft
prefill/decode PIECEWISE and FULL captures completed. The route allocated
998,784 KV tokens and loaded 85.93 GiB of model memory per card.

Dense compressed-tensors W8A8 selected `TritonInt8ScaledMMLinearKernel`.
The requested AITER INT8 MoE route performed its config lookup, but the
observed M=1 `E=160,N=384` shape had no supported AITER solution and explicitly
fell back to the official vLLM Triton MoE implementation on every rank. This
validates the ordered lookup and fallback path, not AITER kernel execution for
that shape. Raw EvalScope and independent normalization passed HumanEval16 at
16/16. All 18 chat requests, including the two prefix probes, returned HTTP
200 with no ERROR or Traceback. The final metric window reported 35.2% prefix
hits and 85.1% draft-token acceptance. The report observed 21.21 output
tokens/s, 6.755 s mean latency, 1.474 s mean TTFT, and 36.25 ms mean TPOT.
Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm47-w8a8-mtp2-kvfp8-tp4`.

The later `/models/Kimi-K2.6` checkpoint was validated as a language-only TP8
route with HcuGPUModelRunnerV2, regular FLASHMLA, Triton WNA16 MoE, prefix
caching, and BF16/auto LBNHC KV. The checkpoint has no MTP layers. Two runtime
ownership hypotheses were investigated, but only one reproduced the fault:
the HCU fused-MoE replacement omitted upstream ROCm's guard against
auxiliary-stream shared-expert overlap when routed/shared inputs alias.
WNA16's unquantized routed-input path now serializes shared experts. Restoring
BF16 FlashMLA's advertised full-graph capability passed the high-risk batch
and HumanEval 64/64, disproving the scheduler-metadata hypothesis. The final
command needs no debug environment switch and retains default
FULL_AND_PIECEWISE Graph capture and prefix reuse.

The deterministic Instant-mode run uses temperature zero, `thinking=false`,
batch eight, and a 1,024-token output cap. It produced 64 predictions and
reviews and passed independently normalized HumanEval 64/64. EvalScope's raw
checker undercounted complete fenced or horizontally indented modules, so the
accepted gate uses the repository's syntax-aware normalizer inside the
explicit isolated execution boundary. Earlier 51/64, 52/64, 54/64, and 55/64
controls remain diagnostic history from before the stream-race fix. Full
evidence and exact commands are in
`docs/validation/kimi-k26-gfx938-humaneval64.md`.

### Qwen3.8 Flash-Next SlimQuant W4A8 current-head rerun

The `/models/Qwen3.8-Flash-Next-w4a8-slimquant` checkpoint was rerun at branch
head `6ebd45f` with the pinned vLLM 0.28.1 wheel. The accepted service command
was:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve /models/Qwen3.8-Flash-Next-w4a8-slimquant \
  --served-model-name Qwen3.8-Flash-Next-w4a8-slimquant \
  --port 10237 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --quantization slimquant_w4a8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --gpu-memory-utilization 0.60 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.8-Flash-Next-w4a8-slimquant \
  --api-url http://127.0.0.1:10237/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen38-flash-next-w4a8-current-tp4-mtp3-kvfp8-run1-20261007 \
  --no-timestamp
```

All four ranks constructed `HcuGPUModelRunnerV2`. The public E4M3 route
selected the QSA FP8 sparse-GQA reader, native `BLNHC`, a 64-token HCU kernel
page, and a 1,600-token hybrid manager page. The default policy captured
target and MTP prefill PIECEWISE/FULL Graphs and MTP decode FULL Graphs. Model
loading used 63.67 GiB/rank and the service allocated 666,586 KV tokens.

HumanEval raw Accuracy and Pass@1 both passed 16/16. The report observed
25.32 output tok/s, 3,107.4 ms mean TTFT, and 17.4 ms mean TPOT. Three
identical 3,525-token prompts all returned `17` with a normal stop and added
3,200 prefix-hit tokens, which covers two 1,600-token manager pages. The
complete request window accepted 1,673/1,881 MTP draft tokens (88.9%).

The explicit AITER lookup found no supported target W4A8 solution for
`E=512,N=160,K=2560,top_k=10` and no supported W16A16 solution for the MTP
layer, so both shapes fell back to official vLLM Triton. Record this as
ordered backend lookup plus fallback, not AITER kernel coverage. No ERROR,
Traceback, VM fault, or dead engine occurred, and exact process-group teardown
returned all eight cards to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen38-flash-next-w4a8-current-tp4-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/qwen38-flash-next-w4a8-current-tp4-mtp3-kvfp8-run1-20261007`

### Qwen3.8 Flash-Next Channel-FP8 current-head rerun

The `/models/Qwen3.8-Flash-Next-FP8-Channelwise` checkpoint was independently
rerun at current MR head `87ef138`. The accepted server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve /models/Qwen3.8-Flash-Next-FP8-Channelwise \
  --served-model-name Qwen3.8-Flash-Next-FP8-Channelwise \
  --port 10240 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --gpu-memory-utilization 0.60 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen38-flash-fp8 \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.8-Flash-Next-FP8-Channelwise \
  --api-url http://127.0.0.1:10240/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen38-flash-next-fp8-channelwise-current-tp4-mtp3-kvfp8-run1-20261007 \
  --no-timestamp
```

All four ranks constructed `HcuGPUModelRunnerV2` and selected the QSA FP8
sparse-GQA reader. Leaving `VLLM_KV_CACHE_LAYOUT` unset resolved native
`BLNHC`, a 64-token HCU kernel page, and a 1,600-token hybrid manager page.
Model loading used 58.35 GiB/rank. The first exact-shape target and MTP AOT
compilations took 262.94 and 132.01 seconds; total engine initialization was
553.99 seconds, including 394.95 seconds of compilation. Target and MTP
prefill captured PIECEWISE/FULL Graphs, MTP decode captured FULL Graphs, and
the service allocated 680,542 KV tokens.

The ordered AITER lookup loaded the channel-shuffle table but found no
supported FP8 solution for `E=512,N=160,K=2560,top_k=10`; the observed shape
therefore fell back explicitly to official vLLM Triton MoE. This is fallback
coverage, not an AITER kernel-hit claim.

HumanEval raw Accuracy and Pass@1 passed 16/16. The report observed 23.74
output tok/s, 2,914.5 ms mean TTFT, 22.8 ms mean TPOT, and 6.343 s mean
latency. Three identical 3,521-token prompts returned `17`; the second and
third each reused one 1,600-token manager page, adding 3,200 prefix-hit tokens
in total. This align-only route proves whole-manager-page reuse and does not
claim fine-grained matching. The complete request window accepted
1,778/1,962 MTP draft tokens (90.62%). No ERROR, Traceback, VM fault, OOM, or
dead engine occurred. Exact PGID teardown closed the port and returned the
owned cards 0--3 to the 3 MiB idle reading; unrelated contexts appeared on
cards 4--7 during the run and were not modified or attributed to this test.

Evidence:

- `/tmp/vllm-hcu-validation/qwen38-flash-next-fp8-channelwise-current-tp4-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/qwen38-flash-next-fp8-channelwise-current-tp4-mtp3-kvfp8-run1-20261007`

### Qwen3.8 27B Channel INT8 current-head rerun

The `/models/Qwen3.8-27B-Channel-INT8-w8a8` checkpoint was rerun after the
SlimQuant gate at MR head `d8c2e2b`. The accepted service command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --port 10238 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.40 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen38-27b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.8-27B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10238/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen38-27b-channel-int8-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

Both ranks constructed `HcuGPUModelRunnerV2`. Compressed-tensors selected
`TritonInt8ScaledMMLinearKernel`; FLASH_ATTN used a 64-token kernel page and
the hybrid cache manager used a 1,600-token page with physical `LBHNC` layout.
The public `fp8_e4m3` route was accepted. Target and MTP prefill captured both
PIECEWISE and FULL Graphs, while MTP decode captured FULL Graphs. Model loading
used 17.27 GiB/rank and the service allocated 612,839 KV tokens.

The owner/junction/sibling probe used a new 3,535-token shared-prefix family.
Its per-request prefix hits were 0, 1,600, and 3,136 tokens. The sibling hit is
49 times the configured 64-token match unit and is not a multiple of the
1,600-token manager page, proving that fine-grained matching, rather than only
whole-page reuse, was active. All three requests returned `17`.

HumanEval raw Accuracy and Pass@1 both passed 16/16. The report observed
37.01 output tok/s, 1,957 ms mean TTFT, 15.1 ms mean TPOT, and 4.252 s mean
latency. The complete request window accepted 1,898/1,980 MTP draft tokens
(95.86%). No ERROR, Traceback, VM fault, or dead engine occurred. Exact
process-group teardown returned all eight cards to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen38-27b-channel-int8-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen38-27b-channel-int8-current-tp2-mtp3-kvfp8-fine-run1-20261007`

### Qwen3 8B current-head E4M3 rerun

The `/models/Qwen3-8B` checkpoint was rerun at MR head `9d6e4ef`. The
accepted service command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen3-8b-current \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --port 10247 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen3-8b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3-8B \
  --api-url http://127.0.0.1:10247/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen3-8b-current-tp2-kvfp8-run1-20261007 \
  --no-timestamp
```

Both ranks constructed `HcuGPUModelRunnerV2`; the engine resolved
`kv_cache_dtype=fp8_e4m3`, physical `LBHNC`, and a 64-token FLASH_ATTN block.
Default compilation selected `FULL_AND_PIECEWISE`, and both PIECEWISE and FULL
graphs were captured. The service allocated 1,163,200 KV tokens. This dense
checkpoint has no MTP layer, so no speculative configuration was added.

HumanEval raw Accuracy and Pass@1 both passed 16/16. The report observed
105.03 output tok/s, 113.2 ms mean TTFT, 8.7 ms mean TPOT, and 1.23 s mean
latency. Two identical long-prefix requests returned coherent content and
increased `vllm:prefix_cache_hits_total` from 0 to 2,688 tokens; the final
session counters were 2,688/7,588 hit/query tokens. All 18 chat requests
returned HTTP 200, and the log contained no ERROR, Traceback, VM fault, or
dead engine. Ctrl-C teardown returned cards 0 and 1 to the 2 MiB baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen3-8b-current-tp2-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/qwen3-8b-current-tp2-kvfp8-run1-20261007`

### Qwen3.5 35B A3B W8A8 current-head fine-prefix rerun

The `/models/Qwen3.5-35B-A3B-W8A8` checkpoint was rerun at plugin commit
`dcf047e`. The accepted service command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-35b-w8a8-current \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name Qwen3.5-35B-A3B-W8A8 \
  --port 10248 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.45 \
  --max-model-len 8192 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen35-35b-w8a8 \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.5-35B-A3B-W8A8 \
  --api-url http://127.0.0.1:10248/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen35-35b-w8a8-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

Both ranks constructed `HcuGPUModelRunnerV2`. Compressed-tensors selected its
W8A8 frontend while the default HCU wrapper retained the LightOp dense GEMM
route. AITER loaded the ordinary and bottom-layer gfx938
`E=256,N=256,dtype=int8_w8a8` configurations plus the channel-shuffle table.
Public E4M3 KV resolved to physical LBHNC with a 64-token kernel block and a
2,176-token hybrid manager page. Target and MTP prefill captured PIECEWISE and
FULL Graphs, MTP decode captured FULL Graphs, and the service allocated
1,704,367 KV tokens.

Raw HumanEval Accuracy and Pass@1 passed 16/16. The report observed 62.99
output tok/s, 509.2 ms mean TTFT, 11.1 ms mean TPOT, and 1.599 s mean latency.
The HumanEval window accepted 1,181/1,281 MTP draft tokens (92.19%).

The final owner/junction/sibling probe used three 3,656-token prompts with an
actual 2,283-token common prefix. All returned `17`; their prefix-hit deltas
were 0, 0, and 2,112 tokens. The last value is 33 times the configured
64-token match unit and is not divisible by the 2,176-token manager page,
proving fine-grained junction reuse. A diagnostic probe whose common prefix
was shorter than one physical manager page correctly produced no junction;
the probe must cross the physical page before the EAGLE/MTP one-unit rollback
can expose a 2,112-token reusable boundary.

All 35 accepted chat requests, including diagnostics, returned HTTP 200. The
complete session accepted 1,238/1,338 MTP draft tokens (92.53%) and contained
no ERROR, Traceback, VM fault, or dead engine. Ctrl-C teardown returned all
cards to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen35-35b-w8a8-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen35-35b-w8a8-current-tp2-mtp3-kvfp8-fine-run1-20261007`

### Qwen3.5 35B A3B BF16 current-head E4M3 fine-prefix rerun

The `/models/Qwen3.5-35B-A3B` checkpoint was rerun at plugin commit `e3fcf0c`.
The accepted service command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-35b-current \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name Qwen3.5-35B-A3B \
  --port 10249 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.55 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The matching isolated HumanEval16 client command was:

```bash
env -i \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen35-35b \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.5-35B-A3B \
  --api-url http://127.0.0.1:10249/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/qwen35-35b-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

Both ranks constructed `HcuGPUModelRunnerV2`. AITER selected
`AiterExperts`, loaded its tuned ASM shuffle table, and successfully loaded
gfx938 BF16 W16A16 stage1/stage2 HSA modules. Public E4M3 KV resolved to
physical LBHNC with a 64-token kernel block and a 2,176-token hybrid manager
page. Target and MTP prefill captured PIECEWISE and FULL Graphs, MTP decode
captured FULL Graphs, and the service allocated 916,540 KV tokens.

Raw HumanEval Accuracy and Pass@1 passed 16/16. The report observed 88.52
output tok/s, 386.4 ms mean TTFT, 7.9 ms mean TPOT, and 1.12 s mean latency.
The HumanEval window accepted 1,166/1,248 MTP draft tokens (93.43%).

The three 3,651-token owner/junction/sibling prompts had an actual 2,291-token
common prefix and all returned `17`. Their prefix-hit deltas were 0, 0, and
2,112 tokens; the sibling hit is 33 times 64 and is not divisible by the
2,176-token manager page, proving fine-grained junction reuse. All 19 chat
requests returned HTTP 200. Complete-session MTP counters were 1,175/1,257
(93.48%), and the log contained no ERROR, Traceback, VM fault, or dead engine.
Ctrl-C teardown returned all cards to the 2 MiB idle baseline.

Evidence:

- `/tmp/vllm-hcu-validation/qwen35-35b-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen35-35b-current-tp2-mtp3-kvfp8-fine-run1-20261007`

Additional checkpoints under `/llm-models-2/hygon` were then exercised with
the same pinned runtime. The detailed score matrix, exact server/client
commands, hybrid fine-grained prefix evidence, Qwen3.8 long-prefix regression,
and the Qwen4Exp BLNHC layout exception are recorded in
`docs/validation/llm-models-2-hygon-v0281.md`.

## Source closure after runtime fixes

- Full `tests/runtime_patch`: `1679 passed, 14 warnings` after the final Kimi
  force-control fix and review updates.
- Full `tests/models/hy_v4`: `222 passed, 14 warnings`.
- HY4 live TP8 gate: `16/16`, 16 predictions and reviews, final observed
  prefix hit rate about 34.3%, final-window MTP acceptance about 91.3%.
- Hy3 Channel-FP8 live gates: TP8/AITER and DP8/TP1/EP8
  DeepEP-low-latency/DeepGEMM profiles each passed raw and normalized
  HumanEval `16/16` with MTP2, E4M3 KV, prefix caching, and default target and
  draft FULL plus PIECEWISE Graphs. The DP profile used nine identical prefix
  requests so at least one request returned to the same rank.
- DeepSeek-V3.2 live TP8 gates: plain and MTP3+E4M3-KV profiles each `16/16`,
  with 16 predictions, reviews, and successful code executions per profile.
- DeepSeek-R1-0528 Channel-INT8 diagnostic TP8+E4M3-KV gate: no-MTP produced
  16 predictions/reviews and `15/16`; MTP3 produced `14/16` at both 4K and 8K
  output budgets. All three runnable services used default Graphs and cleaned
  back to the 2 MiB-per-card idle baseline.
- GLM-5.2 Channel-INT8 live TP8 gate: MTP3+E4M3-KV profile `16/16`, with
  16 predictions, reviews, and successful code executions.
- GLM-4.7 W8A8 live TP4 gate: MTP2+E4M3-KV profile `16/16`, with target/draft
  FULL plus PIECEWISE Graphs, 18/18 HTTP 200 responses, and an explicit AITER
  config-miss-to-Triton MoE fallback for the observed M=1 shape.
- GLM-5 W8A8 post-shared-expert live TP8 gate: MTP3 profile `16/16`, with
  target/draft FULL plus PIECEWISE Graphs, 18/18 HTTP 200 responses including
  prefix probes, and no ERROR or Traceback.
- GLM-5.3 Channel-FP8 post-shared-expert live TP8 gates: MTP3+E4M3-KV passed
  `16/16`, then the extended accuracy gate passed `32/32`; the final service
  recorded 34/34 HTTP 200 responses and no ERROR or Traceback.
- Kimi-K2.6 live TP8 gate: deterministic Instant mode produced 64
  predictions/reviews and normalized HumanEval `64/64`; service, prefix,
  FULL plus PIECEWISE Graphs, FLASHMLA/LBNHC, and Triton WNA16 routes passed
  without a shared-expert stream environment override. The same service first
  passed the eight high-risk prompts, for 72/72 HTTP 200 responses total and
  no ERROR or Traceback.
- Kimi-K2.6 explicit E4M3-KV smoke: the TP8 route resolved
  `kv_cache_dtype=fp8_e4m3`, captured FULL plus PIECEWISE Graphs, retained
  LBNHC, allocated 1,718,272 KV tokens, and completed the eight historical
  high-risk concurrent prompts with HTTP 200 and normal stops.
- Kimi-K2.6 benchmark-style Thinking diagnostic: 64 predictions/reviews,
  normalized HumanEval `55/64`, with four 16K reasoning-only truncations. This
  predates the shared-expert stream-race fix and is retained only as history.
- Kimi changed-file focused suite includes explicit BF16/FP8 FlashMLA Graph
  capability, shared-expert stream-safety, and force-control regression
  coverage. The final deterministic profile enforces normalized score `1.0`;
  the final focused re-review passed `40` tests with one deselection and both
  static command contracts passed (`2 passed, 1 deselected`).
- Current profile/report focused suite: `69 passed, 1 skipped`.
- Skill validation: `Skill is valid!`; repository/skill PAT-pattern scan:
  clean; `git diff --check`: clean.

HumanEval executes model-generated code. The harness now requires an explicit
EvalScope sandbox or the operator assertion
`VLLM_HCU_HUMANEVAL_ISOLATED=1`; merely detecting a container is not treated
as a security boundary. Sandbox results are not re-executed by the local
normalizer. Diagnostic profiles normally check only artifact counts; an
explicit `record_normalized_score` option may re-run HumanEval checks only at
an already isolated host boundary, recording but not enforcing the score.
The evaluator receives an isolated HOME and an explicit
credential denylist. EvalScope API keys are passed through a protected
environment to a Python launcher and do not appear in OS argv or persisted
command logs.
Inherited outbound proxies are preserved for dataset access while loopback is
always added to `NO_PROXY`; the vLLM server ignores inherited proxies but
honors an explicitly configured server proxy.

## Final reviewed plugin wheel

The reviewed runtime/test change is commit
`cdea830d36ba79a8dc2410253d871954498ab72c`. The clean tracked worktree at
that commit produced:

| Package | Filename | SHA256 |
| --- | --- | --- |
| vLLM HCU plugin | `vllm_hcu-0.28.1rc1.dev491+das.cdea830.dtk26041-cp310-cp310-linux_x86_64.whl` | `73898c9400f4c0476498573a243b7c9feb7494719cbe4aca1a492f26c85f37be` |

It was installed with `--no-deps` under
`/models/.installs/vllm-plugin-v0281-gfx938-cdea830`. With
`PYTHONNOUSERSITE=1` and the pinned vLLM root first in `PYTHONPATH`, both the
parent and a fresh child interpreter resolved:

```text
vllm      /models/.installs/vllm-v0281-gfx938-g77acaf/vllm/__init__.py
vllm_hcu  /models/.installs/vllm-plugin-v0281-gfx938-cdea830/vllm_hcu/__init__.py
hcu_ops   /models/.installs/vllm-plugin-v0281-gfx938-cdea830/vllm_hcu/hcu_ops.cpython-310-x86_64-linux-gnu.so
```

The installed-root compatibility, packaging, and lifecycle gate completed
with `62 passed, 14 warnings in 126.35s`. The exact committed runtime patch
suite completed with `1669 passed, 14 warnings in 426.33s`; the final complete
changed-file suite completed with `581 passed, 3 skipped, 14 warnings in
151.81s`.
