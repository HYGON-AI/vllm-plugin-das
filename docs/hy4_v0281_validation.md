# Hy4 v0.28.1-dev HCU validation record

## Frozen baseline, 2026-09-25 UTC

- Plugin source after proxy-free fetch: origin/v0.25.1 at 6ea7b12f3d77d4564d613aac31c5e6c5e7a8641d.
- Plugin target after proxy-free fetch: origin/v0.28.1-dev at 93021650a5c768121507d5b8241d0e848756e44e.
- Feature worktree: feat/hy4-v0281-alignment, based on the target SHA above.
- OpenDAS vLLM: 0.28.1+dtk2604.torch2110.2609171627.g77acaf, imported from /usr/local/lib/python3.10/dist-packages/vllm. This version reports the 77acaf lineage; the installed wheel artifact and checksum are not available in the inspected /models locations or dist-info direct_url.json, so exact delivery provenance remains unverified.
- Installed plugin distribution: vllm-hcu 0.28.1rc1.dev491+dtk2604.torch2110.2609191510.g06f72c. Tests import plugin source from this worktree, not the installed distribution.
- Python 3.10.12; PyTorch 2.11.0, HIP 6.3.26113; DTK 2604 is encoded in the installed package versions.
- Proprietary providers: aiter 0.1.6+dtk2604.torch2110.2609200940.g0cb699; flash_attn 2.8.4+dtk2604.torch2110.2609241509.g624d7b; flash_mla 1.2.0+dtk2604.torch2110.2609161027.g5b030d; lightop 0.6.0+dtk2604.torch2110.2609201832.g8cd3e1.
- Model: /models/Hy4-preview-Channel-FP8-w8a8-v2; config.json SHA-256 d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441; architecture HYV4ForCausalLM, model_type hy_v4, 78 hidden layers, one checkpoint MTP layer.
- Eight HCU devices were visible; each reported 147440 MiB total and 2 MiB used before implementation.
- The configured 127.0.0.1:7897 HTTP proxy was unavailable. Fetch succeeded with proxy environment variables unset and git http.proxy cleared; no credentials were put into commands or this record.

## Model Runner V2 gate

Every Hy4 server command must explicitly set:

~~~bash
export VLLM_USE_V2_MODEL_RUNNER=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
~~~

The target HCU worker rejects Model Runner V1. The existing plugin's automatic V2 normalization applies only to its GLM DSA architecture and is not a Hy4 selector. The live TP8, MTP3, and FP8-KV gates must each confirm VllmConfig.use_v2_model_runner=true and actual HcuGPUModelRunnerV2 construction, not just the environment variable.

Baseline characterization:

~~~text
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/patch/test_plugin_lifecycle.py -k 'model_runner or worker_rejects'
4 passed, 32 deselected, 14 torch deprecation warnings, 18.65s
~~~

## Live-gate attempts

- First TP8/MRV2 launch log: `/models/hy4-v0281-validation-20260925/target_server.log`. Eight workers reached distributed initialization but could not import `vllm_hcu.hcu_ops`: the source worktree has no compiled extension. All workers exited and device memory returned to idle. This is an artifact-layout failure, not model inference evidence.
- A validation-only `sitecustomize.py` under `/models/hy4-v0281-validation-20260925/` extends the source package search path to the installed extension. The installed `hcu_ops.cpython-310-x86_64-linux-gnu.so` SHA-256 is `a90c6728f22affd5c63379b9425c07a90aa3ecd09a19f7bdc0854ca8f01c9e26`. This overlay is outside the MR and is not an isolated build/provenance proof.
- Second TP8/MRV2 launch log: `/models/hy4-v0281-validation-20260925/target_server_overlay.log`. Eight workers loaded all 131 safetensors shards (736.24 GiB) and selected ChannelWiseTorchFP8ScaledMMLinearKernel, AITER FP8 MoE, and default `FULL_AND_PIECEWISE` Graph policy, then failed at KV-cache initialization: `No common block size for 16`. Hy4's sparse MLA backend requires 64-token pages. The harness now pins `--block-size 64`.
- Third launch with `--block-size 64`, log `target_server_block64.log`, completed weights and KV-cache profiling but failed during Graph capture: the upstream default `FULL_AND_PIECEWISE` policy needs `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for this non-torch-compiled model. The target plugin now auto-enables that opt-in for HYV4ForCausalLM and HYV4MTPModel unless explicitly overridden; `test_registration.py` covers both.
- Fourth launch with `--block-size 64` and the explicit breakable opt-in, log `target_server_breakable.log`, passed all 131 shards, captured PIECEWISE and FULL CUDA Graphs on all eight workers, and reached `Application startup complete`. The HCU worker factory rejects V1 and constructs `HcuGPUModelRunnerV2` when `VLLM_USE_V2_MODEL_RUNNER=1`; a later review added an explicit runtime class-name log that will be checked in the isolated-wheel run. Eight `/v1/chat/completions` requests completed with HTTP 200.

The fourth launch used this command (the `PYTHONPATH` prefix is a validation-only bridge to the installed binary extension, not an MR dependency):

~~~bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NO_PROXY=127.0.0.1,localhost \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/models/hy4-v0281-validation-20260925:/models/vllm-plugin-das/.worktrees/feat-hy4-v0281-alignment \
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 8 --moe-backend aiter --enable-prefix-caching \
  --max-model-len 4096 --block-size 64 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' --seed 0
~~~

HumanEval/0-7 was sent through the same deterministic chat prompt template,
temperature 0, seed 0 and max_tokens 2048. Raw completions are in
`/models/hy4-v0281-validation-20260925/humaneval8_target.json`; the EvalScope
1.11.0 functional checker reported **8/8 passed** in
`humaneval8_target_score.json`. This is an eight-sample smoke accuracy gate,
not a full HumanEval benchmark or a statistical quality estimate.

The MTP3 command adds only
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` to the
baseline command and omits the diagnostic breakable-graph environment
variable. The plugin auto-enabled breakable CUDA Graph, main and speculative
Graph captures completed on all eight workers, and the API became ready;
log: `mtp3_server.log`. Fused multi-step metadata update was unavailable for
`DEEPSEEK_V32_INDEXER`/`FLASHMLA_SPARSE`, so official MRV2 rebuilt metadata
between draft steps. SpecDecoding windows accepted 195/213, 214/225 and
213/225 drafted tokens; these are windowed log values, not a whole-run rate.
The same eight prompts passed **8/8** under the EvalScope checker; artifacts
`humaneval8_mtp3.json` and `humaneval8_mtp3_score.json`. Seven completions
matched baseline byte-for-byte; HumanEval/6 differed but passed both runs.

The FP8 E4M3 KV command adds `--kv-cache-dtype fp8_e4m3` to the MTP3
command (with no explicit breakable-graph environment variable). The service
loaded all weights, selected ChannelWiseTorchFP8ScaledMMLinearKernel and AITER
FP8 MoE, logged `Using fp8_e4m3 data type to store kv cache` and
`Padding num_heads from 8 to 64 for FP8 sparse decode kernel`, captured
PIECEWISE and FULL CUDA Graphs, and reached API readiness; log:
`fp8_e4m3_server.log`. The HCU sparse-MLA cache-update patch routes ROCm
requests to `torch.ops.hcu_ops.concat_and_cache_mla`; there is no per-call
native-writer trace in this run. Windowed MTP acceptance included 229/240,
244/255, 246/255, and 73/75 drafted tokens. HumanEval/0-7 passed **8/8**;
artifacts: `humaneval8_fp8_e4m3.json` and
`humaneval8_fp8_e4m3_score.json`. Four completions differed byte-for-byte
from BF16 MTP3; all eight passed functional tests.

Eight repeated requests on the same FP8 service passed **8/8** again;
artifacts: `humaneval8_fp8_e4m3_repeat.json` and
`humaneval8_fp8_e4m3_repeat_score.json`. Prefix-cache hit count increased
from 0 to 512 during the repeat, and the server reported a 20.7% hit rate
after it. Six repeated completions matched the first FP8 run byte-for-byte;
HumanEval/0 and /1 differed but both passed. These source-plus-installed-
binary-overlay runs show functional behavior, not an isolated plugin wheel;
the final isolated-wheel gates follow.

## Final isolated plugin wheel, TP8/MRV2, 2026-09-25

- Plugin code boundary: `691f8baff0ae6244eaccdbc638131aca1b1771da` on `feat/hy4-v0281-alignment`, still based on refreshed `origin/v0.28.1-dev@93021650a5c768121507d5b8241d0e848756e44e`. The later validation-document commit does not change the tested package code.
- Built wheel: `/models/vllm-plugin-das/.worktrees/feat-hy4-v0281-alignment/dist/vllm_hcu-0.28.1rc1.dev491+das.691f8ba.dtk2604-cp310-cp310-linux_x86_64.whl`; SHA-256 `213d34eb370ab20f0fd07583d48eb7cdcd18b5d4ec6c6fd6311bd513fc1444f3`.
- Installed with `pip install --no-deps --target /models/hy4-v0281-validation-20260925/plugin-wheel-final`. Both `vllm_hcu.__file__` and the compiled `hcu_ops` resolved below that directory; the latter SHA-256 is `cbf3556050efe4372b5eaba8f8a7cfa1532b2747efc2fe6cfee894d0b0d491a6`. A live TP0 worker's mapped `hcu_ops` also pointed there. vLLM resolved to the installed `/usr/local/lib/python3.10/dist-packages/vllm`, version `0.28.1+dtk2604.torch2110.2609171627.g77acaf`.
- An independent cold-import/bootstrap check loaded 110 enabled target modules, armed 136 plugin registrations and reported zero failed patches from the isolated install. It used no source worktree or validation-only `sitecustomize.py` on its import path. Retained output: `/models/hy4-v0281-validation-20260925/final_wheel_bootstrap.log`.

The exact final FP8 E4M3 service command was run from `/models` (not from
the plugin source worktree):

~~~bash
set -o pipefail
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/models/hy4-v0281-validation-20260925/plugin-wheel-final \
  python3 -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 8 --moe-backend aiter --enable-prefix-caching \
  --max-model-len 4096 --block-size 64 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  2>&1 | tee /models/hy4-v0281-validation-20260925/final_wheel_fp8_e4m3_server.log
~~~

The BF16+MTP3 service used the same command without
`--kv-cache-dtype fp8_e4m3` and logged to `final_wheel_mtp3_server.log`.
The BF16 target-only service additionally omitted `--speculative-config` and
logged to `final_wheel_target_server.log`. All three used the same installed
plugin wheel, eight GPUs, model, request template and seed. No explicit
Graph/eager mode or manual breakable-Graph opt-in was supplied.

| Final wheel mode | HumanEval/0-7 functional checker | Raw predictions / score | Live evidence |
| --- | --- | --- | --- |
| BF16 target only | **8/8** | `final_wheel_humaneval8_target.json`, `final_wheel_humaneval8_target_score.json` | `final_wheel_target_server.log` |
| BF16 + MTP3 | **8/8** | `final_wheel_humaneval8_mtp3.json`, `final_wheel_humaneval8_mtp3_score.json` | `final_wheel_mtp3_server.log` |
| FP8 E4M3 KV + MTP3 | **8/8** | `final_wheel_humaneval8_fp8_e4m3.json`, `final_wheel_humaneval8_fp8_e4m3_score.json` | `final_wheel_fp8_e4m3_server.log` |
| FP8 repeated on same server | **8/8** | `final_wheel_humaneval8_fp8_e4m3_repeat.json`, `final_wheel_humaneval8_fp8_e4m3_repeat_score.json` | `final_wheel_fp8_e4m3_server.log` |

All prediction and score files above are under
`/models/hy4-v0281-validation-20260925/`. Each response had
`finish_reason=stop`; generated code was inspected for risky imports/calls
before invoking EvalScope 1.11.0's functional checker. This is an eight-item
smoke gate, not a full HumanEval or statistical FP8-parity estimate.

Each final-wheel service logged eight actual
`HCU model runner constructed: HcuGPUModelRunnerV2` instances, resolved
`FULL_AND_PIECEWISE`, completed main PIECEWISE/FULL Graph capture, and reached
API readiness. Both MTP3 modes also captured speculative prefill/decode
Graphs and accepted drafted tokens. BF16 MTP3 windows included 211/234,
203/213, 206/225, and 97/102 accepted/drafted tokens; FP8 windows included
221/240, 251/267, 241/252, and 70/72. These are window samples, not
whole-run acceptance rates. Dense Channel-FP8 and AITER FP8 MoE were
selected independently; the AITER runtime found its `gfx938/fp8_w8a8`
configuration. The FP8 service logged its native sparse-decode head padding
from 8 to 64 and successful capture/inference. The ROCm sparse-MLA cache
patch is wired to `torch.ops.hcu_ops.concat_and_cache_mla`, but these logs do
not contain a per-call writer trace, so exact call counts are not claimed.

The first FP8 wheel run had 1,236 prefix-cache query tokens and zero hits;
after repeating the eight requests on the same server, it had 2,472 queries
and 512 hits (20.7% reported hit rate). BF16 target versus BF16 MTP3 changed
HumanEval/1, /5 and /6 byte-for-byte, while both scored 8/8. BF16 MTP3 and
FP8 E4M3 first-run completions were byte-identical for all eight; the FP8
repeat changed /0 and /6, still 8/8. Each service and its eight workers
exited after SIGINT; all eight cards returned to 2 MiB used, and a later
`rocm-smi --showpids` reported no KFD processes. Retained outputs:
`/models/hy4-v0281-validation-20260925/final_wheel_teardown_smi.log`
(device memory) and `final_wheel_teardown_pids.log` (clean PID retry).
The preinstalled OpenDAS vLLM wheel's exact delivery artifact and checksum
remain **unverified**. The independently built plugin wheel is checksummed,
but this is not full paired-wheel release acceptance; keep the MR in Draft
until the target vLLM artifact is pinned and that gate is rerun.

## DP2 × TP4 + EP8 + MTP3 default-Graph gate, 2026-09-25

This is an additional, **failed accuracy gate**, not part of the passing TP8
smoke result. The same isolated plugin-wheel install and Hy4 checkpoint were
used on devices 0–7. The intended non-eager service command was:

~~~bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/models/hy4-v0281-validation-20260925/plugin-wheel-final \
  python3 -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8012 --trust-remote-code \
  --tensor-parallel-size 4 --data-parallel-size 2 \
  --enable-expert-parallel --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm --enable-prefix-caching \
  --max-model-len 4096 --block-size 64 --max-num-seqs 16 \
  --max-num-batched-tokens 256 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' --seed 0 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
~~~

The test also isolated the proxy environment, cache directory, and log file;
those do not change the model flags. Its log is
`/models/hy4-dp-ep-mtp3-20260925.KLOg/dp2_tp4_ep_ll_deepgemm_mtp3_batch256_server.log`.
Eight workers constructed `HcuGPUModelRunnerV2`, selected
`DeepEPLLAll2AllManager` and DeepGEMM LL experts, captured main and draft
PIECEWISE/FULL Graphs under `FULL_AND_PIECEWISE`, and reached a healthy API.
Two concurrent HumanEval requests were routed to separate DP engines;
both returned HTTP 200 but ended at 2048 tokens with repeated, unusable text.
The partial `/0–1` prediction file is
`dp2_tp4_ep_ll_deepgemm_mtp3_humaneval8.json` in that evidence directory.
Speculative draft acceptance was observed, but does **not** establish
accuracy. The eight-item checker was not run because this gate failed on its
first two items.

Controls on the same wheel/model/request template narrowed, but did not
identify, the defect:

| Configuration delta | HumanEval/0–1 observation | Evidence file prefix in the same directory |
| --- | --- | --- |
| DP2×TP4/EP8/DeepEP-LL/DeepGEMM, no MTP, default Graph | Both repeated/length at 512 | `dp2_tp4_ep_ll_deepgemm_target_humaneval2_diag` |
| Same, `--enforce-eager` | Both repeated/length at 512 | `dp2_tp4_ep_ll_deepgemm_target_eager_humaneval2_diag` |
| DP2×TP4/EP8/AgRs/DeepGEMM, no MTP, default Graph | Both repeated/length at 512 | `dp2_tp4_ep_allgather_deepgemm_target_humaneval2_diag` |
| DP1×TP8/EP8/AgRs/DeepGEMM, no MTP, default Graph, two concurrent requests | `/0` normal `stop` at 189 tokens; `/1` repeated/length at 512 | `dp1_tp8_ep_allgather_deepgemm_target_humaneval2_diag` |
| Same TP8/EP8 service, `/1` alone | Normal `stop` at 116 tokens, plausible code | `dp1_tp8_ep_allgather_deepgemm_target_single_h1` |
| DP2×TP4/EP8/DeepEP-LL/DeepGEMM, no MTP, default Graph, `/1` alone | Repeated/length at 384 | `dp2_tp4_ep_ll_deepgemm_target_single_h1` |

The original EP/DP/TP inference from these controls was superseded by the
sink-routing diagnosis below: TP8 without EP also reproduced the mixed-batch
failure. MTP, Graph capture, and DeepEP-LL are not individually necessary for
the corruption. With `--moe-backend aiter`, the DP2/EP model
fails earlier because AITER rejects the checkpoint's `batched_experts`
activation format; `triton` rejects its Channel-FP8 quantization. At
`--max-num-batched-tokens 4096`, DeepEP-LL RocSHMEM buffer allocation OOMs;
256 allows startup. The corresponding server logs are in the same evidence
directory. At this pre-fix revision, the DP2×TP4/EP8/MTP3 default-Graph
accuracy requirement was **not met**.

## Hy4 learnable-sink mixed-batch fix and accuracy recheck, 2026-09-25

The v0.25.1 DP8/TP1/EP8 MTP3 result was a batch-1 EvalScope 32/32 result,
not a concurrent-request gate. On the target branch before this fix, the same
DP8/TP1/EP8 MTP3 Graph topology scored 8/8 when HumanEval/0–7 were sent
sequentially, but four of eight concurrent requests repeated
`</think:opensource>` until the token limit. TP8 without EP, MTP, Graphs,
FP8 MLA KV, or prefix caching still reproduced the two-request failure.
Layer-0 tracing showed identical pre-attention activations and identical
indexer selections (all 165 tokens for HumanEval/1); the attention output
first diverged when a new short prefill shared a batch with an existing
decode. The target vLLM's short-prefill dense MLA split omits Hy4's learnable
sink. Its own Hy4 NVIDIA implementation forces sparse MQA when the sink is
enabled; this HCU adapter now does the same before constructing MLA attention.

The diagnosis first changed only the public
`--attention-config '{"sparse_mla_force_mqa":true}'` option on the original
wheel: TP8 two-request output normalized, and DP8/EP8/MTP3/FP8-KV Graph
concurrent HumanEval passed 8/8. The final source fix was then copied into an
otherwise unchanged isolated plugin tree. The plugin tree's patched
`attention.py` SHA-256 matched the feature worktree source exactly. Neither
final service below supplied an explicit attention-config override:

| Final gate | Result | Evidence under `/models/hy4-dp8-ep8-diagnosis-20260925.VWpz/` |
| --- | --- | --- |
| DP8/TP1/EP8, DeepEP-LL/DeepGEMM, MTP3, FP8 E4M3 KV, default Graph; eight concurrent HumanEval requests | **8/8**, all `stop` | `dp8_ep8_mtp3_graph_auto_sink_fix_server.log`, `dp8_ep8_mtp3_graph_auto_sink_fix_humaneval8_concurrent.json`, `_score.json` |
| DP2×TP4/EP8, DeepEP-LL/DeepGEMM, MTP3, BF16 KV, default Graph; eight concurrent HumanEval requests | **8/8**, all `stop` | `dp2_tp4_ep8_mtp3_graph_auto_sink_fix_server.log`, `dp2_tp4_ep8_mtp3_graph_auto_sink_fix_humaneval8_concurrent.json`, `_score.json` |

The DP8 launch retained `--tensor-parallel-size 1 --data-parallel-size 8
--enable-expert-parallel --all2all-backend deepep_low_latency --moe-backend
deep_gemm --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 536870912
--gpu-memory-utilization 0.95 --max-num-batched-tokens 256
--speculative-config.method mtp --speculative-config.num_speculative_tokens 3`.
The DP2×TP4 launch retained the pre-fix command above (port 8013 instead of
8012). Both used Model Runner V2, eight devices, max model length 4096,
block size 64 and default non-eager Graph. These eight-sample gates resolve the
documented short-prefill accuracy failure, but do not replace the full
v0.25.1 32-sample gate or the still-unverified paired vLLM-wheel release gate.

## Final source regression suite

The pre-fix source regressions passed in isolated test processes. Commands
below were run from the feature worktree before the learnable-sink routing
fix with
`PYTHONNOUSERSITE=1 PYTHONPATH=.` and `--tb=short`, using separate pytest
processes and `set -o pipefail` so the retained `tee` logs reflect the test
exit status:

| Pytest selection (`pytest -q ... --tb=short`) | Result | Retained output under `/models/hy4-v0281-validation-20260925/` |
| --- | --- | --- |
| `tests/models/hy_v4 tests/runtime_patch/test_hcu_model_runner_v2_api.py tests/integration/server/test_evalscope_hy4_humaneval.py` | **157 passed** | `final_pytest_hy4_mrv2_humaneval.log` |
| `tests/runtime_patch/test_sparse_indexer_torch_fallback.py tests/runtime_patch/test_sparse_indexer_loading.py tests/runtime_patch/test_flashmla_sparse_head_sizes.py tests/runtime_patch/test_indexer_cache_backing_view.py` | **41 passed** | `final_pytest_sparse_indexer.log` |
| `tests/patch/test_plugin_lifecycle.py tests/patch/test_platform_dispatcher.py` | **43 passed** | `final_pytest_plugin_platform.log` |
| `tests/patch/test_clean_process_bootstrap.py` | **2 passed** | `final_pytest_clean_bootstrap.log` |

An earlier broad combined pytest process (before these isolated reruns)
reported nine `HIPBLAS_STATUS_ALLOC_FAILED` errors during
`hipblasCreate(handle)` plus three stale Task-7 assertions that were then
corrected. Its console output was not retained as a file, so that run is not
counted as a passing gate. The sparse-indexer group passed in a fresh
process, and final groups above were rerun after the corrections. The skill
at `/models/upgrading-vllm-hcu/SKILL.md` passed structure validation after
updating the release-line audit and this gate.

After the learnable-sink fix, `python3 -m pytest -q tests/models/hy_v4
tests/runtime_patch/test_hcu_model_runner_v2_api.py
tests/integration/server/test_evalscope_hy4_humaneval.py` passed **160/160**
tests, including short-prefill routing and sink-enabled/sinkless constructor
regressions.
The first full-repository pytest attempt stopped during collection because
the worktree-relative `vllm_0251` source checkout was absent; the actual
v0.25.1 source tree is `/models/zb/vllm_025/vllm`.
With `VLLM_V0251_SOURCE_ROOT` set to that tree, a broad pytest run reached
650 passed and 64 skipped, with five failures outside the changed Hy4 files
(W4A8/DeepGEMM accuracy, two clean-process v0.25.1 bootstrap checks,
LightOp API boundary and patch-coverage audit). It was interrupted after
137 seconds rather than counted as a full-suite pass.
