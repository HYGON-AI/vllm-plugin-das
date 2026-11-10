# HC-SP PIECEWISE capture correctness validation

## Failure and fix

With HC-SP enabled, the default `FULL_AND_PIECEWISE` policy could generate
repeated text such as `ductduct...` on the Qwen3.8-Flash-Next checkpoint.
The non-breakable PIECEWISE path in vLLM's `CudaGraphManager.capture` reuses
the forward closure created with `warmup=True`: it first calls the closure
with `CUDAGraphMode.NONE`, then with `CUDAGraphMode.PIECEWISE`.

The HCU wrapper previously disabled the TP PyNccl communicator for every
invocation of that closure. During PIECEWISE capture, HC-SP's reduce-scatter
therefore reached a disabled communicator and returned without writing its
output. The captured graph omitted the collective, leaving subsequent
computation to consume an unwritten output buffer. Restoring the Python
communicator flag after capture cannot add the missing graph operation.

The fix selects the warmup collective scope using the actual invocation's
`cg_mode`. Only `NONE` calls on a warmup closure use the separate eager
collective path; PIECEWISE capture retains TP PyNccl communication. The
existing RCCL plan ownership workaround and FULL capture behavior remain
in place.

The regression test invokes one `warmup=True` closure with `NONE` followed
by `PIECEWISE` and checks that communication is enabled for the latter.
Both positional and keyword `cg_mode` calls are covered.

## Branch and evaluation configuration

Validation date: 2026-11-07. Branch: `gy-sp-error`. Base commit:
`5f181a36c0e5b97498cc59aa16f3c19fdbded89a`. The fix is a working-tree patch
on that commit. Its production source SHA-256 is
`b594254c5d8e6158cd53de17695d200787e3070041e31a5f6b288e47fbdd18c7`.

- Checkpoint: `Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp`.
- Runtime: vLLM 0.28.1 with the HCU plugin and model runner V2.
- Topology: TP=2, DP=2, EP=4 on devices 0–3; AITER MoE and EP weight filtering.
- All cases: `--engram-config.cpu_offload=true` and
  `VLLM_HCU_PLE_PREFETCH_STREAM=1`.
- MoE sequence parallelism remains enabled under this topology; HC-SP is
  controlled independently.
- Graph policy: default `FULL_AND_PIECEWISE`; Eager: `--enforce-eager`.
- MTP enabled: `method=mtp`, `num_speculative_tokens=3`.
- KV cache: `fp8_e5m2`; model length 32768; maximum 64 sequences and 8192
  batched tokens; prefix caching disabled; thinking disabled.
- Accuracy entry: the workspace's original `test.sh acc`, using evalscope
  1.14.1, HumanEval/0–31, limit=32, batch=32, seed=0, temperature=0,
  top_p=1, max_tokens=16384, retries=2, timeout=180000.

Every case starts a new service and generates its own predictions. Results
are judged by executing the HumanEval tests. An HTTP success or nonempty
response alone does not establish correctness.

## Current branch rerun

Status: complete. All eight cases passed 32/32 tasks, with Accuracy and
Pass@1 both 100%. The results below come from this branch rerun's original
evalscope reports and per-sample execution reviews.

| Mode | MTP | HC-SP | Passed | Pass@1 |
| --- | --- | --- | --- | --- |
| FULL_AND_PIECEWISE | off | 0 | 32/32 | 100% |
| FULL_AND_PIECEWISE | off | 1 | 32/32 | 100% |
| FULL_AND_PIECEWISE | 3 tokens | 0 | 32/32 | 100% |
| FULL_AND_PIECEWISE | 3 tokens | 1 | 32/32 | 100% |
| Eager | off | 0 | 32/32 (retry) | 100% |
| Eager | off | 1 | 32/32 | 100% |
| Eager | 3 tokens | 0 | 32/32 | 100% |
| Eager | 3 tokens | 1 | 32/32 | 100% |

The rerun artifacts are in the sibling workspace directory
`qwen3_8-next-flash/accuracy_matrix_gy_sp_error_20261107/`. The previous
accuracy campaign remains in `accuracy_matrix_20261107/`.

All 256 predictions have no API error and finish with `stop`. All 256
execution reviews report success, `acc=true` and test execution passed.
The verifier checked identical inputs across cases, fresh generation IDs,
no prediction cache, actual serving/evaluation configurations, source
hashes and report denominators. Its status is `passed`.

Compared with the previous accuracy campaign, every case has identical
inputs and a Pass@1 delta of zero. The two sets of 256 completion IDs are
disjoint. Outputs can differ while passing the same tests; this validation
does not assert identical generated text or logits.

The campaign ran from 12:44:57 to 14:02:36 UTC on 2026-11-07.

The three related runtime-patch test files were rerun on this branch:
`27 passed, 14 warnings in 122.85s`, exit status 0. The warnings are existing
PyTorch deprecation warnings. The initial pytest startup also reports an
unset pytest-asyncio fixture loop scope. Full output is preserved in
`related_tests.log` in the rerun artifact directory.

## Startup failure and retry

The first Eager, MTP-off, HC-SP=0 attempt failed before the accuracy test.
At 21:37:28 UTC+8, worker DP1/TP1/EP3 reported
`HIPBLAS_STATUS_INTERNAL_ERROR` in a BF16 `hipblasGemmEx` during visual
encoder profiling. The call passes through
`vllm_hcu/model_executor/layers/linear.py:280`,
`torch.matmul(x, layer.weight) + bias`. Other workers subsequently waited
for communication. The owned service group was stopped after preserving
the raw log and process information.

The complete failed attempt is preserved in
`failed_attempts/05_eager_mtp0_hcsp0_attempt1/`. A new attempt used identical
argv, recorded environment and case configuration, became ready in 181
seconds and passed 32/32. `retry_05_verification.json` confirms that the
configuration is identical and the failed artifacts' hashes are unchanged.
The underlying HIPBLAS failure has not been diagnosed; successful restart
does not establish a fix for that separate startup failure. No inference
parameters or library versions were changed for the retry.

The failed startup generated no accuracy samples. Evaluation processes for
all eight successful attempts exited with status 0. EngineDeadError lines
after completed evaluations occur during deliberate service shutdown.
The verifier confirmed that port 9645 was free and no live members remained
in the successful service process groups; the retry verifier also checked
the failed attempt's process group.

## Reproduction and evidence

The HC-SP=1, Graph, MTP-off service can be reproduced from the plugin
repository root with the following command. The model path is local to the
validation host.

```bash
export PYTHONPATH=".:../vllm:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HCU_PLE_PREFETCH_STREAM=1
export VLLM_ROCM_USE_AITER_CUSTOM_AR=1
export VLLM_HCU_QWEN4_EXP_HC_SP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HIP_VISIBLE_DEVICES=0,1,2,3

vllm serve /public/opendas/DL_DATA/llm-models/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp \
  -tp 2 -dp 2 \
  --enable-expert-parallel --enable-ep-weight-filter --moe-backend aiter \
  --port 9645 --host 0.0.0.0 --served-model-name qwen38-next \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --engram-config.cpu_offload=true --no-enable-prefix-caching \
  --max-model-len 32768 --max-num-seqs 64 \
  --kv-cache-dtype fp8_e5m2 --max-num-batched-tokens 8192
```

For MTP-on cases, append `--speculative-config.method mtp` and
`--speculative-config.num_speculative_tokens 3`. For Eager, append
`--enforce-eager`. Set `VLLM_HCU_QWEN4_EXP_HC_SP=0` for the HC-SP-off cases.

After the service is healthy, run each evaluation from a fresh output
directory. The accuracy command below is the command used by `test.sh acc`:

```bash
evalscope eval \
  --model qwen38-next --eval-type openai_api \
  --api-url http://127.0.0.1:9645/v1 --api-key EMPTY \
  --generation-config '{"timeout":180000,"max_tokens":16384,"temperature":0,"top_p":1,"retries":2}' \
  --datasets humaneval --seed 0 --limit 32 --eval-batch-size 32
```

From the `qwen3_8-next-flash` workspace:

```bash
python accuracy_matrix_gy_sp_error_20261107/run.py
python accuracy_matrix_gy_sp_error_20261107/run.py verify

PYTHONPATH="../_site_v0281/vllm-plugin-das:../_site_v0281/vllm:${PYTHONPATH}" \
  python -m pytest \
  ../_site_v0281/vllm-plugin-das/tests/runtime_patch/test_hcu_mrv2_cudagraph_stream.py \
  ../_site_v0281/vllm-plugin-das/tests/runtime_patch/test_qwen4_exp_sp.py \
  ../_site_v0281/vllm-plugin-das/tests/runtime_patch/test_hcu_model_runner_cudagraph_contract.py -q
```

Each `runs/<case>/` directory contains `launch.json`, `serve.log`,
`test_acc.log`, and the evalscope `outputs/` directory with predictions,
execution reviews, reports and the actual task configuration.
`branch_before.json` and `branch_after.json` record the branch, base commit,
fix hashes and original test script hashes. `source_diff.patch` preserves
the production and regression-test changes used for evaluation.

This is a regression on the first 32 HumanEval tasks, not a full 164-task
HumanEval score. Passing these tasks does not establish token-for-token
equivalence or rule out differences on other inputs or datasets.
