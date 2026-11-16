# RMSNorm policy and AOT cache isolation

Changing `VLLM_HCU_USE_CUSTOM_OPS` and `VLLM_HCU_USE_CUSTOM_RMS_NORM`
could reuse an AOT model compiled with the opposite RMSNorm policy. In a
shared cache-root Qwen2-0.5B-Instruct run, both the ON and OFF model loaded
the same AOT artifact and exported identical graphs containing 98 LightOp
RMSNorm calls. Different first-level compilation directories did not prevent
the collision: their model config hashes were both `204e6517a5`.

The extension now registers `VLLM_HCU_DAS_RMS_NORM_ENABLED` as a derived
getter in the target's environment registry before model construction.
vLLM 0.32's existing `compile_factors()` includes that effective boolean in
both its normal and AOT keys. Only master AND child enables RMSNorm;
equivalent disabled combinations share a factor. Another owner of the same
factor, or a missing registry API, prevents activation explicitly.

This is a narrow integration with the 0.32 environment registry, not a
replacement of either cache implementation. Revisit it when upstream or
the Adapter extension protocol provides a dedicated compilation-policy
registration hook; preserve effective-policy equivalence and test both cache
layers when replacing this integration.

## Validation on gfx936, 2026-10-31

Target OpenDAS vLLM source: `a198423b7be0ccf9c9ac3fb01c02ac03ab145e93`.
Adapter candidate: `35f8f0569df4ffbed045bf0dc18974eaafd5ab1a`.
Python 3.12, Torch 2.11, DTK 26.10, default FULL_AND_PIECEWISE Graph.

- 30 focused environment/activation tests passed. The regression test calls
  the installed target's actual `aot_compile_hash_factors()` and checks
  ON/OFF separation, equivalent disabled combinations, and boolean aliases.
- The complete portable suite passed 141 tests.
- Both live RMSNorm numerical/fallback and compile/replay tests passed.
- With the same checkpoint and the original shared cache root, fixed OFF
  generated zero LightOp RMSNorm calls; fixed ON generated 98 calls.
  Their first-level hashes were `f67cf72426` and `e3622cf486`.
- Their AOT artifacts were also distinct:
  OFF `8955736e75e69892c20911106ee8d9807f14e75277ea86e6b29ad8a5a4bd33db`,
  ON `98a57b76e9306295198a89ffb04a7b4fc1c6fcd073d027e0470939c9423e5864`.

Run OFF and ON against a common `VLLM_CACHE_ROOT`, with only the two RMS
switches changed. Inspect both `cache_key_factors.json` and
`computation_graph.py`, together with the AOT paths in each server log.
Successful capture and different first-level directories alone are
insufficient. Keep the checkpoint, device, sampling, token budget and
HumanEval task set fixed for the independent accuracy comparison.

The original shared-cache OFF evaluation is invalid as an OFF control.
Its samples and graph evidence are retained in the task's evidence bundle;
they must not be relabelled as a valid official-path result.

The independent corrected-cache HumanEval/0–31 run scored raw 0/32 and
format-normalized 16/32 for OFF, versus raw 0/32 and normalized 15/32 for ON.
Earlier ON repetitions were 15/32 and 16/32. These scores do not close the
RMSNorm kernel's accuracy gate; this patch fixes policy isolation and does
not establish a performance or model-accuracy improvement.
