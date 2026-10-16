# PR #239: SiTU fallback intermediate precision

FP16 inputs could lose nonzero outputs in the HCU SiTU eager fallback. For
`beta=1`, `linear_beta=None`, and input `[[-18,8192]]`, the intermediate
`sigmoid(-18)` rounded to FP16 zero before multiplication by the up projection.
The old fallback returned `-0.0`; the native reference returns the representable
FP16 value `-0.00012481212615966797`.

The fallback now converts gate and up to FP32, performs SiTU and optional up
clipping in FP32, then casts the final result to the input dtype. This follows
`vllm.model_executor.layers.activation.SituAndMul.forward_native`.

## Validation

Environment: `xiesl_kimi3_plugin_028`, installed vLLM
`0.28.1+das.77acaf6.dtk2604`, plugin source checkout. Devices are hidden for
these CPU checks; no service is started and no dependency is changed.

From the plugin checkout in the container:

```bash
VLLM_PLUGINS=__disabled__ CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
python -m pytest \
  tests/runtime_patch/test_kimi_k3_reference_adapters.py \
  tests/runtime_patch/test_kimi_k3_slimquant_ht.py \
  -q --disable-warnings
```

- Before the fix, the new FP16 underflow regression fails with absolute error
  `0.00012481212615966797` and relative error 1.
- After the fix: **41 passed**, 10.89 seconds. New numerical cases compare
  exactly against the native method for FP16/BF16/FP32, three beta/clipping
  settings, and contiguous/noncontiguous inputs. The existing custom-op switch
  and adapter tests also pass.
- A CPU probe extracts the native SiTU method from the repository-declared
  OpenDAS baseline `462fdb097c66b487ef4826e8009431c10fe88fb8`. Its AST matches
  the installed native method. Calling that baseline method reproduces the
  reference value above, and the fixed fallback matches exactly. Other runtime
  dependencies remain those of the installed environment; this is not a full
  installation of the declared baseline.
- Production boundary: 401 Python files clean. `git diff --check` passes.

Host artifacts under `task_dir/debug_028/`: `situ_fp16_before.log`,
`situ_fp16_regression.log`, `situ_fp16_baseline_probe.py`, and
`situ_fp16_baseline_probe.log`.

These checks cover fallback arithmetic and dispatch. They do not measure HCU
kernel performance or full-model accuracy.
