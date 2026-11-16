# PR #239: KDA warmup logical input width

The constructor warmup inferred its input width from `weight.shape[-1]`.
With `VLLM_USE_NN=1`, unquantized weights are stored as
`[input_width, local_output_width]`, so the scratch tensor used the output
width. A non-square projection then failed during model construction, before
checkpoint loading. For input width 4 and local output width 12, the failure
was `mat1 and mat2 shapes cannot be multiplied (6288x12 and 4x12)`.
`VLLM_USE_NN` also defaults to true at runtime; an explicit setting of 0
avoided the failing branch but did not fix it.

The warmup now takes `linear.input_size`, which is independent of weight
storage layout. Its diagnostic output uses `output_partition_sizes` to report
the local fused output width correctly in either layout.

## Validation

Environment: `xiesl_kimi3_plugin_028`, installed vLLM
`0.28.1+das.77acaf6.dtk2604`, current plugin checkout. Devices are hidden;
no service is started and no dependency is installed or upgraded.

From the plugin checkout in that container:

```bash
VLLM_PLUGINS=__disabled__ CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
python -m pytest \
  tests/runtime_patch/test_kimi_k3_reference_adapters.py \
  tests/runtime_patch/test_gate_linear_nn_layout.py \
  tests/runtime_patch/test_kimi_k3_slimquant_ht.py \
  -q --disable-warnings
```

Result: **59 passed**, 12.31 seconds. The new warmup cases cover NN on/off and
an unset environment variable, non-square BF16 weights, projection and
synchronization failures with successful retry, and one-time warmup behavior.
The existing CPU-weight skip and gate-layout checks also pass.

The device branch is reached with explicit device/TP metadata and scratch
allocation/synchronization doubles. Tests execute the production HCU
`UnquantizedLinearMethod.create_weights` and `apply` methods with real CPU
matmuls; the non-NN GEMM dispatcher selects a CPU `functional.linear` boundary.
The HCU linear source is imported with an isolated layer registration table
to avoid collisions with native classes already imported by pytest.
Before the fix, the NN case reproduces the matrix-dimension error above.

Production boundary: 401 Python files clean; `git diff --check` passes.
Host logs: `task_dir/debug_028/kda_nn_warmup_before.log` and
`task_dir/debug_028/kda_nn_warmup_regression.log`.

These tests verify constructor control flow and projection dimensions. They
do not validate an actual HCU warmup, full-model NN startup, accuracy, or
performance.
