# Official clamp SwiGLU ownership

The legacy `vllm_hcu/patch/worker/op_opt/patch_activation.py` replaced
`SiluAndMulWithClamp.__init__` to force enablement and resolve the ROCm kernel.
The v0.33 production project is `adapter_extension`; the legacy root package
is retained as reference and is not part of that wheel.

At Core `c440c089fe9621c5b03e1618cf86a13e95de8bf1`, the real constructor
already resolves `_C.silu_and_mul_with_clamp` on CUDA-like platforms. Its
`forward_hip` uses that kernel for `alpha=1, beta=0` and delegates other
parameters to official Torch math to retain the ROCm precision guard.
`CustomOp` owns enablement, including its native fallback. Carrying the old
constructor replacement would bypass that enablement policy. No new Plugin
OOT, Adapter bridge, kernel or Core patch is needed for this dispatch.

The old patch also tolerated an absent native export. These tests use a build
that provides the real `_C` operator; they do not prove that missing-export
fallback. Keep that subcase separate in the migration ledger rather than
retiring the entire composite patch from the successful dispatch result.

Keep the upstream eager matrix in
`tests/kernels/core/test_activation.py::test_silu_and_mul_with_clamp`.
The additional Plugin test
`tests/vllm033/test_official_clamp_swiglu_hcu.py` exercises the actual installed
class and public configuration constructor, rejects a legacy replacement or
foreign OOT owner, and observes real dispatcher calls without replacing the
operator. It covers FP16/BF16, default/alpha/beta variants, empty/2D/3D inputs,
an independent CPU FP64 oracle, unchanged inputs, and three states replayed at
the same input address. Graph and eager outputs must match exactly.

Run it through the existing installed `device` tier, which selects its `hcu`
marker. The portable tier excludes this device file. Record the actual
Core/Adapter/Plugin sources, physical device and package RECORD checks with
each result; neither an older wheel nor another architecture borrows a pass.

These are operator/Graph ownership checks. They do not qualify shared-expert
GEMMs, routing, serialized checkpoints, model accuracy, HumanEval, performance
or model lifecycle. DeepSeek V4.1 checkpoint validation remains deferred.
