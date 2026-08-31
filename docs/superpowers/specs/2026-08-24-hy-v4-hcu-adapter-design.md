# HY V4 HCU Adapter Design

## Goal

Add accuracy-first HY V4 inference support to the `v0.25.1` branch of
`HYGON-AI/vllm-plugin-das`. The first delivery must load the checkpoint at
`/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2` without silently dropping
model semantics and must complete ordinary prefill and decode on eight BW1100
devices.

The first delivery excludes MTP speculative decoding, reasoning parsing, and
tool-call parsing. It has no performance target beyond remaining usable for
functional and numerical validation.

## Baselines and Inputs

- Plugin baseline: `v0.25.1`, commit
  `f1f7a06489e7535bd63711027e876ea1d3301b23`.
- vLLM API reference: `/models/zb/vllm_025/vllm` on `v0.25.1`.
- HY V4 architecture delta: `/models/zb/Hy4-p_vLLM`.
- Validation checkpoint:
  `/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2`.
- Target hardware: eight BW1100 devices with approximately 144 GiB VRAM per
  device.
- Initial parallel layout: tensor parallel size 8 and expert parallel size 8.
- Initial MoE backend: Triton.

The HY V4 delta is NVIDIA-only. It is an architecture reference, not code that
can be copied without platform review. The adapter must not import HY V4 code
from the external delta at runtime.

## Chosen Approach

Implement HY V4 as a native `vllm_hcu` model while preserving HY-specific
architecture and weight semantics. Reuse the plugin's existing HCU
compressed-tensors, sparse-MLA, indexer, attention-sink, and Triton MoE
infrastructure. Add narrowly scoped Triton code only when an existing HCU path
is absent or fails numerical validation.

This is preferred over adapting the HCU DeepSeek V4 model because the HY V4
projection layout, independent Hyper-Connections, routing behavior, and
checkpoint names differ. It is preferred over overlaying vLLM core because the
deliverable must remain an independently installable plugin.

## Components

### Configuration and registration

Add a plugin-owned `HYV4Config` with `model_type = "hy_v4"`. Register it early
enough for vLLM configuration loading to resolve the local checkpoint without
`trust_remote_code`. Registration must be idempotent and must use the plugin's
existing lifecycle and failure-latching conventions.

Register only this architecture in the first delivery:

```text
HYV4ForCausalLM -> vllm_hcu.models.hy_v4:HYV4ForCausalLM
```

Do not register `HYV4MTPModel` until MTP is implemented and validated.

### Backbone and iHC

Port the HY V4 decoder, embedding, normalization, language-model head, and
pipeline-parallel contracts into `vllm_hcu.models.hy_v4`. Preserve both the
ordinary residual path and the checkpoint's enabled independent
Hyper-Connection path.

The iHC pre, post, and head computations must retain the reference tensor
shapes, gate ordering, FP32-sensitive calculations, and parameter names. Use
supported PyTorch/HCU operations first. A Triton replacement is allowed only
after a reference comparison demonstrates that the existing path is
unsupported or numerically incorrect.

### Sparse MLA, indexer, gating, and sink

Preserve the checkpoint's HY-specific MLA projections, elementwise output
gate, lightning indexer, per-layer `full`/`shared` indexer pattern, and
per-head learnable attention sink.

Reuse the plugin's HCU sparse-MLA and indexer infrastructure. The adapter must
meet all of these requirements:

- Sparse layers never fall back to dense attention.
- Shared-indexer layers reuse the intended preceding full indexer results.
- Indexer query/key RoPE uses the checkpoint's interleaved layout.
- The local TP shard of the learnable sink is loaded as FP32.
- Both prefill and decode apply the learnable sink.
- A backend that cannot consume the sink is rejected rather than selected with
  the sink disabled.
- Top-k index buffers are sized and shared according to vLLM scheduler limits.

If the existing HCU wrapper does not expose the required sink argument, add an
HY V4-specific HCU backend adapter over the existing sink-capable sparse MLA
operations. Do not change global attention behavior for unrelated models.

### MoE and FP8 W8A8

Preserve sigmoid routing, FP32 router weights and logits, expert correction
bias, normalized top-k, routed scaling factor `2.827`, eight selected experts,
one shared expert, and the routed-expert SwiGLU clamp.

Use vLLM's `FusedMoE` abstraction and the plugin's compressed-tensors FP8 W8A8
integration. The first delivery accepts only the Triton MoE backend for this
checkpoint and emits an actionable error for another backend. It must not
introduce a second quantization framework or silently dequantize the complete
model to BF16.

### Weight loading

The loader must support the checkpoint's exact naming and layouts, including:

- merged gate/up projections;
- split or fused routed-expert projections and their scales;
- indexer `wk` plus `weights_proj` packing;
- TP-sharded attention sinks;
- expert correction bias;
- iHC projection names stored without a trailing `.weight` in the checkpoint;
- compressed-tensors ignored modules and scale parameters;
- pipeline-parallel missing layers.

Loading must return an exact loaded-parameter set. Unknown, missing, or
shape-incompatible required weights are errors. Intentional checkpoint-only
weights must be filtered by explicit, tested rules.

## Runtime Flow

```text
checkpoint config
  -> register HYV4Config
  -> resolve HYV4ForCausalLM through ModelRegistry
  -> create TP=8 / EP=8 parameter shards
  -> load compressed-tensors FP8 W8A8 weights
  -> iHC pre
  -> gated sparse MLA + lightning indexer + learnable sink
  -> iHC post
  -> Triton FP8 MoE + shared expert
  -> final iHC merge and normalization
  -> language-model head
  -> ordinary sampling
```

## Failure Semantics

The adapter fails closed in the following cases:

- `hy_v4` configuration or model registration is unavailable;
- a sparse MLA or indexer path cannot be selected;
- the selected attention backend cannot apply sinks in both prefill and
  decode;
- a non-Triton MoE backend is requested for this first delivery;
- a required checkpoint tensor is missing, unknown, or shape-incompatible;
- a TP or EP layout violates the architecture's divisibility constraints;
- a kernel produces NaN or Inf during validation.

Errors must identify the failed component and the expected configuration. No
failure above may be converted into a warning followed by a semantically
different execution path.

## Accuracy-First Validation

Implementation follows test-driven development. Every behavior change begins
with a failing test, followed by the minimum implementation needed to pass it.

Validation proceeds in increasing cost order:

1. Test idempotent config and model registration without loading the real
   checkpoint.
2. Compare iHC pre, post, and head outputs with a small PyTorch reference.
3. Compare router top-k IDs, weights, shared-expert contribution, and SwiGLU
   clamping with a small reference implementation.
4. Verify the attention backend applies the same sink formula in prefill and
   decode and rejects a sink-incapable backend.
5. Verify full/shared indexer layer selection and top-k buffer reuse.
6. Construct a small synthetic HY V4 model and prove every expected parameter
   is loaded exactly once across dense, MoE, iHC, indexer, gate, and sink
   variants.
7. Compare every new Triton kernel with its PyTorch reference using multiple
   shapes, boundary sizes, and representative BF16/FP8 inputs. Define
   tolerances per operation from the reference dtype and reduction behavior;
   do not loosen tolerances merely to make a failing kernel pass.
8. Start the real checkpoint on eight BW1100 devices with TP=8, EP=8, Triton
   MoE, and eager execution for initial diagnosis. Confirm there are no
   required missing or unexpected weights.
9. Run a deterministic short prompt through prefill and at least one decode
   step. Assert non-empty output and no NaN or Inf in sampled logits.
10. Record the exact command, environment, logs, numerical comparisons, and
    remaining limitations.

When a trusted reference output for the full checkpoint is available, compare
token IDs and logits in addition to structural and kernel-level checks. The
absence of such an output does not permit skipping component-level numerical
comparisons.

## Deliverables

- Plugin-owned HY V4 configuration and model implementation.
- HCU-specific attention adapter or Triton kernels required by HY V4.
- Unit and integration tests for registration, numerical behavior, weight
  loading, and fail-closed policies.
- An eight-device launch command and validation record for the supplied
  checkpoint.
- Documented first-delivery limitations: no MTP, reasoning parser, tool parser,
  or performance guarantee.
