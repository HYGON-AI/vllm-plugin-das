# Hy4 PP2 + PCP4 Design

## Goal

Enable the eight-HCU `HYV4ForCausalLM` topology `PP=2, TP=1, PCP=4,
DP=1, DCP=1, EP=4` in Model Runner V2 eager mode. Validate the target path
with DeepEP high-throughput, DeepGEMM, and FP8 E4M3 KV cache.

## Scope

- Permit pipeline parallelism with PCP only for the exact Hy4 topology above.
- Keep GLM-5.2, GQA, and every other PP+PCP topology fail-closed.
- Keep PP+PCP speculative decoding disabled; MTP is a separate validation.
- Preserve the existing PCP=1, PP=1, and DeepEP communicator behavior.
- Do not introduce a PP-boundary all-gather. Each PP stage keeps the same
  PCP-local virtual batch during model execution.

## Runtime ownership

vLLM constructs PP groups between equal PCP ranks, while each PP stage owns a
separate PCP group. Consequently, the first stage sends PCP-local intermediate
tensors to the matching PCP rank in the second stage. Only the final PP stage
has final hidden states and gathers those states across its PCP group before
sampling.

All PP ranks still enter `sample_tokens`. A non-final PP stage has
`execute_model_state.hidden_states is None`, but upstream PP receive/broadcast
uses the request layout passed in `InputBatch`. It must therefore restore the
saved global batch without attempting a hidden-state collective. The final
stage restores both hidden states and the same global batch.

`HcuPCPManager` will expose a checked accessor for the saved global sampling
batch. `HcuGPUModelRunnerV2.sample_tokens` will select batch-only restoration
when hidden states are absent and the existing hidden-state restoration when
they are present.

## Configuration contract

The existing generic PP rejection becomes a model-aware exception. PP+PCP is
accepted only when all of these values match:

- architecture: `HYV4ForCausalLM`
- pipeline parallel size: `2`
- tensor parallel size: `1`
- prefill context parallel size: `4`
- data parallel size: `1`
- decode context parallel size: `1`
- expert parallelism: enabled
- eager execution: enabled
- speculative configuration: absent

Existing checks continue to reject LoRA, multimodal execution, KV offload,
P/D disaggregation, lightly-CP, and HCU multi-layer MTP.

## Validation

CPU regression tests must prove that the exact Hy4 topology is accepted, that
nearby Hy4 topologies and Hy4 PP+PCP+MTP are rejected, and that GLM/GQA PP+PCP
remain rejected. A runner regression test must model a non-final PP rank with
`hidden_states=None`, prove there is no PCP hidden-state gather, restore the
global batch, and delegate to upstream sampling.

Live validation uses `/models/Hy4-preview-Channel-FP8-w8a8-v2` with:

```bash
--pipeline-parallel-size 2
--tensor-parallel-size 1
--prefill-context-parallel-size 4
--enable-expert-parallel
--all2all-backend deepep_high_throughput
--moe-backend deep_gemm
--kv-cache-dtype fp8_e4m3
--enforce-eager
```

Acceptance requires all eight workers to initialize, logs to show two PP
stages with four PCP/EP ranks per stage, a successful short request and long
prefill request, and exact HumanEval-32 correctness counts.
