# HYV4 MTP and boltops iHC Design

## Goal

Run `/models/Hy4-preview-FP8-Testing` on the HCU vLLM v0.25.1 plugin with
native HYV4 MTP (`method=mtp`, three speculative tokens, TP8) while replacing
the eager HYV4 backbone iHC boundaries with the installed `boltops.ihc`
kernels on HIP/CUDA tensors.

## Constraints

- Keep the checkpoint's ModelOpt MXFP8 block-wise representation and the
  existing Triton emulation path required by gfx938.
- Keep the current split `q_a_proj` and `kv_a_proj_with_mqa` layout; the HCU
  implementation must not adopt the newer upstream fused Q/KV projection.
- The MTP checkpoint has one `model.mtp_layers.0` block and no MTP iHC
  pre/post parameters. The MTP decoder block therefore runs with iHC disabled,
  while the target backbone continues to use iHC.
- TP8 and expert parallel disabled are the acceptance topology.
- Existing uncommitted user changes in `/models/zb/hy4` must be preserved.

## boltops iHC adapter

`vllm_hcu.models.hy_v4.hc` imports `ihc_pre`, `ihc_post`, and `ihc_head` from
`boltops.ihc`. A small device gate selects the fused functions for CUDA/HIP
tensors and retains the mathematically equivalent eager implementation for CPU
tests or an installation without boltops. Kernel exceptions are not swallowed:
once a supported-device fused call is selected, an operator failure is allowed
to fail startup or inference rather than silently changing execution.

The calls use the existing unquantized FP32 parameters directly:

- `ihc_pre(residual, hc_fn.weight, hc_scale, hc_base, rms_eps, hc_eps,
  magnitude)` returns the BF16 layer input and FP32 post gates.
- `ihc_post(x, residual, post)` returns the BF16 multi-channel residual.
- `ihc_head(residual, hc_head_fn.weight, hc_head_scale, hc_head_base,
  rms_eps, hc_eps)` returns the dense BF16 hidden state.

No parameter names or checkpoint mappings change.

## v0.25.1 speculative configuration

The installed vLLM does not include HYV4 in `SpeculativeConfig`'s MTP model
types. A platform core adapter augments the runtime MTP type literal with
`hy_v4_mtp` and wraps `SpeculativeConfig.hf_config_override`. For a target
`model_type == "hy_v4"`, the wrapper changes the draft config to:

```python
{
    "model_type": "hy_v4_mtp",
    "architectures": ["HYV4MTPModel"],
    "n_predict": config.num_nextn_predict_layers,
}
```

All other model types are delegated unchanged to the original vLLM method.
The adapter is installed through the existing exact-import coordinator and is
idempotent.

## HYV4 draft model

`vllm_hcu.models.hy_v4.mtp` is a dedicated port of PR #54160 rather than a
subclass of HYV3 or DeepSeek V4 MTP. It owns:

- the shared embedding and shared target LM head;
- `enorm`, `hnorm`, `eh_proj`, one HYV4 decoder block, and final RMSNorm;
- a sparse indexer whose top-k buffer is replaced with the target model's
  buffer by the Eagle/MTP loader;
- the vLLM sampler and standard single-tensor MTP forward contract.

The copied draft config extends per-layer type lists to index 78, forces a
sparse MoE MTP layer, and sets `enable_ihc=False`. The target hidden state fed
to MTP is the dense post-iHC-head hidden state already returned by
`HYV4ForCausalLM`.

## Quantization and weight loading

The MTP quant config inherits the target ModelOpt MXFP8 config when
`mtp_quant_algo` is absent. The config is shallow-copied and checkpoint-side
exclusions such as `model.mtp_layers.0.eh_proj` are remapped to
`model.layers.78.eh_proj`, preserving wildcard suffixes. `lm_head` keeps the
same quant prefix as the target so its exclusion remains effective.

Weight loading rewrites `model.mtp_layers.0.*` to `model.layers.78.*` and
inserts `.mtp_block` for decoder weights. It reuses the validated target
helpers for MXFP8 indexer dequantization, TP sink slicing, router-gate loading,
and exact checkpoint-name normalization. Fused routed-expert weights and their
`*_scale` tensors are resolved independently so scales cannot be loaded into
weight parameters. Runtime-only KV cache scale parameters may remain
unassigned; all true MTP parameters must receive checkpoint values.

## Validation

CPU tests cover fused iHC dispatch and eager fallback, speculative-config
rewriting, registry wiring, MTP config construction, exclusion remapping,
weight-name rewriting, fused expert scale resolution, sink slicing, and top-k
buffer propagation. Integration uses TP8, Triton MoE, eager mode, MXFP8
runtime emulation, and three speculative tokens. Acceptance requires startup,
successful no-thinking generation, concurrent request correctness, clean logs,
and a HumanEval subset result compared with the existing 4/5 baseline.
