# vLLM source patches

vLLM-HCU does not modify vLLM source at install or runtime. These patches are
applied to the installed vLLM package separately when a model needs vLLM core
changes that are not yet available from the plugin.

The patches form one ordered series against the vLLM `0.28.1` package tree
(source commit `2d1f703`). Apply them in numeric order; each patch depends on
the previous ones. Applying `0001`–`0024` reproduces source commit `d0ff4a9`.

| Patch | Source commit | Purpose |
| --- | --- | --- |
| `0001` | `00afa28` | Step5 model support (`Step5ForCausalLM`, `Step5ForConditionalGeneration`, Step5 configs, Step4 DSA backend, `VLLM_STEP5_*` env vars) |
| `0002` | `751f31a` | Fix a Step5 inference crash |
| `0003` | `2f911c4` | HCU fallbacks for DeepGEMM functions |
| `0004` | `1bed64d` | Step5 MTP3 |
| `0005` | `76227f2` | Remove the pipeline bubble introduced by Step5 MTP3 |
| `0006` | `637c774` | Remove the bubble at the start of each Step5 decode step |
| `0007` | `eee48e6` | Step5 router: Triton router-bias top-k kernel on HCU |
| `0008` | `bbb28a8` | Step5: hand the shared expert to FusedMoE outside sequence parallel |
| `0009` | `809979e` | Step4 DSA: per-request spec indexer kernel; region top-k `num_warps=8` |
| `0010` | `68bf68d` | Step4 DSA: fuse spec summaries, round q in norm_rope, cache decode rows meta per step |
| `0011` | `77fad47` | graph_capture: register the EP group's custom all-reduce graph buffers |
| `0012` | `8616a6a` | Compile cache: hash `VLLM_HCU_*` env vars |
| `0013` | `a94334e` | Naive DP/EP finalize: optional platform combine hook |
| `0014` | `6a492a2` | EPLB: fix Step5 EPLB; add touched-load and dump options |
| `0015` | `b8ac9c5` | Step5: `qkv_proj` and `g_proj` as one GEMM (`VLLM_HCU_STEP5_FUSE_QKVG`, default on) |
| `0016` | `3ce36c9` | Step5: bf16 × bf16 → fp32 Triton router GEMM (`VLLM_HCU_STEP5_BF16_ROUTER`, default on) |
| `0017` | `0d01a7e` | fused_moe: finalize returns the combined tensor instead of copying into output |
| `0018` | `326dcb3` | Step5: fuse router GEMM + shared-expert gate_up + SwigluStep into one split-K kernel |
| `0019` | `c103a37` | Step5: optional fp8 for the bf16 dense linears and the layer 88–90 MoE |
| `0020` | `3388fae` | Step4 DSA: faster verify-batch indexer and top-k selection |
| `0021` | `fd4e9db` | Step4 DSA: sort-free decode pack |
| `0022` | `02569a0` | logits_processor: skinny Triton GEMM for the bf16 lm_head at M ≤ 64 |
| `0023` | `daa2b87` | Step4 DSA: one GEMM for the four indexer projections |
| `0024` | `d0ff4a9` | Step5: fused SWA q/k norm + RoPE + cache write and gate/fp8-quant epilogue |

Paths in the patches are relative to the `vllm` package directory. Apply the
series with:

```bash
VLLM_DIR=$(python -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
for p in patches/vllm/0*.patch; do
    patch -p1 -d "$VLLM_DIR" < "$p" || break
done
```

Several of these patches (for example `0011`, `0013`, `0017`) add hooks that
the matching `vllm_hcu` changes rely on, such as the EP fused-combine finalize
path. Keep the applied patch level in step with the `vllm_hcu` revision.
