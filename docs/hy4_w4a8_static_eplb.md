# HYV4 compressed-tensors W4A8 static EPLB (experimental)

This adds direct loading of a previously generated expert placement for HYV4
compressed-tensors channel W4A8. It is a draft integration, not a validated
end-to-end performance improvement. Dynamic expert transfers and redundant
experts remain unsupported by this quantization method.

## Supported contract and core dependency

The tested native MTP configuration is PP1 / TP1 / PCP16 / DP1 / DCP1 / EP16,
DeepEP high-throughput, MoE backend `deep_gemm`, eager execution,
and the existing native one-layer MTP3 / FP8 E4M3 KV-cache contract. Other
existing HYV4 PCP/MTP guards still apply. There are 256 logical and 256 physical
experts, with 16 physical experts per rank and zero redundant experts.

The matching vLLM core must count PCP in the EPLB parallel-world eligibility
check. The plugin cannot override that constructor check. The companion
patch is provided in `patches/vllm-pcp-eplb-validation.patch`; it is **not
applied automatically** by installing this plugin. In a matching vLLM source
checkout, first inspect and apply it:

```bash
git -C "$VLLM_SOURCE" apply --check \
  "$PLUGIN_SOURCE/patches/vllm-pcp-eplb-validation.patch"
git -C "$VLLM_SOURCE" apply \
  "$PLUGIN_SOURCE/patches/vllm-pcp-eplb-validation.patch"
```

Build/install that modified vLLM checkout using the deployment's existing
procedure. The tested original `vllm/config/parallel.py` SHA256 was
`2bb0c2d54a0c3211c5ea1ddeed731d0e2a5d95fb58972f92845268959f0c818e`;
the patched file was
`3a879a72ba53bff5f758805a7018c7f7239117e0b25f743d8c22584344a9845f`.
Other core revisions need their own compatibility review. The companion
patch retains the platform, expert-parallel and true-single-rank checks.

Supply a version-2 static expert-map file covering both `HYV4ForCausalLM`
(77 MoE rows) and `HYV4MTP` (one row). Each row must be a permutation of the
256 logical experts in physical-slot order. Append the following to the
existing matching PCP16/native-MTP3 server command:

```bash
--enable-eplb \
--eplb-config '{"expert_map_path":"/absolute/path/offline-map.json","num_redundant_experts":0,"static_dispatch_policy":"nearest"}'
```

Use real expert routing, not a routing simulator. The experiment used P
context 88,000, batch-token budget 65,536 and framework-default request/workspace
limits; it did not explicitly set `max-num-seqs` or workspace factor.
These workload settings are not new defaults in this patch.

## Implementation and tests

The quantization method advertises EPLB support only for the guarded static
configuration, snapshotting that capability while the vLLM config is active.
Composed checkpoint loaders remain actual bound methods so the existing
static binder can remap logical expert IDs before packed INT4 conversion and
FP32 scale conversion. The conversion math is unchanged.

After applying the core patch, run in the matching vLLM/HCU environment:

```bash
python -m pytest -q \
  tests/models/hy_v4/test_compressed_w4a8_static_eplb.py \
  tests/models/hy_v4/test_custom_w4a8.py \
  tests/model_executor/layers/fused_moe/test_static_eplb.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/runtime_patch/test_offline_eplb.py \
  tests/runtime_patch/test_hyv4_pcp_static_eplb_config.py \
  tests/runtime_patch/test_glm52_pcp_config.py --confcutdir tests
```

The positive `ParallelConfig` integration test intentionally requires the
companion core change; the plugin change alone is not sufficient.

## Native observations and remaining validation

The map was generated with the pinned upstream `DefaultEplbPolicy` from 24
real-routing requests; eight separate requests were held out. The loaded-map
SHA256 was `87d2d922963d86f6a76be1e579ee4b2ea6b01b7bd26a29ea4640f20cc500f62c`.
Native balance replay used the same eight 85K-token inputs, a primed 79,050-token
prefix, eight output tokens and concurrency four. All 16 source ranks verified
their main/MTP row bindings, live inverse maps and physical-ID histograms.

| Routing-count metric | Identity placement | Offline placement |
| --- | ---: | ---: |
| Main weighted per-layer maximum / mean | 2.407442 | 1.444667 |
| Main worst-layer maximum / mean | 5.008696 | 2.008979 |
| Main mean per-layer CV | 0.538444 | 0.133304 |
| Main cumulative rank CV | 0.067137 | 0.122083 |
| MTP maximum / mean | 4.155744 | 1.952787 |

The weighted ratio is the sum of layer maxima divided by the sum of layer
rank means. Counts include executed PCP/MTP padding. They do not measure GPU
utilization, padded GEMM FLOPs or speedup. Capture overhead is not benchmark
time. Cumulative balance worsened because heavy groups concentrate on the same
rank across layers. A CPU-only rank-rotation proposal is outside this change.

Post-load comparisons covered 19,968 model-layer/logical-expert records:
39,936 small tensors, including channel scales, were checked in full; 79,872
larger registered/cached weight tensors were checked at 256 uniform byte
positions each. No mismatch was found, but unsampled bytes are not proven equal.

Real-routing, real-MTP3 HumanEval/0..31 passed 32/32 in both configurations.
The no-EPLB control matched all 32 frozen baseline texts; the reordered map
matched only 22/32. Two identity-map next-token probes reproduced all 11
top-20 distributions exactly; two reordered-map probes also reproduced each
other, while identity versus reordered differed in top-1 at 3/11 fixed
contexts. These diagnostics are selected divergence contexts, not an accuracy
benchmark. Placement-dependent local BF16 partial-output rounding is plausible
and demonstrated in a small native LightOp example, but is not established as
the sole full-model cause.

The native deployment also contained previously tested frontend, scheduling
and MQA/tokenizer settings. This change has **no isolated client TTFT result**,
does not inherit the earlier deployment-combination speedup, and remains draft
pending numerical-risk evaluation, core integration and matched real-routing
performance measurement. The 32-case functional result does not establish
general model-quality equivalence or production readiness.
