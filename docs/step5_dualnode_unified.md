# Step5 dual-node unified source

Build and install **one vllm-hcu wheel from this branch on both nodes**. The
same installation serves Native Prefill (TP1/DP16/EP16) and DeepEP/DeepGEMM
Decode (TP4/DP4/EP16). Switch profiles after stopping the current service.
No plugin source overlays, runtime loaders or edits to installed boltops are
required. Model weights, tokenizer, expert placement and tuning data remain
external deployment assets.

The Native profile keeps its FP32 Gemma norm / Torch quantization producer,
EP64 scratch and bounded DBO output pool. The Decode profile selects the
library norms, dynamic quantization and fused decode producers. The two-GiB
sorted-activation addressing fix is now part of the plugin's native EP launch;
the underlying vendor kernels remain unchanged.

## Build and launch

Use the matching Python 3.12 / Torch 2.11 / DTK 26.10 / gfx938 environment:

```bash
python -m pip wheel --no-build-isolation --no-deps -w dist .
python -m pip install --no-deps --force-reinstall dist/vllm_hcu-*.whl
cp scripts/step5_dualnode/config.env.example scripts/step5_dualnode/config.env
```

Fill the two node addresses, interfaces, model/tokenizer paths, asset root and
local cache directory. Use the same wheel and asset contents on both nodes.
Prefill requires the qualified hipBLASLt build matched by its Dense recipe.
This PR uses AITER `hipb_mm` for the tuned Dense plans, including a checksum
of the loaded hipBLASLt library. It does not require the old Step5 Dense C++
extension. Validate the merged PR revision rather than reusing wheel results
from the earlier dual-node branch.
Decode requires DeepEP's layered LL/rail mapping ABI and the HCU DeepGEMM
channel-FP8 build. The earlier Decode measurements used source `7da8cd9` with
code-object V5. The WASP128 profile below uses the pinned nightly API; changing
DeepGEMM requires revalidating Decode as well as Prefill.
For containers, map `/dev/kfd`, `/dev/dri` **and `/dev/infiniband`**, use host
networking, adequate shared memory and unlimited memlock.

Required external asset layout:

```text
assets/
  expert_placement_decode.pt
  prefill/
    dense-qualified.json
    step5_ep16_moe_table.json
    expert_placement_prefill_8192_ep16_20261020.pt
    expert_placement_prefill_e23_static_tokenhash_sz_20261029.json
```

Run each command on both nodes, with `NODE_RANK=0` on the head and `1` on its
peer. Append `--dry-run` to inspect the command without loading a model:

```bash
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh prefill
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh prefill-wasp128
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh decode
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh decode-perf
```

`decode` uses standard MTP3 for quality testing. `decode-perf` uses synthetic
acceptance `[0.838992,0.598322,0.362686]` for the historical performance workload.
Prefill uses no MTP and explicit synchronous scheduling. No profile creates a
96-GiB physical capacity reservation. Prefill retains its historical fixed
4.75-GiB KV budget; compilation cache export is disabled to avoid filling the
system disk. These settings do not change the shared installed source.

Historical results from separate releases are provenance, not validation of
this unified wheel. Record the wheel SHA256 and commit for both profiles,
then check performance and fixed100 GSM8K independently with the corresponding
speculation settings.

The launcher enables the read-only `step5_source_identity` worker RPC. Query
`/collective_rpc` with `{"method":"step5_source_identity","timeout":30}` and
compare installed versions, native-extension hashes and critical source hashes
on all workers for both profiles. This detects stale wheels or source overlays.

## WASP128 with the public nightly wheel

The EP16 integration follows the single-node EP8 implementation in
[PR #504](https://github.com/HYGON-AI/vllm-plugin-das/pull/504), reviewed at
`f4a68aec67965e6cd31a53578109be0a013e91d1`. In the existing Python 3.12 / DTK
26.10 / Torch 2.11 environment, install the same dependency on both machines:

```bash
python -m pip install --no-deps --force-reinstall \
  'deepgemm==2.1.0+dtk2610.torch2110.2611231831.gebe489' \
  --index-url https://pypi.sourcefind.cn/nightly/dtk/
```

The cp312 x86_64 wheel SHA256 is
`e631d987ea05ed10874154fdc4ad718c6ca8fb0f28f9a957e83725bbe507d23c`.
This wheel exports `m_grouped_fp8_gemm_nt_contiguous_wasp_pair32`, rather than
the old private `m_grouped_fp8_gemm_nt_contiguous_step5_ep8` convenience API.
Its ordinary gfx938 Marlin packer selects Pair32 V2, so this profile explicitly
calls `pack_fp8_weight_pair32` for WASP128. The layouts are not interchangeable.

`prefill-wasp128` keeps TP1/DP16/EP16 and the external physical expert map.
It switches to cross-node DeepEP HT/MNVL, separate DBO communication buffers,
fused scatter and 128-row expert slices. Both G1 `[M,4096] x [E,3072,4096]`
and G2 `[M,1536] x [E,4096,1536]` use nightly `MODE=1002` (BM128/BN256).
Both 22 local experts (352 global) and 23 (368 including EPLB replicas) are
supported. Explicit profile guards prevent using this packing in Native
Prefill, TP4 Decode, unrelated models or master-disabled custom-op runs.

Do not copy EP8's `NCCL_IB_DISABLE=1`, intra-node-only flag, local EP8 map or
DP8 startup arguments to a two-node run. The common launcher retains RDMA,
the configured per-node GPU order, DP16 start ranks and the MNVL path.
`prefill` remains the historical Native backend; `prefill-wasp128` is a
different workload configuration and needs its own performance/quality run.
Neither single-node benchmark results nor historical two-node Native results
are validation of end-to-end EP16 WASP128 throughput or model accuracy.
