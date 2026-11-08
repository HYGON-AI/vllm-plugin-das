# HYV4-VL PD + offline EPLB — commands and JSON workflow

These are the actual image2/eager/no-MTP commands used on 2026-11-08. P is127, D is126, each16cards; model `/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8`. Both roles use FP8E4M3 KV, prefix caching OFF and MM processor cache0. P leaves max-num-seqs unspecified; D uses8/32. Max model length4096, image limit2, maxpixels1048576.

Runtime source: v0.25.1 `21065784` + MR338 `c758f415` + adaptation `65c2b496`. Reviewed local head `5ee59b2` additionally fixes disabled-image PCP1 dummy preparation; that compatibility fix has23 native CPU contract tests, but was not hot-patched into the measured image2 services. The image-enabled method order is unchanged. The source evidence and raw runs are retained in the original experiment archive. This branch depends on [MR338](https://github.com/HYGON-AI/vllm-plugin-das/pull/338), which was still open when these instructions were prepared.

The original launch operator also clears inherited CUDA_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES and proxy/observer variables, creates a fresh TMPDIR and records PID/SID/start ticks. Use a new run/output directory for another run; existing capture and client output files deliberately refuse overwrite. All calibration/diagnostic callbacks were stopped before timing. The worker extension/middleware below belongs to the external validation harness; it is idle unless its explicit loopback control endpoints are called.


## Deployment prerequisites and staging

This is the **actual tested deployment**, with the original container paths;
it is not a clean-container installation recipe. The existing vLLM core is
`752a3a5` with the earlier disabled observer adaptation. The `CORE`, `DEPS`,
FlashMLA/native extension, checkpoint, pinned dataset, frozen corpus,
provenance and original captures referenced below must already be present.
Dataset/scorer preparation is documented in [the MR338 guide](hy4_vl_serving.md).
Do not apply its TP8/V1 launch flags to this PCP16/MRV2 experiment.

Install/build the plugin using the matching DTK image and existing native
dependencies. In the original deployment the importable `vllm_hcu` tree and
`tools` were staged under `candidate-v1/package`. When using another checkout,
replace that PYTHONPATH component and the evaluator path with your installed
branch paths. The measured services used `65c2b496`; this published branch also
contains the CPU-tested image-limit-zero guard at `5ee59b2` and these docs.
It has not been rerun on HCU after the compatibility-only guard.

From the plugin checkout, stage the unchanged helpers on the corresponding
nodes (run only the relevant copy for each node):

```bash
mkdir -p /workspace/hy4_mm_20261108/probe
cp tools/hy_v4/vl_pd_validation/{build_vl_maps,vl_calibration_probe,vl_calibration_client_v2,vl_diagnostic_probe,vl_inspection_probe,vl_arrival_bench}.py /workspace/hy4_mm_20261108/probe/
# D node, to reproduce the original measured proxy path:
mkdir -p /workspace/hy4_mm_20261108/pd-r16-inspect-v1
cp tools/hy_v4/vl_pd_validation/{measured_proxy_ka1,mooncake_connector_proxy_ka1}.py /workspace/hy4_mm_20261108/pd-r16-inspect-v1/
```

The helpers' [source and SHA manifest](../tools/hy_v4/vl_pd_validation/README.md)
are committed. Dataset/corpus/provenance/captures and weights are external
inputs. Generation from the preserved captures below requires their original
referenced files. Changing a workload requires a new frozen corpus,
calibration and matching provenance; do not reuse its calibration blindly.
Use fresh output paths, and adjust the proxy wrapper's run directory if moved.

For R0 bootstrap, generate a VL identity map **before** starting P:

```bash
mkdir -p /workspace/hy4_mm_20261108/maps-bootstrap-NEW
python - <<'PY'
import json
from pathlib import Path
key = 'HYV4VLForConditionalGeneration'
mapping = dict(model_class=key,
    model_name='/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8',
    num_logical_experts=256, num_moe_layers=77,
    num_physical_experts=256, num_redundant_experts=0,
    physical_to_logical_map=[list(range(256)) for _ in range(77)])
with Path('/workspace/hy4_mm_20261108/maps-bootstrap-NEW/vl-identity-r0.json').open('x') as f:
    json.dump({'version': 2, 'model_maps': {key: mapping}}, f)
PY
```

For this bootstrap/R0 control, replace only the P launch's EPLB config with
the identity map above and `num_redundant_experts:0`, and set
`HY4_EXPERIMENTAL_STATIC_R16=0`. Keep the other tested settings. The full P
command below is the calibrated R16 configuration used after collection.

## P server (127 container)

```bash
#!/bin/bash
set -eu
unset VLLM_HCU_USE_TILED_KV_GATHER
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP
export PYTHONUNBUFFERED=1
export VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP=10.41.101.127
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_LOGGING_LEVEL=INFO
export NO_PROXY='*'
export no_proxy='*'
export VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8=1
export VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER=0
export NCCL_IB_GID_INDEX=3
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096
export GPU_MAX_HW_QUEUES=4
export LIGHTOP_MQA_LOGITS_BETA=1
export VLLM_HCU_HYV4_MOE_CHUNK_SIZE=1024
export MC_GID_INDEX=3
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
unset VLLM_MOE_ROUTING_SIMULATION_STRATEGY VLLM_RANDOMIZE_DP_DUMMY_INPUTS
export VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=4
unset VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_FACTOR
unset VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
export VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=1
export VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH=1
export VLLM_HCU_STATIC_HT_W4A8_COUNT_DIAGNOSTIC=0
unset VLLM_HCU_ENABLE_TRITON_CP_GATHER HY4_STAGE_PROBE_DIR VLLM_PLUGINS
export HY4_EXPERIMENTAL_STATIC_R16=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/workspace/hy4_mm_20261108/probe:/workspace/hy4_ttft_resume_20261101_0912/pd-eplb-20261103-0452eab/core:/workspace/hy4_mm_20261108/candidate-v1/package:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/installed:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/flashmla:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/deps
export HY4_BENCH_CASE=hy4-mm-20261108-pd-r16-inspect-v1-P
unset AMD_SERIALIZE_KERNEL HY4_SAMPLER_BOUNDARY_DIAGNOSTIC HY4_SAMPLER_BOUNDARY_DIR TORCH_BLAS_PREFER_HIPBLASLT TORCH_BLAS_PREFER_CUBLASLT
exec /usr/bin/python -m vllm.entrypoints.cli.main serve /data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8 --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --enable-expert-parallel --moe-backend deep_gemm --kv-cache-dtype fp8_e4m3 --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4 --kv-transfer-config '{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"mooncake_protocol": "rdma", "num_workers": 80}}' --prefill-context-parallel-size 16 --all2all-backend deepep_high_throughput --hf-overrides '{"head_dtype":"float32"}' --gpu-memory-utilization 0.90 --max-num-batched-tokens 98304 --enforce-eager --max-model-len 4096 --no-async-scheduling --enable-eplb --eplb-config '{"expert_map_path": "/workspace/hy4_mm_20261108/maps-calibrated-v1/vl-r16.json", "num_redundant_experts": 16, "static_dispatch_policy": "nearest"}' --renderer-num-workers 4 --served-model-name hy4-vl-w4a8 --mm-processor-cache-gb 0 --mm-processor-kwargs '{"max_pixels": 1048576}' --limit-mm-per-prompt '{"image": 2}' --no-enable-prefix-caching --worker-extension-cls vl_inspection_probe.WorkerExtension --middleware vl_inspection_probe.InspectionMiddleware
```

## D server (126 container)

```bash
#!/bin/bash
set -eu
unset VLLM_HCU_USE_TILED_KV_GATHER
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP
export PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HOST_IP=10.41.101.126
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_LOGGING_LEVEL=INFO
export NO_PROXY='*'
export no_proxy='*'
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export GPU_MAX_HW_QUEUES=3
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_USE_FUSED_SILU_MUL_QUANT=1
export VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1
export VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS=1
export VLLM_HCU_ENABLE_HYV4_ATTN_INPUT_OVERLAP=1
export MC_GID_INDEX=3
export NCCL_IB_GID_INDEX=3
export VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1
unset VLLM_MOE_ROUTING_SIMULATION_STRATEGY VLLM_RANDOMIZE_DP_DUMMY_INPUTS
unset VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_FACTOR
export TOKENIZERS_PARALLELISM=false
unset RAYON_NUM_THREADS VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING
unset VLLM_HCU_ENABLE_TRITON_CP_GATHER HY4_STAGE_PROBE_DIR VLLM_PLUGINS
export HY4_EXPERIMENTAL_STATIC_R16=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/workspace/hy4_mm_20261108/probe:/workspace/hy4_ttft_resume_20261101_0912/pd-eplb-20261103-0452eab/core:/workspace/hy4_mm_20261108/candidate-v1/package:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/installed:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/flashmla:/workspace/hy4_pr264_dedup_r2p45_n2920_20261030_183955/deps
export HY4_BENCH_CASE=hy4-mm-20261108-pd-r16-inspect-v1-D
unset AMD_SERIALIZE_KERNEL HY4_SAMPLER_BOUNDARY_DIAGNOSTIC HY4_SAMPLER_BOUNDARY_DIR TORCH_BLAS_PREFER_HIPBLASLT TORCH_BLAS_PREFER_CUBLASLT
exec /usr/bin/python -m vllm.entrypoints.cli.main serve /data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8 --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --enable-expert-parallel --moe-backend deep_gemm --kv-cache-dtype fp8_e4m3 --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4 --kv-transfer-config '{"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"mooncake_protocol": "rdma"}}' --data-parallel-size 16 --data-parallel-size-local 16 --data-parallel-start-rank 0 --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 28927 --data-parallel-hybrid-lb --api-server-count 1 --all2all-backend deepep_low_latency --gpu-memory-utilization 0.9 --max-num-seqs 8 --max-num-batched-tokens 32 --max-model-len 4096 --served-model-name hy4-vl-w4a8 --mm-processor-cache-gb 0 --mm-processor-kwargs '{"max_pixels": 1048576}' --limit-mm-per-prompt '{"image": 2}' --no-enable-prefix-caching --enforce-eager --worker-extension-cls vl_inspection_probe.WorkerExtension --middleware vl_inspection_probe.InspectionMiddleware
```

## Proxy (126 container)

```bash
#!/bin/bash
set -eu
export NO_PROXY="*" no_proxy="*" VLLM_PLUGINS=__disabled__
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
exec /usr/bin/python /workspace/hy4_mm_20261108/pd-r16-inspect-v1/measured_proxy_ka1.py --host 0.0.0.0 --port 8002 --prefill http://10.41.101.127:8000 18998 --decode http://10.41.101.126:8000
```

The proxy sends P and D concurrently. D can wait for remote P/KV before first scheduling. P/D engine TTFTs and D queue time therefore overlap; do not sum the two TTFTs or call all D queue time compute contention. The proxy first-byte observation may include a role/empty chunk; client TTFT uses first nonempty content.

## Full accuracy evaluation client (126 container)

```bash
#!/bin/bash
/usr/bin/python /workspace/hy4_mm_20261108/candidate-v1/package/tools/hyv4_realworldqa_eval.py --base-url http://127.0.0.1:8002 --model hy4-vl-w4a8 --data-dir /workspace/hy4_mm_20261108/dataset/data --mcq-extractor /workspace/hy4_mm_20261108/dataset/mcq_extract.py --output-dir /workspace/hy4_mm_20261108/pd-r16-inspect-v1/qa --concurrency 1 --max-tokens 16 --max-pixels 1048576
```

Pinned RealWorldQA765, original questions, official pinned MCQ extraction/open-answer normalization, concurrency1, max_tokens16/temp0/seed0/no_think. The full original TP8 V1 reference was not scored; its7 functional fixtures are separate. Raw output differences are preserved and equal aggregate scores do not establish numerical identity.

## Arrival performance client (126 container)

```bash
#!/bin/bash
/usr/bin/python /workspace/hy4_mm_20261108/probe/vl_arrival_bench.py --base-url http://127.0.0.1:8002 --model hy4-vl-w4a8 --data-dir /workspace/hy4_mm_20261108/dataset/data --corpus /workspace/hy4_mm_20261108/calibration-corpus-v1/corpus.json --output-dir /workspace/hy4_mm_20261108/pd-r16-inspect-v1/arrival --rps 2.55 --count 512 --concurrency 146
```

Resolved defaults:512unique images, selection/arrival seed255, nominal Poisson2.55RPS, max concurrency146, max_tokens16/temp0/top_p1/seed0/no_think. All bodies are encoded and serialized before the timed origin. Fit64/heldout32 IDs and raw-image aliases are excluded; the remaining pool has666unique images. The frozen finite schedule has2.4400167653 event arrivals/s; report actual arrival rate separately from completed RPS. There are no API warmups inside this timed operator, retries or KV priming. Earlier accuracy/functional requests did run, but prefix caching is disabled. Native image-encoder LRU remains enabled; this is not a cold-process/encoder-cache guarantee.

Latency starts before the client semaphore, with HTTP-only TTFT separately recorded. Usage-derived TPOT is (stream end−first content)/(completion tokens−1), including tail overhead; it is not measured token-by-token latency. The budget isP16+D16=32cards; total TPM counts each input once end to end. Completed RPS at a fixed arrival load is not maximum saturated capacity.

## Offline JSON: collect, generate, then reload

Only P loads the map. R0 identity has77MoE rows×256physical experts; calibrated R16 has77×272physical experts (17/rank), covering256logical experts plus16extra slots per MoE row. The only JSON model key is HYV4VLForConditionalGeneration; no text/MTP map is reused. Nearest dispatch is explicitly passed at launch.

1. Start P with the fresh VL identity R0 map and the external calibration/inspection extension. Use the fixed corpus manifest `/workspace/hy4_mm_20261108/calibration-corpus-v1/corpus.json` (fit64,heldout32). Only routing counts are used, no answer labels.

2. During an untimed quiet period, collect fit and heldout separately. The following is the fit example; repeat with group heldout and distinct output/manifest names. Stop restores callbacks before exporting counts. Preserve the full16-worker stop response as the manifest.

```bash
curl --fail --silent --show-error -X POST http://127.0.0.1:8000/hy4/vl-eplb/start -H 'Content-Type: application/json' -d '{}'
/usr/bin/python /workspace/hy4_mm_20261108/probe/vl_calibration_client_v2.py   --corpus /workspace/hy4_mm_20261108/calibration-corpus-v1/corpus.json   --dataset /workspace/hy4_mm_20261108/dataset/data --group fit   --base-url http://127.0.0.1:8000   --output /workspace/hy4_mm_20261108/reproduce-fit-client-NEW --prefill-only
curl --fail --silent --show-error -X POST http://127.0.0.1:8000/hy4/vl-eplb/stop -H 'Content-Type: application/json'   -d '{"output_dir":"/workspace/hy4_mm_20261108/reproduce-fit-counts-NEW"}'   > /workspace/hy4_mm_20261108/reproduce-fit-stop-NEW.json
```

3. Bind source/model/corpus/capture hashes in provenance. Actual original capture manifests are `/workspace/hy4_mm_20261108/pd-r0-v1/capture-fit-stop.json` and `capture-heldout-stop.json`; original provenance is `/workspace/hy4_mm_20261108/maps-calibrated-v1/provenance.json`. The command below regenerates from those preserved original captures into a NEW file. To recalibrate another workload, use its new manifests and provenance instead.

```bash
mkdir -p /workspace/hy4_mm_20261108/maps-reproduce-NEW
env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= VLLM_PLUGINS=__disabled__   /usr/bin/python /workspace/hy4_mm_20261108/probe/build_vl_maps.py   --fit-manifest /workspace/hy4_mm_20261108/pd-r0-v1/capture-fit-stop.json   --heldout-manifest /workspace/hy4_mm_20261108/pd-r0-v1/capture-heldout-stop.json   --provenance /workspace/hy4_mm_20261108/maps-calibrated-v1/provenance.json   --policy-source /workspace/hy4_ttft_resume_20261101_0912/pd-eplb-20261103-0452eab/core/vllm/distributed/eplb/policy/default.py   --redundancy 16 --output /workspace/hy4_mm_20261108/maps-reproduce-NEW/vl-r16.json
```

The generator verifies all16ranks, all77rows, physical→logical scatter equality, model/config/corpus/source SHA values and native DefaultEplbPolicy source. Policy uses fit only; heldout is excluded from optimization. Sidecar predictions assume ideal replica shares and are not observed balance. Source files referenced by absolute provenance paths must be preserved/restored for reproduction.

4. Gracefully stop the owned P service, then reload with the generated JSON. The actual relevant P settings were:

```bash
export HY4_EXPERIMENTAL_STATIC_R16=1
# Add these to the full P command above:
--enable-eplb --eplb-config '{"expert_map_path":"/workspace/hy4_mm_20261108/maps-calibrated-v1/vl-r16.json","num_redundant_experts":16,"static_dispatch_policy":"nearest"}'
```

D keeps EPLB/MTP disabled. Restart the proxy after replacing P so its bootstrap engine_id cache is current. A fresh heldout physical capture after R16 loading verifies actual replica dispatch; stop it before timing. Existing actual R16 heldout capture/report is under `/workspace/hy4_mm_20261108/pd-r16-inspect-v1/capture-heldout-direct-p` / `ACTUAL_HELDOUT_BALANCE.json`. Counts include executed PCP padding and exclude only−1; they are not FLOPs or utilization.

Actual map SHA256: `93cc07bcdaf29ced5919fb0652c88c1b86454e87dd6a967034df97cfab2bdf0a`. The R0→R16 comparison changes placement and replication together; an optimized R0 JSON was also generated but not benchmarked, so redundancy necessity is not isolated. Current image encoder execution is replicated across PCP ranks. P/D graph performance and globally best deployment are not established by these eager runs.

See the [calibration contract](hy4_vl_pd_calibration.md) for recorder/provenance details. Raw QA/SSE responses, metrics and reviews remain in the original experiment archive; they are not files in this public repository.

## Results and limits (2026-11-08)

Both configurations used P16+D16 cards, real routing, the same image workload
and client, no MTP and no prefix-cache priming. R0 is identity placement;
R16 changes placement and replication together. An optimized R0 was generated
but not benchmarked, so these runs do not isolate the need for extra experts.

| Metric | R0 identity | Calibrated R16 |
| --- | ---: | ---: |
| Full RealWorldQA accuracy | 554/765 (72.42%) | 556/765 (72.68%) |
| Successful timed requests | 512/512 | 512/512 |
| Client mean TTFT | 4160.53 ms | 3986.57 ms |
| Client p95 TTFT | 5572.75 ms | 5429.54 ms |
| Completed RPS | 2.40433 | 2.40702 |
| P prefill metric | 1305.32 ms | 1317.58 ms |
| Heldout weighted layer max/mean | 1.695845 | 1.057106 |
| Heldout worst layer max/mean | 2.928239 | 1.134620 |

The 4.18% lower observed mean TTFT is one finite fixed-arrival comparison,
not an established stable kernel speedup or saturated capacity improvement.
P prefill did not improve. Nominal Poisson rate was 2.55 RPS, while the frozen
finite schedule actually arrived at about 2.44002 RPS. The 512 timed requests
exclude calibration items, and are distinct from the 765 serial accuracy run.

R16 versus R0 changed 59 raw answers: 18 became correct and 16 became incorrect;
other changed answers retained their correctness category. These aggregate
scores do not prove numerical equivalence. No full 765-question TP8/V1 golden
reference was evaluated in this experiment. Canonical expert checks compared
32 weight points and complete scales per expert, not complete weight bytes.
The P16 native position/RoPE checks and D position handoff checks passed within
the tested eager scope. Native KV sizing was used without an arbitrary cap.

Prior affected native CPU sets recorded 627 passes (461+166), 195 subtests and
95 deselections; CI selector checks recorded 81 passes. The image-disabled
dummy regression was separately red/green tested (23 passes after the fix).
These are scoped validation results, not complete CI: full collection still
requires native dependencies, and touched-file full Ruff was not clean.
Graph, video, MTP, other distributed layouts and globally best deployment are
not validated by this experiment. Cards were released after testing.
