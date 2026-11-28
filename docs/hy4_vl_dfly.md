# Standard HY4 VL and checkpoint DFly

Uses the unchanged OpenDAS Core 0.33 commit `c440c089`. Adapter owns the
platform and shared operator ABI. This Plugin owns the standard HY4 VL config,
processor, model, M-RoPE and checkpoint DFly adaptation. No Core source overlay
or editable installation is required.

The checkpoint keeps `model_type=hy_v4_vl` and its standard `hy_v4` text tower.
Vision parameters retain their original precision. The existing CT policy
selects the checkpoint's INT4 experts and INT8 linear/draft weights.

DFly uses the checkpoint's own draft layers. For `dfly_block_size=7`, the
default is six speculative tokens plus the anchor; five draft layers do not
imply five speculative tokens.

## DP16 configuration for hardware validation

Run in the installed Adapter/Plugin environment, with the checkpoint at
`/models/hy4`:

```bash
export HCU_ADAPTER_HOST=vllm
export VLLM_PLUGINS=hcu_adapter
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hy4-ct-w4a8,hy4-vl,hy4-dfly
export VLLM_HOST_IP="<NODE_IP>"
export NCCL_SOCKET_IFNAME="<IFACE>"
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export ROCSHMEM_GDR_DISABLE_XDP=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=4096
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve /models/hy4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port 29330 \
  --api-server-count 1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 16 \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 29331 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --attention-backend FLASHMLA_SPARSE \
  --kv-cache-dtype auto \
  --speculative-config '{"method":"dfly"}' \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":2,"video":1}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --enforce-eager
```

The distributed target is TP1/DP16/EP16. Dense draft replicas use their local
target's KV/index snapshots and do not join expert collectives. PCP, DCP and PP
remain one; KV transfer, static/online EPLB, DBO and elastic EP are not admitted.
The existing Flash DFly path retains its own topology restrictions.

For target-only validation, remove `--speculative-config` and the `hy4-dfly`
required capability. Keep the same TP1/DP16/EP16 topology. The VL renderer
delegates to the native HF renderer for other model types. DFly uses guarded
in-process compatibility bindings; the installed Core files remain unchanged.

CPU checks and hardware results are separate validation lanes. The command above
is a functional test configuration, not a measured performance recommendation.

## Validation on 2026-11-28

Installed wheels on DTK2610 / Torch2.11 / Triton3.6: stock Core `c440c089`,
Adapter main `2fc1f16`, Plugin runtime `852fb81`. Both nodes passed the 52 DFly
and 27 VL CPU checks and actual-checkpoint parent/spawn configuration checks.
The two native vision tests passed separately on HCU. The Core's 3,594 packaged
files and Plugin's 283 Python files were checked against the installed wheels
and source; no source overlay was used.

On POD1_CN1, the full checkpoint loaded and served text, images and a two-frame
video with and without DFly. Both configurations answered `7 × 8` with `56`,
identified opposite red/blue image layouts, and described the video's color
swap correctly. DFly's first three requests produced 30 draft tokens and
18 accepted tokens. These are functional samples, not a dataset acceptance rate.
An additional 16 concurrent requests with input lengths of 39–99 tokens all
returned their expected arithmetic answers.
HumanEval, long-context accuracy, Graph execution and comparative throughput
remain separate validation work.
