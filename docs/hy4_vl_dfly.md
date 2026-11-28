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

vllm serve /models/hy4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 16 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend deep_gemm \
  --attention-backend FLASHMLA_SPARSE \
  --kv-cache-dtype auto \
  --speculative-config '{"method":"dfly"}' \
  --max-model-len 4096 \
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
