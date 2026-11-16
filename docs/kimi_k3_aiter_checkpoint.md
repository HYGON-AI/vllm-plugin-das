# Kimi K3 AITER state and extra checkpoints

The optional KDA state implementation uses SGLang's AITER HIP entry point for
K=V=128. Gate, intra and output remain BoltOps. Select it with:

```sh
export VLLM_KIMI_KDA_PREFILL_BACKEND=boltops
export VLLM_KIMI_KDA_CACHE_STRATEGY=extra_buffer
export VLLM_KIMI_KDA_STATE_BACKEND=aiter
```

The default state backend is `boltops`; this keeps the prior FP32 chunk history
path. The AITER kernel returns BF16 chunk history and updates its initial state
pool in place. The adapter copies the initial states into a private FP32 pool,
returns its final states to vLLM, and exports the BF16 boundary to the existing
checkpoint slot. If the recurrent cache itself is FP32 the BF16 checkpoint is
upcast for storage; its values still have BF16 precision. An AITER interface
incompatible with the expected SGLang signature fails explicitly.

The numerical tests require an HCU, SGLang source, and an AITER build with
`chunk_gated_delta_rule_fwd_sglang_hip_blockdim64`. Set the plugin and SGLang
source directories in `PYTHONPATH` and run:

```sh
SGLANG_KDA_USE_HCU_OP=1 SGLANG_KDA_DISABLE_TRITON_CHUNK_H=1 \
VLLM_PLUGINS=__disabled__ python3 -m pytest -q -s \
  tests/kernels/test_kimi_k3_kda_aiter_sglang.py
```

The tests compare outputs, final state, BF16 checkpoint and resumed outputs to
SGLang, and cover packed variable lengths. The deployed layer replay test can be
run with `VLLM_KIMI_KDA_STATE_BACKEND=aiter`; its cache-hit vs cold outputs
are tolerance-based because the BF16 boundary rounds the recurrent state.
`tests/kernels/test_kimi_k3_kda_aiter_timing.py` gives a small, non-gating
state-stage timing, including adapter overhead. No service is needed for these
tests. Full model accuracy and service latency remain to be validated separately.
