# Kimi-K2.6 gfx938 TP8 HumanEval-64 validation

This record covers the local `/models/Kimi-K2.6` checkpoint on the eight-card
gfx938 host with the pinned vLLM 0.28.1 DTK runtime recorded in
`v0281-gfx938-provenance.md`. HumanEval generated code was executed only after
the operator asserted the isolated evaluation boundary with
`VLLM_HCU_HUMANEVAL_ISOLATED=1`.

## Checkpoint and route

- 64 safetensors shards, about 555 GiB, with 208,550 indexed tensors and no
  missing or extra shard entries.
- `KimiK25ForConditionalGeneration` multimodal wrapper with a
  `DeepseekV3ForCausalLM` language model, compressed-tensors W4A16 group
  quantization, 61 layers, 384 routed experts, top-8 routing, and one shared
  expert.
- `num_nextn_predict_layers=0`, so the checkpoint does not support MTP.
- Code evaluation uses `--language-model-only`, TP8, HcuGPUModelRunnerV2,
  regular `FLASHMLA`, Triton WNA16 MoE, the `kimi_k2` reasoning parser,
  prefix caching, BF16/auto KV, and the default FULL_AND_PIECEWISE Graph
  policy. FLASHMLA retains its native LBNHC KV layout and 64-token cache page.

The validated service command is:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /models/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA \
  --moe-backend triton \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --port 10223
```

The final runtime loaded 74.92 GiB per rank, allocated an 859,136-token KV
cache, captured both FULL and PIECEWISE graph sizes, and produced prefix-cache
hits. It reported `CompressedTensorsWNA16MoEMethod`, `Using 'TRITON' WNA16 MoE
backend`, and `Using TritonWNA16Experts`. AITER rejects this WNA16 MoE route
by contract, while Humming has no HIP implementation; neither is presented as
a viable backend for this checkpoint. The command deliberately contains no
`VLLM_DISABLE_SHARED_EXPERTS_STREAM` debug override.

An explicit quantized-KV smoke used the same command with
`--kv-cache-dtype fp8_e4m3 --port 10224`. The resolved engine config retained
`FULL_AND_PIECEWISE`, captured both FULL and PIECEWISE graphs, kept the LBNHC
layout, and allocated 1,718,272 KV tokens. The historical high-risk concurrent
batch `/19,/21,/25,/31,/36,/39,/46,/60` completed 8/8 HTTP 200 requests with
normal stops between 102 and 295 output tokens. This live check confirms that
the chip's explicit E4M3 KV route also retains full-graph support.

The benchmark-style Thinking run used the same command except for the two
capacity arguments below; its full reproducible profile is
`tests/models/kimi_k26_gfx938_humaneval64_thinking.yaml`:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /models/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA \
  --moe-backend triton \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --port 10222
```

## Root cause and fix

The plugin's fused-MoE runner came from an older vLLM implementation
and omitted current upstream ROCm's shared-expert multi-stream safety gate.
Kimi's WNA16 path quantizes weights but leaves activations unquantized, so the
routed-expert and shared-expert inputs alias. Running the shared expert on an
auxiliary stream during graph replay raced the routed expert and caused
batch-sensitive repeated or unrelated generation. The HCU runner now passes
the upstream-equivalent activation-quantization capability to `SharedExperts`;
an aliased unquantized input uses `NO_OVERLAP`, while activation-quantized
routes may retain auxiliary-stream overlap. The HCU early-launch and
stream-force controls are also gated and cannot override this safety result.

The isolation controls were decisive:

- eager execution and several `max-num-seqs=1` requests passed;
- compile-only/no-Graph execution passed HumanEval/3 and /31;
- forcing shared-expert serialization while keeping Graph replay made the
  full deterministic HumanEval-64 run pass 64/64;
- after the code fix, the unmodified command above passed /3 and /31 plus a
  concurrent high-risk batch `/19,/21,/25,/31,/36,/39,/46,/60`, all with
  normal stops between 102 and 295 output tokens; the final post-review rerun
  repeated that concurrent batch without force controls and stopped normally
  between 102 and 273 output tokens;
- restoring BF16 FlashMLA's advertised full-graph capability retained FULL
  plus PIECEWISE capture and passed the same concurrent batch followed by
  normalized HumanEval 64/64; this disproved the earlier scheduler-metadata
  hypothesis and demonstrated that no BF16 graph downgrade is required;
- the explicit `fp8_e4m3` KV route also retained FULL plus PIECEWISE capture and
  completed the same eight concurrent prompts with normal 102-to-295-token
  stops.

`cudagraph_copy_inputs=true` did not fix the issue, direct
`_moe_C::moe_align_block_size` replay was correct, and splitting MoE custom
ops into Inductor graph partitions caused an HSA VMFault. Those approaches are
not part of the accepted fix.

## HumanEval-64 acceptance

The accepted profile uses deterministic Instant mode:

```json
{
  "temperature": 0,
  "do_sample": false,
  "max_tokens": 1024,
  "extra_body": {"chat_template_kwargs": {"thinking": false}}
}
```

It runs batch eight and enforces an independently normalized HumanEval score
of 1.0. The final FULL_AND_PIECEWISE run produced 64 predictions and reviews
in 411.60 seconds and passed normalized HumanEval 64/64 with an empty failure
list. It measured 46.398 s mean latency, 1,796.8 ms mean TTFT, 310.4 ms mean
TPOT, and 3.13 output tokens/s. Together with the preceding high-risk batch,
the same service returned 72/72 HTTP 200 responses without an ERROR or
Traceback.

EvalScope's raw checker scored only 8/64 because it does not consistently
handle the model's complete fenced or horizontally indented module. The
normalizer removes one complete or truncated Markdown fence and removes
leading horizontal whitespace only when the result parses and contains the
expected top-level HumanEval entry point. This preserves valid indented
body-only completions and fails closed on syntax or parser resource errors.
Host-side execution is allowed only after the operator asserts
`VLLM_HCU_HUMANEVAL_ISOLATED=1`; sandboxed output is never re-executed outside
the sandbox.

The exact evaluation command is:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Kimi-K2.6 \
  --api-url http://127.0.0.1:10223/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":1024,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 64 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/kimi-k26-gfx938-full-and-piecewise-final \
  --no-timestamp
```

Earlier greedy E4M3/BF16, official-sampling Instant, and long Thinking runs
scored 52/64, 51/64, 54/64, and 55/64 respectively while the stream race was
still active. They remain useful diagnostic history but no longer define the
accepted route. Repetition penalties 1.05 and 1.1 did not repair representative
failures.

Evidence directories:

- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-full-and-piecewise-final`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-patched-default`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-fp8-graph-smoke`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-shared-serial-graph`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-thinking-16k-b8`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-official-sampling`
- `/tmp/vllm-hcu-evalscope/kimi-k26-gfx938-humaneval64-kvbf16`
