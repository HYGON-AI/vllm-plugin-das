# vLLM 0.32 HCU FlashAttention varlen validation

This record validates the explicit Plugin-owned `FLASH_ATTN` route. vLLM owns
the default-preserving KV allocation and kernel seams, the Adapter owns HCU
platform selection and configuration validation, and the Plugin owns the
optional public `flash_attn.flash_attn_varlen_func` implementation plus the
narrow E5M2 cache writer selected only by that implementation. The
default HCU attention backend remains vLLM's upstream Triton backend.

## Candidate ownership

| Component | Revision | Wheel | SHA256 |
| --- | --- | --- | --- |
| vLLM | `923e3284c` | `vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl` | `3e2798384365084cbcbcb61d0fff713a8c27fae22c0980fb85b6dd93d3e47c22` |
| Adapter | `d7487a8` | `sgl_kernel-0.1.0-py3-none-any.whl` | `a8e2b7fea4a13352c7f3dc3365f8c4579cc8ac941bd4d2f408100f24286affd7` |
| Plugin | `57687161` | `hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl` | `0814ebc46b59bcac09882b7cf56cb620c1ad012be71a2e162d8e47ab48a5b47b` |

Installed-state inspection resolved all three packages from the isolated
candidate environment. The installed upstream class exposed
`_get_kv_cache_views`, `_get_kv_cache_write_views`, and
`_run_flash_attention`; the Plugin reported DCP disabled, exact block-size 64
support, and `auto`, `float16`, `bfloat16`, and `fp8_e5m2` KV cache dtypes.

## Numerical and Graph gate

The live HCU test writes K/V through vLLM's official stride-aware cache writer,
reads the same storage through the Plugin's zero-copy HND aliases, and compares
the public vendor varlen result with a Torch float32 reference. It covers BF16
prefill, BF16 decode, Qwen-style 14/2 short GQA, FP16 MHA, FP16 GQA, and Graph
replay after changing Q/K/V at the same addresses.

```bash
HCU_ADAPTER_HOST=none \
VLLM_PLUGINS=hcu_adapter \
VLLM_CUDART_SO_PATH=/opt/dtk/lib/libamdhip64.so \
HIP_VISIBLE_DEVICES=0 \
VLLM032_HCU_TEST=1 \
PYTHONPATH=<plugin-source>:<candidate-site-packages> \
<candidate-python> -m pytest -q \
  tests/vllm032/test_flash_attention_hcu.py
```

Result: `6 passed` in 35.27 seconds. The audited absolute and relative
tolerance was 0.016. Portable Plugin tests were `96 passed, 12 skipped`;
Adapter bridge/extension tests were `49 passed`; the Adapter L0 gate was GREEN
with 32 pre-existing bare-environment warnings; vLLM seam tests were `2 passed,
158 deselected`.

Negative tests verify that a stock vLLM 0.32 build without all three seams
fails during explicit Plugin import. Explicit Adapter selection calls the
Plugin backend's normal `validate_configuration()` path. DCP, E4M3 KV, page
sizes other than 64, sinks, RSWA, multimodal prefix masks, and active FA4-only
arguments fail closed. Encoder/cross attention, sliding-window attention,
non-causal attention, and batch-invariance mode also fail validation because
they are outside this release's tested decoder matrix.

## Non-eager model service gate

The final wheel set was launched without `--enforce-eager`:

```bash
HCU_ADAPTER_HOST=none \
VLLM_PLUGINS=hcu_adapter \
VLLM_CUDART_SO_PATH=/opt/dtk/lib/libamdhip64.so \
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=<candidate-site-packages> \
vllm serve /model/Qwen2-0.5B-Instruct \
  --host 127.0.0.1 --port 18083 \
  --dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --block-size 64 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 512 \
  --max-num-seqs 4
```

The engine reported vLLM `0.32.0+rocm723`, `enforce_eager=False`, LBHNC KV
layout, and `FULL_AND_PIECEWISE`. Both PIECEWISE and FULL Graph captures
completed; the second capture reported 0.17 GiB actual Graph memory. A prompt
with five tokens exercised the short-prefill boundary and returned HTTP 200,
24 completion tokens, and the `vllm-0.32.0+rocm723` fingerprint. The service
then shut down cleanly and left no owned API or engine process.

## HumanEval/0-15 control

Before the fail-closed review patch, the same kernel implementation and model
served the ordered HumanEval/0-15 prefix from HumanEval commit
`6d43fb980f9fee3c892a914eda09951f772ad10d`. Requests used temperature 0,
seed 0, thinking disabled, and 320 maximum tokens. All 16 requests completed
in 13.08 seconds with 4,184 completion tokens; finish reasons were 12 stop and
4 length.

Sandboxed scoring produced **5/16, pass@1 0.3125**, with tasks 3, 7, 8, 13,
and 15 passing. The Triton control produced the same score and exact passing
task set. The later review patch changes only compatibility/configuration
fail-closed checks and test coverage; the final wheels were revalidated by the
numerical, Graph, and model-service gates above. Raw evidence is under
`/dev/shm/vllm032-flash-candidate.KQDH7R/humaneval16-flash`.

## Scope

This initial gate validated the available single-device HCU topology,
BF16/FP16 dense decoder MHA/GQA, LBHNC storage, 64-token pages, and the public
vendor varlen interface. It did not claim DCP, hybrid cache layouts, alternate
page sizes, FA4-only masks, tensor parallelism, or a throughput improvement.

## Dual-layout follow-up (2026-10-31)

The follow-up branch based on Plugin `11f331601` adds the second block-compact
layout without changing vLLM or Adapter. The Plugin reads the layout already
resolved by vLLM and applies these zero-copy contracts:

| vLLM layout | Legacy alias | Dense vendor view | Vendor argument |
| --- | --- | --- | --- |
| `LBHNC` | `HND` | `[B, H, N, D]` | `layout="bhsd"` |
| `LBNHC` | `NHD` | `[B, N, H, D]` | `layout="bshd"` |

Portable vLLM 0.32 tests completed with `126 passed, 18 skipped`. Live HCU
tests covered BF16 and FP16, MHA and GQA, prefill and decode, plus changed-input
Graph replay for each layout: `12 passed` in 34.69 seconds at
`atol=rtol=0.016` from the final installed wheel. The official
`reshape_and_cache_flash` writer populated the second of two padded cache
blocks in both physical layouts before the public vendor reader was compared
with the Torch float32 reference. Capability tests also reject head sizes not
divisible by 64 before vendor dispatch, matching the installed public vendor
entry point; the supported set through 256 is 64, 128, 192, and 256.

Graph-enabled services were then started twice with the same
`/model/Qwen2-0.5B-Instruct` arguments, first with
`VLLM_KV_CACHE_LAYOUT=HND` and then `NHD`. The engine reported LBHNC and LBNHC
respectively, retained `enforce_eager=False` and `FULL_AND_PIECEWISE`, and
completed both PIECEWISE and FULL capture. Identical deterministic completion
requests returned HTTP 200, five prompt tokens, sixteen completion tokens, and
the same response text on both layouts. Both owned service trees were stopped
and no API server or EngineCore process remained.

The final installed wheel was also served with explicit NHD for the ordered
HumanEval/0-15 control, using the same dataset commit, prompts, temperature 0,
seed 0, thinking-disabled setting, and 320-token limit as the historical HND
run. All 16 requests completed in 12.72 seconds with 4,184 completion tokens
and the same 12 stop / 4 length finish-reason split. Every generated content,
finish reason, prompt-token count, and completion-token count matched the HND
record; both `samples.jsonl` files have SHA256
`32f43cabdda392f515946d6b417423d808b08f31e71fb83cc37b8283f9429cc5`.
The already sandboxed score therefore remains **5/16, pass@1 0.3125**, with
the same passing task set, without re-executing identical generated code. Raw
NHD evidence is under
`/dev/shm/vllm032-flash-dual-layout-he16-nhd-63395e0d`.

The two model-service runs above used the isolated vLLM seam candidate
`923e3284c` and loaded the Plugin branch directly through `PYTHONPATH`; they do
not replace the historical installed-wheel identities above. The subsequent
HCU matrix used an isolated installation of the final wheel built as
`hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl`, SHA256
`25964ff51e99b5a29d66cafaeca310e3767b6d81d42c065bfcab4c4853e0db87`;
package-boundary, installed-verifier fixture, and compat-lock gates completed
with `18 passed`. An isolated target installation then passed the real
installed-state verifier: Plugin version `0.2.0.dev0`, Plugin RECORD SHA256
`cd25ede7f641db6dcb2f64cd62ee4faeb27f4e5fc8010908c25adb8b342f8733`,
active `das_hcu` extension, expected RMSNorm/AITER owners, Adapter platform,
and vLLM `0.32.0+rocm723` ownership were all confirmed. The target directory
was layered over the isolated candidate environment and did not modify its
installed Plugin.

### Additional model-service matrix

The final installed Plugin wheel was then exercised on additional dense GQA
checkpoints with one visible HCU, BF16 weights, `FLASH_ATTN`, 64-token pages,
the V2 model runner, and the default `FULL_AND_PIECEWISE` Graph mode. No run
used `--enforce-eager`. Requests used temperature 0, seed 0, thinking disabled,
and two different prompts; the Qwen3 requests were submitted concurrently.

| Checkpoint | Q/KV heads | Layout | GPU memory utilization | Result |
| --- | ---: | --- | ---: | --- |
| `/model/Qwen2-1.5B-Instruct` | 12/2 | `LBHNC` (`HND`) | 0.08 | Both Graph modes captured; two changed-input requests returned HTTP 200 and normal `stop`, with correct definition and arithmetic answers. |
| `/model/Qwen3-1.7B` | 16/8 | `LBNHC` (`NHD`) | 0.10 | Both Graph modes captured; two concurrent requests returned HTTP 200 and normal `stop`, with correct definition and arithmetic answers. |
| `/model/Qwen3-8B` | 32/8 | `LBNHC` (`NHD`) | 0.30 | Both Graph modes captured; two concurrent requests returned HTTP 200 and normal `stop`, with correct definition and arithmetic answers. |

The first Qwen3-1.7B launch used a 0.08 memory budget. It completed model load
and both Graph captures, but vLLM measured -0.07 GiB available for KV blocks
and failed its official cache-capacity check. Repeating with only the memory
budget raised to 0.10 left 1.96 GiB for KV cache and passed. This is recorded as
a launch-configuration control, not an attention or numerical failure. All
owned services were stopped after the requests; no API server or EngineCore
remained and `rocm-smi --showmemuse --json` reported zero percent memory use on
all eight cards.

### MLA and sparse-MLA isolation

The Adapter route was checked together with the installed vLLM 0.32 backend
classes. It substitutes this Plugin only when the selected top-level enum is
exactly `FLASH_ATTN`; `FLASHMLA`, `FLASHMLA_SPARSE`,
`ROCM_AITER_MLA_SPARSE`, `FLASH_ATTN_MLA`, and other selections continue to
the upstream ROCm/platform implementation. The Plugin therefore does not
replace an MLA or sparse-MLA kernel.

The global KV-layout resolver was also executed with the real backend classes:

| Backend combination | Declared/common layouts | Result |
| --- | --- | --- |
| Plugin + `FlashMLABackend` | MLA declares any layout | `LBHNC`, `LBNHC` |
| Plugin + `FlashMLASparseBackend` | sparse MLA declares any layout | `LBHNC`, `LBNHC` |
| Plugin + `ROCMAiterMLASparseBackend` | AITER declares `LBNHC`, `LBHNC`, `BLHNC` | common `LBHNC`, `LBNHC` |
| Plugin + `DeepseekV4IndexerBackend` | indexer requires `BLHNC` or `BLNHC` | explicit `ValueError`: no common layout |
| Plugin + `KpoolTailBackend` | tail requires `BLHNC` | explicit `ValueError`: no common layout |

Thus ordinary MLA and sparse-MLA backend selection is not affected by this
Plugin change. In a valid composite worker, adding NHD expands the common
FlashAttention/AITER sparse-MLA set to both block-compact HND and NHD. An
explicitly forced composite that mixes this ordinary FlashAttention backend
with a BL-preferring DeepSeek-V4 indexer or kpool tail fails during layout
resolution instead of silently reinterpreting its cache. Large MLA and sparse
MLA checkpoints were not loaded in this single-card follow-up, so this section
is a routing/layout-contract result rather than an MLA model-accuracy claim.

## E5M2 KV-cache follow-up (2026-10-31)

The ordinary decoder `FLASH_ATTN` route now accepts `--kv-cache-dtype
fp8_e5m2` without model-name routing. vLLM remains responsible for cache
allocation, slot mapping, scales, and every non-E5M2 write. For E5M2 only,
`HcuFlashAttentionImpl.do_kv_cache_update()` selects a Plugin-owned Triton
writer that stores OCP E5M2 bytes, then the existing public vendor
`flash_attn_varlen_func` reads the same storage through an E5M2 view. Query
compute remains BF16/FP16.

This writer was required by live RED evidence rather than capability
assumption: the installed HCU vLLM `reshape_and_cache_flash` raised
`Unsupported data type of kv cache: fp8_e5m2` before reader dispatch. Relabeling
the request as E4M3 was rejected because the formats have different byte
semantics. The Plugin override is confined to its implementation subclass;
non-E5M2 calls continue to delegate to vLLM unchanged.

The source-tree portable gate used the HCU vLLM source at `923e3284c`:

```bash
PYTHONNOUSERSITE=1 HCU_ADAPTER_HOST=none \
VLLM_TARGET_ROOT=/dev/shm/hcu-vllm032-compat-20261027/src/vllm-f8caf39 \
<candidate-python> -m pytest -q tests/vllm032 \
  --ignore=tests/vllm032/test_flash_attention_hcu.py
```

Result: `137 passed, 6 skipped` in 155.85 seconds. The live source-tree HCU
matrix completed with `20 passed` in 35.10 seconds. It covers LBHNC/BHSD and
LBNHC/BSHD, BF16 and FP16 controls, E5M2 prefill and GQA decode, padded block-1
writes, scalar and per-head scales, and changed-input Graph replay. Writer
bytes are compared exactly (`rtol=atol=0`) with `torch.float8_e5m2`; the
writer-to-reader output uses the explicit audited E5M2 tolerance
`rtol=atol=0.03`. Existing BF16/FP16 rows retain `rtol=atol=0.016`.

The final Plugin wheel for this gate is:

```text
/dev/shm/vllm-plugin-e5m2-wheel.G7dfij/
  hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl
SHA256 0814ebc46b59bcac09882b7cf56cb620c1ad012be71a2e162d8e47ab48a5b47b
```

It was installed with `--no-deps --target` at
`/dev/shm/vllm-plugin-e5m2-install.FjzYme`. Module-origin inspection resolved
`hcu_vllm_das`, `attention.flash_attn`, and `attention.e5m2_cache` from that
target rather than the source tree. Package-boundary, extension, compat-lock,
and installed-verifier fixture tests completed with `44 passed`; the complete
live HCU matrix repeated from the isolated wheel with `20 passed` in 35.10
seconds. The real installed-state verifier reported Plugin RECORD SHA256
`b61731639f7214a1d77c2647af0b06e82f0c8403b50b9a742baa08a85363250b`,
active `das_hcu`, Adapter version `0.1.0`, and vLLM
`0.32.0+rocm723` from their isolated candidate locations.

This route still does not replace MLA, sparse MLA, multimodal-prefix, hybrid,
sliding-window, sink, DCP, PCP, or FA4-only paths. Those continue to follow
their existing backend selection or fail-closed validation. Cross-model
BF16-KV/E5M2-KV service results are recorded separately below when available;
passing the operator matrix alone is not an accuracy claim.
