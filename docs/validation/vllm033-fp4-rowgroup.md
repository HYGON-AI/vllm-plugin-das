# FP4 request-row reuse candidate

The native FP4 planner and scorer can reuse KV loads for consecutive query rows
belonging to one request. The old reader always passed rows_per_request=1 and,
for noncanonical allocator pages, staged the same request prefix once per token.
The new reader uses Core's original decode_is_uniform/write_max_decode_len,
original request count and exact flattened row count to prove request ownership.
It stages the last (longest causal prefix) row once per request and expands only
the small canonical page table. Each query retains its original device length.
Canonical 64-record pages already share their underlying physical addresses.

Admission requires the existing validated gfx938/64CU device and live native
planner, unpadded uniform metadata, no PCP-remapped indices, and the native
bounded TopK consumer's supported K values (512/1024/2048). Ragged, remapped,
padded or unsupported consumers keep the old row schedule. Request count,
query group width, context length and allocator page count are dynamic. There
are no new production environment variables or arguments.

The original K128 Torch fallback boundary test failed on all-equal scores.
The diagnostic retained identical live FP32 logits, identical output strides,
and three unequal repetitions of the unchanged native TopK itself. The
candidate now leaves this consumer on the old schedule. No TopK implementation,
score tolerance or existing failed artifact was changed.

Local gfx938 evidence from 2026-11-23:

- 160 native scorer cases: request counts 1/2/7/17, group widths
  1/2/3/5/6/7/8/9, contexts 64/512/4096/32768/65536; live scores bit-exact.
- 180 real staging/scoring cases, allocator pages 32/64/128 records, including
  dead requests and different causal row lengths: payload, scale and live
  logits bit-exact; 27 changing-input Graph replays.
- 216 actual reader + native selector cases for K512/1024/2048: random scores,
  finite all-equal scores, no candidate bank, candidate writes and candidate
  masks. Final TopK indices and banks exact; 54 changing-input Graph replays.

The device tests used the real installed LightOp APIs and actual reader, not a
replacement score oracle. The unchanged row schedule is the comparison arm.
Raw gate, scripts, logs and JSON are under
run/dsv4.1/033/fp8_optimization_20261123/:
FP4_ROWGROUP_GATE.json, microbench/fp4_rowgroup_provider_nmz30.json,
microbench/fp4_rowgroup_staging_nmz30.json,
microbench/fp4_rowgroup_reader_supported_nmz30.json. Initial K128 failures and
the diagnostic tensors remain alongside the successful checks.

Paired timing, full HumanEval164, eight-rank profile and unprofiled serving are
still required before accepting a model performance benefit.
