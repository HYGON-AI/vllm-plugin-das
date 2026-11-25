# v0.33 hybrid Mamba prefix-cache guidance

This note applies to the Plugin's official vLLM v0.33.0 release line. Core
owns the prefix-cache scheduler, page allocation, hash registration and
Mamba state checkpoint policy. The Plugin may provide HCU KV-group or provider
integration, but must not recreate Core's checkpoint or replay policy.

## Public command surface

The v0.33 CLI does not expose
`--enable-mamba-fine-grained-prefix-cache`. A command containing that option
is a historical fork command and must be rejected or rewritten before it is
used for v0.33 validation. The supported starting configuration is:

```bash
--enable-prefix-caching \
--prefix-match-unit 64 \
--enable-mamba-shared-prefix-checkpoint
```

`--mamba-cache-mode` is deprecated in this release line. It can remain
parseable for compatibility, but it is not a fine-grained prefix-cache switch
and must not be used to claim that behavior.

`prefix_match_unit` is a hash/match granularity. It does not resize the
physical attention page or the hybrid Mamba manager page. Resolve the actual
`MambaSpec.block_size` after model and speculative configuration are loaded;
the value can change with MTP/EAGLE allocation and topology. Keep the match
unit only when every prefix-cacheable KV group has a block size divisible by
it. A 64-token attention kernel page therefore does not by itself make 64 a
valid match unit.

KV precision follows a separate platform capability check. In the current
gfx936 v0.33 environment, `fp8_e5m2` was accepted for the Qwen3.8-27B TP2
route, while `fp8_e4m3` was rejected by the official runtime's gfx92a/gfx938
gate. Record this result with the target hardware and do not generalize one
FP8 format's admission to the other.

For Qwen3.8-27B, 64 is the validated starting value used by the local BF16
TP2 smoke. The observed manager page may be 832 or 1600 depending on the
resolved model/speculative layout. Record the page from startup/runtime
introspection in the validation artifact; never hard-code it in Plugin code.

## Validation contract

The shared-prefix flag alone does not prove a junction checkpoint. The frozen
v0.33 Core path additionally requires an applicable EAGLE/MTP cache group,
partial hash hits and no reprefillable tokens. A complete validation probe
uses three requests:

1. An owner request seeds the shared prefix.
2. A different suffix causes the junction state to be materialized.
3. A sibling suffix consumes that junction.

Record the hit delta for each request, the resolved manager page and the
effective `prefix_match_unit`. The sibling hit must be divisible by the match
unit and not divisible by the manager page. Repeating one identical prompt or
hitting only complete physical pages proves aligned reuse, not fine-grained
junction reuse. The shared prefix must cross at least one complete manager
page; shorter probes can correctly report zero hits.

Run the probe with the requested default Graph mode, attention backend, KV
dtype and MTP settings. Compare generated text with a cold request. Keep
parameter acceptance, nonzero cache hits, output equality, MTP acceptance and
HumanEval accuracy as separate gates. A healthy HTTP response or successful
Graph capture alone is insufficient.

## Review boundary

When vLLM already implements the required option or checkpoint behavior,
reuse that implementation and keep the Plugin change limited to its public
extension boundary. A new Plugin wrapper around `MambaManager.cache_blocks`,
manual replay re-keying or a forced scheduler block size is a duplicate owner
unless a target-specific ABI gap is demonstrated. Any such gap needs a focused
CPU contract, installed integration test and the owner/junction/sibling
device evidence above before an MR is marked ready.
