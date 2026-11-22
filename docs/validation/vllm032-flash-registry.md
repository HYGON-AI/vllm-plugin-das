# Independent Flash registry owner

The `das_hcu_flash` entry point publishes `HcuFlashAttentionBackend` using the
official `AttentionBackendEnum.FLASH_ATTN` / `register_backend` API. Readback
uses public `get_path`, `is_overridden`, and `get_class`; no private registry
table or platform architecture disguise is used. Official vLLM configuration
validation remains in force.

Preparation resolves the existing PR326 writer (`e5m2_cache`) and vendor ABI
before publication. It does not import W4, FLA, model-load hooks, RMS cache
extensions or the later E5M2 scale provider. Numerical provider bodies remain
PR326's implementation. No additional compute feature is implied.

Foreign overrides are refused before publication. Repeated activation verifies
state; fork requires explicit activation. The API1 synchronous health callback
checks actual registry identity, allowing Adapter's generic report/PID binding
to reject clear/foreign drift before delegating to the official selector.
Public concurrent registry mutation has no CAS and is not claimed atomic.

After attempted publication, failure is commit-fatal. Store the owner failure
sentinel BEFORE formatting or cleanup, safely render diagnostics even when
`__str__` raises BaseException, and retain the original typed cause. Caught
errors, direct retry and fork cannot heal the owner. The same rule covers
health, state readback and registry-availability failures.

CPU gates: lifecycle tests plus actual target selector clear/foreign/config,
optional/required, duplicate owner, early import/unpickle, fork/spawn; and the
real registry write-then-broken-`__str__` fault. Record source vs installed
origins, wheel RECORD and new immutable evidence. These gates do not establish
HCU numerical, performance or model accuracy support.
