# Frozen HYV4-VL PD validation helpers

The files listed in `SHA256SUMS.json` are copies of the opt-in external experiment helpers used on 2026-11-08.
Only the upstream proxy snapshot's final blank line was removed; the other
files retain their original bytes. They are not imported by the production plugin. See
[commands, prerequisites and results](../../../docs/hy4_vl_pd_serving.md).

The scope is the fixed HYV4-VL W4A8 checkpoint, PCP16/EP16, 77 MoE layers,
256 logical experts, static R0/R16 and eager/no-MTP execution. Paths, checkpoint
hashes and the native EPLB policy hash are deliberately pinned. These helpers
are an experiment snapshot, not a general calibration API. `SHA256SUMS.json`
records those 12 frozen files' exact bytes. New corrected proxy code and its
tests are additional files and are not claimed to have produced the old data.

Calibration needs CPU torch/numpy plus the native serving environment; clients
need requests, httpx, Pillow and pyarrow. Use the original matching DTK/vLLM
environment for server probes. The data, corpus, provenance, captures, model
weights and raw responses are external inputs and are not shipped here.
Do not substitute synthetic load or old text-only maps for real image routing.

`measured_proxy_ka1.py` deliberately uses the original run directory and loads
the adjacent `mooncake_connector_proxy_ka1.py` from that directory. The latter
is the upstream Mooncake proxy snapshot with keepalive=1 and retries=0.
Stage both files at the documented path to reproduce the measured transport.

For new runs, use `managed_proxy_ka1.py` with the adjacent
`proxy_task_lifecycle.py` and `mooncake_connector_proxy_ka1.py`. This corrected
entry point owns both P/D tasks, propagates failures, joins cancelled tasks,
and explicitly closes the response iterator on ASGI disconnect. Set
`HY4_PD_PROXY_TIMEOUT_SECONDS` to a finite positive upstream wait budget
(default 300 seconds). FastAPI, Starlette, AnyIO and httpx are required.
P/D remain concurrent, but the failure/cancellation semantics have changed;
this entry point has local CPU/ASGI tests and has not been rebenchmarked on HCU.
The old measured wrapper remains an unchanged historical snapshot, including
its cancelled-request phase-log limitations.

The corrected entry point also owns its startup initializer. It retries both
P health and bootstrap queries after transient HTTP/connection/JSON failures,
using five-second request timeouts and a one-second retry interval. It stays
unready until bootstrap supplies nonempty engine IDs on contiguous DP ranks.
Shutdown cancels and joins initialization before closing the HTTP clients.
This does not retry model inference requests or refresh engine IDs after a
ready producer restarts; restart the proxy when replacing P.

Diagnostics are enabled explicitly by worker extension/middleware CLI options;
their HTTP controls are loopback-only. Counts include executed PCP padding.
Stop every capture and check all workers before a timed benchmark. They are
neither FLOP counts nor utilization measurements. See
[calibration contract](../../../docs/hy4_vl_pd_calibration.md).

Offline regression checks, without network or accelerator use:

```bash
python -m unittest discover -s tools/hy_v4/vl_pd_validation -p 'test_vl_*.py'
python -m pytest tests/models/hy_v4/test_vl_proxy_lifecycle.py tests/models/hy_v4/test_vl_proxy_asgi.py
```
