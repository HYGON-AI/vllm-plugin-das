# Frozen HYV4-VL PD validation helpers

These are copies of the opt-in external experiment helpers used on 2026-11-08.
Only the upstream proxy snapshot's final blank line was removed; the other
files retain their original bytes. They are not imported by the production plugin. See
[commands, prerequisites and results](../../../docs/hy4_vl_pd_serving.md).

The scope is the fixed HYV4-VL W4A8 checkpoint, PCP16/EP16, 77 MoE layers,
256 logical experts, static R0/R16 and eager/no-MTP execution. Paths, checkpoint
hashes and the native EPLB policy hash are deliberately pinned. These helpers
are an experiment snapshot, not a general calibration API. `SHA256SUMS.json`
records the exact published helper bytes.

Calibration needs CPU torch/numpy plus the native serving environment; clients
need requests, httpx, Pillow and pyarrow. Use the original matching DTK/vLLM
environment for server probes. The data, corpus, provenance, captures, model
weights and raw responses are external inputs and are not shipped here.
Do not substitute synthetic load or old text-only maps for real image routing.

`measured_proxy_ka1.py` deliberately uses the original run directory and loads
the adjacent `mooncake_connector_proxy_ka1.py` from that directory. The latter
is the upstream Mooncake proxy snapshot with keepalive=1 and retries=0.
Stage both files at the documented path to reproduce the measured transport.

Diagnostics are enabled explicitly by worker extension/middleware CLI options;
their HTTP controls are loopback-only. Counts include executed PCP padding.
Stop every capture and check all workers before a timed benchmark. They are
neither FLOP counts nor utilization measurements. See
[calibration contract](../../../docs/hy4_vl_pd_calibration.md).

Offline regression checks, without network or accelerator use:

```bash
python -m unittest discover -s tools/hy_v4/vl_pd_validation -p 'test_vl_*.py'
```
