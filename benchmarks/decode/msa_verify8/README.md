# MiniMax M3.1 Verify8 page selection

The workspace builds fixed-width Stage1 metadata once. Each layer still reads
the current causal ends and page table using the deployed FP8 decode scanner.
Stage2 uses the same full-length BlockTopkSort and ascending page-ID order as
the reference. It masks unwritten pages, forces the last causal page, and emits
the planner's query-major tensor in one kernel.

Set `MSA_VENDOR_SOURCE` to the `vendor_source` directory of the deployed
private MSA build. Build in the HCU container with Python 3.12.

```sh
MAX_JOBS=2 python3 setup.py build_ext --build-lib build_lib
```

Put `build_lib`, the reference MSA build, and the plugin checkout on
`PYTHONPATH`. Run `python3 test_verify8_msa.py` with one gfx938 GPU visible.
The fixture uses synthetic scores and FP8 data, and compares exact page IDs
for eager calls and graph replays. Add `--measure` on an isolated card for
the paired graph timing.

Launch the offline Verify8 runner with `HCU_MINIMAX31_MSA_FUSED_VERIFY8=1`
and put the candidate `build_lib` on `PYTHONPATH` to select the new path.
The workspace belongs to one runtime. The caller schedules the page planner
on the same stream before reusing its output buffer.
