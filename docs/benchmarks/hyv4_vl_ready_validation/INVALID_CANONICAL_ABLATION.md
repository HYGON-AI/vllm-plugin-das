# Excluded diagnostic intervention

The three requests in `topk-canonical-ablation.json` completed, but their
`selector=canonical` label describes the requested intervention, not an
executed sort. The already-installed probe closure did not contain that branch;
editing its source file did not refresh the closure. These requests therefore
used the original native selector and are excluded from conclusions about
sorting indices. The external manifest retains their tensor files for audit.

`canonical_probe.py` preserves the attempted invocation. The archived
`topk_probe_install.py` reconstructs the original installed closure; the later
source-file edit had inserted an `if selector == 'canonical'` sort after the
original native call, inside the installation guard, and was ineffective on
the already-loaded process. No performance or correctness result is claimed
for canonical sorting.
