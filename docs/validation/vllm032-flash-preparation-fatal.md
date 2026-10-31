# Flash preparation, fatal publication and fork regression

This follow-up to the independent Flash public-registry owner changes only
`flash_activation.py`, its CPU contracts, and this validation note. It retains
the official attention registry and selector. It adds no quantization, W4,
FLA, model-load lifecycle, numerical kernel, or shared test-fixture change.

The owner marks the entire locked activation attempt as in progress, including
backend/dependency preparation. A child forked during preparation must refuse
inherited incomplete state before re-entering provider imports. The parent can
finish its original attempt normally. The owner classifies the public fatal
exception chain before formatting an ordinary preparation error; a dependency
that signals an unsafe publication cannot become an optional clean failure.
The original ordinary Exception/BaseException boundary is otherwise retained.
There is no rollback, recovery/reset API, or global registry replacement.

The minimum composition is Adapter PR21 plus corrected PR23, both on PR17.
The independently validated Adapter snapshot is
`39662d9c1dd116b8c728b70abd80040416760081`, which includes PR23
`09bd79a77019a3f6848e8e534946a70a2962489b` and does not include or require
PR19's model-load hooks. These revisions identify validation artifacts; they
are not exact-wheel admission requirements. The Plugin uses the public
`ExtensionCommitError`, `is_fatal_extension_error`, API1 health callback/sink,
and existing contexts/reports. It does not inspect private Adapter registries.

Independent CPU reproduction on the published Plugin baseline
`2e2c862ac07ca53f88d0f1f8a6e0f2a4bcb150ab`:

- Fork inside actual activation preparation: child incorrectly accepts.
- Real Torch dispatcher schema declaration followed by public fatal dependency
  error with a broken RuntimeError diagnostic: fatal/latch assertion fails.
- The same declaration with a KeyboardInterrupt diagnostic: typed fatal is lost.

All three new cases fail on that baseline. With the exact owner/test bytes from
`66858e8d72f99bc9fef519a5e7e87a660fb5cff6`, all 38 Flash activation CPU
contracts pass. The schema declaration is real; dependency failure and provider
availability are explicit CPU fixtures. No kernel execution is claimed.

Release verification builds ordinary wheels from clean Git archives into new
isolated environments. Source checks explicitly select the two source roots;
installed checks clear PYTHONPATH, use an outside-repository cwd, and prebind
installed modules before collection. Record source-only exclusions explicitly.
Exercise all eleven cases in `test_flash_public_selector.py`, actual installed
metadata/official general-plugin loading, clean optional and required rejection,
commit-fatal and fork/spawn refusal. Bind Git package bytes, complete wheel
payloads, installed RECORD, and parent/spawn origins to the immutable artifacts.
Keep the Adapter-only metadata fixture in a separate environment without the
production Plugin rather than filtering discovery. CPU acceptance does not
transfer any previous HCU, Graph, numerical, performance or model evidence.
