# Split Adapter distribution compatibility

Adapter main now builds pure sgl-kernel0.0.0+hcu.dev0 and hcu-adapter0.5.22+hcu.1, moving platform/general entry points and hcu_adapter namespace to the latter. Plugin still directly requires sgl-kernel>=0.1.0 and its install verifier assumes that distribution owns Adapter. That dependency conflicts with the new Adapter kernel pin and misattributes modern host ownership.

Use hcu-adapter>=0.1.0 as the host dependency; its declared kernel dependency owns kernel version pairing. Runtime admission stays public API/capability/release-line based, with no new exact-wheel gates or numerical changes. The installed verifier and CI guard resolve either modern hcu-adapter ownership or a genuine legacy combined sgl-kernel owner via RECORD, rejecting duplicate host owners and pure-kernel-only installations. Share a small stdlib ownership helper loaded from an explicit tools file path; no framework imports to find metadata. Update installation/migration instructions and preserve Core/Plugin entrypoint boundaries.

Alternative keeping sgll>=0.1 is incompatible with current upstream pin. Installing a new host over the legacy combined wheel risks overlapping RECORD ownership; use a fresh verification environment and document the upstream clean migration rather than modify existing runtimes.

Tests first: dependency/wheel metadata, modern/legacy ownership, mixed-owner refusal and existing source-shadow guards. Build actual split wheels from a pinned upstream checkout, inspect exact payload/entry points and validate install/discovery in isolation. Preserve the original combined-wheel proof. Recheck fresh target and equivalent requests before MR; numerical/model results are not claimed by metadata tests.
