# Installed compatibility module ownership

The v0.33 boundary migration moves five implementations into Adapter while
preserving their historical Plugin import paths. Those facades intentionally
bind the Adapter module object in `sys.modules`, so writes and reads reach the
same state. The installed pytest guard previously required every Plugin-prefixed
module to appear in the Plugin RECORD. The current release therefore stopped
during collection on `hcu_vllm_das.quantization.dsv41_channel_fp8`, even when both
packages were independently installed from non-editable wheels.

The guard now recognizes these exact bindings:

| Plugin compatibility path | Adapter implementation |
| --- | --- |
| `hcu_vllm_das.dsv41_activation` | `hcu_adapter.dsv41_activation` |
| `hcu_vllm_das.ct_w4a8_activation` | `hcu_adapter.ct_w4a8_registration` |
| `hcu_vllm_das.dsv41_dspark_registration` | `hcu_adapter.dsv41_dspark_registration` |
| `hcu_vllm_das.quantization.dsv41_channel_fp8` | `hcu_adapter.dsv41_channel_fp8` |
| `hcu_vllm_das.quantization.dsv41_int8_linear` | `hcu_adapter.dsv41_int8_linear` |

An exception requires the declared facade in the installed Plugin manifest,
the exact destination file in the installed Adapter manifest, the canonical
Adapter module name, and identity with its canonical `sys.modules` binding.
Declared aliases cannot use the namespace exemption or a Plugin-recorded file
to bypass those checks, including when their `__file__` is absent.
Other Plugin modules retain their original owner checks. The guard neither
imports a replacement implementation nor accepts arbitrary Adapter files as
Plugin modules. Production packages, discovery, kernels and numerical
tolerances are unchanged.

Run the stdlib CI contracts from an interpreter with inherited third-party
paths removed:

```sh
env -u PYTHONPATH -u PYTHONHOME HCU_ADAPTER_HOST=none \
  python -I -S -B -m unittest discover -s tests/ci -v
```

The new negatives cover missing facade/destination records, missing canonical
bindings, another module object at the same path, another owned Adapter module,
an undeclared alias, a changed canonical name, and a foreign checkout containing
the same bytes. Existing source-shadow and duplicate-host rejection tests stay
in the suite. Run the complete installed portable tier against independently
prepared v0.33 Core/Adapter wheels as well:

```sh
python -I tools/run_v033_ci.py --tier portable --junit v033-portable.xml
```

Collection ownership, portable behavior, native device tests and model quality
are separate evidence. In particular, this CPU guard correction does not supply
a prepared CI image, change Worker execution ownership or qualify any device.

The complete portable selection also needs the real upstream JIT metadata.
Core's driver admission treats `-1` masks as an incompatible Triton runtime
when no driver is active and substitutes identity decorators. Empty HIP, ROCR
and CUDA masks still hide devices while allowing Core's deferred-driver import
path. This preserves the actual installed FP4 function and its original audit;
no JIT compilation or device execution is added. Check both visible device count
and `torch.cuda.is_initialized()` when qualifying a new runtime.

The four-process Engram Gloo regression explicitly selects the official
`CpuPlatform` inside its spawned children. Disabling the device communicator
alone does not stop `GroupCoordinator` from mapping ranks through HCU device
visibility. The test retains four real processes, real native group constructors,
both rank partitions, distinct groups, all-reduce assertions and the original
timeouts; it also checks CPU group devices and the parent's unchanged platform.
