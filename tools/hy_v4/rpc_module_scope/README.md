# Legacy RPC copy-module scope workaround

This optional `LD_PRELOAD` shim avoids loading unused Mooncake copy modules in
the validated Hy4 1P1D configuration: TP1, PCP16/EP16 on prefill and DP16/EP16 on
decode. It leaves the installed Mooncake wheel unchanged and is not enabled by
installing vllm-plugin-das.

The legacy RPC wheel loads `MCCopyKernel` on every visible GPU in every process
when constructing its HIP transport. In this configuration, only the canonical
PCP replica initiates the KV writes; the other prefill workers and the passive
decode receivers do not need copy modules. The shim intercepts only
`mooncake::HipTransport::loadCopyModule()` and skips that initialization in those
processes. The selected sender still calls the original loader, including its
all-device initialization.

## Compatibility

The original workaround was validated with
`mooncake-transfer-engine-rpc==0.3.13.post1+dtk26041.2609222249.g466e70`, CPython
3.10, and the library installed at
`/usr/local/lib/python3.10/dist-packages/mooncake/libtransfer_engine.so`.
Both this path and the C++ symbol are fixed in the source. Other wheel versions,
Python versions, installation paths, and transport implementations require
separate validation.

`pcp0` tests the current **visible HIP device number**, not the distributed rank.
Use it only when the canonical sender is on visible device 0 and the other
prefill processes use the other visible devices. Do not use it for a topology
with additional senders, or assume it selects rank 0 when each process sees only
one GPU. Use `target` only for a process that never initiates GPU copies; a
decode process that pulls KV with READ requests is not such a receiver.

This is a workaround for unnecessary module initialization. It does not
establish the cause of an arbitrary startup hang or shared-memory timeout.

## Build

Run inside the Linux service container, from the repository root:

```bash
g++ -shared -fPIC -O2 -Wall -Wextra -Werror \
  tools/hy_v4/rpc_module_scope/rpc_module_scope.cpp \
  -o tools/hy_v4/rpc_module_scope/rpc_module_scope.so -ldl

export RPC_SCOPE_SO="$(realpath tools/hy_v4/rpc_module_scope/rpc_module_scope.so)"
```

Build or copy the shared object into both service containers. The path must be
absolute and visible inside each container. Keep the existing RPC, Fabric KV
allocation, and service settings; the following settings are additional.

## Enable for the validated topology

In the **prefill** launch shell:

```bash
export LD_PRELOAD="$RPC_SCOPE_SO${LD_PRELOAD:+:$LD_PRELOAD}"
export MC_RPC_COPY_MODULE_SCOPE=pcp0
# Start the existing prefill command from this shell.
```

In the **decode** launch shell:

```bash
export LD_PRELOAD="$RPC_SCOPE_SO${LD_PRELOAD:+:$LD_PRELOAD}"
export MC_RPC_COPY_MODULE_SCOPE=target
# Start the existing decode command from this shell.
```

Set these before starting vLLM so worker processes inherit them. They can be
placed in the corresponding launch scripts. Do not preload this shim for the
RDMA or Hylink benchmark profiles. To disable it, remove this shared object
from the launch environment's `LD_PRELOAD`, unset `MC_RPC_COPY_MODULE_SCOPE`,
and restart the services, preserving any unrelated preload libraries.

| Scope | Behavior when the shim is preloaded |
| --- | --- |
| Unset or `full` | Call the original loader in every process. |
| `pcp0` | Call the original loader on visible device 0; skip on other devices. |
| `target` | Skip the loader in every process. |

Setting the scope alone has no effect without `LD_PRELOAD`. Invalid scope
values or failure to resolve the original library/symbols abort the process.

## Logs and prior validation

Each intercepted call logs `RPC_MODULE_SCOPE`, PID, device, scope, and
`action=original_module_load` or `action=skip_unused_module` to stderr.
For the validated 16-worker-per-host layout, prefill logs one original load and
15 skips; decode logs 16 skips. All workers still initialize their HIP
transports. A loader log alone does not prove successful KV transfer.

The unchanged shim was used in three independent RPC runs, each with a service
restart, 146 warmup requests, and 2920 measured requests. All three completed
with zero request errors. These results apply to the wheel and topology above;
they are not validation of other environments.
