"""Opt-in Qwen GDN decode providers; importing this package activates nothing.

Activation must register_compile_factors(vllm.envs), then call
install(architecture=<Adapter's detected HCU architecture>) before compilation.
Keep flags fixed for a compiled model's lifetime. See provider.py for the
temporary alias boundary and GPU gates still required before enabling it.
"""

from .provider import (
    COMPILE_FACTOR,
    FlaInstallError,
    effective_compile_factor,
    install,
    is_requested,
    register_compile_factors,
    uninstall,
)

__all__ = [
    "COMPILE_FACTOR",
    "FlaInstallError",
    "effective_compile_factor",
    "install",
    "is_requested",
    "register_compile_factors",
    "uninstall",
]
