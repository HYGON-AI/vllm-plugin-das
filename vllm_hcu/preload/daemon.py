# SPDX-License-Identifier: Apache-2.0
"""Opt-in launcher for the installed vLLM weight-cache daemon.

This launches full model loading and must not be used during single-layer-only
experiments. The explicit flag prevents accidental full-model startup.
"""
from __future__ import annotations

import sys

_FULL_MODEL_FLAG = "--enable-full-model-preload"


def _prepare():
    from vllm.plugins import load_general_plugins
    from vllm.platforms import current_platform
    if not current_platform.is_cuda_alike():
        raise RuntimeError("weight-cache IPC requires a CUDA/ROCm-compatible platform")
    load_general_plugins()
    from vllm.model_executor.model_loader.weight_cache import daemon
    from vllm_hcu.runtime_compat.weight_cache import (
        install_expert_export_metadata, install_key_fingerprint,
        install_ipc_quant_support, install_per_client_export, install_uuid_daemon,
    )
    install_key_fingerprint(daemon)
    install_per_client_export(daemon)
    install_expert_export_metadata(daemon)
    install_uuid_daemon(daemon)
    from vllm.model_executor.model_loader.weight_cache import protocol
    install_ipc_quant_support(protocol)
    return daemon


def run_daemon(*args):
    """Spawn target: initialize plugin and repairs in each fresh process."""
    daemon = _prepare()
    original = daemon._run_daemon
    if original is run_daemon:
        raise RuntimeError("weight-cache daemon spawn target was not isolated")
    original(*args)


def main():
    if _FULL_MODEL_FLAG not in sys.argv[1:]:
        raise SystemExit(
            "This launcher loads the FULL model. It is prohibited during the "
            "current single-layer-only phase. When full-model loading is "
            f"authorized, explicitly pass {_FULL_MODEL_FLAG}."
        )
    sys.argv.remove(_FULL_MODEL_FLAG)
    daemon = _prepare()
    from vllm_hcu.preload.pp_daemon import argv_requests_pp, main as pp_main
    if argv_requests_pp(sys.argv[1:]):
        return pp_main(daemon)
    # Spawn imports this module by name and executes run_daemon in every rank.
    daemon._run_daemon = run_daemon
    daemon.main()


if __name__ == "__main__":
    main()
