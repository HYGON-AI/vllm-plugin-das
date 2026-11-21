# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Isolated startup contract; never imports an inference engine or uses a GPU."""
import os
import sys
from types import ModuleType

from vllm_hcu.patch import locateanything as adapter
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator, LateModuleReplacementError
from vllm_hcu.patch.runtime_state import PatchRegistry


def main():
    contract = adapter.fast_source_contract()
    names = ["vllm." + name[:-3].replace("/", ".") for name in contract]
    assert not any(name in sys.modules for name in names)
    os.environ["VLLM_HCU_LOCATEANYTHING_MODE"] = "fast"
    # Source identity is independently tested; this fixture only checks cold
    # registration and never loads or replaces any target implementation.
    adapter.validate_fast_sources = lambda contract: None
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    adapter.register_locateanything_exchanges(coordinator)
    first = coordinator.registrations()
    adapter.register_locateanything_exchanges(coordinator)
    assert coordinator.registrations() == first and len(first) == 7
    assert not any(name in sys.modules for name in names)

    sys.modules[names[0]] = ModuleType(names[0])
    try:
        adapter.register_locateanything_exchanges(ExactImportCoordinator(registry=PatchRegistry()))
    except LateModuleReplacementError:
        pass
    else:
        raise AssertionError("Late imports must still fail closed")
    print("cold registration, reentry and late-import rejection passed")


if __name__ == "__main__":
    main()
