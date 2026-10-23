# SPDX-License-Identifier: Apache-2.0
"""CPU PLE orchestration contracts without global custom-op registration."""

import functools
import importlib.util
from pathlib import Path

import pytest


@functools.lru_cache(maxsize=1)
def load_ple_layer_contract():
    import vllm.utils.torch_utils as torch_utils

    original = torch_utils.direct_register_custom_op
    owned_ops = {"qwen4_exp_amd_ple_ngram_embedding", "qwen4_exp_ple_short_conv"}

    def register(*args, **kwargs):
        name = kwargs.get("op_name", args[0] if args else None)
        if name not in owned_ops:
            return original(*args, **kwargs)
        # These tests call Python orchestration methods with explicit kernel
        # doubles. Registering the mirrored schemas clashes with upstream PLE.

    path = Path(__file__).resolve().parents[2] / "vllm_hcu/models/qwen4_exp/amd/ple_layer.py"
    spec = importlib.util.spec_from_file_location("hcu_ple_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch_utils, "direct_register_custom_op", register)
        spec.loader.exec_module(module)
    return module
