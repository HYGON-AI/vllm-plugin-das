import os
import subprocess
import sys

import torch

from vllm_hcu.patch.platform.core_fix import patch_layer_name


def test_layer_name_value_member_is_graph_safe():
    from torch._library.opaque_object import (
        MemberType,
        get_opaque_obj_info,
    )
    import vllm.utils.torch_utils as torch_utils

    type_info = get_opaque_obj_info(torch_utils.LayerName)
    original_members = type_info.members.copy()
    type_info.members.pop("value", None)

    try:
        patch_layer_name.apply_to_module(torch_utils)
        patch_layer_name.apply_to_module(torch_utils)

        assert type_info.members["value"] is MemberType.USE_REAL

        def read_value(layer_name):
            return layer_name.value

        compiled = torch.compile(read_value, backend="eager", fullgraph=True)
        assert (
            compiled(torch_utils.LayerName("model.layers.0"))
            == "model.layers.0"
        )
    finally:
        type_info.members.clear()
        type_info.members.update(original_members)


def test_vllm_import_registers_layer_name_without_cycle():
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "hcu"
    script = """
import vllm
from torch._library.opaque_object import MemberType, get_opaque_obj_info
from vllm.utils.torch_utils import LayerName
assert get_opaque_obj_info(LayerName).members["value"] is MemberType.USE_REAL
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
