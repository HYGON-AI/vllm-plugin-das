# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""INT8 quantization must remain callable after an FX cache process restart."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
    _per_token_quant_int8,
)


@pytest.mark.parametrize("shape", [(5, 3072), (2, 3, 6144)])
def test_int8_quant_fake_preserves_token_dimensions(shape):
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16)
        quantized, scales = _per_token_quant_int8(x)
        assert quantized.shape == x.shape
        assert quantized.dtype == torch.int8
        assert scales.shape == (*x.shape[:-1], 1)
        assert scales.dtype == torch.float32
        assert quantized.device == scales.device == x.device


@pytest.mark.isolated_process
def test_int8_quant_fx_cache_loads_in_fresh_process(tmp_path):
    # A CPU LightOp stand-in keeps this serialization regression independent
    # of accelerator availability. Hardware numerical tests exercise LightOp.
    script = tmp_path / "roundtrip.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from types import ModuleType

            import torch
            from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            from vllm.compilation.caching import VllmSerializableFunction
            from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
                _per_token_quant_int8,
            )

            def reference(x):
                scales = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127
                return (x.float() / scales).round().clamp(-127, 127).to(torch.int8), scales

            lightop = ModuleType("lightop")
            lightop.__path__ = []
            quant = ModuleType("lightop.quant")
            quant.per_token_quant_int8 = reference
            lightop.quant = quant
            sys.modules["lightop"] = lightop
            sys.modules["lightop.quant"] = quant

            root = Path(sys.argv[1])
            x = torch.arange(2 * 3 * 64, dtype=torch.float32).reshape(2, 3, 64)
            assert not kernel_side_table.id_to_kernel

            if sys.argv[2] == "save":
                def backend(graph, inputs):
                    assert "vllm_hcu.per_token_quant_int8" in graph.code
                    assert "triton_kernel_wrapper" not in graph.code
                    data = VllmSerializableFunction.serialize_graph_module(graph)
                    (root / "graph.pkl").write_bytes(data)
                    return graph.forward

                actual = torch.compile(
                    _per_token_quant_int8, backend=backend, fullgraph=True
                )(x)
            else:
                graph = VllmSerializableFunction.deserialize_graph_module(
                    (root / "graph.pkl").read_bytes(),
                    FakeTensorMode(shape_env=ShapeEnv()),
                )
                graph.recompile()
                assert not kernel_side_table.id_to_kernel
                actual = graph(x)

            expected = reference(x)
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1], expected[1])
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "VLLM_PLUGINS": "__disabled__"}
    # The script lives outside the checkout. Prefer the PR source over an older
    # installed wheel even if CI puts site-packages first in PYTHONPATH.
    source_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_root, env.get("PYTHONPATH")) if path
    )
    for mode in ("save", "load", "load"):
        result = subprocess.run(
            [sys.executable, str(script), str(tmp_path), mode],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
