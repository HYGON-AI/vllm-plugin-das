# SPDX-License-Identifier: Apache-2.0
"""Step5 runtime ownership, adapter lifecycle and CPU behavior contracts."""

import importlib
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal
import numpy as np
import pytest
import torch

ADAPTERS = (
    "vllm_hcu.patch.platform.core_fix.patch_deep_gemm_foundation",
    "vllm_hcu.patch.platform.core_fix.patch_step5_envs",
    "vllm_hcu.patch.platform.core_fix.patch_step5_config_exports",
    "vllm_hcu.patch.platform.core_fix.patch_step5_config_registry",
    "vllm_hcu.patch.platform.core_fix.patch_step5_speculative_config",
    "vllm_hcu.patch.platform.core_fix.patch_step5_splitting_ops",
    "vllm_hcu.patch.platform.core_fix.patch_step5_compile_prefix",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_block_tables",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_model_state",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_mtp_heads",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_multi_module_mtp",
    "vllm_hcu.patch.platform.core_fix.patch_step5_ep_graph_capture",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_moe_combine",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_token_fp8",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_skinny_lm_head",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_eplb_communicator",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_eplb_transfer",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_eplb_async_worker",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_eplb_state",
    "vllm_hcu.patch.worker.framework_opt.patch_step5_deep_gemm_api",
)


@pytest.mark.parametrize("name", ADAPTERS)
def test_exact_target_and_repeat_application(name, request):
    adapter = importlib.import_module(name)
    with pytest.raises(RuntimeError, match="expected module"):
        adapter.apply_to_module(ModuleType("unrelated.module"))
    module = importlib.import_module(adapter.TARGET_MODULE)
    if adapter.PATCH_ID.startswith("worker.framework_opt.step5_eplb"):
        # Offline EPLB tests patch the upstream module globals. Restore the
        # EPLB methods/aliases so those tests do not inherit these replacements.
        missing = object()
        originals = []
        for symbol in (*adapter.TARGETS, f"{module.__name__}.{adapter._MARKER}"):
            parts = symbol.removeprefix(module.__name__ + ".").split(".")
            owner = module
            for part in parts[:-1]:
                owner = getattr(owner, part)
            original = inspect.getattr_static(owner, parts[-1], missing)
            originals.append((owner, parts[-1], original))

        def restore():
            for owner, attribute, original in reversed(originals):
                if original is missing:
                    if hasattr(owner, attribute):
                        delattr(owner, attribute)
                else:
                    setattr(owner, attribute, original)

        request.addfinalizer(restore)
    adapter.apply_to_module(module)
    from vllm_hcu.runtime_compat.step5_patch_contract import resolve

    before = tuple(resolve(module, t) for t in adapter.TARGETS)
    assert adapter.apply_to_module(module) is False
    assert all(resolve(module, t) is v for t, v in zip(adapter.TARGETS, before))


def test_signature_drift_and_stale_marker_fail_closed():
    from vllm_hcu.patch.platform.core_fix import patch_step5_speculative_config as a

    m = ModuleType(a.TARGET_MODULE)

    class Config:
        hf_config_override = staticmethod(lambda incompatible: None)

    m.SpeculativeConfig = Config
    m.MTPModelTypes = Literal["step3p5_mtp"]
    with pytest.raises(RuntimeError, match="signature"):
        a.apply_to_module(m)
    assert not hasattr(m, a._MARKER)
    Config.hf_config_override = staticmethod(lambda hf_config: hf_config)
    assert a.apply_to_module(m)
    Config.hf_config_override = staticmethod(lambda hf_config: hf_config)
    with pytest.raises(RuntimeError, match="Stale"):
        a.apply_to_module(m)


def test_step5_mtp_config_keeps_quantization_and_delegates_other_models():
    from vllm_hcu.patch.platform.core_fix import patch_step5_speculative_config as a

    m = ModuleType(a.TARGET_MODULE)
    calls = []

    class Config:
        @staticmethod
        def hf_config_override(hf_config):
            calls.append(hf_config)
            return hf_config

    class HF(SimpleNamespace):
        def update(self, values):
            self.__dict__.update(values)

    m.SpeculativeConfig = Config
    m.MTPModelTypes = Literal["step3p5_mtp"]
    a.apply_to_module(m)
    text = HF(model_type="step5", num_nextn_predict_layers=3)
    quant = {"quant_method": "compressed-tensors"}
    vl = HF(
        model_type="step5_vl",
        architectures=["Step5ForConditionalGeneration"],
        text_config=text,
        quantization_config=quant,
    )
    assert Config.hf_config_override(vl) is text
    assert text.architectures == ["Step5MTP"] and text.n_predict == 3
    assert text.quantization_config is quant and not calls
    other = HF(model_type="llama", architectures=["LlamaForCausalLM"])
    assert Config.hf_config_override(other) is other and calls == [other]


def test_block_mirror_matches_expansion_overwrite_row_order_and_padding():
    from vllm_hcu.patch.worker.framework_opt import patch_step5_block_tables as a

    m = ModuleType(a.TARGET_MODULE)

    class Tables:
        def __init__(self):
            self.blocks_per_kv_block = [2]
            self.block_tables = [SimpleNamespace(gpu=torch.empty(3, 16))]
            self.num_blocks = SimpleNamespace(np=np.zeros((1, 3), dtype=np.int32))

        def append_block_ids(self, req_index, new_block_ids, overwrite):
            for i, ids in enumerate(new_block_ids):
                start = 0 if overwrite else self.num_blocks.np[i, req_index]
                self.num_blocks.np[i, req_index] = (
                    start + len(ids) * self.blocks_per_kv_block[i]
                )

    m.BlockTables = Tables
    a.apply_to_module(m)
    t = Tables()
    t.enable_host_mirror()
    t.append_block_ids(1, ([3, 5],), True)
    t.append_block_ids(1, ([7],), False)
    t.append_block_ids(0, ([2],), True)
    gathered = t.gather_block_tables_np(np.array([1, 0]), 3)[0]
    assert gathered[0, :6].tolist() == [6, 7, 10, 11, 14, 15]
    assert gathered[1, :2].tolist() == [4, 5]
    assert not gathered[2].any()
    t.append_block_ids(1, ([9],), True)
    assert t.gather_block_tables_np(np.array([1]), 1)[0][0, :2].tolist() == [18, 19]


def test_metadata_extension_is_owned_without_changing_upstream_dataclass():
    from vllm.v1.attention.backend import CommonAttentionMetadata as Upstream
    from vllm_hcu.v1.step5_metadata import CommonAttentionMetadata as Owned

    assert "block_table_np" not in Upstream.__dataclass_fields__
    assert "block_table_np" in Owned.__dataclass_fields__
    q = torch.tensor([0, 2, 4], dtype=torch.int32)
    m = Owned(
        query_start_loc=q,
        query_start_loc_cpu=q,
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.zeros(2, 1, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        block_table_np=np.array([[1], [2]], dtype=np.int32),
    )
    unpadded = m.unpadded(2, 1)
    assert unpadded.num_reqs == 1
    assert unpadded.block_table_np.tolist() == [[1]]
    assert m.block_table_np.tolist() == [[1], [2]]


def test_combine_can_return_result_without_copying_output(monkeypatch):
    from vllm_hcu.model_executor.layers.fused_moe import combine_runtime as c

    output = torch.zeros(2, 3)
    expert = torch.ones(2, 3)

    class Reduce:
        def apply(self, *args):
            return expert

    c.set_finalize_combine_hook(lambda x: x + 4)
    try:
        result = c.finalize(
            SimpleNamespace(is_sequence_parallel=False),
            output,
            expert,
            None,
            None,
            False,
            Reduce(),
        )
        assert torch.equal(result, expert + 4)
        assert not output.any()
    finally:
        c.set_finalize_combine_hook(None)


def test_deep_gemm_native_signature_does_not_receive_ue8m0_keyword():
    from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_api_runtime import (
        install_api,
    )

    seen = []

    def kernel(x):
        seen.append(x)
        return x + 1

    m = SimpleNamespace(
        _import_deep_gemm=lambda: None,
        _lazy_init=lambda: None,
        get_mk_alignment_for_contiguous_layout=lambda: (_ for _ in ()).throw(
            RuntimeError()
        ),
        set_mk_alignment_for_contiguous_layout=lambda value: None,
        _get_mk_alignment_for_contiguous_layout_impl=None,
        _grouped_impl=kernel,
        _grouped_fp4_impl=kernel,
        _grouped_masked_impl=kernel,
        m_grouped_fp8_gemm_nt_contiguous=lambda x: None,
        m_grouped_fp8_fp4_gemm_nt_contiguous=lambda x: None,
        fp8_m_grouped_gemm_nt_masked=lambda x: None,
        is_deep_gemm_e8m0_used=lambda: False,
    )
    install_api(m)
    assert m.get_mk_alignment_for_contiguous_layout() == [16, 16]
    assert m.m_grouped_fp8_gemm_nt_contiguous(4) == 5 and seen == [4]


def test_eplb_uses_each_layers_weight_format(monkeypatch):
    from vllm_hcu.distributed import eplb_transfer_runtime as r

    received = []
    monkeypatch.setattr(r, "move_to_buffer", lambda **kw: received.append(kw) or None)
    monkeypatch.setattr(r, "move_from_buffer", lambda **kw: None)
    w0 = torch.empty(1, 2, dtype=torch.bfloat16)
    w1 = torch.empty(1, 4, dtype=torch.float8_e4m3fn)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    indices = torch.tensor([[0], [0]])
    r.rearrange_expert_weights_inplace(
        indices, indices, [[w0], [w1]], [torch.empty_like(w0)], group, SimpleNamespace()
    )
    assert [x["expert_weights_buffers"][0].dtype for x in received] == [
        w0.dtype,
        w1.dtype,
    ]
    assert [tuple(x["expert_weights_buffers"][0].shape) for x in received] == [
        (1, 2),
        (1, 4),
    ]


def test_step5_projection_and_attention_modules_have_plugin_ownership():
    import subprocess, sys, json, os

    # Initialize vLLM before HCU registration to avoid cold import cycles.
    code = (
        "import json;import vllm;import vllm_hcu;"
        "from vllm_hcu import hcu_platform_register_model;"
        "hcu_platform_register_model();"
        "from vllm.model_executor.models.step5_text import Step5ForCausalLM;"
        "from vllm.model_executor.layers import step4_dsa;"
        "print(json.dumps([Step5ForCausalLM.__module__,"
        "step4_dsa.__file__,vllm_hcu.__file__]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    model, backend, plugin = json.loads(result.stdout.splitlines()[-1])
    assert model == "vllm_hcu.models.step5_text"
    plugin_root = Path(plugin).resolve().parent
    assert Path(backend).resolve().is_relative_to(plugin_root), backend


def test_ep_capture_scopes_ep_buffers_and_keeps_export_identity():
    from contextlib import contextmanager
    from vllm_hcu.patch.platform.core_fix import patch_step5_ep_graph_capture as a

    events = []

    @contextmanager
    def original(device, graph_capture_context=None):
        events.append("native_enter")
        try:
            yield "capture_context"
        finally:
            events.append("native_exit")

    @contextmanager
    def capture():
        events.append("ep_enter")
        try:
            yield
        finally:
            events.append("ep_exit")

    m = ModuleType(a.TARGET_MODULE)
    m.graph_capture = original
    m._EP = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=SimpleNamespace(capture=capture))
    )
    a.apply_to_module(m)
    with pytest.raises(ValueError):
        with m.graph_capture("cpu") as ctx:
            assert ctx == "capture_context"
            events.append("body")
            raise ValueError("caller failed")
    assert events == ["native_enter", "ep_enter", "body", "ep_exit", "native_exit"]
    import os, subprocess, sys

    script = "from vllm_hcu import hcu_platform_register_model; hcu_platform_register_model(); import vllm.distributed as d; import vllm.distributed.parallel_state as p; assert d.graph_capture is p.graph_capture; assert hasattr(p, '_vllm_hcu_step5_ep_graph_capture_applied')"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compile_prefix_distinguishes_step5_layers_and_preserves_other_models():
    from vllm_hcu.patch.platform.core_fix import patch_step5_compile_prefix as a

    m = ModuleType(a.TARGET_MODULE)

    class Wrapper:
        def __init__(self, compile_prefix="", is_encoder=False):
            self._compile_prefix = compile_prefix

    def support(cls, dynamic_arg_dims, mark_unbacked_dims, enable_if, is_encoder):
        return cls

    def model_key(fn):
        return "same-forward-body"

    m.TorchCompileWithNoGuardsWrapper = Wrapper
    m._support_torch_compile = support
    m._model_hash_key = model_key
    a.apply_to_module(m)

    class Model:
        def __init__(self, *, prefix=""):
            Wrapper.__init__(self)

        def forward(self):
            pass

    Model.__module__ = "vllm_hcu.models.step5_text"
    a_model = m._support_torch_compile(Model, None, None, None, False)
    first, second = a_model(prefix="layers.0"), a_model(prefix="layers.1")
    assert first._compile_prefix == "layers.0"
    assert second._compile_prefix == "layers.1"
    assert m._model_hash_key(first.forward) != m._model_hash_key(second.forward)

    class Other:
        def forward(self):
            pass

    assert m._model_hash_key(Other().forward) == "same-forward-body"
    assert Wrapper()._compile_prefix == ""


def test_router_and_launcher_import_without_legacy_model_tree():
    import os, subprocess, sys

    script = "from vllm_hcu import hcu_platform_register_model; hcu_platform_register_model(); import vllm_hcu.ops.step5.router_bias as router; import vllm_hcu.ops.step5.router_gemm; import vllm_hcu.ops.step5.fp8_dense; assert router.KernelLaunchSpec.__module__ == 'vllm_hcu.ops.step5.driver_cache'; import sys; assert not any(n.startswith('vllm.models.step5') for n in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_driver_launcher_reuses_matching_signature_and_separates_shapes(monkeypatch):
    from vllm_hcu.ops.step5 import driver_cache as d

    calls = []

    def make_call(kernel, meta, best_config):
        return lambda grid, launch, *args: calls.append(("cached", args))

    monkeypatch.setattr(d, "_build_driver_call", make_call)
    registry = d.DriverCacheRegistry()
    launcher = registry.get_launcher("router")
    assert launcher is registry.get_launcher("router")
    assert launcher is not registry.get_launcher("other")

    def spec(rows):
        def build(grid):
            calls.append(("build", rows))
            return object()

        return d.KernelLaunchSpec(
            "router",
            (rows,),
            lambda: (rows,),
            build,
            signature_fn=lambda grid, meta, args: (args[0],),
            enforce_driver_launch_on_register=True,
        )

    launcher(spec(4))
    launcher(spec(4))
    launcher(spec(8))
    assert calls == [
        ("build", 4),
        ("cached", (4,)),
        ("cached", (4,)),
        ("build", 8),
        ("cached", (8,)),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Cold native MoE imports query GPU architecture during initialization",
)
def test_quantization_import_before_worker_dispatch_keeps_combine_hook_owned():
    import os, subprocess, sys

    env = os.environ.copy()
    env["VLLM_HCU_EP_FUSED_COMBINE"] = "1"
    script = "from vllm_hcu.patch.platform import apply_platform_patches; apply_platform_patches(); import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors; from vllm_hcu.model_executor.layers.fused_moe import combine_runtime; assert callable(combine_runtime._combine_hook); from vllm_hcu.patch import IMPORT_COORDINATOR; assert not any(r.patch_id == 'worker.framework_opt.step5_moe_combine' for r in IMPORT_COORDINATOR.registrations())"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Cold native MoE imports query GPU architecture during initialization",
)
def test_cold_moe_consumers_bind_compatible_deep_gemm_exports():
    import os, subprocess, sys

    env = os.environ.copy()
    env["VLLM_HCU_EP_FUSED_COMBINE"] = "1"
    script = "from vllm_hcu.patch.platform import apply_platform_patches; apply_platform_patches(); import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors; from vllm_hcu.model_executor.layers.fused_moe.experts import deep_gemm_moe as moe; import vllm.utils.deep_gemm as api; assert moe.get_mk_alignment_for_contiguous_layout is api.get_mk_alignment_for_contiguous_layout; assert api.get_mk_alignment_for_contiguous_layout() == [16,16]; import torch; sf=torch.ones(2,1); assert api.transform_sf_into_required_layout(sf) is sf"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
