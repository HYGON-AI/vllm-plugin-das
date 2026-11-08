"""CPU contracts executing production config and fixed DeepEP gate bodies.

AST loading avoids importing optional vLLM/HCU/GPU packages. The feature-config
parser and architecture reader are real; only platform/operator imports are
substituted. Run directly with Python when vLLM is not installed.
"""
from __future__ import annotations
import __future__

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
VL = "HYV4VLForConditionalGeneration"
TEXT = "HYV4ForCausalLM"


def functions(path, names, namespace=None):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {} if namespace is None else namespace
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope


def feature_config_module():
    name = "_hyv4_vl_test_feature_config"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_hcu/patch/config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config():
    return NS(
        use_v2_model_runner=True,
        model_config=NS(architectures=[VL], use_mla=True, enforce_eager=True,
                        is_multimodal_model=True, is_hybrid=False,
                        hf_config=NS(model_type="hy_v4_vl", quantization_config={
                            "quant_method": "compressed-tensors", "format": "pack-quantized",
                            "config_groups": {"experts": {
                                "targets": ["re:.*experts.*"],
                                "weights": {"num_bits": 4, "type": "int",
                                            "strategy": "channel", "symmetric": True,
                                            "dynamic": False},
                                "input_activations": {"num_bits": 8, "type": "int",
                                                      "strategy": "token", "symmetric": True,
                                                      "dynamic": True}}}}),
                        quantization="compressed-tensors"),
        parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1,
                           prefill_context_parallel_size=16, data_parallel_size=1,
                           decode_context_parallel_size=1, enable_expert_parallel=True,
                           all2all_backend="deepep_high_throughput", enable_eplb=True,
                           eplb_config=NS(num_redundant_experts=0)),
        kernel_config=NS(moe_backend="deep_gemm"),
        attention_config=NS(backend="FLASH_ATTN"),
        speculative_config=None, lora_config=None,
        cache_config=NS(kv_offloading_size=None),
        kv_transfer_config=NS(kv_connector="MooncakeConnector", kv_role="kv_producer",
                              kv_connector_module_path=None),
        additional_config={"hcu": {"expert_map_path": "/tmp/vl-static.json"}},
    )


def set_field(value, path, replacement):
    fields = path.split(".")
    for field in fields[:-1]:
        value = getattr(value, field)
    setattr(value, fields[-1], replacement)


class VLConfigGateTests(unittest.TestCase):
    def setUp(self):
        registry = ModuleType("vllm.v1.attention.backends.registry")
        registry.AttentionBackendEnum = NS(FLASH_ATTN="FLASH_ATTN")
        imports = patch.dict(sys.modules, {registry.__name__: registry})
        imports.start()
        self.addCleanup(imports.stop)

    @classmethod
    def setUpClass(cls):
        scope = functions(
            "vllm_hcu/patch/platform/core_fix/patch_vllm_config.py",
            {"_require_hcu_pcp_attribute", "_require_hyv4_pcp_mtp_contract",
             "_require_hyv4_vl_pcp_contract", "_require_mrv2_pcp_contract",
             "_validate_hcu_pcp_scope"},
            {"os": os, "PatchCompatibilityError": RuntimeError,
             "get_hcu_config": feature_config_module().get_hcu_config})
        cls.validate = staticmethod(scope["_validate_hcu_pcp_scope"])

    def test_vl_static_r0_producer_is_admitted(self):
        self.assertIs(self.validate(config()), True)

    def test_vl_compressed_container_does_not_admit_ordinary_fp8_or_int8(self):
        for raw in (None, {}, "pack-quantized", NS(format="pack-quantized"),
                    {"format": "float-quantized", "quant_method": "compressed-tensors",
                     "config_groups": {"experts": {
                         "weights": {"num_bits": 8, "type": "float"},
                         "input_activations": {"num_bits": 8, "type": "float"}}}},
                    {"format": "int-quantized", "quant_method": "compressed-tensors",
                     "config_groups": {"experts": {
                         "weights": {"num_bits": 8, "type": "int"}}}}):
            with self.subTest(raw=raw):
                current = config()
                current.model_config.hf_config.quantization_config = raw
                with self.assertRaises(ValueError):
                    self.validate(current)

    def test_vl_r16_requires_exact_experimental_flag(self):
        for flag in (None, "0", "1", "true", " 1"):
            with self.subTest(flag=flag), patch.dict(os.environ):
                os.environ.pop("HY4_EXPERIMENTAL_STATIC_R16", None)
                if flag is not None:
                    os.environ["HY4_EXPERIMENTAL_STATIC_R16"] = flag
                current = config()
                current.parallel_config.eplb_config.num_redundant_experts = 16
                if flag == "1":
                    self.assertIs(self.validate(current), True)
                else:
                    with self.assertRaises(ValueError):
                        self.validate(current)

    def test_vl_neighbors_remain_closed(self):
        cases = [
            ("use_v2_model_runner", False),
            ("model_config.architectures", [VL, TEXT]),
            ("model_config.architectures", ["UnknownMultimodal"]),
            ("model_config.architectures", [TEXT]),
            ("model_config.hf_config.model_type", "hy_v4"),
            ("model_config.use_mla", False),
            ("model_config.enforce_eager", False),
            ("model_config.quantization", "fp8"),
            ("parallel_config.pipeline_parallel_size", 2),
            ("parallel_config.tensor_parallel_size", 2),
            ("parallel_config.prefill_context_parallel_size", 8),
            ("parallel_config.data_parallel_size", 2),
            ("parallel_config.decode_context_parallel_size", 2),
            ("parallel_config.enable_expert_parallel", False),
            ("parallel_config.all2all_backend", "deepep_low_latency"),
            ("parallel_config.all2all_backend", "deepep_auto"),
            ("parallel_config.enable_eplb", False),
            ("parallel_config.eplb_config.num_redundant_experts", 1),
            ("kernel_config.moe_backend", "auto"),
            ("speculative_config", NS(method="mtp")),
            ("lora_config", NS()), ("cache_config.kv_offloading_size", 1.0),
            ("kv_transfer_config", None),
            ("kv_transfer_config.kv_connector", "OtherConnector"),
            ("kv_transfer_config.kv_role", "kv_consumer"),
            ("kv_transfer_config.kv_role", "kv_both"),
            ("kv_transfer_config.kv_connector_module_path", "custom.connector"),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                current = config()
                set_field(current, path, value)
                with self.assertRaises(ValueError):
                    self.validate(current)

    def test_vl_static_map_and_feature_neighbors_remain_closed(self):
        for changes in ({"expert_map_path": None},
                        {"expert_map_record_path": "/tmp/record.json"},
                        {"expert_map_path": None, "expert_map_record_path": "/tmp/record.json"},
                        {"eplb_disable_rearrange": True},
                        {"enable_lightly_cp": True},
                        {"enable_multi_layers_mtp": True}):
            with self.subTest(changes=changes):
                current = config()
                current.additional_config["hcu"].update(changes)
                with self.assertRaises(ValueError):
                    self.validate(current)

    def test_existing_text_pcp_scope_is_unchanged(self):
        current = config()
        current.model_config.architectures = [TEXT]
        current.model_config.is_multimodal_model = False
        current.parallel_config.prefill_context_parallel_size = 8
        current.parallel_config.enable_eplb = False
        current.additional_config = {"hcu": {}}
        self.assertIs(self.validate(current), True)
        current.kv_transfer_config = None
        self.assertIs(self.validate(current), True)

    def test_existing_text_checkpoint_native_mtp_scope_is_unchanged(self):
        for depth in (1, 3, 8):
            with self.subTest(depth=depth):
                current = config()
                current.model_config.architectures = [TEXT]
                current.model_config.is_multimodal_model = False
                current.model_config.model = "native-hyv4"
                current.model_config.hf_config.num_nextn_predict_layers = 1
                current.cache_config.cache_dtype = "fp8_ds_mla"
                current.speculative_config = NS(
                    method="mtp", num_speculative_tokens=depth,
                    draft_model_config=NS(
                        architectures=["HYV4MTPModel"], model="native-hyv4",
                        hf_config=NS(num_nextn_predict_layers=1, n_predict=1)))
                self.assertIs(self.validate(current), True)
        current.kv_transfer_config = None
        self.assertIs(self.validate(current), True)


class VLFixedRouteTests(unittest.TestCase):
    def setUp(self):
        self.active = None
        self.calls = []
        config_import = ModuleType("vllm.config")
        config_import.get_current_vllm_config_or_none = lambda: self.active
        arch = functions("vllm_hcu/deepseek_v4_runtime.py", {"model_architectures"})
        arch_import = ModuleType("vllm_hcu.deepseek_v4_runtime")
        arch_import.model_architectures = arch["model_architectures"]
        self.imports = patch.dict(sys.modules, {
            "vllm.config": config_import,
            "vllm_hcu.deepseek_v4_runtime": arch_import})
        self.imports.start()
        self.addCleanup(self.imports.stop)
        self.platform = NS(is_rocm=lambda: True)
        self.routes = functions(
            "vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py",
            {"slimquant_w4a8_uses_deepep_ht", "slimquant_w4a8_uses_deepep_ll"},
            {"current_platform": self.platform,
             "_require_slimquant_w4a8_hipc_runtime": lambda **kw: self.calls.append(kw)})

    def moe(self, route, architecture=VL):
        pcp, dp, backend = ((16, 1, "deepep_high_throughput") if route == "ht"
                            else (1, 16, "deepep_low_latency"))
        return NS(moe_parallel_config=NS(pcp_size=pcp, dp_size=dp, ep_size=16,
                                          tp_size=1, use_ep=True,
                                          all2all_backend=backend,
                                          use_deepep_auto_kernels=False),
                  moe_backend="deep_gemm", activation="silu", swiglu_limit=7.0,
                  _hcu_vllm_config=NS(model_config=NS(architectures=[architecture]),
                                      parallel_config=NS(
                                          pipeline_parallel_size=1, tensor_parallel_size=1,
                                          prefill_context_parallel_size=pcp,
                                          data_parallel_size=dp, decode_context_parallel_size=1),
                                      speculative_config=None))

    def run_route(self, route, moe):
        return self.routes["slimquant_w4a8_uses_deepep_" + route](moe)

    def test_exact_vl_and_text_fixed_ht_ll_routes(self):
        for route in ("ht", "ll"):
            for architecture in (VL, TEXT):
                with self.subTest(route=route, architecture=architecture):
                    self.assertIs(self.run_route(route, self.moe(route, architecture)), True)
        self.assertEqual(self.calls, [dict(ht_only=True, clamp=True)] * 2
                         + [dict(ll_only=True, clamp=True)] * 2)

    def test_vl_speculative_and_mixed_architecture_routes_rejected(self):
        for route in ("ht", "ll"):
            for path, value in [
                ("_hcu_vllm_config.speculative_config", NS(method="mtp")),
                ("_hcu_vllm_config.model_config.architectures", [VL, TEXT]),
                ("_hcu_vllm_config.model_config.architectures", ["UnknownMultimodal"]),
                ("moe_backend", "triton"), ("activation", "gelu"),
                ("moe_parallel_config.use_deepep_auto_kernels", True),
            ]:
                with self.subTest(route=route, path=path):
                    moe = self.moe(route)
                    set_field(moe, path, value)
                    with self.assertRaises(ValueError):
                        self.run_route(route, moe)
        self.assertEqual(self.calls, [])

    def test_fixed_route_topology_and_platform_guards(self):
        for route in ("ht", "ll"):
            for path, value in [("dp_size", 1 if route == "ll" else 2),
                                ("use_ep", False),
                                ("all2all_backend", "deepep_auto")]:
                with self.subTest(route=route, path=path):
                    moe = self.moe(route)
                    setattr(moe.moe_parallel_config, path, value)
                    if route == "ll" and path != "all2all_backend":
                        with self.assertRaises(ValueError):
                            self.run_route(route, moe)
                    else:
                        self.assertIs(self.run_route(route, moe), False)
            self.platform.is_rocm = lambda: False
            with self.assertRaises(RuntimeError):
                self.run_route(route, self.moe(route))
            self.platform.is_rocm = lambda: True
        self.assertEqual(self.calls, [])

    def test_current_config_is_authoritative_over_saved_config(self):
        for route in ("ht", "ll"):
            self.active = NS(model_config=NS(architectures=["UnknownMultimodal"]))
            with self.assertRaises(ValueError):
                self.run_route(route, self.moe(route))

    def test_vl_ll_topology_is_exact_on_outer_and_effective_configs(self):
        for path, value in [
            ("moe_parallel_config.dp_size", 2),
            ("moe_parallel_config.ep_size", 2),
            ("moe_parallel_config.tp_size", 2),
            ("moe_parallel_config.pcp_size", 2),
            ("_hcu_vllm_config.parallel_config.pipeline_parallel_size", 2),
            ("_hcu_vllm_config.parallel_config.tensor_parallel_size", 2),
            ("_hcu_vllm_config.parallel_config.prefill_context_parallel_size", 2),
            ("_hcu_vllm_config.parallel_config.data_parallel_size", 2),
            ("_hcu_vllm_config.parallel_config.decode_context_parallel_size", 2),
            ("_hcu_vllm_config.parallel_config", None),
        ]:
            with self.subTest(path=path):
                moe = self.moe("ll")
                set_field(moe, path, value)
                with self.assertRaises(ValueError):
                    self.run_route("ll", moe)
        self.assertEqual(self.calls, [])

    def test_existing_text_ll_dp2_ep2_and_outer_tp2_are_unchanged(self):
        moe = self.moe("ll", TEXT)
        moe.moe_parallel_config.dp_size = moe.moe_parallel_config.ep_size = 2
        moe._hcu_vllm_config.parallel_config.tensor_parallel_size = 2
        self.assertIs(self.run_route("ll", moe), True)


class VLResolvedQuantizationTests(unittest.TestCase):
    def setUp(self):
        marker_module = ModuleType(
            "vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe")
        marker_module.HCU_COMPRESSED_W4A8_METHOD_MARKER = (
            "_vllm_hcu_compressed_tensors_w4a8_int8_moe")
        imports = patch.dict(sys.modules, {marker_module.__name__: marker_module})
        imports.start()
        self.addCleanup(imports.stop)
        self.scope = functions(
            "vllm_hcu/patch/platform/core_fix/patch_vllm_config.py",
            {"validate_hyv4_vl_pcp_quantization"})

    def validate(self, current, model):
        self.assertIn("validate_hyv4_vl_pcp_quantization", self.scope)
        return self.scope["validate_hyv4_vl_pcp_quantization"](current, model)

    def layer(self, marker=True, static=True, supports=True):
        return NS(routed_experts=NS(quant_method=NS(
            _vllm_hcu_compressed_tensors_w4a8_int8_moe=marker,
            _hcu_static_eplb=static, supports_eplb=supports)))

    def test_all_resolved_moe_layers_must_be_static_w4a8(self):
        self.assertIsNone(self.validate(config(), NS(moe_layers=[self.layer(), self.layer()])))
        for layers in (None, [], 1, [NS()], [NS(routed_experts=None)],
                       [NS(routed_experts=NS(quant_method=None))],
                       [self.layer(), self.layer(marker=False)],
                       [self.layer(static=False)], [self.layer(supports=False)],
                       [self.layer(marker=1)], [self.layer(static=1)],
                       [self.layer(supports=1)]):
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(ValueError, "W4A8"):
                    self.validate(config(), NS(moe_layers=layers))

    def test_mixed_or_malformed_raw_groups_cannot_bypass_resolved_guard(self):
        for groups in ({}, {"experts": None},
                       {"experts": {"targets": ["re:.*"],
                                    "weights": {"num_bits": 4}},
                        "other": {"weights": {"num_bits": 8, "type": "float"}}}):
            with self.subTest(groups=groups):
                current = config()
                current.model_config.hf_config.quantization_config["config_groups"] = groups
                with self.assertRaisesRegex(ValueError, "W4A8"):
                    self.validate(current, NS(moe_layers=[self.layer(), self.layer(marker=False)]))

    def test_resolved_guard_is_noop_for_text_and_pcp1(self):
        for architecture, pcp in ((TEXT, 16), (VL, 1), ("Unrelated", 16)):
            current = config()
            current.model_config.architectures = [architecture]
            current.parallel_config.prefill_context_parallel_size = pcp
            self.assertIsNone(self.validate(current, object()))


if __name__ == "__main__":
    unittest.main()
