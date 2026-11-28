# SPDX-License-Identifier: Apache-2.0
"""Portable tests for graph opt-in and spawn provenance."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import gc
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[2] / "adapter_extension" / "src" / "hcu_vllm_das"
          / "models" / "hy_v4_flash" / "hisparse_config_compat.py")
MODULE = spec_from_file_location("_hisparse_graph_config_under_test", SOURCE)
OWNER = module_from_spec(MODULE)
MODULE.loader.exec_module(OWNER)


class GraphConfig(SimpleNamespace):
    pass


class CompilationMode(Enum):
    NONE = auto()
    VLLM_COMPILE = auto()


class GraphMode(Enum):
    NONE = auto()
    FULL_AND_PIECEWISE = auto()


def candidate():
    return GraphConfig(
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        kv_transfer_config=SimpleNamespace(
            kv_connector="HCUHiSparseConnector",
            kv_connector_module_path="hcu_vllm_das.models.hy_v4_flash.hisparse_planner_compat",
        ),
    )


class HandshakeConfig(GraphConfig):
    def __post_init__(self):
        self.visibility.append(self.attention_config.hisparse_config)
        if self.fail:
            raise RuntimeError("native post-init failed")


class EngineProc:
    def __init__(self, vllm_config, *, complete=True):
        self.armed = OWNER.arm_hisparse_engine_handshake(vllm_config)
        self.ready_attention = vllm_config.attention_config.hisparse_config
        if complete:
            self.finish_handshake(vllm_config)

    def finish_handshake(self, vllm_config):
        vllm_config.__post_init__()


class TestHiSparseGraphConfig(unittest.TestCase):
    def test_startup_keeps_the_cli_reasoning_parser_configuration(self):
        @dataclass
        class HiSparse:
            pass

        @dataclass
        class AttentionConfig:
            hisparse_config: HiSparse | None

        class EngineArgs(SimpleNamespace):
            def __post_init__(self):
                pass

            def create_model_config(self):
                return SimpleNamespace()

            def create_structured_outputs_config(self):
                self.structured_outputs_config.reasoning_parser = self.reasoning_parser
                self.structured_outputs_config.reasoning_parser_plugin = self.reasoning_parser_plugin
                return self.structured_outputs_config

        namespace = SimpleNamespace(reasoning_parser="", reasoning_parser_plugin="")
        args = EngineArgs(attention_config=AttentionConfig(HiSparse()),
                         structured_outputs_config=namespace, block_size=64,
                         reasoning_parser="hy_v4", reasoning_parser_plugin="external.py")

        def native(clone, **kwargs):
            return GraphConfig(model_config=clone.create_model_config(),
                               attention_config=clone.attention_config,
                               structured_outputs_config=clone.create_structured_outputs_config())

        bindings = [SimpleNamespace(EngineArgs=EngineArgs)]
        with (patch.object(OWNER, "_require_hisparse"),
              patch.object(OWNER, "_is_flash", return_value=True),
              patch.object(OWNER, "_request_bounds", return_value=True),
              patch.object(OWNER, "_require_model"),
              patch.object(OWNER, "_graph_requested", return_value=True),
              patch.object(OWNER, "_validate_result"),
              patch.object(OWNER, "check_health"),
              patch.object(OWNER, "_graph_snapshot"),
              patch.object(OWNER, "_record"),
              patch.object(OWNER, "validate_hisparse_config")):
            config = OWNER._construct(native, args, None, False, bindings)

        self.assertIs(args.structured_outputs_config, namespace)
        self.assertEqual(namespace.reasoning_parser, "hy_v4")
        self.assertEqual(namespace.reasoning_parser_plugin, "external.py")
        self.assertEqual(config.structured_outputs_config, namespace)

    def test_graph_opt_in_requires_environment_connector_and_cache_dtype(self):
        cfg = candidate()
        with patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "0"}):
            self.assertFalse(OWNER._graph_requested(cfg, final=True))
        with patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "1"}):
            self.assertTrue(OWNER._graph_requested(cfg, final=True))
            cfg.kv_transfer_config.kv_connector_module_path = "other.module"
            self.assertFalse(OWNER._graph_requested(cfg, final=True))
            cfg = candidate()
            cfg.cache_config.cache_dtype = "bfloat16"
            self.assertFalse(OWNER._graph_requested(cfg, final=True))

    def test_engine_request_preserves_eager_bf16_and_requires_the_exact_graph_shape(self):
        bindings = [None, None, None, SimpleNamespace(
            CompilationMode=CompilationMode, CUDAGraphMode=GraphMode,
        )]
        args = SimpleNamespace(
            data_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            tensor_parallel_size=4, kv_transfer_config=candidate().kv_transfer_config,
            enable_eplb=False, enable_prefix_caching=False, max_model_len=65792,
            kv_cache_dtype="fp8_ds_mla", enforce_eager=False, block_size=64,
            compilation_config=SimpleNamespace(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=GraphMode.FULL_AND_PIECEWISE,
            ),
        )
        with (patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "1"}),
              patch.object(OWNER, "_planner_scope", return_value=True)):
            self.assertTrue(OWNER._request_bounds(args, bindings))
            with patch.object(args, "block_size", None):
                self.assertTrue(OWNER._request_bounds(args, bindings))
            for name, value in (("tensor_parallel_size", 8), ("max_model_len", 65536), ("block_size", 256),
                                ("enable_prefix_caching", True), ("enforce_eager", True)):
                with self.subTest(name=name), patch.object(args, name, value):
                    with self.assertRaisesRegex(ValueError, "HiSparse FP8 graph"):
                        OWNER._request_bounds(args, bindings)
            args.kv_cache_dtype = "bfloat16"
            args.enforce_eager = True
            args.compilation_config.mode = CompilationMode.NONE
            args.compilation_config.cudagraph_mode = GraphMode.NONE
            self.assertFalse(OWNER._request_bounds(args, bindings))

    def test_spawn_revalidates_before_restoring_only_intact_graph_provenance(self):
        cfg = candidate()
        bindings = [None, SimpleNamespace(VllmConfig=GraphConfig)]
        checked = []
        recorded = []
        with (patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "1"}),
              patch.object(OWNER, "_validate_result", side_effect=lambda obj, _: checked.append(obj)),
              patch.object(OWNER, "_record", side_effect=recorded.append),
              patch.object(OWNER, "_graph_snapshot", side_effect=lambda obj: obj.cache_config.cache_dtype)):
            self.assertFalse(OWNER._restore_graph_record(cfg, bindings))
            setattr(cfg.attention_config.hisparse_config, OWNER._GRAPH_ATTESTATION,
                    (os.getpid() + 1, "fp8_ds_mla"))
            copy = deepcopy(cfg)
            self.assertTrue(OWNER._restore_graph_record(copy, bindings))
            self.assertEqual(checked, [copy])
            self.assertEqual(recorded, [copy])
            setattr(copy.attention_config.hisparse_config, OWNER._GRAPH_ATTESTATION,
                    (os.getpid() + 1, "different"))
            with self.assertRaisesRegex(ValueError, "graph construction provenance"):
                OWNER._restore_graph_record(copy, bindings)
            setattr(copy.attention_config.hisparse_config, OWNER._GRAPH_ATTESTATION,
                    (os.getpid(), "fp8_ds_mla"))
            self.assertFalse(OWNER._restore_graph_record(copy, bindings))
            self.assertEqual(recorded, [copy])

    def test_handshake_restores_owned_attention_and_clears_instance_override(self):
        cfg = HandshakeConfig(**vars(candidate()), fail=False, visibility=[])
        hisparse = object()
        attention = SimpleNamespace(hisparse_config=hisparse)
        cfg.attention_config = attention
        scope = (EngineProc, EngineProc.__init__.__code__, EngineProc.finish_handshake.__code__)
        with (patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "1"}),
              patch.object(OWNER, "_state", {"engine_handshake": scope}),
              patch.object(OWNER, "check_health", return_value={}),
              patch.object(OWNER, "validate_hisparse_config",
                           side_effect=lambda value: self.assertIs(value.attention_config.hisparse_config, hisparse))):
            self.assertFalse(OWNER.arm_hisparse_engine_handshake(cfg))
            engine = EngineProc(cfg, complete=False)
            self.assertTrue(engine.armed)
            self.assertIs(engine.ready_attention, hisparse)
            cfg.__post_init__()
            self.assertEqual(cfg.visibility, [hisparse], "an unrelated call remains native")
            engine.finish_handshake(cfg)
            self.assertEqual(cfg.visibility, [hisparse, None])
            self.assertIs(cfg.attention_config, attention)
            self.assertIs(attention.hisparse_config, hisparse)
            self.assertNotIn("__post_init__", vars(cfg))

            cfg.fail = True
            with self.assertRaisesRegex(RuntimeError, "native post-init failed"):
                EngineProc(cfg)
            self.assertIs(cfg.attention_config, attention)
            self.assertIs(attention.hisparse_config, hisparse)
            self.assertNotIn("__post_init__", vars(cfg))

            engine = EngineProc(cfg, complete=False)
            self.assertTrue(engine.armed)
            del engine
            gc.collect()
            self.assertNotIn("__post_init__", vars(cfg))

    def test_text_config_exposes_hisparse_only_after_core_derives_the_submodel(self):
        @dataclass
        class AttentionConfig:
            hisparse_config: object

        class OuterConfig(SimpleNamespace):
            def with_hf_config(self, text_config, *, architectures):
                if vars(self).keys() != {"attention_config"}:
                    raise ValueError("non-Core config fields")
                if self.attention_config.hisparse_config is not None:
                    raise ValueError("Core CUDA-only HiSparse validation")
                return SimpleNamespace(attention_config=self.attention_config,
                                       hf_config=text_config, architectures=architectures)

        hisparse = SimpleNamespace()
        parent = OuterConfig(attention_config=AttentionConfig(hisparse))
        hf_text = SimpleNamespace()
        with patch.object(OWNER, "validate_hisparse_config") as validate:
            child = OWNER.derive_hisparse_text_config(parent, hf_text, "HYV4FlashForCausalLM")
        validate.assert_called_once_with(parent)
        self.assertIs(parent.attention_config.hisparse_config, hisparse)
        self.assertIs(child.attention_config.hisparse_config, hisparse)
        self.assertIs(child.hf_config, hf_text)
        self.assertEqual(child.architectures, ["HYV4FlashForCausalLM"])


if __name__ == "__main__":
    unittest.main()
