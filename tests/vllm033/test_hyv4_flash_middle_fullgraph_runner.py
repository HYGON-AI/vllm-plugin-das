"""Execute actual runner methods; Core/device interfaces are explicit doubles."""

import copy
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from test_hyv4_flash_middle_fullgraph_banks import SOURCE, Mode, Manager, owner


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def config():
    w = lambda bits: dict(
        num_bits=bits, type="int", strategy="channel", symmetric=True, dynamic=False
    )
    a = dict(num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True)
    recipe = dict(
        quant_method="compressed-tensors",
        format="pack-quantized",
        quantization_status="compressed",
        config_groups={
            str(bits): dict(
                targets=["test"],
                weights=w(bits),
                input_activations=copy.deepcopy(a),
                format="pack-quantized" if bits == 4 else "int-quantized",
            )
            for bits in (4, 8)
        },
    )
    parallel = NS(
        tensor_parallel_size=4,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
        enable_eplb=False,
        disable_custom_all_reduce=True,
        enable_expert_parallel=False,
        enable_elastic_ep=False,
        use_sequence_parallel_moe=False,
        eplb_config=NS(num_redundant_experts=0),
    )
    return NS(
        use_v2_model_runner=True,
        model_config=NS(
            enforce_eager=False,
            enable_prompt_embeds=False,
            hf_text_config=NS(model_type="hy_v4_flash"),
            multimodal_config=None,
        ),
        parallel_config=parallel,
        compilation_config=NS(
            mode=NS(name="VLLM_COMPILE"), cudagraph_mode=NS(name="FULL_AND_PIECEWISE")
        ),
        speculative_config=NS(method="dfly", num_speculative_tokens=5),
        cache_config=NS(cache_dtype="fp8_ds_mla", enable_prefix_caching=False),
        lora_config=None,
        kv_transfer_config=None,
        quant_config=NS(config=recipe),
    )


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.environment_patch = patch.dict(os.environ, {}, clear=False)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        os.environ.pop("VLLM_HCU_USE_CUSTOM_OPS", None)
        self.owners = [owner(), owner()]
        self.constructed = []
        self.properties_calls = []
        self.platform = True
        self.breakable = False
        self.properties = NS(gcnArchName="gfx938:sramecc+", multi_processor_count=64)
        self.cfg = config()
        test = self

        class CoreManager(Manager):
            def __init__(s, cfg, device, mode, **kwargs):
                super().__init__(
                    ["wide", "short", "middle"][len(test.constructed) % 3], test.owners
                )
                s.cudagraph_mode = mode
                s.lora_capture_cases = kwargs.get("lora_capture_cases", [0])
                s.varlen_decode = kwargs.get("varlen_decode", True)
                s.arguments = kwargs
                test.constructed.append(s)

        class BudgetArray:
            def __init__(s, values):
                s.values = values

            def __getitem__(s, mapping):
                if mapping == "invalid":
                    raise IndexError("unavailable mapping")
                return [s.values[i] for i in mapping]

        class BaseRunner:
            def __init__(s, cfg, device):
                s.vllm_config = cfg
                s.device = device
                s.max_model_len = 1048576
                s.decode_query_len = 6
                s.model = NS(modules=lambda: iter(test.owners))
                s.req_states = NS()

            def initialize_kv_cache(s, *args, **kwargs):
                s.cudagraph_manager = CoreManager(
                    s.vllm_config, s.device, Mode.FULL_AND_PIECEWISE, decode_query_len=6
                )

            def gather_batch_req_state(s, request, dummy):
                s.req_states.max_seq_len = BudgetArray(request["budgets"])
                return (
                    None if dummy else NS(idx_mapping_np=request.get("mapping", [0])),
                    6,
                )

        self.CoreManager = CoreManager
        modules = {}
        for name in (
            "_middle_runner",
            "vllm",
            "vllm.config",
            "vllm.config.compilation",
            "vllm.compilation",
            "vllm.compilation.breakable_cudagraph",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.cudagraph_utils",
            "vllm.v1.worker.gpu.model_runner",
            "vllm.platforms",
            "hcu_vllm_das",
            "torch",
        ):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules["vllm.config.compilation"].CUDAGraphMode = Mode
        modules[
            "vllm.v1.worker.gpu.cudagraph_utils"
        ].ModelCudaGraphManager = CoreManager
        modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = BaseRunner
        modules["vllm.platforms"].current_platform = NS(graph_pool_handle=object)
        modules[
            "vllm.compilation.breakable_cudagraph"
        ].is_breakable_cudagraph_enabled = lambda: self.breakable
        modules["torch"].version = NS(hip="test-HIP")

        def properties(device):
            self.properties_calls.append(device)
            return self.properties

        modules["torch"].cuda = NS(get_device_properties=properties)
        idx = ModuleType("_middle_runner.compressed_indexer")
        idx.FlashCompressedIndexer = NS
        modules[idx.__name__] = idx
        runtime = ModuleType("_middle_runner.runtime")
        runtime.is_hcu_platform = lambda: self.platform
        modules[runtime.__name__] = runtime
        quantization_source = SOURCE / "quantization.py"
        environment_source = SOURCE.parents[1] / "environment.py"
        modules["_middle_runner.quantization"] = module_from_file(
            "_middle_runner.quantization", quantization_source
        )
        modules["hcu_vllm_das.environment"] = module_from_file(
            "hcu_vllm_das.environment", environment_source
        )
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.banks = module_from_file(
            "_middle_runner.fullgraph_banks", SOURCE / "fullgraph_banks.py"
        )
        sys.modules["_middle_runner.fullgraph_banks"] = self.banks
        self.addCleanup(sys.modules.pop, "_middle_runner.fullgraph_banks", None)
        self.runner_api = module_from_file(
            "_middle_runner.fullgraph_runner", SOURCE / "fullgraph_runner.py"
        )
        self.runner = self.runner_api.runner_class()(self.cfg, "cuda:0")

    def test_middle_qualification_default_and_real_validators(self):
        self.assertTrue(self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0"))
        self.assertEqual(self.properties_calls, ["cuda:0"])

    def test_master_off_spellings_keep_original_admission(self):
        for value in ("0", "false", "no", "off", "OFF", " false "):
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = value
            self.assertTrue(
                self.runner_api.is_qualified_config(self.cfg, breakable=False)
            )
            self.assertFalse(self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0"))
        self.assertEqual(self.properties_calls, [])

    def test_invalid_master_value_refuses(self):
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "maybe"
        with self.assertRaises(ValueError):
            self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0")

    def test_fp8_unknown_recipe_and_unqualified_parallel_keep_middle_off(self):
        cases = []
        fp8 = copy.deepcopy(self.cfg)
        fp8.quant_config.config = dict(
            quant_method="fp8",
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        cases.append(fp8)
        unknown = copy.deepcopy(self.cfg)
        unknown.quant_config = object()
        cases.append(unknown)
        for field, value in [
            ("enable_expert_parallel", True),
            ("enable_elastic_ep", True),
            ("use_sequence_parallel_moe", True),
            ("tensor_parallel_size", 2),
            ("data_parallel_size", 2),
        ]:
            cfg = copy.deepcopy(self.cfg)
            setattr(cfg.parallel_config, field, value)
            cases.append(cfg)
        for cfg in cases:
            self.assertFalse(self.runner_api.middle_scoring_enabled(cfg, "cuda:0"))
        self.assertEqual(self.properties_calls, [])

    def test_device_and_platform_and_breakable_gate(self):
        for arch, cus in [("gfx936", 64), ("gfx938", 128), ("gfx938", 63)]:
            self.properties = NS(gcnArchName=arch, multi_processor_count=cus)
            self.assertFalse(self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0"))
        self.platform = False
        self.assertFalse(self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0"))
        self.platform = True
        self.properties = NS(gcnArchName="gfx938", multi_processor_count=64)
        self.breakable = True
        self.assertFalse(self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0"))

    def test_missing_device_capability_keeps_middle_off(self):
        for properties in (
            NS(gcnArchName="gfx938"),
            NS(gcnArchName=None, multi_processor_count=64),
        ):
            self.properties = properties
            with self.subTest(properties=properties):
                self.assertFalse(
                    self.runner_api.middle_scoring_enabled(self.cfg, "cuda:0")
                )

    def test_initialize_builds_one_middle_with_original_core_arguments(self):
        self.runner.initialize_kv_cache()
        b = self.runner.cudagraph_manager
        self.assertEqual(len(self.constructed), 3)
        self.assertEqual(b._scoring_banks, ((16384, self.constructed[2]),))
        for m in self.constructed[1:]:
            self.assertEqual(
                m.arguments,
                dict(
                    decode_query_len=6,
                    lora_capture_cases=[0],
                    varlen_decode=True,
                    ubatch_runner=None,
                ),
            )
            self.assertNotIn(Mode.PIECEWISE, m._capture_descs)
        self.assertIn(Mode.PIECEWISE, self.constructed[0]._capture_descs)

    def test_master_off_builds_two_without_disabling_w4_graph(self):
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "off"
        self.runner.initialize_kv_cache()
        self.assertEqual(len(self.constructed), 2)
        self.assertEqual(self.runner.cudagraph_manager._scoring_banks, ())

    def test_startup_policy_not_reread_between_profile_and_real_initialize(self):
        self.runner.initialize_kv_cache()
        self.assertEqual(len(self.constructed), 3)
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "off"
        self.runner.initialize_kv_cache()
        self.assertEqual(len(self.constructed), 6)
        self.assertEqual(self.properties_calls, ["cuda:0"])

    def test_replay_does_not_lookup_configuration_or_device(self):
        self.runner.initialize_kv_cache()
        b = self.runner.cudagraph_manager
        b.capture("model")
        with patch.object(
            self.runner_api,
            "middle_scoring_enabled",
            side_effect=AssertionError("init-only policy read"),
        ):
            self.runner.gather_batch_req_state(
                dict(budgets=[4352, 100], mapping=[1, 0]), False
            )
            self.assertEqual(b.run_fullgraph(b.dispatch()), "middle-owned-output")
        self.assertEqual(self.properties_calls, ["cuda:0"])

    def test_gather_mapping_mixed_and_dummy_reset(self):
        self.runner.initialize_kv_cache()
        b = self.runner.cudagraph_manager
        b.capture("model")
        for request, dummy, expected in [
            (dict(budgets=[4352, 100], mapping=[1, 0]), False, "middle"),
            (dict(budgets=[4352, 33000], mapping=[0, 1]), False, "wide"),
            (dict(budgets=[10]), False, "short"),
            (dict(budgets=[10]), True, "wide"),
            (dict(budgets=[4352], mapping=None), False, "wide"),
            (dict(budgets=[4352], mapping="invalid"), False, "wide"),
        ]:
            (state, uniform) = self.runner.gather_batch_req_state(request, dummy)
            self.assertEqual(uniform, 6)
            self.assertEqual(b.run_fullgraph(b.dispatch()), expected + "-owned-output")

    def test_model_limit_does_not_duplicate_middle_as_wide(self):
        self.runner.max_model_len = 16384
        for o in self.owners:
            o.max_model_len = 16384
        self.runner.initialize_kv_cache()
        self.assertEqual(len(self.constructed), 2)

    def test_shared_predicate_remains_independent_of_master_and_quant_family(self):
        for value in ("1", "off"):
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = value
            self.assertTrue(
                self.runner_api.is_qualified_config(self.cfg, breakable=False)
            )
            fp8 = copy.deepcopy(self.cfg)
            fp8.quant_config.config = dict(
                quant_method="fp8",
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
            self.assertTrue(self.runner_api.is_qualified_config(fp8, breakable=False))
        rejected = copy.deepcopy(self.cfg)
        rejected.parallel_config.tensor_parallel_size = 2
        self.assertFalse(self.runner_api.is_qualified_config(rejected, breakable=False))
        self.assertFalse(self.runner_api.is_qualified_config(self.cfg, breakable=True))


class CustomAllReduceRunnerTests(RunnerTests):
    """Keep the same bank construction/replay contracts with Core AR enabled.

    Core/device managers remain explicit CPU doubles in these tests. Native
    collective numerical correctness and model capture are separate checks.
    """

    def setUp(self):
        super().setUp()
        self.cfg.parallel_config.disable_custom_all_reduce = False


if __name__ == "__main__":
    unittest.main()
