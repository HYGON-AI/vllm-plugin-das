# SPDX-License-Identifier: Apache-2.0
"""CPU shape contracts; execute the real receiver/apply up to vendor GEMM.

Shape-only device tensors keep this suite runnable without torch or HCU
extensions. It checks layout selection and validation, not kernel numerics.
"""
from __future__ import annotations

import ast
import importlib.util
from contextlib import nullcontext
from math import prod
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
RUNTIME = "vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py"
EXPERTS = "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
SIZING = "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"


_env_spec = importlib.util.spec_from_file_location(
    "step5_scatter_test_envs", ROOT / "vllm_hcu/platforms/envs.py")
HENVS = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(HENVS)


def load(path, *names, scope=None):
    scope = {} if scope is None else scope
    scope.setdefault("henvs", HENVS)
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    functions = [next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == name)
                 for name in names]
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(
        body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), scope)
    return scope


class Tensor:
    def __init__(self, shape, dtype="fp8", device="hcu"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim = len(self.shape)
        self.cleared = False

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return True

    def view(self, *, dtype):
        return Tensor(self.shape, dtype, self.device)

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.start is None
        return Tensor((min(self.shape[0], index.stop), *self.shape[1:]),
                      self.dtype, self.device)

    def __add__(self, offset):
        return self

    def record_stream(self, stream):
        pass

    def zero_(self):
        self.cleared = True


class Counts:
    """CPU arithmetic used by the unchanged real sizing helper."""
    device = SimpleNamespace(type="cpu")

    def __init__(self, values):
        self.values = list(values)

    def to(self, dtype):
        return self

    def numel(self):
        return len(self.values)

    def __add__(self, value):
        return Counts(c + value for c in self.values)

    def __floordiv__(self, value):
        return Counts(c // value for c in self.values)

    def __mul__(self, value):
        return Counts(c * value for c in self.values)

    def __lt__(self, value):
        return Counts(c < value for c in self.values)

    def any(self):
        return any(self.values)


TORCH = SimpleNamespace(
    float8_e4m3fn="fp8", float32="float32", int32="int32", int64="int64",
    where=lambda condition, a, b: b,
    sum=lambda counts: SimpleNamespace(item=lambda: sum(counts.values)),
    cuda=SimpleNamespace(current_stream=lambda: object()),
)


def log_modules():
    module = ModuleType("vllm.logger")
    module.init_logger = lambda _: SimpleNamespace(
        info=lambda *a, **k: None, info_once=lambda *a, **k: None)
    return {"vllm.logger": module}


class GemmReached(Exception):
    pass


class FusedScatterContractTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.scope = dict(
            torch=TORCH, _HCU_TOKEN_ALIGNMENT=256, _COMPACT_ROWS_LOGGED=False,
            _HCU_HT_EP_TOKEN_ALIGNMENT=256, _EP_FUSED_MOE=False,
            current_platform=SimpleNamespace(is_rocm=lambda: True),
            henvs=SimpleNamespace(VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT=False,
                                  optional_custom_op_enabled=HENVS.optional_custom_op_enabled,
                                  custom_ops_enabled=HENVS.custom_ops_enabled),
            round_up=lambda value, align: (value + align - 1) // align * align,
            init_logger=lambda _: SimpleNamespace(info=lambda *a: None),
            MoEActivation=SimpleNamespace(SWIGLUSTEP="swiglustep"),
            mk_alignment_scope=lambda align: nullcontext(),
            _resize_cache=lambda tensor, shape: Tensor(shape, tensor.dtype),
            m_grouped_fp8_gemm_nt_contiguous=self.gemm,
            deepgemm_moe_permute=self.permute,
        )
        load(SIZING, "expert_num_tokens_round_up_and_sum",
             "compute_aligned_M_and_alignment", scope=self.scope)
        load(EXPERTS, "_step5_fused_scatter_rows", "apply", scope=self.scope)
        self.runtime = load(RUNTIME, "ht_receiver",
                            "_uses_post_dispatch_channel_fp8_quant")

    def gemm(self, inputs, weights, output, expert_ids, **kwargs):
        self.calls.append((inputs, output, expert_ids))
        raise GemmReached

    def permute(self, **kwargs):
        self.calls.append("permute")
        rows = kwargs["aq_out"].size(0)
        return (kwargs["aq_out"], Tensor((rows, 1), "float32"),
                Tensor((rows,), "int32"),
                Tensor(kwargs["topk_ids"].shape, "int32"), 256)

    def receive(self, experts=22, rows=256, counts=None, scatter=True):
        counts = [rows] + [0] * (experts - 1) if counts is None else counts
        buffer = object()
        key = (id(buffer), 0)
        pf = SimpleNamespace(
            rank_expert_offset=0, buffer=buffer,
            _hcu_step5_fused_scatter_enabled=scatter,
            _hcu_step5_scatter_buffers={key: (
                Tensor((sum(counts), 4096)),
                Tensor((sum(counts), 1), "float32"),
                Tensor((rows, 8), "int32"),
                Tensor((sum(counts),), "int32"),
            )}, _hcu_step5_scatter_guards={key: {}},
        )
        module = SimpleNamespace(
            torch=TORCH, dbo_current_ubatch_id=lambda: 0,
            mk=SimpleNamespace(ExpertTokensMetadata=SimpleNamespace(
                make_from_list=lambda counts, **kwargs: SimpleNamespace(
                    expert_num_tokens_cpu=Counts(counts)))),
        )
        quant = SimpleNamespace(
            use_fp8_w8a8=True, is_per_act_token=True,
            is_block_quantized=False, is_per_act_token_quant=True,
        )
        with patch.dict("sys.modules", log_modules()):
            return self.runtime["ht_receiver"](
                module, pf, SimpleNamespace(event=None), True,
                (Tensor((rows, 4096)), Tensor((rows, 1), "float32")),
                Tensor((rows, 8), "int32"), experts * 16, counts,
                Tensor((rows, 8), "float32"), None, quant, True,
            )

    def apply(self, result, *, experts=22, compact=False, scatter=True, owner=True):
        x, scale, meta, ids, weights = result
        expert = SimpleNamespace(
            _hcu_step5_owner=owner, _hcu_step5_compact_ht=compact,
            _hcu_step5_fused_scatter_enabled=scatter,
            _hcu_logical_n=3072, _hcu_logical_k=4096, block_shape=None,
            w1_scale=object(), w2_scale=object(), mxfp8=False,
            quant_config=SimpleNamespace(use_fp8_w8a8=True, use_int8_w8a8=False,
                                        per_out_ch_quant=True, per_act_token_quant=True),
        )
        self.output = Tensor(x.shape, "bf16")
        self.scope["apply"](
            expert, self.output, x, Tensor((experts,) + (1,) * 6),
            Tensor((experts,) + (1,) * 6), weights, ids, "swiglustep",
            experts * 16, None, scale, None, Tensor((1,)), Tensor((1,)), meta, False,
        )

    def test_all_compact_and_scatter_combinations_for_e22_and_e23(self):
        for experts in (22, 23):
            for compact in (False, True):
                for scatter in (False, True):
                    with self.subTest(experts=experts, compact=compact, scatter=scatter):
                        self.calls.clear()
                        result = self.receive(experts, scatter=scatter)
                        with self.assertRaises(GemmReached):
                            self.apply(result, experts=experts, compact=compact, scatter=scatter)
                        expected = 256 if compact or scatter else {22: 7680, 23: 7936}[experts]
                        inputs, output, expert_ids = self.calls[-1]
                        self.assertEqual(inputs[0].shape, (expected, 4096))
                        self.assertEqual(inputs[1].shape, (expected, 1))
                        self.assertEqual(output.shape, (expected, 3072))
                        self.assertEqual(expert_ids.shape, (expected,))
                        self.assertEqual("permute" in self.calls, not scatter)
                        if scatter:
                            self.assertIs(inputs[0], result[2]._hcu_step5_fused_scatter[0])

    def test_scatter_does_not_depend_on_optional_ht_exact_layout(self):
        self.scope["henvs"].VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT = True
        result = self.receive()
        self.assertFalse(hasattr(result[2], "_hcu_deepep_ht_layout"))
        with self.assertRaises(GemmReached):
            self.apply(result)
        self.assertEqual(self.calls[-1][0][0].shape, (256, 4096))

    def test_construction_accepts_scatter_without_compact_flag(self):
        configure = load(RUNTIME, "configure_step5_ht_kernel", scope={"os": os})[
            "configure_step5_ht_kernel"]
        primary, secondary = object(), object()
        pf = type("DeepEPHTPrepareAndFinalize", (), {})()
        pf.buffer, pf.num_dispatchers_ = primary, 16
        expert = type("DeepGemmExperts", (), {})()
        expert._hcu_step5_owner = True
        expert._hcu_logical_n, expert._hcu_logical_k = 3072, 4096
        quant = SimpleNamespace(use_fp8_w8a8=True, per_out_ch_quant=True,
                                per_act_token_quant=True, block_shape=None)
        kernel = SimpleNamespace(prepare_finalize=pf, fused_experts=expert)
        config = ModuleType("vllm.config")
        config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_dbo=True))
        step5 = ModuleType("vllm_hcu.runtime_compat.step5")
        step5.is_step5_owner = lambda _: True
        manager = SimpleNamespace(
            get_handle=lambda _: primary,
            _make_all2all_kwargs=lambda: {"num_nvl_bytes": 4096},
            handle_cache=SimpleNamespace(get_or_create=lambda *args: secondary),
        )
        all2all = ModuleType("vllm.model_executor.layers.fused_moe.all2all_utils")
        all2all.get_ep_all2all_manager = lambda: manager
        deep_ep = ModuleType("deep_ep")
        deep_ep.Buffer = object
        modules = {"vllm.config": config, "vllm_hcu.runtime_compat.step5": step5,
                   all2all.__name__: all2all, "deep_ep": deep_ep, **log_modules()}
        env = {"HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT": "1",
               "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR": "1",
               "HCU_STEP5_DEEPEP_FUSED_SCATTER": "1", "DEEP_EP_NORMAL_MNVL": "1"}
        with patch.dict(os.environ, env, clear=True), patch.dict("sys.modules", modules):
            self.assertNotIn("HCU_STEP5_DEEPEP_COMPACT_ROWS", os.environ)
            self.assertIs(configure(kernel, quant, object()), kernel)
        self.assertTrue(pf._hcu_step5_fused_scatter_enabled)
        self.assertTrue(expert._hcu_step5_fused_scatter_enabled)
        self.assertEqual(pf._hcu_step5_buffer_pair, (primary, secondary))

    def test_scatter_uses_all_aligned_expert_slices(self):
        result = self.receive(rows=512, counts=[512, 256] + [0] * 20)
        with self.assertRaises(GemmReached):
            self.apply(result)
        self.assertEqual(self.calls[-1][0][0].shape, (768, 4096))

    def test_empty_scatter_batch_skips_gemm(self):
        for experts in (22, 23):
            for compact in (False, True):
                self.apply(self.receive(experts, rows=0), experts=experts, compact=compact)
                self.assertTrue(self.output.cleared)
        self.assertEqual(self.calls, [])

    def test_non_step5_compact_flag_keeps_conservative_sizing(self):
        result = self.receive(scatter=False)
        with self.assertRaises(GemmReached):
            self.apply(result, compact=True, scatter=False, owner=False)
        self.assertEqual(self.calls[-1][0][0].shape, (7680, 4096))

    def test_planner_without_dispatch_metadata_keeps_conservative_reservation(self):
        for experts, expected in ((22, 7680), (23, 7936)):
            self.assertEqual(self.scope["compute_aligned_M_and_alignment"](
                256, 8, experts, 256, None), (expected, 256))

    def test_receiver_rejects_unaligned_negative_and_wrong_expert_counts(self):
        for counts in ([255] + [0] * 21, [-256] + [0] * 21,
                       [256.0] + [0] * 21, [256] + [0] * 20):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "dispatch counts"):
                self.receive(counts=counts)

    def test_stale_or_invalid_metadata_is_rejected_before_gemm(self):
        def mutate(meta, field, value):
            setattr(meta, field, value)

        changes = {
            "missing_layout": lambda m: delattr(m, "_hcu_step5_fused_scatter_layout"),
            "batch": lambda m: mutate(m, "_hcu_step5_fused_scatter_layout", ((128, 8), (256,) + (0,) * 21)),
            "expert_count": lambda m: mutate(m, "_hcu_step5_fused_scatter_layout", ((256, 8), (256,) + (0,) * 22)),
            "unaligned_count": lambda m: mutate(m, "_hcu_step5_fused_scatter_layout", ((256, 8), (255,) + (0,) * 21)),
            "missing_guard": lambda m: delattr(m, "_hcu_step5_fused_scatter_guard"),
            "short_source": lambda m: mutate(m, "_hcu_step5_fused_scatter", (Tensor((0, 4096)), *m._hcu_step5_fused_scatter[1:])),
            "scale_dtype": lambda m: setattr(m._hcu_step5_fused_scatter[1], "dtype", "fp8"),
            "inverse_shape": lambda m: setattr(m._hcu_step5_fused_scatter[3], "shape", (128, 8)),
            "device": lambda m: setattr(m._hcu_step5_fused_scatter[0], "device", "other"),
            "strides": lambda m: setattr(m._hcu_step5_fused_scatter[0], "is_contiguous", lambda: False),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                result = self.receive()
                change(result[2])
                with self.assertRaisesRegex(ValueError, "Step5 fused scatter"):
                    self.apply(result)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
