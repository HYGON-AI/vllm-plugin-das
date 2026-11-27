"""Portable contract checks; target qualification separately executes real Core."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/dfly.py'


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.ndim = 1

    def narrow(self, dim, start, count):
        assert dim == 0
        return Tensor(self.values[start:start + count])

    def contiguous(self):
        return self


class SinkContract(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_slice_sink_for_tp', '_sink_weight_for_loader')]
        self.global_loader = lambda param, weights: None
        self.ns = {'weight_utils': SimpleNamespace(
            load_attention_sink=self.global_loader)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.ns)
        self.select = self.ns['_sink_weight_for_loader']

    def test_modern_loader_preserves_global_object_every_rank(self):
        weights = Tensor(range(32))
        param = SimpleNamespace(weight_loader=self.global_loader)
        for tp in (4, 8):
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    self.assertIs(self.select(param, weights, num_heads=32,
                                              tp_size=tp, tp_rank=rank), weights)

    def test_raw_custom_and_missing_loader_receive_local_slice(self):
        for param in (SimpleNamespace(weight_loader=lambda p, w: None),
                      SimpleNamespace()):
            for tp in (4, 8):
                for rank in range(tp):
                    with self.subTest(tp=tp, rank=rank):
                        selected = self.select(param, Tensor(range(32)),
                                               num_heads=32, tp_size=tp, tp_rank=rank)
                        self.assertEqual(selected.values,
                                         list(range(rank * (32 // tp), (rank + 1) * (32 // tp))))

    def test_old_core_without_export_keeps_local_contract(self):
        del self.ns['weight_utils'].load_attention_sink
        selected = self.select(SimpleNamespace(), Tensor(range(32)),
                               num_heads=32, tp_size=8, tp_rank=7)
        self.assertEqual(selected.values, [28, 29, 30, 31])

    def test_global_loader_does_not_bypass_layout_validation(self):
        p = SimpleNamespace(weight_loader=self.global_loader)
        for size, heads, tp, rank in ((31, 32, 8, 0), (32, 31, 8, 0),
                                      (32, 32, 0, 0), (32, 32, True, 0),
                                      (32, 32, 8, -1), (32, 32, 8, 8)):
            with self.subTest(size=size, heads=heads, tp=tp, rank=rank):
                with self.assertRaises(ValueError):
                    self.select(p, Tensor(range(size)), num_heads=heads,
                                tp_size=tp, tp_rank=rank)

    def test_load_weights_uses_actual_sink_parameter(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'HYV4DFlyForCausalLM')
        load = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'load_weights')
        calls = [n for n in ast.walk(load) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == '_sink_weight_for_loader']
        self.assertEqual(len(calls), 1)
        param = calls[0].args[0]
        self.assertIsInstance(param, ast.Call)
        self.assertEqual(ast.unparse(param.func), 'self.model.get_parameter')
        self.assertEqual(ast.unparse(param.args[0]), 'name')


if __name__ == '__main__':
    unittest.main()
