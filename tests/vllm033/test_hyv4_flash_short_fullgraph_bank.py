"""CPU control-plane contracts; native capture is qualified separately."""
import ast
import importlib.util
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@pytest.fixture
def api():
    source = Path(__file__).resolve().parents[2] / (
        "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/fullgraph_banks.py"
    )
    assert source.is_file(), "request-budget-safe FULL bank implementation absent"
    spec = importlib.util.spec_from_file_location("flash_bank_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indexer(budget=511, ratio=4):
    return NS(group_budget=budget, ratio=ratio, max_model_len=1048576,
              _hcu_fullgraph_capture_bound=None)


class Manager:
    """Explicit Core control-plane double; no GPU/model math is simulated."""
    def __init__(self, name, owners):
        self.name, self.owners = name, owners
        self._capture_descs = {Mode.FULL: [6, 12, 384], Mode.PIECEWISE: [128]}
        self._max_full_descs_to_capture = None
        self._capture_mem_samples = None
        self.pool = object()
        self.graphs = {}
        self.calls = []
        self.captured = False
        self.fail_capture = False

    def capture(self, *args, **kwargs):
        self.calls.append(("capture", tuple(x._hcu_fullgraph_capture_bound
                                             for x in self.owners), args, kwargs))
        if self.fail_capture:
            raise RuntimeError("capture witness failure")
        descs = self._capture_descs[Mode.FULL]
        if self._max_full_descs_to_capture is not None:
            descs = descs[:self._max_full_descs_to_capture]
        for desc in descs:
            self.graphs[desc] = object()
            if self._capture_mem_samples is not None:
                self._capture_mem_samples.append((self.name, desc))
        self.captured = True

    def needs_capture(self):
        return not self.captured

    def release_graphs(self):
        self.calls.append(("release",))
        self.graphs.clear()
        self.captured = False

    def dispatch(self, *args, **kwargs):
        self.calls.append(("dispatch", args, kwargs))
        assert self.name == "wide", "short must not introduce a new dispatch policy"
        return NS(cg_mode=kwargs.get("mode", Mode.FULL), num_tokens=12)

    def run_fullgraph(self, desc):
        self.calls.append(("full", desc))
        return self.name + "-output"

    def run_pw_graph(self, *args, **kwargs):
        self.calls.append(("piecewise", args, kwargs))
        return self.name + "-piecewise-output"

    def captured_token_counts(self):
        return sorted(self.graphs)


def banks(api):
    owners = [indexer(), indexer(31, 128)]
    wide, short = Manager("wide", owners), Manager("short", owners)
    proxy = api.FlashFullGraphBanks(wide, short, owners, 2048, Mode.FULL,
                                   short_pool_factory=object)
    return proxy, wide, short, owners


def test_derive_bound_from_every_indexer(api):
    assert api.derive_short_bound([indexer(), indexer(31, 128)]) == 2048
    assert api.derive_short_bound([]) is None
    for invalid in [indexer(-1), indexer(511, 0), NS(group_budget=511)]:
        assert api.derive_short_bound([indexer(), invalid]) is None


def test_admission_counts_every_request_and_lookahead(api):
    assert api.budget_fits_short([100, 2042], 6, 2048)
    assert not api.budget_fits_short([100, 2043], 6, 2048)
    for limits in [[], [0], [-1], [True], [1.5], [None]]:
        assert not api.budget_fits_short(limits, 6, 2048)


def test_tensor_like_values_never_convert_or_synchronize(api):
    class TensorLike:
        def __index__(self):
            raise AssertionError("would synchronize a device tensor")
    assert not api.budget_fits_short([TensorLike()], 6, 2048)


def test_original_dispatch_remains_authoritative(api):
    proxy, wide, short, owners = banks(api)
    proxy.capture("model", descriptor_marker=object())
    proxy.set_request_limits([2042], 6)
    desc = proxy.dispatch(64, 384, 6, mode=Mode.FULL)
    assert proxy.run_fullgraph(desc) == "short-output"
    assert any(x[0] == "dispatch" for x in wide.calls)
    assert not any(x[0] == "dispatch" for x in short.calls)
    proxy.set_request_limits([2043], 6)
    desc = proxy.dispatch(64, 384, 6, mode=Mode.FULL)
    assert proxy.run_fullgraph(desc) == "wide-output"


def test_piecewise_and_dummy_do_not_inherit_short_selection(api):
    proxy, wide, short, owners = banks(api)
    proxy.capture("model")
    proxy.set_request_limits([500], 6)
    desc = proxy.dispatch(mode=Mode.FULL)
    assert proxy.run_fullgraph(desc) == "short-output"
    proxy.dispatch(mode=Mode.PIECEWISE)
    assert proxy.run_pw_graph("model", {}) == "wide-piecewise-output"
    proxy.use_wide_bank()
    desc = proxy.dispatch(mode=Mode.FULL)
    assert proxy.run_fullgraph(desc) == "wide-output"


def test_capture_bound_restores_and_arguments_are_preserved(api):
    proxy, wide, short, owners = banks(api)
    marker = object()
    proxy.capture(marker, use_aux_hidden_state_outputs=True)
    assert wide.calls[0][1] == (None, None)
    assert short.calls[0][1] == (2048, 2048)
    assert wide.calls[0][2:] == short.calls[0][2:]
    assert all(x._hcu_fullgraph_capture_bound is None for x in owners)
    assert Mode.PIECEWISE in wide._capture_descs
    assert Mode.PIECEWISE not in short._capture_descs


def test_core_profile_pool_samples_and_count_cover_both_banks(api):
    proxy, wide, short, owners = banks(api)
    short_pool = short.pool
    samples = []
    pool = object()
    proxy.pool = pool
    proxy._max_full_descs_to_capture = 2
    proxy._capture_mem_samples = samples
    assert wide.pool is pool
    assert short.pool is short_pool
    assert wide._capture_mem_samples is samples
    assert short._capture_mem_samples is None
    assert short._max_full_descs_to_capture is None
    proxy.capture("profile-model")
    assert samples == [("wide", 6), ("wide", 12)]
    assert len(short.graphs) == 3
    assert len(proxy._capture_descs[Mode.FULL]) == 3
    assert proxy.captured_token_counts() == [6, 12]
    assert not proxy.needs_capture()


def _actual_core_extrapolator():
    root = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
    source = root / "v1/worker/gpu/cudagraph_utils.py"
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "_extrapolate_full_graph_memory")
    ns = {"_MIN_PER_GRAPH_BYTES": 1 << 20}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source) + "::<pure-memory-extrapolator>", "exec"), ns)
    return ns[fn.name]


@pytest.mark.parametrize("wide_costs,short_costs", [
    ([8], [5]),
    ([8, 1], [5, 2]),
    ([8, 1, 1, 1], [5, 2, 2, 2]),
    ([8, 0.25, 0.25], [5, 2, 3]),
    ([8, -1, 1], [5, 1, -1]),
])
def test_complete_core_reservation_charges_all_short_bytes(api, wide_costs, short_costs):
    unit = 1 << 20
    extrapolate = _actual_core_extrapolator()
    owners = [indexer()]

    class MemoryManager(Manager):
        def __init__(self, name, costs):
            super().__init__(name, owners)
            self.costs = [int(value * unit) for value in costs]
            self._capture_descs = {Mode.FULL: list(range(len(costs)))}
            self.measured = []

        def capture(self, *args, **kwargs):
            descs = self._capture_descs[Mode.FULL]
            if self._max_full_descs_to_capture is not None:
                descs = descs[:self._max_full_descs_to_capture]
            for desc in descs:
                self.graphs[desc] = object()
                self.measured.append(self.costs[desc])
                if self._capture_mem_samples is not None:
                    self._capture_mem_samples.append(self.costs[desc])

    wide = MemoryManager("wide", wide_costs)
    short = MemoryManager("short", short_costs)
    proxy = api.FlashFullGraphBanks(wide, short, owners, 2048, Mode.FULL,
                                   short_pool_factory=object)
    samples = []
    proxy._max_full_descs_to_capture = 2
    proxy._capture_mem_samples = samples
    proxy.capture("profile-model")
    measured = sum(wide.measured) + sum(short.measured)
    # This is the complete return expression of the real Core profiler.
    reservation = max(measured - sum(samples) + extrapolate(
        samples, len(proxy._capture_descs[Mode.FULL])), 0)
    expected = max(sum(short.costs) + extrapolate(wide.costs[:2], len(wide.costs)), 0)
    assert reservation == expected
    assert samples == wide.costs[:2]
    assert short.measured == short.costs


def test_release_clears_both_banks_and_requires_new_dispatch(api):
    proxy, wide, short, owners = banks(api)
    previous_short_pool = short.pool
    proxy.capture("model")
    proxy.set_request_limits([100], 6)
    proxy.dispatch(mode=Mode.FULL)
    proxy.release_graphs()
    assert not wide.graphs and not short.graphs
    assert short.pool is not previous_short_pool
    with pytest.raises(RuntimeError, match="dispatch"):
        proxy.run_fullgraph(NS(cg_mode=Mode.FULL))
    proxy.capture("model")
    proxy.set_request_limits([100], 6)
    desc = proxy.dispatch(mode=Mode.FULL)
    assert proxy.run_fullgraph(desc) == "short-output"


def test_partial_capture_latches_failure_and_releases_owner_bound(api):
    proxy, wide, short, owners = banks(api)
    short.fail_capture = True
    with pytest.raises(RuntimeError, match="capture witness"):
        proxy.capture("model")
    assert all(x._hcu_fullgraph_capture_bound is None for x in owners)
    with pytest.raises(RuntimeError):
        proxy.dispatch(mode=Mode.FULL)


def test_preexisting_capture_owner_is_not_overwritten(api):
    proxy, wide, short, owners = banks(api)
    owners[1]._hcu_fullgraph_capture_bound = 4096
    with pytest.raises(RuntimeError):
        proxy.capture("model")
    assert owners[0]._hcu_fullgraph_capture_bound is None
    assert owners[1]._hcu_fullgraph_capture_bound == 4096


@pytest.mark.parametrize("bound,mode,expected", [
    (2048, Mode.FULL, 2048),
    (2048, Mode.NONE, 2048),
    (None, Mode.FULL, 1048576),
    (None, Mode.NONE, 53),
])
def test_production_bound_policy_covers_capture_warmup(bound, mode, expected):
    source = Path(__file__).resolve().parents[2] / (
        "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/compressed_indexer.py"
    )
    tree = ast.parse(source.read_text())
    forward = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assignment = next(n for n in ast.walk(forward)
                      if isinstance(n, ast.Assign) and any(
                          isinstance(t, ast.Name) and t.id == "max_seq_len"
                          for t in n.targets))
    namespace = dict(self=NS(_hcu_fullgraph_capture_bound=bound,
                             max_model_len=1048576), km=NS(max_seq_len=53),
                     get_forward_context=lambda: NS(cudagraph_runtime_mode=mode),
                     CUDAGraphMode=Mode)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],
                                                      type_ignores=[])),
                 "<production-bound-policy-only>", "exec"), namespace)
    assert namespace["max_seq_len"] == expected
