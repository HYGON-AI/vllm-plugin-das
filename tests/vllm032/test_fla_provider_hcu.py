"""Opt-in real HCU tests of the FLA provider, without a model.

Run only with an exclusive physical GPU1 allocation. HCU_FLA_SOURCE_ROOT points
to adapter_extension/src; vLLM must come from the exact a198 target venv. All
scratch/cache paths must be supplied under /dev/shm by the launcher. These tests
do not activate the extension entry point or certify an integration wheel.

For installed-wheel validation set HCU_FLA_TEST_MODE=installed-wheel and
HCU_VLLM032_FLA_LIVE=1. Use features-integration-venv (override its expected path
with HCU_FLA_WHEEL_VENV), unset PYTHONPATH/HCU_FLA_SOURCE_ROOT/ROCR_VISIBLE_DEVICES,
set HIP_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES=HCU_FLA_PHYSICAL_GPU=2 and
VLLM_PLUGINS=hcu_adapter. This mode never inserts source or calls install();
it requires the official general-plugin lifecycle to mount the installed FLA.
Use -k 'hk16-hv16 and not performance' for the tiny model's real head geometry.
The caller owns the exclusive device allocation and all shm cache/temp paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import ctypes
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import statistics
import sys

import pytest


pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(os.getenv("HCU_VLLM032_FLA_LIVE") != "1", reason="explicit exclusive GPU allocation required"),
]
TEST_MODE = os.getenv("HCU_FLA_TEST_MODE", "source")
GEOMETRIES = [(16, 32), (8, 16), (4, 8), (4, 12)]
if TEST_MODE == "installed-wheel":
    GEOMETRIES = [pytest.param(hk, hv, id=f"hk{hk}-hv{hv}")
                  for hk, hv in [(16, 16), *GEOMETRIES]]
DIM = 128
SLOTS = 32


def record(kind, **values):
    row = {"kind": kind, **values}
    line = json.dumps(row, sort_keys=True)
    print("FLA_HCU_RECORD=" + line, flush=True)
    evidence = Path(os.environ["HCU_FLA_EVIDENCE_ROOT"])
    evidence.mkdir(parents=True, exist_ok=True)
    with (evidence / "records.jsonl").open("a") as stream:
        stream.write(line + "\n")


class KernelWitness:
    """Observe a completed call to the real Triton launch, not a selector."""

    def __init__(self, kernel, name, records):
        self.kernel, self.name, self.records = kernel, name, records

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            self.records.append({
                "entry": self.name,
                "null_state_index": kwargs.get("NULL_STATE_INDEX"),
                "compiled_kernel": getattr(compiled, "name", None),
            })
            return compiled

        return observed


def installed_record(distribution_name, module_path, *, venv=None):
    """Verify the actual origin against the installed distribution's RECORD."""
    dist = metadata.distribution(distribution_name)
    files = dist.files
    assert files, f"{distribution_name}: missing RECORD"
    path = Path(module_path).resolve(strict=True)
    if venv is not None:
        assert path.is_relative_to(venv / "lib"), f"source/shared shadow: {path}"
    matches = [f for f in files if Path(dist.locate_file(f)).resolve() == path]
    assert len(matches) == 1, f"{path} is not uniquely owned by {distribution_name}"
    entry = matches[0]
    assert entry.hash is not None and entry.hash.mode == "sha256", f"missing SHA256: {path}"
    data = path.read_bytes()
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    assert encoded == entry.hash.value, f"RECORD hash mismatch: {path}"
    assert entry.size == len(data), f"RECORD size mismatch: {path}"
    records = [f for f in files if str(f).endswith(".dist-info/RECORD")]
    assert len(records) == 1, f"{distribution_name}: ambiguous RECORD"
    record_path = Path(dist.locate_file(records[0])).resolve(strict=True)
    if venv is not None:
        assert record_path.is_relative_to(venv / "lib"), f"shared RECORD: {record_path}"
    # Do not print direct_url.json: it can contain authenticated source URLs.
    direct = dist.read_text("direct_url.json")
    assert not direct or not json.loads(direct).get("dir_info", {}).get("editable"), "editable installs are not wheel evidence"
    return {"distribution": distribution_name, "version": dist.version,
            "origin": str(path), "sha256": digest.hex(), "record_entry": str(entry),
            "record": str(record_path),
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()}


@pytest.fixture(scope="module")
def runtime():
    assert TEST_MODE in {"source", "installed-wheel"}, f"unknown FLA test mode: {TEST_MODE}"
    wheel_mode = TEST_MODE == "installed-wheel"
    physical_gpu = int(os.environ.get("HCU_FLA_PHYSICAL_GPU", "2")) if wheel_mode else 1
    assert physical_gpu >= 0
    assert os.environ.get("HIP_VISIBLE_DEVICES") == str(physical_gpu)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu)
    assert not os.environ.get("ROCR_VISIBLE_DEVICES")
    if wheel_mode:
        venv = Path(os.environ.get("HCU_FLA_WHEEL_VENV",
                    "/dev/shm/vllm032-upgrade-20261031/features-integration-venv")).resolve(strict=True)
        assert Path(sys.prefix).resolve() == venv, "use the independent integration venv"
        assert not os.environ.get("PYTHONPATH"), "installed-wheel mode forbids PYTHONPATH"
        assert not os.environ.get("HCU_FLA_SOURCE_ROOT"), "installed-wheel mode forbids source insertion"
        allowed = os.environ.get("VLLM_PLUGINS")
        assert allowed is None or "hcu_adapter" in allowed.split(","), "general-plugin lifecycle is disabled"
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "1"
        os.environ["VLLM_HCU_USE_CUSTOM_AITER_FLA"] = "1"
    else:
        assert sys.prefix == "/data/models/vllm032-upgrade-20261031/target-venv"
    import torch
    import vllm
    import vllm.envs as envs

    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    hip = ctypes.CDLL("libamdhip64.so")
    pci = ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(pci, 32, 0) == 0
    render = Path(f"/sys/class/drm/renderD{128 + physical_gpu}/device")
    expected = render.resolve(strict=True).name
    assert pci.value.decode().lower() == expected.lower()
    assert vllm.__version__ == "0.32.0+das.a198423b7.dtk2610"
    assert Path(vllm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    arch = props.gcnArchName.split(":")[0]
    wheel_evidence = {}
    if wheel_mode:
        from vllm.plugins import load_general_plugins

        entries = [e for e in metadata.entry_points(group="vllm.general_plugins")
                   if e.name == "hcu_adapter"]
        assert len(entries) == 1 and entries[0].value == "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
        load_general_plugins()
        from hcu_adapter import extensions
        from hcu_adapter.bridges import vllm as bridge
        from hcu_vllm_das import environment, extension, fla
        from hcu_vllm_das.fla import provider as p

        reports = [r for r in bridge.extension_reports() if r.extension_name == "das-hcu"]
        assert len(reports) == 1 and reports[0].status is extensions.ExtensionStatus.ACTIVE, reports
        assert "boltops-fla-decode" in reports[0].capabilities
        policy = json.loads(p.effective_compile_factor())
        assert policy["route"] == "boltops" and policy["mounted"] and policy["owned"], policy
        assert p._installation is not None, "lifecycle did not mount the real provider"
        modules = [("vllm", vllm), ("vllm", envs), ("vllm", import_module("vllm.plugins")),
                   ("sgl-kernel", bridge), ("sgl-kernel", extensions),
                   ("hcu-vllm-das-plugin", fla), ("hcu-vllm-das-plugin", p),
                   ("hcu-vllm-das-plugin", extension), ("hcu-vllm-das-plugin", environment),
                   ("vllm", p._installation.consumer)]
        wheel_evidence["files"] = [installed_record(dist, mod.__file__, venv=venv) for dist, mod in modules]
        provider_proof = next(r for r in wheel_evidence["files"] if r["origin"] == str(Path(p.__file__).resolve()))
        assert policy["identity"]["adapter"] == provider_proof["sha256"]
        expected_record = os.environ.get("HCU_FLA_EXPECT_PLUGIN_RECORD_SHA256")
        assert expected_record is None or provider_proof["record_sha256"] == expected_record, "unexpected installed Plugin RECORD"
        wheel_evidence["entry_point"] = entries[0].value
        wheel_evidence["activation_report"] = {"name": reports[0].extension_name,
            "status": reports[0].status.value, "capabilities": sorted(reports[0].capabilities)}
    else:
        source = Path(os.environ["HCU_FLA_SOURCE_ROOT"]).resolve()
        sys.path.insert(0, str(source))
        from hcu_vllm_das.fla import provider as p

        assert Path(p.__file__).resolve() == source / "hcu_vllm_das/fla/provider.py"
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "1"
        os.environ["VLLM_HCU_USE_CUSTOM_AITER_FLA"] = "1"
        p.register_compile_factors(envs)
        assert p.install(architecture=arch), "provider unavailable: not a native pass"
    originals = p._installation.originals.copy()
    vendor = import_module("boltops.fla.gdn")
    if wheel_mode:
        wheel_evidence["vendor_files"] = [installed_record("boltops", vendor.__file__)]
        for name in (p.PACKED, p.SIGMOID):
            owner = import_module(getattr(vendor, name).__module__)
            wheel_evidence["vendor_files"].append(installed_record("boltops", owner.__file__))
    launched = []
    patch = pytest.MonkeyPatch()
    for name in (p.PACKED, p.SIGMOID):
        function = getattr(vendor, name)
        owner = import_module(function.__module__)
        kernel_name = name + "_kernel"
        kernel = getattr(owner, kernel_name)
        patch.setattr(owner, kernel_name, KernelWitness(kernel, name, launched))
    manifest = {
        "python": sys.executable, "vllm_version": vllm.__version__,
        "vllm_file": vllm.__file__, "plugin_source": p.__file__,
        "stage": ("installed integration wheel via official general-plugin lifecycle" if wheel_mode else
                  "FLA worktree source plus exact target vLLM wheel; not integration wheel"),
        "mode": TEST_MODE, "installed_wheel": wheel_evidence,
        "architecture": arch, "compute_units": props.multi_processor_count,
        "physical_gpu": physical_gpu, "pci": pci.value.decode(), "visible_count": 1,
        "pci_sysfs": str(render), "plugin_file": p.__file__,
        "versions": {name: metadata.version(name) for name in ("torch", "triton", "boltops", "vllm")},
        "factor": json.loads(p.effective_compile_factor()),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": {"output_rtol": 0.02, "output_atol": 0.002, "output_nrmse": 0.005,
                   "state_rtol": 0.002, "state_atol": 0.0002, "state_nrmse": 0.001},
    }
    record("manifest", **manifest)
    rt = Runtime(torch, p, originals, launched)
    yield rt
    torch.cuda.synchronize()
    patch.undo()
    assert p.uninstall()


@dataclass
class Case:
    name: str
    args: dict
    lengths: list[int]
    indices: list[list[int]]
    accepted: list[int] | None
    state_backing: object | None = None

    @property
    def state(self):
        return self.args["initial_state"]

    def pointers(self):
        pointers = {k: v.data_ptr() for k, v in self.args.items() if hasattr(v, "data_ptr")}
        if self.state_backing is not None:
            pointers["state_backing"] = self.state_backing.data_ptr()
        return pointers


class Runtime:
    def __init__(self, torch, provider, originals, launched):
        self.t, self.p, self.originals, self.launched = torch, provider, originals, launched
        self.device = torch.device("cuda", 0)
        self.generator = torch.Generator(device=self.device).manual_seed(20261031)

    def rand(self, shape, dtype=None, gain=0.2):
        return self.t.randn(shape, generator=self.generator, device=self.device,
                            dtype=dtype or self.t.bfloat16) * gain

    def integer(self, values):
        return self.t.tensor(values, dtype=self.t.int32, device=self.device)

    def make(self, name, hk, hv, *, batch=4, mode="varlen", wave=0):
        t = self.t
        common = dict(A_log=self.rand((hv,), t.float32) - 1.5,
                      dt_bias=self.rand((hv,), t.float32),
                      initial_state=self.rand((SLOTS, hv, DIM, DIM), t.float32, 0.05),
                      scale=DIM**-0.5, use_qk_l2norm_in_kernel=True)
        if mode == "pool":
            common["initial_state"] = common["initial_state"][:8]
        if name == self.p.PACKED:
            width = (2 * hk + hv) * DIM
            backing = self.rand((batch, width + 16))
            gates = self.rand((batch, 2 * hv))
            ids = [1 + i for i in range(batch)]
            if batch > 1:
                ids = [((i + wave) % batch) + 1 for i in range(batch)]
                ids[wave % batch] = 0
                if batch > 2:
                    ids[(wave + 1) % batch] = -1
            common.update(mixed_qkv=backing[:, :width], a=gates[:, :hv], b=gates[:, hv:],
                          out=t.full((batch, 1, hv, DIM), float("nan"), device=self.device, dtype=t.bfloat16),
                          ssm_state_indices=self.integer(ids))
            return Case(name, common, [1] * batch, [[i] for i in ids], None)
        if mode == "fixed":
            bsz, total, lengths, count = 4, 1, [1] * 4, 4
            rows = [[i + 1] for i in range(count)]
            rows[wave % count] = [0]
            rows[(wave + 1) % count] = [-1]
            accepted, cu = None, None
            indices = self.integer([row[0] for row in rows])
        elif mode == "pool":
            # Reproduce the witnessed B1/T32, eight-slot pool and [8,4]
            # speculative indices. Each sequence owns a distinct slot; repeated
            # indices within its row exercise sequential in-place MTP updates.
            bsz, total, count, lengths = 1, 32, 8, [4] * 8
            rows = [[(n + wave) % 8] * 4 for n in range(count)]
            if wave % 2:
                rows[next(n for n, row in enumerate(rows) if row[0] == 0)] = [-1] * 4
            accepted = [1 + (wave + n) % 4 for n in range(count)]
            cu, indices = self.integer(list(range(0, 33, 4))), self.integer(rows)
        elif mode == "fixed_multi":
            bsz, total, lengths, count = 2, 4, [4, 4], 2
            rows = [[1 + n * 7 + i + wave % 2 for i in range(4)] for n in range(count)]
            rows[0][-1] = 0
            accepted, cu = None, None
            indices = self.integer(rows)
        else:
            bsz, total, count = 1, 8, 4
            lengths = ([1, 3, 4, 0], [2, 2, 1, 3], [0, 4, 2, 2], [3, 1, 0, 4])[wave % 4]
            rows = [[1 + n * 7 + i + wave % 3 for i in range(4)] for n in range(count)]
            accepted = [1 + (wave + n) % max(1, length) for n, length in enumerate(lengths)]
            # Distinguish the vLLM NULL=0 ABI from the provider's SGLang default.
            null_seq = next(n for n, length in enumerate(lengths) if length)
            rows[null_seq][accepted[null_seq] - 1] = 0 if wave % 2 == 0 else -1
            valid_seq = next(n for n, length in enumerate(lengths) if length and n != null_seq)
            if lengths[valid_seq] > 1:
                # A NULL write target after a valid initial slot must stay untouched.
                final = lengths[valid_seq] - 1
                if final != accepted[valid_seq] - 1:
                    rows[valid_seq][final] = 0
            boundaries = [0]
            for length in lengths:
                boundaries.append(boundaries[-1] + length)
            cu, indices = self.integer(boundaries), self.integer(rows)
        common.update(q=self.rand((bsz, total, hk, DIM)), k=self.rand((bsz, total, hk, DIM)),
                      v=self.rand((bsz, total, hv, DIM)), a=self.rand((bsz * total, hv)),
                      b=self.rand((bsz * total, hv)), beta=1.0, threshold=20.0,
                      inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=indices,
                      num_accepted_tokens=None if accepted is None else self.integer(accepted), is_kda=False)
        return Case(name, common, lengths, rows, accepted)

    def clone(self, case):
        # Preserve source strides, including Qwen's sliced projection rows.
        copied = {}
        backing = None if case.state_backing is None else case.state_backing.clone()
        for k, value in case.args.items():
            if isinstance(value, self.t.Tensor):
                if k == "initial_state" and backing is not None:
                    target = backing.as_strided(value.shape, value.stride(), value.storage_offset())
                else:
                    target = self.t.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
                    target.copy_(value)
                copied[k] = target
            else:
                copied[k] = value
        return Case(case.name, copied, case.lengths, case.indices, case.accepted, backing)

    def pad_state(self, case, *, offset):
        """Model shared-pool slot pitch, with protected prefix/gaps/trailer."""
        assert case.state_backing is None
        state = case.state
        slots, hv, values, keys = state.shape
        pitch = hv * values * keys + 24576  # Real tiny-model slot gap: 96 KiB FP32.
        backing = self.t.full((offset + slots * pitch + 29,), -918.25,
                              dtype=state.dtype, device=state.device)
        view = backing.as_strided(state.shape, (pitch, values * keys, keys, 1), offset)
        view.copy_(state)
        case.args["initial_state"], case.state_backing = view, backing
        return case

    def check_state_padding(self, case, before, *, label):
        t, state, backing = self.t, case.state, case.state_backing
        assert backing is not None and backing.shape == before.shape
        assert state.data_ptr() == backing.data_ptr() + state.storage_offset() * state.element_size()
        protected = t.ones(backing.shape, device=backing.device, dtype=t.bool)
        protected.as_strided(state.shape, state.stride(), state.storage_offset()).fill_(False)
        actual, expected = backing[protected], before[protected]
        assert actual.numel() > 0
        assert t.equal(actual.view(t.uint8), expected.view(t.uint8)), f"{label}: shared-pool padding changed"
        return {"label": label, "shape": list(state.shape), "strides": list(state.stride()),
                "storage_offset": state.storage_offset(), "protected_elements": actual.numel(),
                "padding_bitwise_untouched": True}

    def update(self, destination, source):
        assert destination.args.keys() == source.args.keys()
        if destination.state_backing is not None:
            assert source.state_backing is not None
            assert destination.state.stride() == source.state.stride()
            assert destination.state.storage_offset() == source.state.storage_offset()
            destination.state_backing.copy_(source.state_backing)
        for k, value in destination.args.items():
            if isinstance(value, self.t.Tensor):
                if k == "initial_state" and destination.state_backing is not None:
                    continue
                value.copy_(source.args[k])
            else:
                assert value == source.args[k]
        destination.lengths, destination.indices, destination.accepted = source.lengths, source.indices, source.accepted

    def call(self, case, *, vendor):
        before = len(self.launched)
        fn = getattr(self.p._installation.consumer, case.name) if vendor else self.originals[case.name]
        output, state = fn(**case.args)
        assert state is case.state and state.data_ptr() == case.state.data_ptr()
        if case.name == self.p.PACKED:
            assert output is case.args["out"]
        if vendor:
            calls = self.launched[before:]
            assert len(calls) == 1 and calls[0]["entry"] == case.name, "fallback is not a native pass"
            assert calls[0]["null_state_index"] == 0
            assert calls[0]["compiled_kernel"], "real compiled-kernel witness required"
        return output, state

    def oracle(self, case):
        """Independent FP32 physical recurrence and CPU-known metadata semantics."""
        t, a = self.t, case.args
        state = case.state.clone()
        hv, dim = state.shape[1], state.shape[-1]
        if case.name == self.p.PACKED:
            hk = (a["mixed_qkv"].shape[-1] - hv * dim) // (2 * dim)
            q, k, v = a["mixed_qkv"].float().split((hk * dim, hk * dim, hv * dim), dim=-1)
            q, k, v = q.view(-1, hk, dim), k.view(-1, hk, dim), v.view(-1, hv, dim)
        else:
            q, k, v = (a[key].float().reshape(-1, a[key].shape[-2], dim) for key in ("q", "k", "v"))
            hk = q.shape[-2]
        gates_a, gates_b = a["a"].float().reshape(-1, hv), a["b"].float().reshape(-1, hv)
        out = t.full((v.shape[0], hv, dim), float("nan"), device=self.device, dtype=t.float32)
        valid = [False] * len(out)
        touched = set()
        offset = 0
        for n, length in enumerate(case.lengths):
            pos = 0 if case.accepted is None else case.accepted[n] - 1
            slot = case.indices[n][pos]
            if length and slot > 0:
                h = case.state[slot].float().clone()
                for step in range(length):
                    token = offset + step
                    qt = q[token].repeat_interleave(hv // hk, dim=0)
                    kt = k[token].repeat_interleave(hv // hk, dim=0)
                    if a["use_qk_l2norm_in_kernel"]:
                        qt = qt * t.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                        kt = kt * t.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
                    beta = a.get("beta", 1.0)
                    threshold = a.get("threshold", 20.0)
                    x = gates_a[token] + a["dt_bias"]
                    decay = -a["A_log"].exp() * t.nn.functional.softplus(x, beta=beta, threshold=threshold)
                    h = h * decay.exp()[:, None, None]
                    delta = (v[token] - (h * kt[:, None, :]).sum(-1)) * gates_b[token].sigmoid()[:, None]
                    h = h + delta[:, :, None] * kt[:, None, :]
                    out[token] = (h * qt[:, None, :]).sum(-1) * a["scale"]
                    valid[token] = True
                    target = case.indices[n][step]
                    if target > 0:
                        state[target] = h
                        touched.add(target)
            elif case.name == self.p.PACKED:
                out[offset] = 0
                valid[offset] = True
            offset += length
        return out, state, valid, sorted(touched)

    def compare(self, case, actual, reference, *, label):
        t = self.t
        expected_o, expected_s, valid, touched = reference
        output, state = actual
        mask = self.integer(valid).bool()
        flat = output.reshape(-1, output.shape[-2], DIM).float()
        assert mask.sum().item() > 0 and touched, "nontrivial active tokens/states required"
        defined = flat[mask]
        expected = expected_o.reshape_as(flat)[mask].float()
        assert t.isfinite(defined).all() and t.isfinite(state).all()
        t.testing.assert_close(defined, expected, rtol=0.02, atol=0.002)
        t.testing.assert_close(state, expected_s, rtol=0.002, atol=0.0002)
        active = t.tensor(touched, device=self.device)
        t.testing.assert_close(state[0], case.state[0], rtol=0, atol=0)
        stats = {}
        for kind, x, y, limit in [("output", defined, expected, 0.005),
                                    ("state", state[active], expected_s[active], 0.001)]:
            error = x.float() - y.float()
            rms = y.float().square().mean().sqrt().item()
            nrmse = error.square().mean().sqrt().item() / max(rms, 1e-12)
            assert rms > 1e-5 and nrmse <= limit, (label, kind, rms, nrmse)
            stats[kind] = {"reference_rms": rms, "nrmse": nrmse, "max_abs": error.abs().max().item()}
        untouched = [i for i in range(case.state.shape[0]) if i not in touched]
        t.testing.assert_close(state[untouched], expected_s[untouched], rtol=0, atol=0)
        assert (expected_s[active] != case.state[active]).any().item()
        return {"label": label, "metrics": stats, "defined_tokens": sum(valid),
                "undefined_null_or_empty_tokens": len(valid) - sum(valid), "touched_slots": touched}


    def compare_pair(self, provider, official, reference, *, label):
        """Report direct equality separately from FP32-oracle/model gates."""
        t = self.t
        valid, touched = reference[2:]
        mask, active = self.integer(valid).bool(), self.integer(touched).long()
        output = provider[0].reshape(-1, provider[0].shape[-2], DIM)[mask]
        baseline = official[0].reshape(-1, official[0].shape[-2], DIM)[mask]
        pairs = {"output": (output, baseline), "state": (provider[1], official[1]),
                 "state_updated": (provider[1][active], official[1][active])}
        metrics = {}
        for kind, (x, y) in pairs.items():
            assert x.shape == y.shape and x.dtype == y.dtype and x.numel() > 0
            error = x.float() - y.float()
            rms = y.float().square().mean().sqrt().item()
            metrics[kind] = {
                "exact": t.equal(x, y),
                "bitwise_exact": t.equal(x.contiguous().view(t.uint8), y.contiguous().view(t.uint8)),
                "num_elements": x.numel(), "num_mismatched": (x != y).sum().item(),
                "max_abs": error.abs().max().item(), "reference_rms": rms,
                "nrmse": error.square().mean().sqrt().item() / max(rms, 1e-12),
            }
        result = {"label": label, "defined_tokens": sum(valid),
                  "undefined_null_or_empty_tokens": len(valid) - sum(valid),
                  "output_scope": "defined tokens only", "state_scope": "full pool and updated slots",
                  "metrics": metrics}
        # Preserve diagnostic metrics even when the direct comparison fails.
        record("provider_official_comparison", **result)
        t.testing.assert_close(output, baseline, rtol=0.02, atol=0.002)
        t.testing.assert_close(provider[1], official[1], rtol=0.002, atol=0.0002)
        return result


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_packed_native_matches_official_state_alias_and_null(runtime, hk, hv, batch):
    rt = runtime
    base = rt.make(rt.p.PACKED, hk, hv, batch=batch)
    reference = rt.oracle(base)
    official, vendor = rt.clone(base), rt.clone(base)
    a, b = rt.call(official, vendor=False), rt.call(vendor, vendor=True)
    rt.t.cuda.synchronize()
    pair = rt.compare_pair(b, a, reference, label="packed-provider-vs-official")
    rows = [rt.compare(base, a, reference, label="official-vs-physical"),
            rt.compare(base, b, reference, label="provider-vs-physical")]
    record("native_pass", entry="packed", hk=hk, hv=hv, dim=DIM, batch=batch,
           checks=rows, provider_vs_official=pair, kernel=rt.launched[-1], aliases=True)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("mode", ["fixed", "fixed_multi", "varlen"])
def test_sigmoid_native_matches_official_state_alias_mtp_varlen(runtime, hk, hv, mode):
    rt = runtime
    base = rt.make(rt.p.SIGMOID, hk, hv, mode=mode)
    reference = rt.oracle(base)
    official, vendor = rt.clone(base), rt.clone(base)
    a, b = rt.call(official, vendor=False), rt.call(vendor, vendor=True)
    rt.t.cuda.synchronize()
    pair = rt.compare_pair(b, a, reference, label="sigmoid-provider-vs-official")
    rows = [rt.compare(base, a, reference, label="official-vs-physical"),
            rt.compare(base, b, reference, label="provider-vs-physical")]
    record("native_pass", entry="sigmoid", hk=hk, hv=hv, dim=DIM, mode=mode,
           lengths=base.lengths, accepted=base.accepted, checks=rows,
           provider_vs_official=pair, kernel=rt.launched[-1], aliases=True)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("entry", ["packed", "sigmoid"])
@pytest.mark.parametrize("offset", [0, 37], ids=["padded-slot", "padded-slot-offset"])
def test_slot_pitched_state_native_preserves_storage_offset_and_padding(runtime, hk, hv, entry, offset):
    rt = runtime
    name = rt.p.PACKED if entry == "packed" else rt.p.SIGMOID
    base = rt.pad_state(rt.make(name, hk, hv, mode="pool"), offset=offset)
    reference = rt.oracle(base)
    official, vendor = rt.clone(base), rt.clone(base)
    before = base.state_backing.clone()
    a, b = rt.call(official, vendor=False), rt.call(vendor, vendor=True)
    rt.t.cuda.synchronize()
    pair = rt.compare_pair(b, a, reference, label=f"{entry}-slot-pitch-provider-vs-official")
    checks = [rt.compare(base, a, reference, label="official-slot-pitch-vs-physical"),
              rt.compare(base, b, reference, label="provider-slot-pitch-vs-physical")]
    padding = [rt.check_state_padding(case, before, label=label)
               for case, label in [(official, "official"), (vendor, "provider")]]
    record("slot_pitched_native_pass", entry=entry, hk=hk, hv=hv, dim=DIM,
           checks=checks, padding=padding, provider_vs_official=pair,
           kernel=rt.launched[-1], aliases=True)


def capture(rt, case, *, vendor):
    stream = rt.t.cuda.Stream()
    stream.wait_stream(rt.t.cuda.current_stream())
    with rt.t.cuda.stream(stream):
        for _ in range(3):
            rt.call(case, vendor=vendor)
    rt.t.cuda.current_stream().wait_stream(stream)
    rt.t.cuda.synchronize()
    graph = rt.t.cuda.CUDAGraph()
    with rt.t.cuda.graph(graph, stream=stream):
        outputs = rt.call(case, vendor=vendor)
    return graph, outputs


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("entry", ["packed", "sigmoid"])
def test_slot_pitched_state_graph_same_addresses_offset_padding_all_waves(runtime, hk, hv, entry):
    rt = runtime
    name = rt.p.PACKED if entry == "packed" else rt.p.SIGMOID
    base = rt.pad_state(rt.make(name, hk, hv, mode="pool"), offset=37)
    official, vendor = rt.clone(base), rt.clone(base)
    before = base.state_backing.clone()
    reference_graph, reference_out = capture(rt, official, vendor=False)
    provider_graph, provider_out = capture(rt, vendor, vendor=True)
    capture_padding = [rt.check_state_padding(case, before, label=f"{label}-capture")
                       for case, label in [(official, "official"), (vendor, "provider")]]
    pointers = [case.pointers() for case in (official, vendor)]
    output_pointers = [result[0].data_ptr() for result in (reference_out, provider_out)]
    waves, digests = [], []
    for wave in range(8):
        fresh = rt.pad_state(rt.make(name, hk, hv, mode="pool", wave=wave), offset=37)
        expected = rt.oracle(fresh)
        rt.update(official, fresh)
        rt.update(vendor, fresh)
        reference_graph.replay()
        provider_graph.replay()
        rt.t.cuda.synchronize()
        assert [case.pointers() for case in (official, vendor)] == pointers
        assert [result[0].data_ptr() for result in (reference_out, provider_out)] == output_pointers
        assert reference_out[1] is official.state and provider_out[1] is vendor.state
        pair = rt.compare_pair(provider_out, reference_out, expected,
                               label=f"{entry}-slot-pitch-graph-wave-{wave}")
        checks = [rt.compare(fresh, reference_out, expected, label="official-slot-pitch-graph-vs-physical"),
                  rt.compare(fresh, provider_out, expected, label="provider-slot-pitch-graph-vs-physical")]
        padding = [rt.check_state_padding(case, fresh.state_backing, label=label)
                   for case, label in [(official, "official"), (vendor, "provider")]]
        defined = provider_out[0].reshape(-1, hv, DIM)[rt.integer(expected[2]).bool()]
        digests.append(hashlib.sha256(defined.float().cpu().numpy().tobytes()).hexdigest())
        waves.append({"wave": wave, "lengths": fresh.lengths, "indices": fresh.indices,
                      "accepted": fresh.accepted, "checks": checks, "padding": padding,
                      "provider_vs_official": pair})
    assert len(set(digests)) == len(digests)
    record("slot_pitched_graph_pass", entry=entry, hk=hk, hv=hv, dim=DIM,
           capture_padding=capture_padding, waves=waves, kernel=rt.launched[-1],
           input_addresses_stable=True, output_address_stable=True, output_digests=digests)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("entry", ["packed", "sigmoid"])
def test_graph_multiple_waves_same_addresses_changed_inputs_indices_accepted(runtime, hk, hv, entry):
    rt = runtime
    name = rt.p.PACKED if entry == "packed" else rt.p.SIGMOID
    base = rt.make(name, hk, hv)
    official, vendor = rt.clone(base), rt.clone(base)
    reference_graph, reference_out = capture(rt, official, vendor=False)
    provider_graph, provider_out = capture(rt, vendor, vendor=True)
    ptrs = vendor.pointers()
    output_ptr = provider_out[0].data_ptr()
    waves, digests = [], []
    for wave in range(8):
        fresh = rt.make(name, hk, hv, wave=wave)
        expected = rt.oracle(fresh)
        rt.update(official, fresh)
        rt.update(vendor, fresh)
        reference_graph.replay()
        provider_graph.replay()
        rt.t.cuda.synchronize()
        assert vendor.pointers() == ptrs and provider_out[0].data_ptr() == output_ptr
        assert provider_out[1] is vendor.state
        pair = rt.compare_pair(provider_out, reference_out, expected,
                               label=f"{entry}-graph-wave-{wave}-provider-vs-official")
        a = rt.compare(fresh, reference_out, expected, label="official-graph-vs-physical")
        b = rt.compare(fresh, provider_out, expected, label="provider-graph-vs-physical")
        defined = provider_out[0].reshape(-1, hv, DIM)[rt.integer(expected[2]).bool()]
        digests.append(hashlib.sha256(defined.float().cpu().numpy().tobytes()).hexdigest())
        waves.append({"wave": wave, "lengths": fresh.lengths, "indices": fresh.indices,
                      "accepted": fresh.accepted, "checks": [a, b],
                      "provider_vs_official": pair})
    assert len(set(digests)) == len(digests)
    record("graph_pass", entry=entry, hk=hk, hv=hv, dim=DIM, waves=waves,
           input_addresses_stable=True, output_address_stable=True,
           output_digests=digests, kernel=rt.launched[-1])


@pytest.mark.parametrize("entry", ["packed", "sigmoid"])
@pytest.mark.parametrize("reason", ["off", "fp16-unsupported"])
def test_fallback_is_separate_from_native_coverage(runtime, entry, reason):
    rt = runtime
    name = rt.p.PACKED if entry == "packed" else rt.p.SIGMOID
    base = rt.make(name, 4, 8)
    if reason == "fp16-unsupported":
        keys = ("mixed_qkv", "out", "a", "b") if entry == "packed" else ("q", "k", "v", "a", "b")
        for key in keys:
            base.args[key] = base.args[key].half()
    official, candidate = rt.clone(base), rt.clone(base)
    expected = rt.call(official, vendor=False)
    before = len(rt.launched)
    try:
        if reason == "off":
            os.environ["VLLM_HCU_USE_CUSTOM_AITER_FLA"] = "0"
        actual = getattr(rt.p._installation.consumer, name)(**candidate.args)
        rt.t.cuda.synchronize()
    finally:
        os.environ["VLLM_HCU_USE_CUSTOM_AITER_FLA"] = "1"
    assert len(rt.launched) == before
    rt.t.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    if entry == "packed":
        mask = [True] * len(base.lengths)
    else:
        mask = rt.oracle(base)[2]
    valid = rt.integer(mask).bool()
    rt.t.testing.assert_close(actual[0].reshape(-1, 8, DIM)[valid], expected[0].reshape(-1, 8, DIM)[valid], rtol=0, atol=0)
    record("fallback_pass", entry=entry, reason=reason, vendor_launches=0)


@pytest.mark.skipif(os.getenv("HCU_VLLM032_FLA_PERF") != "1", reason="run only after numerical/Graph phase passes")
@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("entry", ["packed", "sigmoid"])
def test_performance_three_rounds_of_graph_comparison(runtime, hk, hv, entry):
    rt = runtime
    name = rt.p.PACKED if entry == "packed" else rt.p.SIGMOID
    base = rt.make(name, hk, hv)
    oracle = rt.oracle(base)
    # Recheck numerics in this performance process before any timing.
    for native in (False, True):
        case = rt.clone(base)
        rt.compare(base, rt.call(case, vendor=native), oracle, label="pre-performance")
    official, vendor = rt.clone(base), rt.clone(base)
    a, _ = capture(rt, official, vendor=False)
    b, _ = capture(rt, vendor, vendor=True)
    rounds = []
    for round_id in range(3):
        samples = {"official": [], "provider": []}
        order = [("official", a), ("provider", b)]
        if round_id % 2:
            order.reverse()
        for kind, graph in order:
            for _ in range(10):
                graph.replay()
            rt.t.cuda.synchronize()
            for _ in range(5):
                start, end = rt.t.cuda.Event(enable_timing=True), rt.t.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[kind].append(start.elapsed_time(end) * 1000 / 200)
        medians = {k: statistics.median(v) for k, v in samples.items()}
        rounds.append({"round": round_id, "microseconds_samples": samples,
                       "microseconds_median": medians, "speedup": medians["official"] / medians["provider"]})
    record("performance", entry=entry, hk=hk, hv=hv, dim=DIM, rounds=rounds,
           scope="Graph kernel path; no model, provider sigmoid allocation included in capture; reset excluded equally")
