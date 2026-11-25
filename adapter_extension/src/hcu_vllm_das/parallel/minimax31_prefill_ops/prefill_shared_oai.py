"""Optional shared-expert down projection with the existing LightOP INT8 fusion."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sysconfig
from typing import NamedTuple

import torch


ROWS = (2048, 4096, 6144, 8192)
OAI = (1.702, 1.0, 7.0)
_plan = None
_split_only = False
_native = None
_gate_table = None
_original_forward = None
_patched_forward = None
_hits = Counter()
_verified = Counter()
_mismatches = Counter()
_split_hits = Counter()
_split_down_verified = Counter()


class PreparedDown(NamedTuple):
    q: torch.Tensor
    scales: torch.Tensor
    weight: torch.Tensor
    weight_scale: torch.Tensor
    reference_activation: torch.Tensor | None


def _gated_down(mlp, hidden):
    down = mlp.down_proj
    gate = mlp.gate_up_proj
    if (hidden.ndim != 2 or hidden.shape[0] not in ROWS or hidden.shape[1] != 6144
            or hidden.dtype != torch.bfloat16 or not hidden.is_cuda
            or getattr(down, "tp_size", None) != 1 or getattr(gate, "tp_size", None) != 1
            or getattr(down, "reduce_results", None) is not False
            or getattr(down, "bias", None) is not None or getattr(gate, "bias", None) is not None
            or (mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit) != OAI):
        return None
    kernel = getattr(getattr(down, "scheme", None), "kernel", None)
    if type(kernel).__name__ != "MiniMax31HipBlasLtInt8Kernel":
        return None
    w, scale, input_scale, input_zp, adjustment = kernel._get_layer_params(down)
    if (input_scale is not None or input_zp is not None or adjustment is not None
            or w.shape != (3072, 6144) or w.stride() != (1, 3072) or w.dtype != torch.int8
            or scale.shape != (6144, 1) or scale.dtype != torch.float32 or not scale.is_contiguous()
            or w.device != hidden.device or scale.device != hidden.device):
        return None
    return w.t(), scale


def forward(original, mlp, hidden):
    if _plan is None or _split_only:
        return original(mlp, hidden)
    down = _gated_down(mlp, hidden)
    if down is None:
        return original(mlp, hidden)

    gate, _ = mlp.gate_up_proj(hidden)
    prepared = _prepare(gate, down)
    if prepared is None:
        from vllm.models.minimax_m3.amd.model import swiglu_oai_split
        result, _ = mlp.down_proj(swiglu_oai_split(
            gate, alpha=OAI[0], beta=OAI[1], limit=OAI[2],
        ))
        return result
    return _run(prepared)


def _prepare(gate, down):
    m = gate.shape[0]
    if gate.shape != (m, 6144) or gate.dtype != torch.bfloat16 or not gate.is_contiguous():
        return None
    q = torch.empty((m, 3072), dtype=torch.int8, device=gate.device)
    scales = torch.empty((m, 1), dtype=torch.float32, device=gate.device)
    _native.quantize(gate, q, scales, *OAI, _gate_table, 64)
    activated = None
    if _plan:
        from vllm import _custom_ops as ops
        from vllm.models.minimax_m3.amd.model import swiglu_oai_split
        activated = swiglu_oai_split(gate, alpha=OAI[0], beta=OAI[1], limit=OAI[2])
        ref_q, ref_scales, zp = ops.scaled_int8_quant(activated, None, None, symmetric=True)
        if zp is not None or not torch.equal(ref_q, q) or not torch.equal(ref_scales, scales):
            # Keep this rank on the normal collective sequence; the caller
            # rejects this candidate after reading the eight-rank counters.
            _mismatches[m] += 1
            return None
        _verified[m] += 1
    weight, weight_scale = down
    _hits[m] += 1
    return PreparedDown(q, scales, weight, weight_scale, activated)


def _run(prepared):
    from lightop._lmslim_native.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
    q, scales, weight, weight_scale, _ = prepared
    m = q.shape[0]
    ok, out = hipblaslt_w8a8_channelwise_gemm(
        q, weight, scales, weight_scale, m, 6144, 3072, "NT", torch.bfloat16, None,
    )
    if (not ok or out.shape not in ((m, 6144), (1, m, 6144)) or not out.is_contiguous()
            or out.dtype != torch.bfloat16 or out.device != q.device):
        raise RuntimeError("Unexpected MiniMax shared expert down GEMM output")
    return out.view(m, 6144)


def prepare_split_down(mlp, hidden, gate):
    if _plan is None or not _split_only or hidden.shape[0] != 6144:
        return None
    down = _gated_down(mlp, hidden)
    if down is None:
        return None
    prepared = _prepare(gate, down)
    if prepared is not None:
        _split_hits[hidden.shape[0]] += 1
    return prepared


def run_split_down(mlp, prepared):
    output = _run(prepared)
    if prepared.reference_activation is not None:
        reference, _ = mlp.down_proj(prepared.reference_activation)
        rows = output.shape[0]
        if not torch.equal(reference.view(torch.uint8), output.view(torch.uint8)):
            _mismatches[rows] += 1
            return reference
        _split_down_verified[rows] += 1
    return output


def configure(value):
    global _plan, _split_only, _native, _gate_table, _original_forward, _patched_forward
    if value == "status":
        return {"enabled": _plan is not None, "verify": _plan is True,
                "split_only": _split_only, "split_hits": dict(_split_hits),
                "split_down_verified": dict(_split_down_verified),
                "hits": dict(_hits), "verified": dict(_verified),
                "mismatches": dict(_mismatches)}
    split_only = isinstance(value, dict)
    if split_only:
        if set(value) != {"split_only", "verify"} or value["split_only"] is not True:
            raise ValueError("Expected an explicit split-only shared OAI plan")
        value = value["verify"]
        if type(value) is not bool:
            raise ValueError("The split-only verification flag must be boolean")
    if value is not None and type(value) is not bool:
        raise ValueError("Set shared OAI to null, true for verify, or false for timing")
    if value is not None and _native is None:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if not arch.startswith("gfx938"):
            raise ValueError("Shared OAI experiment requires gfx938")
        receipt = json.loads(Path(os.environ["MINIMAX_PREFILL_SHARED_OAI_RECEIPT"]).read_text())
        module = Path(receipt["module"])
        expected = (str(torch.__version__), torch.version.hip, sysconfig.get_config_var("SOABI"))
        if tuple(receipt[k] for k in ("torch", "hip", "python_abi")) != expected:
            raise ValueError("LightOP isolated extension ABI differs")
        if hashlib.sha256(module.read_bytes()).hexdigest() != receipt["module_sha256"]:
            raise ValueError("LightOP isolated extension SHA256 differs")
        spec = importlib.util.spec_from_file_location("prefill_shared_oai_native", module)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        device = torch.device("cuda", torch.cuda.current_device())
        table = torch.empty((65536,), dtype=torch.float32, device=device)
        candidate.build_gate_table(table, OAI[0], OAI[2])
        torch.cuda.current_stream(device).synchronize()
        _native, _gate_table = candidate, table
    from vllm.models.minimax_m3.amd.model import MiniMaxM3MLP
    if value is not None and not split_only and _original_forward is None:
        original_forward = MiniMaxM3MLP.forward

        def patched_forward(mlp, hidden):
            return forward(original_forward, mlp, hidden)

        _original_forward, _patched_forward = original_forward, patched_forward
        MiniMaxM3MLP.forward = patched_forward
    elif (value is None or split_only) and _original_forward is not None:
        if MiniMaxM3MLP.forward is not _patched_forward:
            raise RuntimeError("MiniMax shared MLP forward changed during the fusion experiment")
        MiniMaxM3MLP.forward = _original_forward
        _original_forward = _patched_forward = None
    _plan = value
    _split_only = split_only
    _hits.clear()
    _verified.clear()
    _mismatches.clear()
    _split_hits.clear()
    _split_down_verified.clear()
    return configure("status")
