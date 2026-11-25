"""O projection for the fused Verify8 attention result."""
import json
import os
from pathlib import Path

import torch


def supports(proj, rows, device):
    kernel = getattr(getattr(proj, "scheme", None), "kernel", None)
    if kernel is None or not hasattr(kernel, "_get_layer_params"):
        return False
    w, ws, ins, zp, correction = kernel._get_layer_params(proj)
    return (
        rows in (64, 80, 96, 128, 144) and proj.tp_size == 1 and proj.bias is None
        and w.shape == (8192, 6144) and w.dtype == torch.int8 and w.stride() == (1, 8192)
        and ws.shape == (6144, 1) and ws.dtype == torch.float32 and ws.is_contiguous()
        and w.device == ws.device == device and all(x is None for x in (ins, zp, correction))
    )


def _lt_configuration():
    solutions = json.loads(os.environ.get("M31_CROSS_VERIFY_LT", "{}"))
    solution = solutions.get("8192")
    if solution is None:
        return None, None
    path = (Path(os.environ["M31_CROSS_LT_DIR"]) / "lt_plans.so").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Verify8 Lt plans library at {path}")
    return path, solution


def warmup(rows, device):
    path, solution = _lt_configuration()
    if solution is not None:
        from lightop.minimax31_decode_lt import warmup as lt_warmup

        lt_warmup(path, torch.device(device), 6144, rows, 8192, solution)


def bind_projector():
    """Freeze the existing offline Lt plan selection before capture."""
    path, solution = _lt_configuration()
    if solution is None:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm

        def project(q, w, scale, ws):
            return triton_scaled_mm(q, w, scale_a=scale, scale_b=ws, out_dtype=torch.bfloat16).contiguous()
    else:
        from lightop.minimax31_decode_lt import project as lt_project

        def project(q, w, scale, ws):
            return lt_project(path, w.t(), ws, q, scale, solution)

    def apply(proj, q, scale):
        rows = q.numel() // 8192
        if not supports(proj, rows, q.device) or q.dtype != torch.int8 or scale.dtype != torch.float32:
            raise ValueError("Unexpected Verify8 O projection input")
        w, ws, _, _, _ = proj.scheme.kernel._get_layer_params(proj)
        return project(q.view(rows, 8192), w, scale.view(rows, 1), ws)

    return apply
