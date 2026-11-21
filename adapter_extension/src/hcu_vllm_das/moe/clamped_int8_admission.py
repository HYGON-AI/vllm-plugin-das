# SPDX-License-Identifier: Apache-2.0
"""Admission for the byte-verified DSV4.1 W4A8 routed activation."""
from __future__ import annotations

from functools import lru_cache
import importlib
import logging

import torch


_LOG = logging.getLogger("vllm." + __name__)


@lru_cache(maxsize=2)
def _log_admission(masked):
    # vLLM's INFO handler lives on its non-propagating namespace. Retain
    # the Plugin identity below it so serving admission is observable.
    logging.getLogger("vllm." + __name__).info(
        "HCU W4A8 live-row INT8 activation admitted: layout=%s H5120 N2304 clamp10",
        "masked" if masked else "contiguous")


@lru_cache(maxsize=1)
def _verified_provider():
    try:
        module = importlib.import_module("lightop.fuse_silu_mul_quant")
        extension = module.op
        functions = (module.fuse_silu_mul_clamp_quant, module.fuse_silu_mul_clamp_quant_ep,
                     extension.fuse_silu_mul_clamp_quant, extension.fuse_silu_mul_clamp_quant_ep)
    except (ImportError, OSError, AttributeError):
        return None
    return module if all(callable(fn) for fn in functions) else None


@lru_cache(maxsize=32)
@torch.inference_mode()
def _matches_native(native, device, experts, masked):
    """Qualify the loaded provider once before capture on this worker."""
    try:
        from triton.errors import TritonError
        from .clamped_int8_triton import clamped_int8_2304
    except (ImportError, OSError) as error:
        _LOG.warning('HCU live-row INT8 preflight unavailable; using native activation: %s', error)
        return False

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        # A full expert-stride probe exercises the same native launch geometry.
        # Bound peak startup memory and keep all probes off the hot path.
        rows = 2048 if masked else 4096
        shape = (experts, rows, 4608) if masked else (rows, 4608)
        elements = (experts if masked else 1) * rows * 4608
        if torch.cuda.mem_get_info(device)[0] < 5 * elements + 64 * 1024**2:
            _LOG.info('HCU live-row INT8 preflight skipped: insufficient free device memory')
            return False
        try:
            gateup = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            generator = torch.Generator(device=device).manual_seed(30419)
            if masked:
                seed = torch.randn((experts, 64, 4608), device=device,
                                   generator=generator, dtype=torch.float32)
                gateup[:, :64] = (seed * 8).to(torch.bfloat16)
                row = torch.arange(rows, device=device)
            else:
                seed = torch.randn((64, 4608), device=device,
                                   generator=generator, dtype=torch.float32)
                gateup[:64].copy_((seed * 8).to(torch.bfloat16))
                row = torch.arange(rows, device=device, dtype=torch.int32)
            for hint in (16, 32, 48):
                if masked:
                    meta = (torch.arange(experts, dtype=torch.int32, device=device)
                            % (hint + 1)).contiguous()
                    meta[-1] = 64
                    live = row[None, :] < meta[:, None]
                    baseline = native(gateup, limit=10.0, mask_m=meta, expect_m=hint)
                else:
                    live = row < hint
                    meta = torch.where(live, 0, -1).to(torch.int32)
                    baseline = native(gateup, limit=10.0)
                candidate = clamped_int8_2304(gateup, meta, masked=masked, expected_m=hint)
                if (not isinstance(baseline, (list, tuple)) or len(baseline) != 2
                        or any(not isinstance(reference, torch.Tensor)
                               or reference.shape != result.shape
                               or reference.dtype != result.dtype
                               or reference.device != result.device
                               or not torch.equal(reference[live].contiguous().view(torch.uint8),
                                                  result[live].contiguous().view(torch.uint8))
                               for reference, result in zip(baseline, candidate))):
                    _LOG.warning('HCU live-row INT8 differs from installed LightOp; using native activation')
                    return False
                del baseline, candidate
        except (RuntimeError, TritonError, TypeError, ValueError) as error:
            # A launch failure can leave a sticky device error. Only decline the
            # candidate if the worker can still use this device.
            if isinstance(error, RuntimeError):
                torch.cuda.synchronize(device)
            _LOG.warning('HCU live-row INT8 preflight unavailable; using native activation: %s', error)
            return False
    return True


def admitted(ops, weights, w1_scale, w2_scale, limit, *, masked):
    """Unknown providers, shapes and devices retain their existing operator."""
    experts = int(w1_scale.shape[0])
    if (limit != 10 or experts not in ((16, 48) if masked else (16,))
            or tuple(w1_scale.shape) != (experts, 4608, 1)
            or tuple(w2_scale.shape) != (experts, 5120, 1)
            or weights[0].device.type != "cuda"):
        return False
    provider = _verified_provider()
    if (provider is None or ops.clamp_quant is not provider.fuse_silu_mul_clamp_quant
            or ops.clamp_quant_ep is not provider.fuse_silu_mul_clamp_quant_ep):
        return False
    props = torch.cuda.get_device_properties(weights[0].device)
    supported = (props.gcnArchName.split(":")[0] == "gfx938"
                 and props.multi_processor_count == 64)
    if supported:
        native = ops.clamp_quant_ep if masked else ops.clamp_quant
        supported = _matches_native(native, weights[0].device, experts, masked)
    if supported:
        _log_admission(masked)
    return supported
