"""Two optional BoltOPs decode entries at Qwen GDN's imported aliases.

The target has no provider callback for these functions. Only the two globals
in qwen_gdn_linear_attn are mounted; the official ops modules stay untouched.
Retire this mount when upstream provides per-function registration or a native
HCU route. Install/uninstall and flags must be settled before model compilation;
uninstall restores Python aliases, not already compiled or captured graphs.

The conservative envelope is BF16 activations, FP32 state/parameters, K=V=128,
indexed in-place GDN (not KDA), and the vendor's unsplit grid limit. All gates
inspect metadata only. Numerical, NULL/padded-slot, MTP and changed-input Graph
validation on real tensors is still required; CPU contract tests are not that
evidence. No prefill, causal-conv, model, scheduler or cache implementation lives
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import hashlib
from importlib import import_module
import inspect
import json
import math
import os
from pathlib import Path
import threading
from types import ModuleType
from typing import Callable


COMPILE_FACTOR = "VLLM_HCU_DAS_FLA_DECODE_POLICY"
CONSUMER = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
OPS = "vllm.third_party.flash_linear_attention.ops"
PACKED = "fused_recurrent_gated_delta_rule_packed_decode"
SIGMOID = "fused_sigmoid_gating_delta_rule_update"
_OWNERS = {PACKED: f"{OPS}.fused_recurrent", SIGMOID: f"{OPS}.fused_sigmoid_gating"}
_ARCHES = frozenset({"gfx936", "gfx938"})
_REQUIRED = inspect.Parameter.empty
_CONTRACTS = {
    PACKED: (
        ("mixed_qkv", "a", "b", "A_log", "dt_bias", "scale", "initial_state",
         "out", "ssm_state_indices", "use_qk_l2norm_in_kernel"),
        (_REQUIRED,) * 9 + (False,),
    ),
    SIGMOID: (
        ("A_log", "a", "b", "dt_bias", "q", "k", "v", "beta", "threshold",
         "scale", "initial_state", "inplace_final_state", "cu_seqlens",
         "ssm_state_indices", "num_accepted_tokens", "use_qk_l2norm_in_kernel",
         "is_kda"),
        (_REQUIRED,) * 7 + (1.0, 20.0, None, None, True, None, None, None, False, False),
    ),
}
_lock = threading.RLock()
_envs: ModuleType | None = None
_installation: _Installation | None = None


class FlaInstallError(RuntimeError):
    """The target ABI, ownership or activation order needs review."""


@dataclass(frozen=True)
class _Installation:
    consumer: ModuleType
    originals: dict[str, Callable]
    wrappers: dict[str, Callable]
    identity: dict


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _enabled() -> bool:
    # One existing family switch, not new per-kernel switches. Both default off.
    master = _flag("VLLM_HCU_USE_CUSTOM_OPS")
    family = _flag("VLLM_HCU_USE_CUSTOM_AITER_FLA")
    return master and family


def _batch_invariant() -> bool:
    return bool(getattr(_envs, "VLLM_BATCH_INVARIANT", False))


def is_requested() -> bool:
    """Read the effective opt-in policy without importing providers or hardware.

    Register compile factors first so vLLM's batch-invariant policy is visible.
    This expresses intent, not evidence that a provider has been mounted.
    """
    return _enabled() and not _batch_invariant()


def effective_compile_factor() -> str:
    """Derived, JSON-serializable factor for ordinary and second-level AOT keys.

    No imports/device queries here. Getter failures are encoded, not raised:
    vLLM silently skips failing getters. Identity describes the mounted provider,
    including its code/config bytes, rather than just the requested env value.
    """
    try:
        enabled = is_requested()
        installed = _installation
        owned = installed is not None and all(
            getattr(installed.consumer, name, None) is wrapper
            for name, wrapper in installed.wrappers.items()
        )
        policy = {
            "abi": 1,
            "enabled": enabled,
            "mounted": installed is not None,
            "owned": owned,
            "route": ("foreign-alias" if installed is not None and not owned else
                      "boltops" if enabled and owned else "official"),
            "identity": None if installed is None else installed.identity,
        }
    except (TypeError, ValueError, AttributeError) as error:
        policy = {"abi": 1, "invalid": str(error)}
    return json.dumps(policy, sort_keys=True, separators=(",", ":"))


def register_compile_factors(vllm_envs: ModuleType) -> None:
    """Explicit integration hook; call in every process before any cache lookup.

    This does not edit environment.py or activate the provider. The caller owns
    activation order and passing the detected architecture to install().
    """
    global _envs
    with _lock:
        registry = getattr(vllm_envs, "environment_variables", None)
        if not isinstance(registry, dict) or not callable(
            getattr(vllm_envs, "compile_factors", None)
        ):
            raise FlaInstallError("vLLM compile-factor registry is unavailable")
        cached = getattr(vllm_envs, "_is_envs_cache_enabled", lambda: False)
        if cached():
            raise FlaInstallError("register FLA factors before vLLM caches env values")
        previous = registry.get(COMPILE_FACTOR)
        if previous is not None and previous is not effective_compile_factor:
            raise FlaInstallError("FLA compile factor has another owner")
        if _envs is not None and _envs is not vllm_envs:
            raise FlaInstallError("FLA factors are already bound to another env module")
        _enabled()  # Reject invalid activation settings before registration.
        registry[COMPILE_FACTOR] = effective_compile_factor
        _envs = vllm_envs


def _signature_matches(function: object, name: str, *, vendor: bool) -> bool:
    if not callable(function):
        return False
    names, defaults = _CONTRACTS[name]
    if vendor:
        names += ("null_state_index", "kernel_cfg")
        defaults += ((-1 if name == PACKED else 0), None)
    try:
        params = tuple(inspect.signature(function, follow_wrapped=False).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(params) == len(names) and all(
        p.name == key and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and type(p.default) is type(default) and p.default == default
        for p, key, default in zip(params, names, defaults)
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def _provider_identity(module: ModuleType, functions: dict, architecture: str) -> dict:
    files = {}
    for name, function in functions.items():
        source = inspect.getsourcefile(function)
        if source is None:
            raise FlaInstallError(f"cannot identify BoltOPs implementation for {name}")
        path = Path(source)
        files[name] = {
            "code": _file_hash(path),
            "config": _file_hash(path.parent / "configs" / name / f"{name}-{architecture}.json"),
        }
    return {
        "architecture": architecture,
        "boundary": CONSUMER,
        "null_state_index": 0,
        "envelope": "bf16-fp32-k128-v128-indexed-inplace-gdn-grid65535-v1",
        "provider": files,
        "public_module": _file_hash(Path(module.__file__)),
        "adapter": _file_hash(Path(__file__)),
    }


def _shape(tensor: object) -> tuple:
    return tuple(getattr(tensor, "shape", ()))


def _dtype(tensor: object) -> str:
    return str(getattr(tensor, "dtype", ""))


def _same_device(tensors: tuple) -> bool:
    device = getattr(tensors[0], "device", None)
    return getattr(device, "type", None) == "cuda" and all(
        getattr(t, "device", None) == device for t in tensors
    )


def _contiguous(tensor: object) -> bool:
    return callable(getattr(tensor, "is_contiguous", None)) and tensor.is_contiguous()


def _row_contiguous(tensor: object) -> bool:
    # Qwen gate projections can be adjacent slices of a wider row. Both public
    # packed kernels accept an explicit row pitch, but assume unit inner stride.
    return (
        callable(getattr(tensor, "stride", None))
        and tensor.stride(-1) == 1
        and tensor.stride(0) >= _shape(tensor)[-1]
    )


def _indices(tensor: object, shape: tuple | None = None) -> bool:
    return (
        _dtype(tensor) in {"torch.int32", "torch.int64"}
        and _contiguous(tensor)
        and (shape is None or _shape(tensor) == shape)
    )


def _positive_scale(scale: object, *, allow_none: bool = False) -> bool:
    return (allow_none and scale is None) or (
        type(scale) in (int, float) and math.isfinite(scale) and scale > 0
    )


def _packed_supported(x, a, b, A_log, dt_bias, scale, state, out, indices, norm) -> bool:
    xs, ss = _shape(x), _shape(state)
    if len(xs) != 2 or len(ss) != 4 or ss[-2:] != (128, 128):
        return False
    batch, width = xs
    slots, hv, _, k = ss
    qk_width = width - hv * 128
    heads = qk_width // (2 * k)
    tensors = (x, a, b, A_log, dt_bias, state, out, indices)
    return (
        batch > 0 and slots > 0 and hv > 0 and heads > 0 and hv % heads == 0
        and qk_width == 2 * heads * k and batch * hv <= 65535
        and _shape(a) == _shape(b) == (batch, hv)
        and _shape(A_log) == _shape(dt_bias) == (hv,)
        and _shape(out) == (batch, 1, hv, 128)
        and _same_device(tensors)
        and _dtype(x) == _dtype(out) == "torch.bfloat16"
        and _dtype(a) in {"torch.bfloat16", "torch.float32"}
        and _dtype(b) == _dtype(a)
        and all(_dtype(t) == "torch.float32" for t in (A_log, dt_bias, state))
        and all(_row_contiguous(t) for t in (x, a, b))
        and all(_contiguous(t) for t in (A_log, dt_bias, state, out))
        and _indices(indices, (batch,))
        and _positive_scale(scale) and type(norm) is bool
    )


def _sigmoid_supported(A_log, a, b, dt_bias, q, k, v, beta, threshold, scale,
                       state, inplace, cu, indices, accepted, norm, is_kda) -> bool:
    qs, vs, ss = _shape(q), _shape(v), _shape(state)
    if len(qs) != 4 or len(vs) != 4 or len(ss) != 4 or _shape(k) != qs:
        return False
    batch, tokens, heads, dim = qs
    hv = vs[2]
    if (
        min(batch, tokens, heads, hv, ss[0]) <= 0 or dim != 128
        or vs != (batch, tokens, hv, 128) or hv % heads != 0
        or ss[1:] != (hv, 128, 128) or inplace is not True or is_kda is not False
        or type(norm) is not bool
    ):
        return False
    count = batch if cu is None else (_shape(cu)[0] - 1 if len(_shape(cu)) == 1 else 0)
    if count <= 0 or count * hv > 65535:
        return False
    index_shape = _shape(indices)
    if not _indices(indices) or len(index_shape) not in (1, 2) or index_shape[0] != count:
        return False
    if len(index_shape) == 1:
        if accepted is not None or (cu is None and tokens != 1) or (cu is not None and tokens != count):
            return False
    elif index_shape[1] <= 0 or (cu is None and index_shape[1] < tokens):
        return False
    if cu is not None and (batch != 1 or not _indices(cu, (count + 1,))):
        return False
    if accepted is not None and (cu is None or not _indices(accepted, (count,))):
        return False
    tensors = (q, k, v, a, b, A_log, dt_bias, state, indices)
    tensors += tuple(t for t in (cu, accepted) if t is not None)
    return (
        _same_device(tensors)
        and all(_dtype(t) == "torch.bfloat16" for t in (q, k, v))
        and _dtype(a) in {"torch.bfloat16", "torch.float32"} and _dtype(b) == _dtype(a)
        and _shape(a) in ((batch * tokens, hv), (batch, tokens, hv))
        and _shape(b) == _shape(a)
        and _shape(A_log) == _shape(dt_bias) == (hv,)
        and all(_dtype(t) == "torch.float32" for t in (A_log, dt_bias, state))
        and all(_contiguous(t) for t in (A_log, dt_bias, state))
        and _positive_scale(beta) and _positive_scale(threshold)
        and _positive_scale(scale, allow_none=True)
    )


def _make_wrappers(originals: dict, providers: dict) -> dict:
    @wraps(originals[PACKED])
    def packed(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out,
               ssm_state_indices, use_qk_l2norm_in_kernel=False):
        args = (mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out,
                ssm_state_indices, use_qk_l2norm_in_kernel)
        if not is_requested() or not _packed_supported(*args):
            return originals[PACKED](*args)
        return providers[PACKED](*args, null_state_index=0)

    @wraps(originals[SIGMOID])
    def sigmoid(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
                scale=None, initial_state=None, inplace_final_state=True,
                cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=False, is_kda=False):
        args = (A_log, a, b, dt_bias, q, k, v, beta, threshold, scale,
                initial_state, inplace_final_state, cu_seqlens, ssm_state_indices,
                num_accepted_tokens, use_qk_l2norm_in_kernel, is_kda)
        if not is_requested() or not _sigmoid_supported(*args):
            return originals[SIGMOID](*args)
        return providers[SIGMOID](*args, null_state_index=0)

    return {PACKED: packed, SIGMOID: sigmoid}


def install(*, architecture: str | None = None) -> bool:
    """Mount both aliases atomically, after explicit opt-in/capability preflight.

    architecture must come from Adapter's existing detection, not GPU_ARCHS.
    Off, unsupported hardware, unavailable imports or provider ABI return False
    with official aliases intact. Target ABI/owner drift raises before mutation.
    A selected provider's runtime errors are never caught by these wrappers.
    """
    global _installation
    if not is_requested():
        return False
    with _lock:
        if _envs is None or _envs.environment_variables.get(COMPILE_FACTOR) is not effective_compile_factor:
            raise FlaInstallError("register FLA compile factors before install/cache lookup")
        if architecture not in _ARCHES:
            return False
        if _installation is not None:
            if architecture != _installation.identity["architecture"] or any(
                getattr(_installation.consumer, name) is not wrapper
                for name, wrapper in _installation.wrappers.items()
            ):
                raise FlaInstallError("FLA alias ownership or architecture changed")
            return True
        try:
            module = import_module("boltops.fla.gdn")
            providers = {name: getattr(module, name) for name in _CONTRACTS}
        except (ImportError, AttributeError):
            return False
        if not all(_signature_matches(fn, name, vendor=True) for name, fn in providers.items()):
            return False
        originals = {name: getattr(import_module(owner), name) for name, owner in _OWNERS.items()}
        if not all(_signature_matches(fn, name, vendor=False) for name, fn in originals.items()):
            raise FlaInstallError("official GDN function signature changed; review mount")
        consumer = import_module(CONSUMER)
        if any(getattr(consumer, name, None) is not fn for name, fn in originals.items()):
            raise FlaInstallError("Qwen GDN aliases have another owner; refusing replacement")
        identity = _provider_identity(module, providers, architecture)
        wrappers = _make_wrappers(originals, providers)
        for name, wrapper in wrappers.items():
            setattr(consumer, name, wrapper)
        _installation = _Installation(consumer, originals, wrappers, identity)
        return True


def uninstall() -> bool:
    """Restore both aliases before destroying/rebuilding compiled models."""
    global _installation
    with _lock:
        if _installation is None:
            return False
        if any(getattr(_installation.consumer, name, None) is not wrapper
               for name, wrapper in _installation.wrappers.items()):
            raise FlaInstallError("FLA alias has another owner; refusing partial restore")
        for name, original in _installation.originals.items():
            setattr(_installation.consumer, name, original)
        _installation = None
        return True
