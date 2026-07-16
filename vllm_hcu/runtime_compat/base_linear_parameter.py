# SPDX-License-Identifier: Apache-2.0
"""In-place NN-layout compatibility for vLLM's parameter classes.

``vllm.model_executor`` imports ``parameter`` before platform discovery can
arm a cold module exchange.  Replacing that module late would split the
parameter class family: the parent package would retain official class
objects while later imports observed HCU copies.  This adapter therefore
keeps the official module and class identities and changes only the six
audited v0.21 loader methods.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import threading
from types import ModuleType
from typing import Any

from vllm_hcu.patch._stage3_common import Stage3CompatibilityError


_TARGET_MODULE = "vllm.model_executor.parameter"
_REMOVED_HCU_MODULE = "vllm_hcu.model_executor.parameter"
_PATCH_MARKER = "_hcu_base_linear_parameter_patch_applied"
_BINDING_MARKER = "_hcu_base_linear_parameter_binding"
_INSTALL_LOCK = threading.RLock()
_MISSING = object()

_BASE_COLUMN = "base.load_column_parallel_weight"
_BASE_ROW = "base.load_row_parallel_weight"
_COLUMN = "column.load_column_parallel_weight"
_MERGED_COLUMN = "column.load_merged_column_weight"
_QKV = "column.load_qkv_weight"
_ROW = "row.load_row_parallel_weight"

_EMPTY = inspect.Parameter.empty
_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD

_ORIGINAL_SIGNATURES = {
    _BASE_COLUMN: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
    ),
    _BASE_ROW: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
    ),
    _COLUMN: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
    ),
    _MERGED_COLUMN: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("kwargs", _VAR_KEYWORD, _EMPTY),
    ),
    _QKV: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("kwargs", _VAR_KEYWORD, _EMPTY),
    ),
    _ROW: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
    ),
}

_PATCHED_SIGNATURES = {
    **_ORIGINAL_SIGNATURES,
    _BASE_COLUMN: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("is_quantization", _POSITIONAL, False),
    ),
    _BASE_ROW: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("is_quantization", _POSITIONAL, False),
    ),
    _COLUMN: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("is_quantization", _POSITIONAL, False),
    ),
    _ROW: (
        ("self", _POSITIONAL, _EMPTY),
        ("loaded_weight", _POSITIONAL, _EMPTY),
        ("is_quantization", _POSITIONAL, False),
    ),
}


def _fail(message: str) -> Stage3CompatibilityError:
    return Stage3CompatibilityError(f"{_TARGET_MODULE}: {message}")


def _require_module(module: ModuleType | None) -> ModuleType:
    if module is None:
        module = importlib.import_module(_TARGET_MODULE)
    if not isinstance(module, ModuleType) or module.__name__ != _TARGET_MODULE:
        actual = getattr(module, "__name__", None)
        raise _fail(f"expected exact module, got {actual!r}")

    removed = sys.modules.get(_REMOVED_HCU_MODULE)
    if removed is not None and removed is not module:
        raise _fail(
            "obsolete HCU parameter module is loaded; refusing split class identity"
        )
    return module


def _require_class(module: ModuleType, name: str) -> type:
    value = vars(module).get(name)
    if not isinstance(value, type):
        raise _fail(f"required class {name} is missing")
    if value.__module__ != _TARGET_MODULE or value.__name__ != name:
        raise _fail(
            f"class {name} has unexpected identity "
            f"{value.__module__}.{value.__name__}"
        )
    return value


def _signature_shape(function: Any) -> tuple[tuple[str, Any, Any], ...]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError) as exc:
        raise _fail(f"cannot inspect callable {function!r}: {exc}") from exc
    return tuple((item.name, item.kind, item.default) for item in parameters)


def _require_binding(
    owner: type,
    attribute: str,
    role: str,
    expected_signature: tuple[tuple[str, Any, Any], ...],
    *,
    patched: bool,
) -> Any:
    function = vars(owner).get(attribute)
    target = f"{owner.__name__}.{attribute}"
    if not callable(function):
        raise _fail(f"required method {target} is missing")
    actual_signature = _signature_shape(function)
    if actual_signature != expected_signature:
        raise _fail(
            f"method {target} signature drifted: "
            f"{inspect.signature(function)}"
        )
    marker = getattr(function, _BINDING_MARKER, None)
    if patched:
        if marker != role:
            raise _fail(f"method {target} lost HCU binding marker {role!r}")
    else:
        if marker is not None or function.__module__ != _TARGET_MODULE:
            raise _fail(f"method {target} is not the audited official binding")
    return function


def _set_binding_metadata(function: Any, owner: type, attribute: str, role: str) -> None:
    function.__name__ = attribute
    function.__qualname__ = f"{owner.__qualname__}.{attribute}"
    setattr(function, _BINDING_MARKER, role)


def _nn_layout_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(henvs.VLLM_USE_NN)


def _nn_storage_dim(data: Any, logical_dim: int) -> int | None:
    """Return the transposed physical dimension for a 2-D NN-layout tensor."""

    if getattr(data, "ndim", None) != 2:
        return None
    if logical_dim not in (0, 1):
        raise _fail(f"NN layout requires logical dimension 0 or 1, got {logical_dim}")
    return 1 - logical_dim


def _verify_parent_exports(module: ModuleType) -> None:
    """Reject an existing parent-package class split without forcing import."""

    parent = sys.modules.get("vllm.model_executor")
    if not isinstance(parent, ModuleType):
        return
    for name in ("BasevLLMParameter", "PackedvLLMParameter"):
        exported = vars(parent).get(name)
        if exported is not None and exported is not vars(module).get(name):
            raise _fail(f"parent export {name} does not share canonical class identity")


def _bindings(module: ModuleType) -> tuple[
    dict[str, tuple[type, str]],
    tuple[type, type],
]:
    base = _require_class(module, "BasevLLMParameter")
    column = _require_class(module, "_ColumnvLLMParameter")
    row = _require_class(module, "RowvLLMParameter")
    model_weight = _require_class(module, "ModelWeightParameter")
    packed_column = _require_class(module, "PackedColumnParameter")
    packed = _require_class(module, "PackedvLLMParameter")

    if not issubclass(column, base) or not issubclass(row, base):
        raise _fail("private column/row parameter inheritance drifted")
    if not issubclass(model_weight, column) or not issubclass(model_weight, row):
        raise _fail("ModelWeightParameter inheritance drifted")
    if not issubclass(packed_column, column) or not issubclass(packed, model_weight):
        raise _fail("packed parameter inheritance drifted")

    return (
        {
            _BASE_COLUMN: (base, "load_column_parallel_weight"),
            _BASE_ROW: (base, "load_row_parallel_weight"),
            _COLUMN: (column, "load_column_parallel_weight"),
            _MERGED_COLUMN: (column, "load_merged_column_weight"),
            _QKV: (column, "load_qkv_weight"),
            _ROW: (row, "load_row_parallel_weight"),
        },
        (packed_column, packed),
    )


def _verify_patched(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, None) is not True:
        raise _fail("module patch marker is missing")
    bindings, _ = _bindings(module)
    for role, (owner, attribute) in bindings.items():
        _require_binding(
            owner,
            attribute,
            role,
            _PATCHED_SIGNATURES[role],
            patched=True,
        )
    _verify_parent_exports(module)
    if sys.modules.get(_REMOVED_HCU_MODULE) not in (None, module):
        raise _fail("obsolete HCU parameter module appeared after installation")


def _install_base_linear_parameter_compat_unlocked(
    parameter_module: ModuleType | None = None,
) -> None:
    module = _require_module(parameter_module)
    if getattr(module, _PATCH_MARKER, False):
        _verify_patched(module)
        return

    bindings, packed_types = _bindings(module)
    originals: dict[str, Any] = {}
    for role, (owner, attribute) in bindings.items():
        originals[role] = _require_binding(
            owner,
            attribute,
            role,
            _ORIGINAL_SIGNATURES[role],
            patched=False,
        )

    def base_column(self, loaded_weight, is_quantization=False):
        return originals[_BASE_COLUMN](self, loaded_weight)

    def base_row(self, loaded_weight, is_quantization=False):
        return originals[_BASE_ROW](self, loaded_weight)

    def column(self, loaded_weight, is_quantization=False):
        if not _nn_layout_enabled() or is_quantization:
            return originals[_COLUMN](self, loaded_weight)
        storage_dim = _nn_storage_dim(self.data, self.output_dim)
        if storage_dim is None:
            return originals[_COLUMN](self, loaded_weight)

        shard_size = self.data.shape[storage_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim,
            self.tp_rank * shard_size,
            shard_size,
        ).t()
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def merged_column(self, loaded_weight, **kwargs):
        is_quantization = bool(kwargs.pop("is_quantization", False))
        if not _nn_layout_enabled() or is_quantization:
            return originals[_MERGED_COLUMN](self, loaded_weight, **kwargs)
        storage_dim = _nn_storage_dim(self.data, self.output_dim)
        if storage_dim is None:
            return originals[_MERGED_COLUMN](self, loaded_weight, **kwargs)

        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]
        if (
            isinstance(self, packed_types)
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset,
                shard_size=shard_size,
            )

        param_data = self.data.narrow(storage_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim,
            self.tp_rank * shard_size,
            shard_size,
        ).t()
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def qkv(self, loaded_weight, **kwargs):
        is_quantization = bool(kwargs.pop("is_quantization", False))
        if not _nn_layout_enabled() or is_quantization:
            return originals[_QKV](self, loaded_weight, **kwargs)
        storage_dim = _nn_storage_dim(self.data, self.output_dim)
        if storage_dim is None:
            return originals[_QKV](self, loaded_weight, **kwargs)

        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]
        shard_id = kwargs["shard_id"]
        num_heads = kwargs["num_heads"]
        if (
            isinstance(self, packed_types)
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset,
                shard_size=shard_size,
            )

        shard_rank = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
        param_data = self.data.narrow(storage_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim,
            shard_rank * shard_size,
            shard_size,
        ).t()
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def row(self, loaded_weight, is_quantization=False):
        if not _nn_layout_enabled() or is_quantization:
            return originals[_ROW](self, loaded_weight)
        storage_dim = _nn_storage_dim(self.data, self.input_dim)
        if storage_dim is None:
            return originals[_ROW](self, loaded_weight)

        shard_size = self.data.shape[storage_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim,
            self.tp_rank * shard_size,
            shard_size,
        )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        loaded_weight = loaded_weight.t()
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    replacements = {
        _BASE_COLUMN: base_column,
        _BASE_ROW: base_row,
        _COLUMN: column,
        _MERGED_COLUMN: merged_column,
        _QKV: qkv,
        _ROW: row,
    }
    for role, function in replacements.items():
        owner, attribute = bindings[role]
        _set_binding_metadata(function, owner, attribute, role)

    previous_marker = vars(module).get(_PATCH_MARKER, _MISSING)
    try:
        for role, function in replacements.items():
            owner, attribute = bindings[role]
            setattr(owner, attribute, function)
        setattr(module, _PATCH_MARKER, True)
        _verify_patched(module)
    except BaseException:
        for role, original in originals.items():
            owner, attribute = bindings[role]
            setattr(owner, attribute, original)
        if previous_marker is _MISSING:
            if hasattr(module, _PATCH_MARKER):
                delattr(module, _PATCH_MARKER)
        else:
            setattr(module, _PATCH_MARKER, previous_marker)
        raise


def install_base_linear_parameter_compat(
    parameter_module: ModuleType | None = None,
) -> None:
    """Patch official vLLM parameter loaders without replacing their classes."""

    with _INSTALL_LOCK:
        _install_base_linear_parameter_compat_unlocked(parameter_module)


__all__ = ["install_base_linear_parameter_compat"]
