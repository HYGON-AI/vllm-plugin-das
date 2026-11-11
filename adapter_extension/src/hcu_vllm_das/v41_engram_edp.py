# SPDX-License-Identifier: Apache-2.0
"""Restore the dedicated Engram EDP communicator without editing vLLM.

vLLM 0.33 reuses the DP group for the Engram hash gather when the Engram
DP shard covers the whole data-parallel group (``parallel_state.py``
assigns ``_ENGRAM_DP = _DP``). The 0.32 line always created a dedicated
``group_name="edp"`` communicator for that gather. This module restores
that behaviour from the adapter: after ``initialize_model_parallel``
returns, an independent coordinator with the same ranks is created and
published through the public getter path. Core destroy logic already
tears an independent ``_ENGRAM_DP`` down, so shutdown needs no changes.
"""
from __future__ import annotations

from functools import wraps as _wraps
from importlib import import_module as _import_module
import inspect
import logging
import os
from threading import RLock

from .environment import _flag

_FLAG = 'HCU_DSV41_ENGRAM_EDP_INDEPENDENT'
_logger = logging.getLogger(__name__)
_lock = RLock()
_failure: str | None = None
_installed = False
_parallel_state = None
_original = None


def _fatal(reason: str) -> None:
    global _failure
    if _failure is None:
        _failure = reason
    raise RuntimeError(_failure)


def _after_fork_child() -> None:
    # A forked child inherits a consistent installed wrapper; only the
    # lock must be rebuilt because its holder may not survive the fork.
    global _lock
    _lock = RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_child)


def _is_hcu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and current_platform.device_name == 'hcu'


def _restore_independent_edp() -> None:
    """Rebuild a dedicated edp communicator when 0.33 reused the DP group."""
    parallel_state = _parallel_state
    dp_group = parallel_state.get_dp_group()
    reused = parallel_state.get_engram_dp_group()
    if reused is None or dp_group.world_size <= 1:
        # No Engram sharding over DP; nothing to isolate.
        return
    if reused is not dp_group:
        # Already an independent communicator (0.32-style behaviour).
        return
    backend = None
    try:
        import torch

        backend = torch.distributed.get_backend(dp_group.device_group)
    except Exception as error:  # pragma: no cover - defensive backend probe
        _fatal(f'cannot derive the DP communicator backend: {error}')
    group = parallel_state.init_model_parallel_group(
        [list(dp_group.ranks)],
        parallel_state.get_world_group().local_rank,
        backend,
        group_name='edp',
    )
    parallel_state._ENGRAM_DP = group
    message = (
        'HCU DSV41: restored independent Engram EDP communicator '
        f'(ranks={list(dp_group.ranks)}, backend={backend})'
    )
    # stdout as well as logging: extension activation can run before vLLM
    # configures its log handlers, so the marker must not depend on them.
    print(message, flush=True)
    _logger.info('%s', message)


def _install_wrapper(parallel_state) -> None:
    global _original
    try:
        signature = inspect.signature(parallel_state.init_model_parallel_group)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f'cannot inspect init_model_parallel_group ABI: {error}'
        ) from error
    positional = [
        name for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ][:3]
    if positional != ['group_ranks', 'local_rank', 'backend']:
        raise RuntimeError(
            'init_model_parallel_group ABI changed: leading positional '
            f'parameters are {positional}'
        )
    for name in (
        'get_dp_group',
        'get_engram_dp_group',
        'get_world_group',
    ):
        if not callable(getattr(parallel_state, name, None)):
            raise RuntimeError(f'required callable is unavailable: {name}')

    original = parallel_state.initialize_model_parallel

    @_wraps(original)
    def wrapper(*args, **kwargs):
        original(*args, **kwargs)
        _restore_independent_edp()

    wrapper.__hcu_das_edp_original__ = original
    parallel_state.initialize_model_parallel = wrapper
    _original = original


def install_engram_edp_guard() -> None:
    """Install the post-init EDP guard once, on the HCU platform only."""
    global _installed, _parallel_state
    with _lock:
        if _failure is not None:
            raise RuntimeError(_failure)
        if _installed:
            return
        if not _flag(_FLAG, default=True):
            _logger.info(
                'HCU DSV41: independent Engram EDP guard disabled by %s', _FLAG
            )
            return
        if not _is_hcu():
            return
        try:
            parallel_state = _import_module('vllm.distributed.parallel_state')
            _install_wrapper(parallel_state)
        except RuntimeError:
            raise
        except BaseException as error:
            raise RuntimeError(
                f'Engram EDP guard installation failed: {error}'
            ) from error
        _parallel_state = parallel_state
        _installed = True
        message = 'HCU DSV41: Engram EDP independent-group guard installed'
        print(message, flush=True)
        _logger.info('%s', message)


def check_engram_edp_guard() -> None:
    """Refuse a lost wrapper owner without repairing it."""
    with _lock:
        if _failure is not None:
            raise RuntimeError(_failure)
        if not _installed:
            return
        current = getattr(
            _parallel_state.initialize_model_parallel,
            '__hcu_das_edp_original__',
            None,
        )
        if current is not _original:
            _fatal('Engram EDP guard lost its wrapper owner; restart process')


__all__ = [
    'install_engram_edp_guard',
    'check_engram_edp_guard',
]
