# SPDX-License-Identifier: Apache-2.0
"""Join model configuration selection with the Adapter runtime lifecycle."""

import os
from threading import RLock


class ModelRuntimeActivation:
    """Publish a model runtime when both its config and the host are ready."""

    def __init__(self, activate, *, spawn_by_default=False):
        self._activate = activate
        self._spawn_by_default = spawn_by_default
        self._context = None
        self._requested = False
        self._active = False
        self._activating = False
        self._inherited_publication = False
        self._lock = RLock()
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self):
        self._lock = RLock()
        self._context = None
        self._inherited_publication |= self._active or self._activating
        self._active = False
        self._activating = False

    @property
    def requested(self):
        with self._lock:
            return self._requested

    def bind(self, context):
        """Called by the host extension after its normal startup checks."""
        with self._lock:
            self._context = context
            self._activate_if_ready()

    def request(self):
        """Called only by a model configuration owned by the extension."""
        with self._lock:
            self._requested = True
            self._activate_if_ready()

    def _activate_if_ready(self):
        if self._inherited_publication:
            raise RuntimeError('Model runtime publication inherited across fork; use spawn')
        if not self._requested or self._context is None or self._active or self._activating:
            return
        self._activating = True
        try:
            self._activate(self._context)
            if self._spawn_by_default:
                os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
            self._active = True
        finally:
            self._activating = False


class ModelRuntimeConfig:
    """Request the owned model runtime on config parsing and worker restore."""

    @classmethod
    def request_runtime(cls):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        cls.request_runtime()
        return config

    def __setstate__(self, state):
        restore = getattr(super(), '__setstate__', None)
        if restore is None:
            self.__dict__.update(state)
        else:
            restore(state)
        self.request_runtime()
