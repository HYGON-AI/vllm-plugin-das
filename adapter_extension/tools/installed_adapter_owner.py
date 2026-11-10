# SPDX-License-Identifier: Apache-2.0
"""Resolve host-distribution ownership without importing either framework."""
from importlib import metadata
from pathlib import PurePosixPath


def _optional_distribution(name):
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _owns_host_namespace(distribution):
    return distribution is not None and any(
        PurePosixPath(str(item)).parts[0]=='hcu_adapter'
        for item in distribution.files or () if PurePosixPath(str(item)).parts)


def adapter_distribution_name():
    """Accept modern split or genuine legacy combined layouts, refuse overlap.

    RECORD existence/editability/module origins are checked by the caller.
    A present but broken modern host is never hidden by falling back to legacy.
    """
    modern=_optional_distribution('hcu-adapter')
    kernel=_optional_distribution('sgl-kernel')
    legacy_owner=_owns_host_namespace(kernel)
    if modern is not None:
        if legacy_owner:
            raise RuntimeError('hcu_adapter is not owned by a unique distribution: both hcu-adapter and legacy sgl-kernel claim it; migrate in a clean environment')
        return 'hcu-adapter'
    if kernel is not None and kernel.files and not legacy_owner:
        raise RuntimeError('hcu_adapter is not owned by the pure sgl-kernel distribution; install hcu-adapter')
    return 'sgl-kernel'
