try:
    __version__ = "0.18.1"
    __version_tuple__ = (0, 18, 1)
    __hcu_version__ = "0.18.1+das.dtk2604"

    from vllm_hcu.version import __version__, __version_tuple__, __hcu_version__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash: {e}", RuntimeWarning)
    __version__ = "dev"
    __version_tuple__ = (0, 0, 0)
