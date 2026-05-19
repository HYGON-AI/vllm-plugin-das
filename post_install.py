"""Compatibility wrapper for old post_install.py entry."""

from vllm_hcu.post_install import main


if __name__ == "__main__":
    raise SystemExit(main())
