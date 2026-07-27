# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""One-release read-only compatibility entry for the former patch command."""

from __future__ import annotations

from typing import Sequence

from vllm_hcu.doctor import main as doctor_main


def apply_post_install_patches(argv: Sequence[str] | None = None) -> int:
    """Run diagnostics only; retained for callers of the legacy command name."""

    print(
        "vllm-hcu-apply-patches is now read-only; vLLM-HCU no longer writes "
        "vLLM source files or creates modular_kernel symlinks."
    )
    return doctor_main(argv)


def main() -> int:
    return apply_post_install_patches()


if __name__ == "__main__":
    raise SystemExit(main())
