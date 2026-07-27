# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Compatibility wrapper for old post_install.py entry."""

from vllm_hcu.post_install import main


if __name__ == "__main__":
    raise SystemExit(main())
