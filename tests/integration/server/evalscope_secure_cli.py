# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Launch EvalScope without placing its API key in the OS command line."""

from __future__ import annotations

import os
import sys

from evalscope.cli.cli import run_cmd

from tests.integration.server.evalscope_server import EVALSCOPE_API_KEY_ENV

_API_KEY_OPTIONS = frozenset({"--api-k", "--api-ke", "--api-key"})


def main() -> int:
    # EvalScope's argparse parser accepts the two unique long-option
    # abbreviations in addition to the full spelling.
    if any(arg.partition("=")[0] in _API_KEY_OPTIONS for arg in sys.argv):
        raise RuntimeError(
            "pass the EvalScope API key through the protected environment"
        )
    api_key = os.environ.pop(EVALSCOPE_API_KEY_ENV, "EMPTY")
    # Python-level argv is not reflected in /proc/<pid>/cmdline, so the key is
    # available to EvalScope's parser without appearing in process listings.
    sys.argv.extend(["--api-key", api_key])
    return int(run_cmd() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
