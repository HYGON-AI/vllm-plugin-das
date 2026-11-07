# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Repair a known gfx938 hipBLASLt packaging defect in disposable CI containers.

The affected DTK 26.10 image can produce invalid INT8 outputs, including NaN
logprobs in DeepSeek-R1 TP/EP tests. Remove this workaround once CI exclusively
uses corrected images. Newer or already corrected tables are left untouched.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


BROKEN_MAP_SHA256 = "931d948a115aceea468b98b691c5376bfe1921ed9aedd74852d55976b9c90f9c"
FUSED_NAMES_SHA256 = "c9b46ded3728ec1a7a25b3ef61f4489dd070b2b58d48ee2defd30acfbcea0bdf"
FUSED_BINARY_SHA256 = "863e7510da73242b2351519909bef08950383d34aaf78110ede4e724deedeb1e"
MISSING_KERNEL = (
    "Cijk_Alik_Bljk_I8BS_MT96x128x256_TT6_4_WG16_32_1_"
    "MFWGS768_SLC_WGM1_bias_channelwise"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_kernel_map(root: Path = Path("/opt/dtk/lib/hipblaslt")) -> bool:
    directory = root / "library_gfx938"
    mapping = directory / "w8a8_customkernel_all.txt"
    if not mapping.is_file():
        return False
    original = mapping.read_bytes()
    if _sha256(original) != BROKEN_MAP_SHA256:
        return False

    # This table omits one exported kernel. hipBLASLt then abandons *all*
    # name-to-index remapping and launches unrelated kernels using old indices.
    # Bind the workaround to the exact affected table, names and code object;
    # never infer an index repair for another runtime release.
    names = (directory / "FusedMatmulKernelName.txt").read_bytes()
    binary = (directory / "FusedMatmulKernel_gfx938.co").read_bytes()
    if (
        _sha256(names) != FUSED_NAMES_SHA256
        or _sha256(binary) != FUSED_BINARY_SHA256
    ):
        raise RuntimeError(
            "Known incomplete hipBLASLt INT8 map has unexpected companion files; "
            "use a corrected HCU CI image"
        )
    missing = set(names.decode().splitlines()) - set(original.decode().splitlines())
    if missing != {MISSING_KERNEL}:
        raise RuntimeError("Cannot safely repair the incomplete hipBLASLt INT8 map")

    repaired = original.rstrip(b"\n") + b"\n" + MISSING_KERNEL.encode() + b"\n"
    # Preserve all existing indices; append only the missing exported name.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(repaired)
            os.fchmod(handle.fileno(), mapping.stat().st_mode & 0o777)
        os.replace(temporary, mapping)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        "Repaired known gfx938 hipBLASLt INT8 kernel map in the CI container: "
        f"appended {MISSING_KERNEL}",
        flush=True,
    )
    return True


if __name__ == "__main__":
    ensure_kernel_map()
