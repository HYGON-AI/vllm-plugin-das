# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prepare the v0.28.1 ARC PR lane without a Docker daemon or extra packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

REGISTRY = "10.16.1.152:5000"
REPOSITORIES = ("jenkins/model_test_env/vllm", "hcu/base_env/vllm")
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class PreparationError(ValueError):
    """Reject an untrusted image or matrix before publishing job outputs."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def resolve_image(image: str) -> str:
    # Tag lookup requires manifest access to the existing internal registry.
    # Do not follow redirects or forward credentials to another endpoint.
    repositories = "(?:" + "|".join(re.escape(p) for p in REPOSITORIES) + ")"
    match = re.fullmatch(
        re.escape(REGISTRY) + "/(" + repositories + ")"
        r"(?::([A-Za-z0-9_][A-Za-z0-9_.-]{0,127})|@(sha256:[0-9a-f]{64}))",
        image,
    )
    if match is None:
        raise PreparationError(
            "image must be a tag or sha256 digest in an approved vLLM repository"
        )
    repository, tag, digest = match.groups()
    if digest:
        return image
    request = Request(
        f"http://{REGISTRY}/v2/{repository}/manifests/{tag}",
        headers={"Accept": ", ".join(MANIFEST_TYPES)},
    )
    with build_opener(_NoRedirect()).open(request, timeout=30) as response:
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        body = response.read(MAX_MANIFEST_BYTES + 1)
        declared = response.headers.get("Docker-Content-Digest")
    if len(body) > MAX_MANIFEST_BYTES or media_type not in MANIFEST_TYPES:
        raise PreparationError("registry returned an oversized or unsupported manifest")
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    if declared != digest:
        raise PreparationError(
            "registry manifest digest is absent or does not match its content"
        )
    return f"{REGISTRY}/{repository}@{digest}"


def split_matrix(matrix: object) -> dict[str, str]:
    if not isinstance(matrix, list):
        raise PreparationError("selected matrix must be a list")
    arc = []
    legacy = []
    seen = set()
    for job in matrix:
        if not isinstance(job, dict):
            raise PreparationError("selected jobs must be mappings")
        job_id = job.get("id")
        if (
            not isinstance(job_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", job_id)
            or job_id in seen
        ):
            raise PreparationError("selected job IDs must be valid and unique")
        seen.add(job_id)
        runner, arch, cards = job.get("runner"), job.get("arch"), job.get("cards")
        expected_arch = (
            {"bw1000": "gfx936", "bw1100": "gfx938"}.get(runner)
            if isinstance(runner, str)
            else None
        )
        if (
            expected_arch is None
            or arch != expected_arch
            or type(cards) is not int
            or cards not in (1, 2, 4, 8)
        ):
            raise PreparationError(f"unsupported ARC topology for {job_id}")
        isolated = job.get("isolate_generated_code", False)
        if type(isolated) is not bool:
            raise PreparationError("isolate_generated_code must be a boolean")
        if isolated:
            # A normal ARC job container is not the hardened generated-code
            # sandbox. Preserve its existing Docker isolation, never omit it.
            legacy.append(dict(job))
        else:
            arc.append(dict(job, runner=f"hygon-{runner}-{cards}card"))
    return {
        "arc_matrix": json.dumps(arc, separators=(",", ":")),
        "legacy_matrix": json.dumps(legacy, separators=(",", ":")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("image", "matrix"))
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "image":
            outputs = {"image": resolve_image(os.environ.get("SOURCE_IMAGE", ""))}
        else:
            outputs = split_matrix(json.loads(os.environ.get("MATRIX_JSON", "")))
    except (PreparationError, OSError, URLError, json.JSONDecodeError) as exc:
        print(f"ARC preparation refused: {exc}", file=sys.stderr)
        return 2
    # All validation precedes output publication. JSON values are single-line.
    with args.github_output.open("a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
