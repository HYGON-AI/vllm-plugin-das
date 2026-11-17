# SPDX-License-Identifier: Apache-2.0
"""Resolve a frozen release/PR checkout before running any submitted code."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys

SPEC = importlib.util.spec_from_file_location(
    "hcu_authorization", Path(__file__).with_name("check_hcu_ci_authorization.py")
)
AUTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTH)


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise AUTH.AuthorizationError("expected a complete lowercase commit SHA")
    return value


def resolve(event, environment, github):
    """Validate the live PR and manual actor; return only bounded output values."""
    kind = environment["GITHUB_EVENT_NAME"]
    repository = environment["GITHUB_REPOSITORY"]
    inputs = event.get("inputs") or {}
    number = inputs.get("pr_number", "")
    expected = inputs.get("expected_head", "")
    if kind not in ("push", "workflow_dispatch", "pull_request_target"):
        raise AUTH.AuthorizationError("unsupported release event")
    if kind != "pull_request_target" and environment["GITHUB_REF"] != "refs/heads/v0.33.0":
        raise AUTH.AuthorizationError("release/manual runs require ref v0.33.0")
    manual_pr = kind == "workflow_dispatch" and bool(number or expected)
    if manual_pr:
        if not isinstance(number, str) or not re.fullmatch(r"[1-9][0-9]*", number):
            raise AUTH.AuthorizationError("manual PR requires a positive pr_number")
        sha(expected)
    elif kind == "pull_request_target":
        pull = event.get("pull_request", {})
        number = pull.get("number")
        if type(number) is not int or number <= 0:
            raise AUTH.AuthorizationError("PR event has no valid number")
        number = str(number)
        expected = sha(pull.get("head", {}).get("sha"))
    else:
        return dict(head_sha=sha(environment["GITHUB_SHA"]), base_sha="",
                    pr_number="", manual_authorized="", release="v0.33.0")

    pull = github(f"/repos/{repository}/pulls/{number}")
    if (not isinstance(pull, dict) or pull.get("state") != "open"
            or pull.get("number") != int(number)
            or pull.get("base", {}).get("ref") != "v0.33.0"
            or pull.get("base", {}).get("repo", {}).get("full_name") != repository):
        raise AUTH.AuthorizationError("PR must be open against this repository's v0.33.0")
    if sha(pull.get("head", {}).get("sha")) != expected:
        raise AUTH.AuthorizationError("PR head moved; dispatch the current expected_head")
    if manual_pr:
        actor = environment["GITHUB_ACTOR"]
        # Explicit dispatch binds this actor's authorization to expected_head.
        # A persistent ready-hcu label never grants manual authorization.
        from urllib.parse import quote
        permission = github(f"/repos/{repository}/collaborators/{quote(actor, safe='')}/permission")
        if permission.get("permission") not in AUTH.TRUSTED_PERMISSIONS:
            raise AUTH.AuthorizationError("manual PR execution requires a trusted dispatch actor")
    return dict(head_sha=expected, base_sha=sha(pull["base"].get("sha")),
                pr_number=number, manual_authorized="true" if manual_pr else "",
                release="v0.33.0")


def main():
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        def github(path):
            return AUTH._github_json(os.environ["GITHUB_API_URL"] + path,
                                     os.environ["GITHUB_TOKEN"])
        output = resolve(event, os.environ, github)
        AUTH._write_outputs(output)
        print(json.dumps(output, sort_keys=True))
    except (AUTH.AuthorizationError, KeyError, OSError, ValueError, TypeError) as exc:
        print(f"v0.33 run resolution failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
