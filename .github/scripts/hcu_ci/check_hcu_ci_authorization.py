# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Verify that trusted actors authorized HCU PR hardware execution."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TRUSTED_PERMISSIONS = {"admin", "maintain", "write", "triage"}


class AuthorizationError(RuntimeError):
    """Raised when an HCU label cannot be attributed to a trusted actor."""


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AuthorizationError(f"required environment variable is unset: {name}")
    return value


def _github_json(url: str, token: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "vllm-hcu-ci-authorization",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise AuthorizationError(f"GitHub API request failed for {url}: {exc}") from exc


def _ready_label_actor(
    *,
    api_url: str,
    repository: str,
    issue_number: int,
    token: str,
) -> str:
    actor: str | None = None
    for page in range(1, 11):
        url = (
            f"{api_url}/repos/{repository}/issues/{issue_number}/events"
            f"?per_page=100&page={page}"
        )
        events = _github_json(url, token)
        if not isinstance(events, list):
            raise AuthorizationError("GitHub issue events response is not a list")
        if not events:
            break
        for event in events:
            if not isinstance(event, dict) or event.get("event") != "labeled":
                continue
            label = event.get("label")
            event_actor = event.get("actor")
            if (
                isinstance(label, dict)
                and label.get("name") == "ready-hcu"
                and isinstance(event_actor, dict)
                and isinstance(event_actor.get("login"), str)
            ):
                actor = event_actor["login"]
    if actor is None:
        raise AuthorizationError(
            "ready-hcu is present but its label event/actor was not found"
        )
    return actor


def _actor_permission(
    *,
    api_url: str,
    repository: str,
    actor: str,
    token: str,
) -> str:
    url = f"{api_url}/repos/{repository}/collaborators/{actor}/permission"
    response = _github_json(url, token)
    if not isinstance(response, dict):
        raise AuthorizationError("GitHub permission response is not a mapping")
    permission = response.get("permission")
    if not isinstance(permission, str):
        raise AuthorizationError(f"permission is unavailable for {actor}")
    return permission


def _write_outputs(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def main() -> int:
    try:
        event_path = Path(_required_environment("GITHUB_EVENT_PATH"))
        repository = _required_environment("GITHUB_REPOSITORY")
        api_url = _required_environment("GITHUB_API_URL")
        token = _required_environment("GITHUB_TOKEN")
        event = json.loads(event_path.read_text(encoding="utf-8"))
        pull_request = event.get("pull_request")
        if not isinstance(pull_request, dict):
            raise AuthorizationError("workflow event has no pull_request object")
        user = pull_request.get("user")
        if not isinstance(user, dict) or not isinstance(user.get("login"), str):
            raise AuthorizationError("pull request author is unavailable")
        author = user["login"]
        author_permission = _actor_permission(
            api_url=api_url,
            repository=repository,
            actor=author,
            token=token,
        )
        if author_permission in TRUSTED_PERMISSIONS:
            print(f"HCU execution auto-authorized for {author} ({author_permission})")
            _write_outputs(
                {
                    "ready": "auto",
                    "authorized": "true",
                    "actor": author,
                    "permission": author_permission,
                }
            )
            return 0

        labels = pull_request.get("labels", [])
        ready = any(
            isinstance(item, dict) and item.get("name") == "ready-hcu"
            for item in labels
        )
        if not ready:
            print("ready-hcu is not present; hardware execution remains locked")
            _write_outputs(
                {
                    "ready": "false",
                    "authorized": "false",
                    "actor": author,
                    "permission": author_permission,
                }
            )
            return 0

        issue_number = pull_request.get("number")
        if not isinstance(issue_number, int):
            raise AuthorizationError("pull request number is unavailable")
        actor = _ready_label_actor(
            api_url=api_url,
            repository=repository,
            issue_number=issue_number,
            token=token,
        )
        permission = _actor_permission(
            api_url=api_url,
            repository=repository,
            actor=actor,
            token=token,
        )
        if permission not in TRUSTED_PERMISSIONS:
            raise AuthorizationError(
                f"ready-hcu was applied by {actor}, whose permission is "
                f"{permission!r}; expected one of {sorted(TRUSTED_PERMISSIONS)}"
            )
        print(f"ready-hcu authorized by {actor} ({permission})")
        _write_outputs(
            {
                "ready": "true",
                "authorized": "true",
                "actor": actor,
                "permission": permission,
            }
        )
    except (AuthorizationError, OSError, json.JSONDecodeError) as exc:
        print(f"HCU CI authorization failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
