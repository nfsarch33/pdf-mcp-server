#!/usr/bin/env python3
"""
Ensure standard repo labels exist (idempotent).

Runs in GitHub Actions using GITHUB_TOKEN.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    name: str
    color: str
    description: str


def _req(method: str, url: str, token: str, body: dict | None = None) -> tuple[int, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310 (GitHub endpoint)
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # owner/name
    if not token or not repo:
        print("Missing GITHUB_TOKEN or GITHUB_REPOSITORY")
        return 2

    api = f"https://api.github.com/repos/{repo}"

    labels = [
        Label("feature", "1D76DB", "Feature request / capability"),
        Label("feature-approved", "0E8A16", "Approved to proceed; auto-creates GitHub Project"),
        Label("bug", "D73A4A", "Bug report"),
        Label("meeting-notes", "C5DEF5", "Meeting distillation issue"),
        Label("docs", "0075CA", "Documentation-only changes"),
        Label("breaking", "B60205", "Breaking change; requires explicit approval"),
        Label("skip-changelog", "FBCA04", "CI override: changelog not required for this PR"),
    ]

    created = 0
    for l in labels:
        status, _ = _req("GET", f"{api}/labels/{urllib.parse.quote(l.name)}", token)
        if status == 200:
            continue
        if status != 404:
            print(f"WARN: label lookup failed for {l.name}: HTTP {status}")
            continue
        status2, body2 = _req(
            "POST",
            f"{api}/labels",
            token,
            {"name": l.name, "color": l.color, "description": l.description},
        )
        if status2 in (200, 201):
            created += 1
            continue
        print(f"ERROR: failed to create label {l.name}: HTTP {status2} body={body2}")
        return 2

    print(f"OK: labels ensured; created={created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


