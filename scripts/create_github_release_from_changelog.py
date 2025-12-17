#!/usr/bin/env python3
"""
Create or update a GitHub Release from CHANGELOG.md when a tag is pushed.

Trigger: tags like vX.Y.Z (see .github/workflows/release.yml).

Behavior:
- Extracts the CHANGELOG section starting with '## X.Y.Z' (optionally followed by ' - <date>')
- Creates the GitHub Release if it doesn't exist
- Updates the Release body if it already exists (idempotent)

No external deps (stdlib only).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


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


def _extract_section(changelog: str, version: str) -> str:
    """
    Extract section starting with '## {version}' (optionally followed by date),
    up to next '## ' heading.
    """
    lines = changelog.splitlines()
    start: int | None = None
    header_re = re.compile(rf"^##\s+{re.escape(version)}(\s+-.*)?$")
    for i, ln in enumerate(lines):
        if header_re.match(ln.strip()):
            start = i
            break
    if start is None:
        raise RuntimeError(f"CHANGELOG section for version {version} not found")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    return "\n".join(lines[start:end]).strip() + "\n"


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # owner/name
    tag = os.environ.get("GITHUB_REF_NAME")  # vX.Y.Z
    if not token or not repo or not tag:
        raise SystemExit("Missing GITHUB_TOKEN, GITHUB_REPOSITORY, or GITHUB_REF_NAME")

    if not tag.startswith("v"):
        print(f"Tag {tag} does not start with v; skipping")
        return 0

    version = tag[1:]
    body = _extract_section(Path("CHANGELOG.md").read_text(encoding="utf-8"), version)

    api = f"https://api.github.com/repos/{repo}"
    payload = {"tag_name": tag, "name": tag, "body": body, "draft": False, "prerelease": False}

    st, txt = _req("GET", f"{api}/releases/tags/{tag}", token)
    if st == 200:
        rel = json.loads(txt)
        rel_id = rel.get("id")
        if not rel_id:
            raise RuntimeError("Release exists but id missing")
        st2, txt2 = _req("PATCH", f"{api}/releases/{rel_id}", token, payload)
        if st2 not in (200, 201):
            raise RuntimeError(f"Failed to update release: HTTP {st2} {txt2}")
        print(f"OK: updated release for {tag}")
        return 0

    if st != 404:
        raise RuntimeError(f"Failed to check release: HTTP {st} {txt}")

    st3, txt3 = _req("POST", f"{api}/releases", token, payload)
    if st3 not in (200, 201):
        raise RuntimeError(f"Failed to create release: HTTP {st3} {txt3}")
    print(f"OK: created release for {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


