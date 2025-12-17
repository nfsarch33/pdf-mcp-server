#!/usr/bin/env python3
"""
Automation: when an Issue gets label `feature-approved`, create a GitHub Project (Projects v2),
add the issue to it, and write the project URL back to the issue body.

Idempotency: if the issue body already contains `<!-- project-url: ... -->`, do nothing.

No external deps (stdlib only).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


GQL_URL = "https://api.github.com/graphql"


@dataclass(frozen=True)
class Ctx:
    token: str
    repo_owner: str
    repo_name: str
    issue_node_id: str
    issue_number: int
    issue_title: str


def _gql(token: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(GQL_URL, data=payload, method="POST")
    req.add_header("Authorization", f"bearer {token}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # nosec B310 (GitHub endpoint)
        data = json.loads(resp.read().decode("utf-8"))
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


def _load_event(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_ctx(event: Dict[str, Any]) -> Ctx:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Missing GITHUB_TOKEN")

    repo = event.get("repository") or {}
    owner = (repo.get("owner") or {}).get("login")
    name = repo.get("name")
    if not owner or not name:
        raise SystemExit("Missing repository.owner.login or repository.name in event")

    issue = event.get("issue") or {}
    node_id = issue.get("node_id")
    number = issue.get("number")
    title = issue.get("title") or ""
    if not node_id or not number:
        raise SystemExit("Missing issue.node_id or issue.number in event")

    return Ctx(token=token, repo_owner=owner, repo_name=name, issue_node_id=node_id, issue_number=int(number), issue_title=title)


def _get_issue_body(ctx: Ctx) -> str:
    q = """
    query($id: ID!) {
      node(id: $id) {
        ... on Issue {
          body
        }
      }
    }
    """
    data = _gql(ctx.token, q, {"id": ctx.issue_node_id})
    return (data["node"] or {}).get("body") or ""


def _update_issue_body(ctx: Ctx, body: str) -> None:
    m = """
    mutation($id: ID!, $body: String!) {
      updateIssue(input: {id: $id, body: $body}) { issue { id } }
    }
    """
    _gql(ctx.token, m, {"id": ctx.issue_node_id, "body": body})


def _ensure_project(ctx: Ctx) -> str:
    """Create a ProjectV2 under the repo owner and return its URL."""
    # Get owner node id
    q_owner = """
    query($login: String!) {
      organization(login: $login) { id }
      user(login: $login) { id }
    }
    """
    d = _gql(ctx.token, q_owner, {"login": ctx.repo_owner})
    owner_id = (d.get("organization") or d.get("user") or {}).get("id")
    if not owner_id:
        raise RuntimeError("Unable to resolve owner id (org/user)")

    title = f"{ctx.issue_title} (feature)"
    m_create = """
    mutation($ownerId: ID!, $title: String!) {
      createProjectV2(input: {ownerId: $ownerId, title: $title}) {
        projectV2 { id url }
      }
    }
    """
    d2 = _gql(ctx.token, m_create, {"ownerId": owner_id, "title": title})
    proj = d2["createProjectV2"]["projectV2"]
    return proj["url"]


def _add_issue_to_project(ctx: Ctx, project_url: str) -> None:
    """Add issue to project by resolving project node id via URL."""
    # Resolve project id from URL (ProjectsV2 doesn't have direct lookup by URL; use repository projects query and match url).
    q = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        projectsV2(first: 50) { nodes { id url title } }
      }
    }
    """
    d = _gql(ctx.token, q, {"owner": ctx.repo_owner, "name": ctx.repo_name})
    nodes = ((d.get("repository") or {}).get("projectsV2") or {}).get("nodes") or []
    proj_id: Optional[str] = None
    for n in nodes:
        if n and n.get("url") == project_url:
            proj_id = n.get("id")
            break
    if not proj_id:
        # Project might be created at org/user scope; fetch owner projects as fallback.
        q2 = """
        query($login: String!) {
          organization(login: $login) { projectsV2(first: 50) { nodes { id url } } }
          user(login: $login) { projectsV2(first: 50) { nodes { id url } } }
        }
        """
        d2 = _gql(ctx.token, q2, {"login": ctx.repo_owner})
        nodes2 = ((d2.get("organization") or {}).get("projectsV2") or {}).get("nodes") or []
        if not nodes2:
            nodes2 = ((d2.get("user") or {}).get("projectsV2") or {}).get("nodes") or []
        for n in nodes2:
            if n and n.get("url") == project_url:
                proj_id = n.get("id")
                break
    if not proj_id:
        raise RuntimeError("Could not resolve ProjectV2 id to add issue")

    m = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item { id }
      }
    }
    """
    _gql(ctx.token, m, {"projectId": proj_id, "contentId": ctx.issue_node_id})


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("No GITHUB_EVENT_PATH; exiting.")
        return 0

    event = _load_event(event_path)
    ctx = _extract_ctx(event)

    body = _get_issue_body(ctx)
    if "<!-- project-url:" in body:
        print("Project already recorded in issue body; skipping.")
        return 0

    project_url = _ensure_project(ctx)
    _add_issue_to_project(ctx, project_url)

    marker = f"\\n\\n<!-- project-url: {project_url} -->\\n\\nProject created: {project_url}\\n"
    _update_issue_body(ctx, body + marker)
    print(f"OK: created project and linked issue: {project_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


