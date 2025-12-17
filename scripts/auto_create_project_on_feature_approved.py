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
from typing import Any, Dict, Optional, Tuple


GQL_URL = "https://api.github.com/graphql"


@dataclass(frozen=True)
class Ctx:
    token: str
    repo_owner: str
    repo_name: str
    issue_node_id: str
    issue_number: int
    issue_title: str


PROJECT_MARKER_PREFIX = "<!-- project-url:"


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


def _resolve_owner_id(ctx: Ctx) -> str:
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
    return owner_id


def _create_project(ctx: Ctx, owner_id: str) -> Tuple[str, str]:
    """Create a ProjectV2 under the repo owner and return (project_id, project_url)."""

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
    return proj["id"], proj["url"]


def _add_issue_to_project(ctx: Ctx, project_id: str) -> str:
    m = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item { id }
      }
    }
    """
    d = _gql(ctx.token, m, {"projectId": project_id, "contentId": ctx.issue_node_id})
    return d["addProjectV2ItemById"]["item"]["id"]


def _get_project_fields(ctx: Ctx, project_id: str) -> Dict[str, Any]:
    q = """
    query($projectId: ID!) {
      node(id: $projectId) {
        ... on ProjectV2 {
          fields(first: 50) {
            nodes {
              ... on ProjectV2FieldCommon {
                id
                name
                dataType
              }
              ... on ProjectV2SingleSelectField {
                id
                name
                dataType
                options { id name }
              }
            }
          }
        }
      }
    }
    """
    return _gql(ctx.token, q, {"projectId": project_id})


def _find_field(fields_data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    nodes = (((fields_data.get("node") or {}).get("fields") or {}).get("nodes") or [])
    for n in nodes:
        if n and n.get("name") == name:
            return n
    return None


def _create_single_select_field(ctx: Ctx, project_id: str, name: str, options: list[str]) -> None:
    # Best-effort: API supports providing options; if it fails, we skip and keep project creation functional.
    m = """
    mutation($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]) {
      createProjectV2Field(input: {projectId: $projectId, name: $name, dataType: SINGLE_SELECT, singleSelectOptions: $options}) {
        projectV2Field { ... on ProjectV2SingleSelectField { id name } }
      }
    }
    """
    colors = ["GRAY", "BLUE", "YELLOW", "ORANGE", "GREEN", "RED", "PURPLE", "PINK"]
    opt_payload = [{"name": o, "color": colors[i % len(colors)]} for i, o in enumerate(options)]
    _gql(ctx.token, m, {"projectId": project_id, "name": name, "options": opt_payload})


def _create_text_field(ctx: Ctx, project_id: str, name: str) -> None:
    m = """
    mutation($projectId: ID!, $name: String!) {
      createProjectV2Field(input: {projectId: $projectId, name: $name, dataType: TEXT}) {
        projectV2Field { ... on ProjectV2FieldCommon { id name } }
      }
    }
    """
    _gql(ctx.token, m, {"projectId": project_id, "name": name})


def _set_single_select(ctx: Ctx, project_id: str, item_id: str, field_id: str, option_id: str) -> None:
    m = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(
        input: {projectId: $projectId, itemId: $itemId, fieldId: $fieldId, value: { singleSelectOptionId: $optionId }}
      ) {
        projectV2Item { id }
      }
    }
    """
    _gql(ctx.token, m, {"projectId": project_id, "itemId": item_id, "fieldId": field_id, "optionId": option_id})


def _set_text(ctx: Ctx, project_id: str, item_id: str, field_id: str, value: str) -> None:
    m = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $text: String!) {
      updateProjectV2ItemFieldValue(
        input: {projectId: $projectId, itemId: $itemId, fieldId: $fieldId, value: { text: $text }}
      ) {
        projectV2Item { id }
      }
    }
    """
    _gql(ctx.token, m, {"projectId": project_id, "itemId": item_id, "fieldId": field_id, "text": value})


def _ensure_project_fields_and_defaults(ctx: Ctx, project_id: str, item_id: str) -> None:
    """
    Best-effort automation:
    - Ensure Status (single-select), Risk (single-select), Target version (text) fields exist
    - Set defaults for the created item (Status=Backlog, Risk=Medium)
    """
    try:
        fields_data = _get_project_fields(ctx, project_id)
        if _find_field(fields_data, "Status") is None:
            _create_single_select_field(ctx, project_id, "Status", ["Backlog", "Ready", "In progress", "In review", "Done"])
        if _find_field(fields_data, "Risk") is None:
            _create_single_select_field(ctx, project_id, "Risk", ["Low", "Medium", "High"])
        if _find_field(fields_data, "Target version") is None:
            _create_text_field(ctx, project_id, "Target version")

        # Re-fetch fields to get ids/options
        fields_data = _get_project_fields(ctx, project_id)
        status = _find_field(fields_data, "Status")
        risk = _find_field(fields_data, "Risk")
        target = _find_field(fields_data, "Target version")

        if status and status.get("options"):
            opt = next((o for o in status["options"] if o.get("name") == "Backlog"), None)
            if opt:
                _set_single_select(ctx, project_id, item_id, status["id"], opt["id"])
        if risk and risk.get("options"):
            opt = next((o for o in risk["options"] if o.get("name") == "Medium"), None)
            if opt:
                _set_single_select(ctx, project_id, item_id, risk["id"], opt["id"])
        if target and target.get("id"):
            _set_text(ctx, project_id, item_id, target["id"], "")
    except Exception as e:
        # Keep core behavior (project creation + linking) working even if Projects API evolves.
        print(f"WARN: project fields/defaults setup skipped due to error: {e}")


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("No GITHUB_EVENT_PATH; exiting.")
        return 0

    event = _load_event(event_path)
    ctx = _extract_ctx(event)

    body = _get_issue_body(ctx)
    if PROJECT_MARKER_PREFIX in body:
        print("Project already recorded in issue body; skipping.")
        return 0

    owner_id = _resolve_owner_id(ctx)
    project_id, project_url = _create_project(ctx, owner_id)
    item_id = _add_issue_to_project(ctx, project_id)
    _ensure_project_fields_and_defaults(ctx, project_id, item_id)

    marker = f"\\n\\n<!-- project-url: {project_url} -->\\n\\nProject created: {project_url}\\n"
    _update_issue_body(ctx, body + marker)
    print(f"OK: created project and linked issue: {project_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


