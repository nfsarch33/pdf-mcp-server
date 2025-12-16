# Memory and rules system (portable)

This repository is intended to be used from Cursor with the `pdf-handler` MCP server enabled.

Because this repo is open source, do not commit personal notes, academic content, or any secrets.

## What is “global” here

Cursor has its own memory system and MCP tooling. This repo keeps a portable, repeatable setup that can be applied across machines and projects:

- Keep project rules in the repo (so they travel with the code).
- Keep personal notes out of git (use `.gitignore`).
- Use MCP tool calls for repeatable end to end testing in Cursor.

## Recommended rules

- Prefer deterministic edits: annotations and managed assets are easier to update and remove than content stream rewriting.
- Always include stable identifiers for edit and remove operations.
- Keep tests for each tool at two levels:
  - Unit tests for `pdf_mcp/pdf_tools.py`
  - MCP layer smoke test invoking tools through `FastMCP.call_tool` (closest to Cursor usage)

## Cursor testing workflow

1) Restart or toggle the `pdf-handler` MCP server after pulling changes.
2) Use absolute paths for inputs and outputs.
3) Test each tool with a small dummy PDF and verify output by reopening it.

## Open source hygiene

- Files like `cursor-memory-complete-setup-guide.md` must remain untracked and ignored.
- Use placeholders in docs instead of personal filesystem paths.


