---
name: release-sop
description: Execute the pdf-mcp-server release SOP end-to-end, including version consistency, tests, tags, and memo/kb sync. Use when preparing a release or running pre-release checks.
---
# Release SOP

## When to Use
- Preparing a new release or release candidate
- Running pre-release checks and documentation updates
- Ensuring version consistency and post-release sync

## Instructions
1. Run health check and confirm local LLM server status.
2. Sync memo and global-kb repos (pull first, push after updates).
3. Run required tests and checks:
   - `make lint` (or `ruff check --fix` then manually fix remaining)
   - `make check-llm`
   - `make test-e2e` (local server required)
   - `make test` or `make prepush`
4. Update version across all required files:
   - `pyproject.toml`
   - `README.md`
   - `CHANGELOG.md`
   - `pdf_mcp/pdf_tools.py` (if version present)
   - `pdf_mcp/server.py` (if version present)
   - `PROJECT_MEMO/GLOBAL_CURSOR_INSTRUCTIONS.md`
5. Update `CHANGELOG.md` with release notes and test counts.
6. Commit, tag, and push using conventional commit style.
7. Post-release housekeeping: delete merged branches, verify CI, re-run lint.
8. Update Pepper memory status and push memo repo.

## Output Format
- Provide a concise release summary with:
  - Version
  - Tests and E2E results
  - Skipped tests summary
  - Sync status for memo and global-kb
