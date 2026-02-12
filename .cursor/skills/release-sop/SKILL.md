---
name: release-sop
description: Executes the pdf-mcp-server release SOP end-to-end, including version consistency checks, test validation, tagging, and memo/kb sync. Use when preparing a release, running pre-release checks, or verifying post-release state.
---
# Release SOP

## When to Use
- Preparing a new release or release candidate
- Running pre-release checks and documentation updates
- Ensuring version consistency and post-release sync

## Workflow

Copy this checklist and track progress:
```
Release Progress:
- [ ] Step 1: Health check and LLM server status
- [ ] Step 2: Sync memo and global-kb repos
- [ ] Step 3: Run lint, tests, and E2E checks
- [ ] Step 4: Update version in all required files
- [ ] Step 5: Update CHANGELOG with release notes
- [ ] Step 6: Commit, tag, and push
- [ ] Step 7: Post-release housekeeping
- [ ] Step 8: Update Pepper memory and push memo
```

**Step 1**: Run health check and confirm local LLM server status.

**Step 2**: Sync memo and global-kb repos (pull first, push after updates).

**Step 3**: Run required tests and checks:
- `ruff check --fix` then manually fix remaining
- `make check-llm`
- `make test-e2e` (local server required)
- `make test` or `make prepush`

**Step 4**: Update version across all required files:
- `pyproject.toml`
- `README.md`
- `CHANGELOG.md`
- `pdf_mcp/pdf_tools.py` (if version present)
- `pdf_mcp/server.py` (if version present)
- `PROJECT_MEMO/GLOBAL_CURSOR_INSTRUCTIONS.md`

**Step 5**: Update `CHANGELOG.md` with release notes and test counts.

**Step 6**: Commit, tag, and push using conventional commit style.

**Step 7**: Post-release housekeeping: delete merged branches, verify CI, re-run lint.

**Step 8**: Update Pepper memory status and push memo repo.

## Output Format
- Provide a concise release summary with:
  - Version
  - Tests and E2E results
  - Skipped tests summary
  - Sync status for memo and global-kb
