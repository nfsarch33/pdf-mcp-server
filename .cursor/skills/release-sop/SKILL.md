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
- [ ] Step 5: Version consistency check (BUG-011 lesson)
- [ ] Step 6: Update CHANGELOG with release notes
- [ ] Step 7: Commit, tag, and push
- [ ] Step 8: Post-release: reinstall package and verify
- [ ] Step 9: Update Pepper memory and push memo
```

**Step 1**: Run health check and confirm local LLM server status.

**Step 2**: Sync memo and global-kb repos (pull first, push after updates).

**Step 3**: Run required tests and checks:
- `ruff check --fix` then manually fix remaining
- `make check-llm`
- `make test-e2e` (local server required)
- `make test` or `make prepush`

**Step 4**: Update version across all required files:
- `pyproject.toml` (single source of truth)
- `CHANGELOG.md`
- `PROJECT_MEMO/PROJECT_STATUS_PROMPT.md` (test counts)
- `PROJECT_MEMO/FORM_FILLING_PROMPT.md` (version header)

**Step 5**: Version consistency check (lesson from BUG-011):
- Verify `python -c "from pdf_mcp import __version__; print(__version__)"` matches `pyproject.toml`
- Run `pytest tests/test_pdf_tools.py::TestServerVersionExposure` to confirm
- After release tag, run `pip install -e .` to refresh package metadata

**Step 6**: Update `CHANGELOG.md` with release notes and test counts.

**Step 7**: Commit, tag, and push using conventional commit style.

**Step 8**: Post-release housekeeping:
- Run `pip install -e .` to refresh installed package metadata
- Delete merged branches, verify CI, re-run lint
- Verify GitHub Release was created automatically

**Step 9**: Update Pepper memory status and push memo repo.

## Output Format
- Provide a concise release summary with:
  - Version
  - Tests and E2E results
  - Skipped tests summary
  - Sync status for memo and global-kb
