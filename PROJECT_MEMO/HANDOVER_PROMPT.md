# Handover Prompt for New Cursor Instance

Copy everything below the line and paste into a new Cursor Agent conversation.

---

You are a Cursor AI agent for Jason Lian. Orient yourself before any work.

## Project: pdf-mcp-server

- **Repo**: git@github.com:nfsarch33/pdf-mcp-server.git
- **Version**: Check `pyproject.toml` (single source of truth)
- **Stack**: Python 3.10+, MCP (Model Context Protocol), PyMuPDF, Qwen3-VL via Ollama
- **Tests**: `make test` (430+ tests)
- **Lint**: `ruff check pdf_mcp/`

Read these files first:
1. `PROJECT_MEMO/HANDOVER.md` -- full project context
2. `.cursor/rules/template.rules` -- version strategy, Pepper pointers
3. `.cursor/rules/memory-system.rules` -- 3-layer memory architecture
4. `.cursor/rules/writing-standards.rules` -- output standards

## MEMORY SYSTEM (three layers)

1. **Layer 1** (`.cursor/rules/`): Auto-injected. Memory routing, version strategy, writing standards.

2. **Layer 2** (Pepper): `~/memo/global-memories/` -- SOPs, skills, project status.
   Remote: `git@github.com:nfsarch33/cursor-memory-bank.git`
   Read on start: `pdf-mcp-server-status.md`, `release-sop-checklist.md`, `memory-sop.md`

3. **Layer 3** (global-kb): `~/Code/global-kb/` -- archive only.
   Remote: `git@github-agtc:nfsarch33/cursor-global-kb.git`

## IDENTITY (nfsarch33 -- personal repo)

- Git: `Jason Lian <jaslian@gmail.com>`
- GH CLI: `nfsarch33` via `GH_TOKEN` in `.envrc` (direnv-managed, gitignored)
- Pre-push hook at `~/.git-hooks/pre-push` blocks Zendesk identity
- Never use Zendesk identity (jlianzendesk) in this repo

## HARD RULES

- No direct main/master pushes -- feature branches + PR always
- No credentials committed -- `.envrc` is gitignored
- No AI attribution in any artifact (commits, PRs, docs, comments)
- No emojis in code, configs, commits, or docs
- Short commit messages: `type: what changed`
- Pass linter + tests before PR
- Delete merged branches. Sync memo/kb after merges.

## RELEASE PROCESS

1. Bump `pyproject.toml` version
2. Add `CHANGELOG.md` section
3. Commit + annotated tag + push
4. CI auto: test-gate -> version-gate -> GitHub Release

## QUICK COMMANDS

```bash
make test              # full suite
ruff check pdf_mcp/    # lint
make check-llm         # LLM backend status
./scripts/run_local_vlm.sh  # start VLM server (auto GPU selection)
```

## SKILLS

Project: cross-platform-setup, llm-e2e-qa, memo-kb-sync, pdf-form-filling, release-sop
Global: find-skills, session-handoff, gh-fix-ci, gh-address-comments

## FIRST ACTIONS

1. Run `health check` to verify memory system
2. Read `PROJECT_MEMO/HANDOVER.md`
3. Check git status and recent commits
4. Read Pepper `pdf-mcp-server-status.md` for current state
5. Pull latest main before any work

## MULTI-REPO WORKSPACE NOTE

This project is moving to a multi-repo workspace. When working alongside other repos:
- Each repo has its own `.envrc` for identity isolation
- MCP server config is in `~/.cursor/mcp.json` (machine-specific)
- Pepper memory is shared across all projects (global-memories/)
- Keep project-specific rules in `.cursor/rules/` within each repo
- Keep project-specific skills in `.cursor/skills/` within each repo

health check
