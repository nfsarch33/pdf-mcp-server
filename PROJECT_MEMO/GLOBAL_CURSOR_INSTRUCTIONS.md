# Global Cursor IDE Instructions

> **Purpose**: This document provides comprehensive instructions for any Cursor IDE instance to understand and work with:
> 1. The `pdf-handler` MCP server
> 2. The multi-layer memory and rules system
> 3. The automated workflows and SOPs

---

## Part 1: pdf-handler MCP Server

### What it is
A local MCP server for PDF manipulation built with Python, `pypdf`, `fillpdf`, and `pymupdf`. It provides **51 tools** for form filling, text/comment/signature editing, page operations, encryption, OCR, and **LLM-powered agentic AI features**.

### LLM Integration (v0.9.0+)
Zero-cost local VLM support with multi-backend:
```bash
# Check backend status
make check-llm

# Ensure Ollama model is present (skips duplicate downloads)
make install-llm-models

# Start local model server (FREE! auto-detects best GPU)
./scripts/run_local_vlm.sh
```

**Backend Priority**: local > ollama > openai (free first!)

**Ollama model override**:
```bash
export PDF_MCP_OLLAMA_MODEL="qwen2.5:7b"
```

**Agentic AI Tools**:
- `get_llm_backend_info()` - Check available backends
- `auto_fill_pdf_form()` - Intelligent form filling
- `extract_structured_data()` - Entity extraction
- `analyze_pdf_content()` - Document analysis/summarization

### Registration (per machine)
Edit `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "pdf-handler": {
      "command": "/ABSOLUTE/PATH/TO/pdf-mcp-server/.venv/bin/python",
      "args": ["-m", "pdf_mcp.server"],
      "description": "Local PDF form filling and editing (stdio)"
    }
  }
}
```
Restart Cursor after saving.

### Hard requirements (verified working)
| Capability | Tools |
|------------|-------|
| Form fill/update/delete | `fill_pdf_form`, `clear_pdf_form_fields`, `get_pdf_form_fields` |
| Comments CRUD | `add_comment`, `update_comment`, `remove_comment` |
| Text CRUD | `insert_text`, `edit_text`, `remove_text` |
| Signature image CRUD | `add_signature_image`, `update_signature_image`, `remove_signature_image` |
| Sign + protect workflow | `add_signature_image` → `encrypt_pdf` |

### Signing options (0.5.2)
- `sign_pdf` and `sign_pdf_pem` support `timestamp_url`, `embed_validation_info`, `allow_fetching`, `docmdp_permissions`.

### Consolidated API (0.6.0)
- `extract_text`: Unified text extraction (replaces 4 separate tools). Engines: native, auto, smart, ocr, force_ocr. Optional confidence scores.
- `split_pdf`: Unified splitting (replaces 2 tools). Modes: pages, bookmarks.
- `export_pdf`: Unified export (replaces 2 tools). Formats: markdown, json.
- `get_pdf_metadata(full=True)`: Extended metadata (replaces `get_full_metadata`).
- Deprecated: `insert_text/edit_text/remove_text`, `extract_text_*`, `split_pdf_by_*`, `export_to_*`, `get_full_metadata`.

### Full tool list
- **Forms**: `get_pdf_form_fields`, `fill_pdf_form`, `clear_pdf_form_fields`, `flatten_pdf`
- **Text (managed FreeText)**: `insert_text`, `edit_text`, `remove_text`, `add_text_annotation`, `update_text_annotation`, `remove_text_annotation`
- **Comments**: `add_comment`, `update_comment`, `remove_comment`
- **Signatures**: `add_signature_image`, `update_signature_image`, `remove_signature_image`
- **Pages**: `merge_pdfs`, `extract_pages`, `insert_pages`, `remove_pages`, `rotate_pages`
- **Metadata**: `get_pdf_metadata`, `set_pdf_metadata`
- **Security**: `encrypt_pdf`
- **Other**: `add_text_watermark`, `remove_annotations`

### Troubleshooting quick reference

| Issue | Fix |
|-------|-----|
| Tools list is stale | Toggle MCP server off/on in Cursor |
| Form fill value appears empty | Re-read with `get_pdf_form_fields` to verify `/V` persisted |
| Encrypt fails on certain PDFs | Our implementation normalizes trailer IDs; retry after toggle |
| Signature xref changed after update | Always use the latest returned xref for subsequent operations |
| PyMuPDF warnings during signature ops | Non-blocking if tests pass and outputs are correct |

### Quality gates before pushing
```bash
# Unit + integration tests
./.venv/bin/python -m pytest -q

# Cursor smoke test
./.venv/bin/python scripts/cursor_smoke.py
```

---

## Part 2: Multi-layer Memory and Rules System

### Architecture overview

```
Layer 1: .cursor/rules/  (always injected into prompts)
         ↓ short, universal rules (~1 paragraph each)

Layer 2: ~/memo/global-memories/  (Pepper memory bank, git)
         ↓ procedures, SOPs, checklists, multi-step workflows

Layer 3: ~/Code/global-kb/  (archive, git)
         ↓ historical investigations, RCAs, old versions

Cursor memory MCP: tiny, stable preferences + searchable entities
```

### Routing SOP ("add to memory/rules")

When asked to "add to memory" or "add to rules":

1. **If short + universal + needed every conversation** → `.cursor/rules/`
2. **If procedural / multi-step / project-agnostic** → Pepper (`~/memo/global-memories/`)
3. **If historical / archive / RCA** → Global KB (`~/Code/global-kb/`)
4. **If tiny + stable preference** → Cursor memory MCP

### Key Pepper files (canonical)

| File | Purpose |
|------|---------|
| `memory-sop.md` | Routing rules for "add to memory" |
| `mcp-index-and-selection-sop.md` | MCP tool selection guide (auto-generated) |
| `pdf-handler-troubleshooting.md` | Quick reference for pdf-handler issues |
| `pdf-handler-manual-qa.md` | Manual QA checklist for Cursor testing |
| `release-sop-tag-based.md` | Tag-based release workflow |
| `coding-standards-v9-1.md` | KISS/DRY/SOLID/clean code standards |

### Automation (zero manual intervention)

**Daily automation** refreshes:
- MCP index from `~/.cursor/mcp.json` (secrets redacted)
- Repo memories sync from configured repos

**One-time install**:
```bash
~/memo/tools/install-daily-automation.sh
```

**Configuration**:
- `~/memo/tools/repos-to-sync.txt` (one repo `PROJECT_MEMO/PEPPER_SYNC/memories-and-kb/` path per line)

**Manual refresh** (if needed):
```bash
~/memo/tools/refresh_mcp_index.py
~/memo/tools/sync_repo_memories_to_pepper.py --src-dir /path/to/repo/PROJECT_MEMO/PEPPER_SYNC/memories-and-kb
```

---

## Part 3: MCP Tool Selection SOP

With 300+ MCP tools available, use this quick guide:

| Task type | Primary MCP server | Key tools |
|-----------|-------------------|-----------|
| Codebase work | (built-in) | `grep`, `read_file`, `codebase_search` |
| Git operations | `git-mcp-server` | `git_branch`, `git_diff`, `git_log`, `git_commit`, `git_push` |
| Word documents | `word-document-server` | `add_paragraph`, `add_heading`, `add_table` |
| PDF operations | `pdf-handler` | (see Part 1) |
| Memory/rules | `allPepper-memory-bank` | `memory_bank_read`, `memory_bank_write` |
| Web search | `duckduckgo`, `fetch` | `search`, `fetch_content` |
| Browser automation | `playwright`, `chrome-devtools` | `browser_navigate`, `browser_snapshot` |
| Kubernetes | `kubernetes` | `list-k8s-resources`, `get-k8s-pod-logs` |
| GitHub | `github-official` | `list_issues`, `create_pull_request` |

---

## Part 4: Git Workflow and Release SOP

### Development workflow
- Use feature branches off `main`
- Keep PRs focused (single tool/capability + tests)
- Delete merged branches and run `git fetch --prune`
- Before releases, review/close outdated PRs and delete stale branches (keep active POCs/features)

### Tag-based release (SemVer)

1. Sync main: `git checkout main && git pull --ff-only`
2. Run quality gates: `pytest -q` + `scripts/cursor_smoke.py`
3. Bump version in `pyproject.toml`
4. Update `CHANGELOG.md` under new version heading
5. Commit: `git commit -am "chore(release): vX.Y.Z"`
6. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`
7. Push: `git push origin main --tags`
8. Create GitHub Release with changelog section

---

## Part 5: Portability / Migration

### Clone + install on new machine
```bash
git clone git@github.com:nfsarch33/pdf-mcp-server.git
cd pdf-mcp-server
make install
make test
```

### Register MCP server
Add to `~/.cursor/mcp.json` (see Part 1).

### Connect memory system (optional)
If you have the private repos:
- `~/memo/` (Pepper memory bank)
- `~/Code/global-kb/` (global KB archive)

Follow their READMEs to install daily automation.

### WSL note
If `systemctl --user` fails ("Failed to connect to bus"), use cron-based installer per Pepper's `memory-sop.md`.

---

## Part 6: Principles and Standards

### Code quality (KISS/DRY/SOLID)
- Prefer deterministic edits (annotations > content stream rewriting)
- Always include stable identifiers for edit/remove operations
- Keep tests at two levels: unit tests + MCP smoke tests
- Default to non-breaking changes; ask before breaking

### Safety rules
- Never store secrets in rules, repo files, or `mcp.json` values in repos
- Prefer env vars for secrets
- Keep personal/academic notes untracked

### Repo hygiene
- `PROJECT_MEMO/PEPPER_SYNC/memories-and-kb/` folder for portable KB items
- `PROJECT_MEMO/` for repo-specific runbooks
- `.cursor/rules/` for always-injected rules

---

## Quick Start Prompt (copy/paste for new IDE)

```text
Context: I'm working with the pdf-handler MCP server and multi-layer memory system.

pdf-handler MCP:
- Repo: https://github.com/nfsarch33/pdf-mcp-server
- 25+ PDF tools: forms, text, comments, signatures, encryption
- Hard requirements: form fill/update/delete, comments CRUD, text CRUD, signature CRUD, encrypt

Memory system:
- Layer 1: .cursor/rules/ (short universal rules)
- Layer 2: ~/memo/global-memories/ (Pepper, SOPs/procedures)
- Layer 3: ~/Code/global-kb/ (archive)
- "Add to memory" → route to correct layer automatically

Key Pepper files:
- memory-sop.md (routing rules)
- mcp-index-and-selection-sop.md (tool selection)
- pdf-handler-troubleshooting.md (quick fixes)
- release-sop-tag-based.md (release workflow)

Quality gates:
- pytest -q
- scripts/cursor_smoke.py

Principles: KISS, DRY, SOLID, non-breaking by default.
```

---

## Appendix: Repository Structure

```
pdf-mcp-server/
├── .cursor/rules/           # Always-injected Cursor rules
├── .github/workflows/       # CI, CodeQL, dependency review
├── docs/                    # CURSOR_SMOKE_TEST.md
├── PROJECT_MEMO/PEPPER_SYNC/memories-and-kb/         # Portable KB items (synced to Pepper)
├── pdf_mcp/                 # Main Python package
│   ├── pdf_tools.py         # Core PDF manipulation functions
│   └── server.py            # FastMCP server
├── PROJECT_MEMO/            # Repo-local runbooks
│   ├── MIGRATION_RUNBOOK.md
│   ├── RELEASE_RUNBOOK.md
│   └── TROUBLESHOOTING.md
├── scripts/                 # cursor_smoke.py
├── tests/                   # Unit tests
├── CHANGELOG.md
├── MEMORY_AND_RULES.md
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

*Last updated: 2026-02-10 | Version: 1.0.7*
