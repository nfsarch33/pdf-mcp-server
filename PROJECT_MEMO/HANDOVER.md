# PDF MCP Server -- Project Handover

## Current State (v1.2.18)

- **Version**: 1.2.18 (pyproject.toml is single source of truth)
- **Tests**: 433 passed, 9 skipped, 0 failures
- **Open issues**: 0
- **Open PRs**: 0
- **Bugs**: BUG-001 through BUG-011 all resolved
- **CI**: All green (CI, Lint, CodeQL, Secret scan, Release)
- **License**: AGPL-3.0
- **Repo**: https://github.com/nfsarch33/pdf-mcp-server

## What This Project Does

MCP server exposing 50+ tools for PDF manipulation via Model Context Protocol:
- Form filling (including passport/MRZ auto-fill with VLM)
- OCR text extraction (native + Tesseract + VLM-enhanced)
- Table, image, link extraction
- PDF signing (PKCS#12 + PEM), encryption
- Multi-run VLM consensus for stochastic error reduction
- Batch processing, split, merge, compare, export (markdown/json)

## Architecture

| Layer | File | Purpose |
|-------|------|---------|
| MCP server | `pdf_mcp/server.py` | FastMCP tool registration, parameter mapping |
| Core logic | `pdf_mcp/pdf_tools.py` | All PDF operations + agentic AI functions |
| LLM config | `pdf_mcp/llm_setup.py` | Backend priority: local > ollama > openai |
| Version | `pdf_mcp/__init__.py` | `__version__` from pyproject.toml (BUG-011 fix) |
| Tests | `tests/` | 7 test modules, 433+ tests |

## LLM/VLM Stack

- **Backend priority**: local vLLM > Ollama > OpenAI (free first)
- **Default model**: Qwen3-VL-8B (Ollama) / Qwen3-VL-30B-A3B (local vLLM)
- **Apple Silicon note**: Ollama VL models run CPU-only (image analysis >300s). Use NVIDIA GPUs for real workloads.
- **Consensus mode**: `consensus_runs=N` for majority-vote across N VLM runs

## Cross-Platform Setup (self-contained)

```bash
./scripts/setup_environment.sh   # one command: venv, deps, Ollama, GPU detection
./scripts/run_local_vlm.sh       # auto-selects best GPU by VRAM
```

- macOS: MLX backend for Apple Silicon
- Linux/WSL: vLLM + CUDA for NVIDIA GPUs
- Fallback: Ollama (cross-platform)
- GPU selection: auto via `nvidia-smi`, override with `CUDA_VISIBLE_DEVICES`

## Release Process

1. Bump `pyproject.toml` version (only place)
2. Add `CHANGELOG.md` section
3. `git commit` + `git tag -a vX.Y.Z` + `git push origin main --tags`
4. CI auto: test-gate -> version-gate -> GitHub Release from CHANGELOG

Rollback: `git push origin --delete vX.Y.Z && git tag -d vX.Y.Z`, fix, re-tag.

## GitHub Identity

- **Account**: nfsarch33 / jaslian@gmail.com
- **Auth**: `.envrc` with `GH_TOKEN` (gitignored, direnv-managed)
- **Pre-push hook**: `~/.git-hooks/pre-push` blocks Zendesk identity
- **SSH**: `github-agtc` host alias for personal repos on macOS

## Memory System (3 layers)

| Layer | Path | Purpose |
|-------|------|---------|
| Rules | `.cursor/rules/` (3 files) | Always-injected context: memory routing, version strategy, writing standards |
| Pepper | `~/memo/global-memories/` | SOPs, skills, project status. Remote: `nfsarch33/cursor-memory-bank` |
| global-kb | `~/Code/global-kb/` | Archive. Remote: `nfsarch33/cursor-global-kb` (SSH: `github-agtc`) |

Key Pepper files for this project:
- `pdf-mcp-server-status.md` -- project tracker
- `release-sop-checklist.md` -- release process
- `tag-based-release-sop.md` -- CI/CD reference
- `pdf-handler-troubleshooting.md` -- known issues
- `wsl-ubuntu-onboarding.md` -- WSL setup guide

## Skills (installed)

**Project** (`.cursor/skills/`): cross-platform-setup, llm-e2e-qa, memo-kb-sync, pdf-form-filling, release-sop

**Global** (`~/.cursor/skills/`): find-skills, session-handoff, gh-fix-ci, gh-address-comments, go-security-review

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_environment.sh` | Cross-platform env setup (Python, Ollama, GPU) |
| `scripts/run_local_vlm.sh` | GPU-aware VLM server |
| `scripts/check_release_ready.py` | CI gate: tag == pyproject.toml == CHANGELOG |
| `scripts/create_github_release_from_changelog.py` | Auto GitHub Release |
| `scripts/ensure_ollama_model.py` | Idempotent model install |
| `scripts/cursor_smoke.py` | MCP smoke test |

## CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR | pytest + smoke |
| `lint.yml` | push/PR | ruff |
| `release.yml` | tag v*.*.* | test-gate + version-gate + release |
| `codeql.yml` | push/PR/schedule | security analysis |
| `secret-scan.yml` | push/PR | gitleaks |

## Critical Lessons

1. Version: bump `pyproject.toml` only. Run `uv pip install -e .` after.
2. Ollama `num_predict` must be >= 4096 for Qwen3 (thinking tokens).
3. Ollama `keep_alive` = `'10m'` to prevent model unloading.
4. VLM image analysis on Apple Silicon via Ollama: >300s (CPU-only). Use NVIDIA.
5. Tag rollback works: delete + recreate triggers fresh CI.
6. `gh` CLI: use `GH_TOKEN` via `.envrc` to override global `GITHUB_TOKEN`.

## Handover Checklist

- [ ] Clone repo: `git clone git@github.com:nfsarch33/pdf-mcp-server.git`
- [ ] Set identity: `git config user.name "Jason Lian" && git config user.email "jaslian@gmail.com"`
- [ ] Create `.envrc` with `GH_TOKEN` and `direnv allow .`
- [ ] Run `./scripts/setup_environment.sh`
- [ ] Run `make test` to verify
- [ ] Say "health check" in Cursor
- [ ] Read Pepper: `pdf-mcp-server-status.md`

*Created: 2026-02-26*
