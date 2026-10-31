# AGENTS.md

Guide for coding agents (and humans) working in this repository. The full
contributor guide is [`CONTRIBUTING.md`](CONTRIBUTING.md); this file is the
short version.

## What this is

`pdf-mcp` is a local-first PDF toolkit exposed two ways from one registry:
an MCP server (`pdf-mcp serve`, stdio) and a CLI (`pdf-mcp <verb> <tool>`).
Tools are declared once in `pdf_mcp/registry.py`; `server.py` and `cli.py`
both read from it. Implementations live in `pdf_mcp/pdf_tools.py` and
`pdf_mcp/agentic_tools.py`.

## Setup and checks

| Goal | Command |
|------|---------|
| Full test suite (75% coverage gate) | `make test` |
| Quick run, no coverage | `pytest -q --no-cov` |
| OCR tests (needs Tesseract) | `make test-ocr` |
| Lint and format changed files | `make lint`, `make format-check` |
| Everything CI runs before a push | `make prepush` |
| Public-repo content check | `bash scripts/ci/deny-pattern.sh` |

Python 3.10 or newer (CI tests 3.10 to 3.12). Re-lock with a current `uv` so `uv.lock` stays at
`revision = 5`.

## Rules for changes

- Add a tool by registering it in `pdf_mcp/registry.py`, then run
  `python scripts/generate_usage_doc.py` to refresh `USAGE.md`. Keep tool
  names, descriptions and order stable; MCP clients depend on them.
- `pdf-mcp --help` and `import pdf_mcp.cli` must not import `pdf_tools`.
  The registry stores lazy callables for that reason.
- Do not send PDF content to a hosted LLM by default. Remote backends need
  `PDF_MCP_ENABLE_REMOTE_LLM=1`; sensitive flows also need
  `PDF_MCP_ALLOW_REMOTE_LLM_FOR_SENSITIVE=1`.
- Write the failing test first, then the fix.
- Add a `CHANGELOG.md` entry under `## Unreleased`.
- Conventional commits (`feat`, `fix`, `docs`, `test`, `chore`, `ci`, ...),
  with no version number in the scope.
- This repository is public. Never commit credentials, private hostnames or
  IP addresses, personal paths, or references to private tooling.
  `scripts/ci/deny-pattern.sh` and gitleaks run on every pull request.

## Releases

Tag `vX.Y.Z` on `main` after the version in `pyproject.toml` and the
`## X.Y.Z` section in `CHANGELOG.md` agree. `.github/workflows/release.yml`
runs the tests, checks the version, and publishes the GitHub Release from
the changelog section.
