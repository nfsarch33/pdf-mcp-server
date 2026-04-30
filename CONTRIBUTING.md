# Contributing to pdf-mcp

Thanks for your interest in pdf-mcp! This guide focuses on the
**architecture** that contributors most need to understand: the
single-source registry that drives both the CLI and the MCP server,
and the workflow for adding a new tool.

For test / lint / release mechanics see [`docs/`](docs/) and
[`README.md`](README.md).

## Repository layout

```
pdf_mcp/
  registry.py          single source of truth for tools (verb, name, description, callable)
  cli.py               Typer CLI entry point; mounts verb groups from the registry
  server.py            FastMCP server; mounts the same tools from the registry
  pdf_tools.py         the actual tool implementations (heavy: pymupdf, pypdf, ...)
  agentic_tools.py     LLM-backed tools (auto_fill, structured extraction, analysis)
  llm_client.py        backend router (local VLM / Ollama / OpenAI)
scripts/
  generate_usage_doc.py  derives USAGE.md from registry.verb_groups()
  cursor_smoke.py        end-to-end smoke that exercises the MCP server
tests/                   pytest suite (526 collected tests, 75% coverage gate)
USAGE.md                 generated; reference of every CLI command
```

## The registry pattern

`pdf_mcp/registry.py` is the only place where tools are declared. Both
the CLI (`pdf_mcp/cli.py`) and the MCP server (`pdf_mcp/server.py`)
iterate over `registry.iter_all()` / `registry.verb_groups()` to mount
their surfaces, so there is no duplicate list of tools to keep in sync.

A tool is a `PdfTool` dataclass with five fields:

| Field | What it carries |
|-------|-----------------|
| `name` | MCP tool name (`snake_case`); also the CLI command after `_` -> `-` translation |
| `verb` | One of `form / pages / text / extract / sign / metadata / ocr / ai / batch / security` (defined in `_VERB_HELP`) |
| `description` | One-sentence human description used in `--help`, USAGE.md, and the MCP schema |
| `callable` | A `LazyCallable("module:attr")` that resolves on first invocation |
| `schema` / `output_schema` | Reserved for a future ticket (auto-generated Typer params) |

### Why lazy callables?

`pdf_mcp.pdf_tools` pulls in `pymupdf`, `pypdf`, optional `openai`, and
many other heavy modules. We do not want `pdf-mcp --help` to pay for
that import. `LazyCallable("pdf_mcp.pdf_tools:fill_pdf_form")` only
calls `importlib.import_module("pdf_mcp.pdf_tools")` when the tool is
actually invoked, so the help / version paths stay sub-second.

This is enforced by tests in
[`tests/test_cli_verb_groups.py::TestLazyImportPreserved`](tests/test_cli_verb_groups.py)
and [`tests/test_registry.py`](tests/test_registry.py).

## Adding a new tool

The work is roughly: write the function, register it, regenerate
docs. There is no separate CLI plumbing or MCP plumbing.

### 1. Implement the function

Put the new function in `pdf_mcp/pdf_tools.py` (or
`pdf_mcp/agentic_tools.py` for LLM-backed tools). Keep the signature
JSON-friendly:

* paths as strings or `pathlib.Path`
* dicts and lists for structured args
* avoid Python objects that don't round-trip JSON

Return either a primitive, a list, or a dict. The CLI auto-serialises
the return value as JSON and the MCP server hands it back to the host.

```python
# pdf_mcp/pdf_tools.py
def shrink_pdf(input_path: str, output_path: str, quality: str = "medium") -> dict:
    """Compress a PDF to reduce file size.

    Args:
        input_path: source PDF path.
        output_path: destination PDF path.
        quality: "low" (max compression), "medium", "high".

    Returns:
        ``{"output_path": "...", "size_before": int, "size_after": int}``
    """
    ...
```

### 2. Register the tool

Add a single `register_tool(...)` call to
`pdf_mcp.registry._seed_default_registry()`:

```python
register_tool(
    name="shrink_pdf",
    verb="pages",
    description="Compress a PDF to reduce file size.",
    import_path=f"{pt}:shrink_pdf",
)
```

The `description` becomes the one-line help shown in `pdf-mcp pages
--help`, in `USAGE.md`, and in the MCP tool schema. Keep it under
~80 characters; longer prose belongs in the docstring.

If you need a brand-new verb, add it to `_VERB_HELP` first with a
short verb-level description.

### 3. Regenerate the usage docs

```bash
python scripts/generate_usage_doc.py
```

`USAGE.md` updates in place. Commit it alongside your tool change so
reviewers see both the registry diff and the doc diff.

### 4. Test it

Add a focused test that exercises the new function directly. The CLI
and MCP layers do not need new tests for routine adds — the existing
parametrised tests in `tests/test_cli_verb_groups.py` and
`tests/test_server_registry_parity.py` automatically cover any new
registry entry on the next run. They will fail if you forget step 2 or
break the lazy-import invariant.

Quick smoke from the CLI:

```bash
pdf-mcp pages shrink-pdf --json '{
  "input_path": "fixture.pdf",
  "output_path": "/tmp/out.pdf",
  "quality": "low"
}' --pretty
```

### 5. Update the changelog

Add an entry under `## Unreleased` in `CHANGELOG.md` under `### Added`,
following the style of existing entries.

## Tests, lint, format

| Goal | Command |
|------|---------|
| Run the full suite (with coverage gate) | `make test` |
| Quick run, no coverage | `pytest -q --no-cov` |
| OCR-specific tests (Tesseract required) | `make test-ocr` |
| LLM mock tests | `make test-llm` |
| Lint changed files (diff vs main) | `make lint` |
| Format changed files (diff vs main) | `make format-check` |
| Pre-push gate (lint + format + test + smoke) | `make prepush` |

Coverage gate: `pytest` is configured with
`--cov=pdf_mcp --cov-fail-under=75`. Below 75% line coverage CI fails.

## Remote LLM safety

`pdf-mcp` is local-first. Do not add code that sends PDF content to a
hosted LLM just because a provider API key exists.

Rules:

* Local model server and Ollama are safe defaults.
* OpenAI / hosted backends require `PDF_MCP_ENABLE_REMOTE_LLM=1` before
  they enter auto-selection.
* Sensitive flows (passport extraction, form-field mapping, or future
  identity / financial-document helpers) must also require
  `PDF_MCP_ALLOW_REMOTE_LLM_FOR_SENSITIVE=1`.
* Add a regression test before any change that touches `_get_llm_backend`,
  `_call_llm`, or the LLM-backed PDF tools.

## Pre-commit hooks

CI runs the same `pre-commit` hooks the local repo does:
`ruff format`, `ruff check`, `end-of-file-fixer`, `trailing-whitespace`,
`check-added-large-files`, `check-yaml`, and a tiny `py_compile` of
repo scripts. The hooks only run on files changed in your branch.

If a hook fails in CI, run the same step locally and re-push:

```bash
ruff format <file>      # or: pre-commit run --files <file>
ruff check --fix <file>
```

## Conventional commits

The repo enforces conventional commits via a CI gate. Use the standard
prefixes: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`,
`perf`, `style`. Scope is optional but should NOT contain version
numbers (e.g. use `feat: ...` or `feat(cli): ...`, not
`feat(v1.3.0): ...`).

Examples:

```
feat: add shrink_pdf tool with three quality presets
fix(extract): handle PDFs without text layer in extract_text
docs: regenerate USAGE.md after shrink_pdf
test: lock shrink_pdf size budget at 50 percent reduction for medium quality
chore: bump pytest-cov pin
```

## Architecture decisions

* **Single source of truth.** Tools are declared once in the registry.
  Anything that needs the canonical tool list (CLI, MCP server, docs
  generator, tests) reads from `pdf_mcp.registry`.
* **Lazy imports.** `pdf-mcp --help` and `import pdf_mcp.cli` MUST NOT
  pull in `pdf_tools`. Adding a tool does not change this invariant
  because the registry only stores a `LazyCallable("module:attr")`,
  not the function itself.
* **JSON over kwargs.** The CLI uses `--json '{...}'` rather than
  per-tool argparse for the v1.3.0 baseline. This keeps the CLI surface
  uniform across all 57 tools and lets future tickets layer
  hand-curated ergonomic flags on top of stable behaviour.
* **Backwards compatibility.** `pdf-mcp serve` is a permanent alias for
  `python -m pdf_mcp.server`. Existing Cursor / Claude Desktop configs
  must not need to change across minor releases.
