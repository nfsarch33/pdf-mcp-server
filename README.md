# pdf-mcp

A first-class **command-line tool** and **Model Context Protocol (MCP)
server** for everything PDF: form filling, page operations, OCR,
metadata, signatures, redaction, image and table extraction, batch
processing, and LLM-backed document understanding.

Built with Python, `pypdf` (BSD), `fillpdf` (MIT), and `pymupdf` (AGPL-3.0).

**Goal**: Extract 99% of information from any PDF file, including
scanned / image-based documents, and fill any PDF form. Works equally
well as a daily CLI for humans, as a CI / automation script, and as an
MCP server inside Cursor or Claude Desktop.

## Status
[![Release](https://img.shields.io/github/v/release/nfsarch33/pdf-mcp-server?style=flat&label=release)](https://github.com/nfsarch33/pdf-mcp-server/releases/latest)
[![CI](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml)
[![CodeQL](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick start

```bash
# 1. Install (project root)
git clone https://github.com/nfsarch33/pdf-mcp-server.git
cd pdf-mcp-server
make install      # uv-based install of runtime requirements

# 2. Try the CLI
pdf-mcp --help                                    # top-level help
pdf-mcp form --help                               # 9 form-handling tools
pdf-mcp ai get-llm-backend-info --pretty          # check LLM backends

# 3. Run as an MCP server (Cursor, Claude Desktop, etc.)
pdf-mcp serve                                     # stdio transport

# 4. Run the tests
make test
```

Run the CLI on a real PDF:

```bash
# Read form fields
pdf-mcp form get-pdf-form-fields \
  --json '{"pdf_path":"sample.pdf"}' --pretty

# Fill a form (writes filled.pdf)
pdf-mcp form fill-pdf-form --json '{
  "input_path": "sample.pdf",
  "output_path": "filled.pdf",
  "data": {"Name": "Jane Doe", "Email": "jane@example.com"},
  "flatten": false
}'

# OCR a scanned PDF
pdf-mcp extract extract-text --json '{
  "pdf_path": "scanned.pdf",
  "engine": "ocr",
  "language": "eng"
}' --pretty
```

## CLI surface (v1.3.0+)

`pdf-mcp` ships **all 57 underlying tools** as CLI subcommands grouped
by verb. The taxonomy comes directly from `pdf_mcp.registry`, so the
CLI and MCP surfaces never drift.

| Verb | Tools | Examples |
|------|-------|----------|
| `pdf-mcp form` | 9 | `fill-pdf-form`, `get-pdf-form-fields`, `flatten-pdf` |
| `pdf-mcp pages` | 8 | `merge-pdfs`, `split-pdf`, `extract-pages`, `rotate-pages` |
| `pdf-mcp text` | 13 | `redact-text-regex`, `add-text-watermark`, `add-bates-numbering` |
| `pdf-mcp extract` | 7 | `extract-text`, `extract-tables`, `extract-images`, `extract-links` |
| `pdf-mcp metadata` | 4 | `get-pdf-metadata`, `set-pdf-metadata`, `sanitize-pdf-metadata` |
| `pdf-mcp sign` | 6 | `sign-pdf`, `verify-digital-signatures`, `add-signature-image` |
| `pdf-mcp ocr` | 2 | `get-ocr-languages`, `get-image-info` |
| `pdf-mcp ai` | 4 | `auto-fill-pdf-form`, `extract-structured-data`, `analyze-pdf-content` |
| `pdf-mcp batch` | 2 | `batch-process`, `compare-pdfs` |
| `pdf-mcp security` | 2 | `encrypt-pdf`, `detect-pii-patterns` |
| `pdf-mcp serve` | — | Run as an MCP server over stdio (drop-in for `python -m pdf_mcp.server`). |

Every command accepts:

```
pdf-mcp <verb> <tool> [--json '{...}'] [--json-file PATH]
                      [--pretty] [--output PATH]
```

* `--json` and `--json-file` are mutually exclusive (`--json` wins).
* `--pretty` indents the JSON output for human reading.
* `--output PATH` writes the JSON result to a file instead of stdout.
* Tool exceptions exit non-zero with `error: <tool> failed: <msg>` on
  stderr, so the CLI is safe to use in pipelines.

The full per-tool reference is in [`USAGE.md`](USAGE.md), generated
directly from the registry by `scripts/generate_usage_doc.py`.

### Why this layout?

* **Verb-first** matches Unix muscle memory (`git commit`, `kubectl apply`).
* **Same code as the MCP surface** — there is one source of truth
  (`pdf_mcp.registry`) so a CLI invocation and an MCP tool call hit
  identical implementations.
* **Lazy imports** — `pdf-mcp --help` and `pdf-mcp <verb> --help`
  do **not** load `pymupdf` or `pypdf`. The heavy dependency tree only
  loads when a tool is actually run, so help is sub-second cold.
* **Backwards compatible** — `pdf-mcp serve` is a drop-in replacement
  for `python -m pdf_mcp.server`. Existing Cursor / Claude Desktop
  configs need no changes.

## CI notes
- Dependency Review requires GitHub Dependency Graph to be enabled in the repository settings.
- AI Review is optional and only runs if you add the `OPENAI_API_KEY` repository secret.

## Setup (uv)
1) Install `uv` if not present:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

2) Install dependencies (project root is this folder):
```bash
cd /path/to/pdf-mcp-server
uv pip install -r requirements.txt
```

Or use the Makefile:
```bash
cd /path/to/pdf-mcp-server
make install
```

For best flatten support, install Poppler:
```bash
sudo apt-get install poppler-utils
```

### OCR Support (Optional)

For OCR capabilities on scanned/image-based PDFs, install Tesseract:

**macOS:**
```bash
brew install tesseract
pip install pytesseract pillow
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract pillow
```

**Additional language packs (optional):**
```bash
sudo apt-get install tesseract-ocr-chi-tra tesseract-ocr-jpn tesseract-ocr-kor
```

Or install with the `ocr` extra:
```bash
pip install -e ".[ocr]"
```

## Run the MCP server
```bash
python -m pdf_mcp.server
```
(It runs over stdio by default.)

## Register with Cursor
Edit `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "pdf-mcp": {
      "command": "/path/to/pdf-mcp-server/.venv/bin/python",
      "args": ["-m", "pdf_mcp.server"],
      "description": "Local PDF form filling and editing (stdio)"
    }
  }
}
```
Restart Cursor after saving.

## Features overview (57 tools across 10 verbs)

| Verb group | Tools | What it does |
|------------|-------|--------------|
| `form` | 9 | Read, fill, clear, flatten, and create PDF forms (AcroForm + label-detection fallback). |
| `pages` | 8 | Merge, split, extract, rotate, reorder, insert, remove, optimise. |
| `text` | 13 | Annotations, redaction, watermarks, comments, page numbers, Bates stamps. |
| `extract` | 7 | Text blocks, tables, images, links, structured data, barcodes/QR, format export. |
| `metadata` | 4 | Read, write, sanitise metadata + classify PDF type (searchable / image / hybrid). |
| `sign` | 6 | PKCS#12 / PEM digital signatures + signature image stamps + verification. |
| `ocr` | 2 | OCR language inventory + image inspection (Tesseract + optional packs). |
| `ai` | 4 | LLM-backed form auto-fill, structured data extraction, document analysis. |
| `batch` | 2 | Multi-file processing and PDF-vs-PDF comparison. |
| `security` | 2 | Password encryption + PII pattern detection. |

**See [`USAGE.md`](USAGE.md) for the full per-tool command reference.**

The same 57 tools are exposed as MCP tools with their original
`snake_case` names (e.g. `fill_pdf_form`, `extract_text`,
`detect_pii_patterns`).

## Available tools (Python API reference)

The Python API on `pdf_mcp.pdf_tools` matches the CLI / MCP surface
1:1. `pdf-mcp form fill-pdf-form ...` calls the same
`pdf_tools.fill_pdf_form(...)` function as an MCP `fill_pdf_form` tool
call. Pick the surface that matches your workflow.

### Form Handling
- `get_pdf_form_fields(pdf_path)`: list fields and count.
- `fill_pdf_form(input_path, output_path, data, flatten=False)`: fill fields; optional flatten.
- `fill_pdf_form_any(input_path, output_path, data, flatten=False)`: fill standard or non-standard forms using label detection when needed.
- `clear_pdf_form_fields(input_path, output_path, fields=None)`: clear values while keeping fields fillable.
- `flatten_pdf(input_path, output_path)`: flatten forms/annotations.
- `create_pdf_form(output_path, fields, page_size=None, pages=1)`: create a new PDF with AcroForm fields.
- `get_form_templates()`: list built-in templates for common workflows.
- `create_pdf_form_from_template(output_path, template_name)`: create a form from a built-in template.
Note: XFA/LiveCycle forms are not supported; convert to AcroForm or flatten first.

### Page Operations
- `merge_pdfs(pdf_list, output_path)`: merge multiple PDFs.
- `extract_pages(input_path, pages, output_path)`: 1-based pages, supports negatives (e.g., -1 = last).
- `rotate_pages(input_path, pages, degrees, output_path)`: degrees must be multiple of 90.
- `reorder_pages(input_path, pages, output_path)`: reorder pages using a 1-based page list.
- `insert_pages(input_path, insert_from_path, at_page, output_path)`: insert pages from another PDF.
- `remove_pages(input_path, pages, output_path)`: remove specific pages.

### Annotations & Text
- `add_text_annotation(input_path, page, text, output_path, rect=None, annotation_id=None)`: add FreeText annotation.
- `update_text_annotation(input_path, output_path, annotation_id, text, pages=None)`: update annotation by id.
- `remove_text_annotation(input_path, output_path, annotation_id, pages=None)`: remove annotation by id.
- `remove_annotations(input_path, output_path, pages, subtype=None)`: remove annotations, optionally by subtype.
- `redact_text_regex(input_path, output_path, pattern, ...)`: redact text using a regex pattern.
- `add_text_watermark(input_path, output_path, text, ...)`: add text watermark/stamp.
- `add_highlight(input_path, output_path, page, text=None, rect=None)`: add highlight annotations.
- `add_date_stamp(input_path, output_path, ...)`: add a date stamp.
- `add_page_numbers(input_path, output_path, ...)`: add page numbers as annotations.
- `add_bates_numbering(input_path, output_path, ...)`: add Bates numbering as annotations.
- `add_comment` / `update_comment` / `remove_comment`: PDF comments (sticky notes).

### Signatures & Security
- `add_signature_image(input_path, output_path, page, image_path, rect)`: add signature image.
- `update_signature_image(...)`: update or resize signature.
- `remove_signature_image(...)`: remove signature image.
- `sign_pdf(input_path, output_path, pfx_path, ...)`: sign with PKCS#12/PFX, optional `timestamp_url`, `embed_validation_info`, `allow_fetching`, `docmdp_permissions`.
- `sign_pdf_pem(input_path, output_path, key_path, cert_path, ...)`: sign with PEM key + cert chain, supports timestamping, revocation checks, DocMDP.
- `encrypt_pdf(input_path, output_path, user_password, ...)`: password-protect PDF.
- `verify_digital_signatures(pdf_path)`: verify digital signatures.

### Metadata
- `get_pdf_metadata(pdf_path, full=False)`: return document metadata; set `full=True` for extended info.
- `set_pdf_metadata(input_path, output_path, title=None, author=None, ...)`: set metadata fields.
- `sanitize_pdf_metadata(input_path, output_path, ...)`: remove metadata keys.

### Export
- `export_pdf(pdf_path, output_path, format="markdown", ...)`: export text to Markdown or JSON. Formats: "markdown", "json".

### OCR and Text Extraction
- `detect_pdf_type(pdf_path)`: analyze PDF to classify as "searchable", "image_based", or "hybrid"; returns page-by-page metrics and OCR recommendation.
- `extract_text(pdf_path, engine="auto", pages=None, include_confidence=False, ...)`: unified text extraction. Engines: "native", "auto", "smart", "ocr", "force_ocr". Set `include_confidence=True` for word-level confidence scores.
- `get_pdf_text_blocks(pdf_path, pages=None)`: extract text blocks with bounding box positions (useful for form field detection).
- `get_ocr_languages()`: get available Tesseract languages and installation status.
Optional language packs: `chi_tra`, `jpn`, `kor` improve OCR for low-quality scans.

### Table Extraction
- `extract_tables(pdf_path, pages=None, output_format="list")`: extract tables as structured data; format "list" or "dict" (with headers).

### Image Extraction
- `extract_images(pdf_path, output_dir, pages=None, min_width=50, min_height=50, image_format="png")`: extract embedded images to files.
- `get_image_info(pdf_path, pages=None)`: get image metadata (dimensions, format, positions) without extracting.

### Form Auto-Detection
- `detect_form_fields(pdf_path, pages=None)`: detect potential form fields in non-AcroForm PDFs using text pattern analysis (labels, checkboxes, underlines).

### PII Detection
- `detect_pii_patterns(pdf_path, pages=None)`: detect common PII patterns (email, phone, SSN, credit cards).

### Link Extraction
- `extract_links(pdf_path, pages=None)`: extract URLs, hyperlinks, and internal references with type categorization.

### PDF Optimization
- `optimize_pdf(pdf_path, output_path, quality="medium")`: compress PDF; quality: "low" (max compression), "medium", "high".

### Barcode/QR Detection
- `detect_barcodes(pdf_path, pages=None, dpi=200)`: detect and decode QR codes, Code128, EAN13, etc. Requires optional `pyzbar`.

### Page Splitting
- `split_pdf(pdf_path, output_dir, mode="pages", pages_per_split=1)`: split PDF by pages or bookmarks. Modes: "pages", "bookmarks".

### PDF Comparison
- `compare_pdfs(pdf1_path, pdf2_path, compare_text=True, compare_images=False)`: diff two PDFs, find text and page differences.

### Batch Processing
- `batch_process(pdf_paths, operation, output_dir=None)`: process multiple PDFs; operations: "get_info", "extract_text", "extract_links", "optimize".

### API Reference

**Unified Text Extraction:**
```python
extract_text(pdf_path, engine="auto", pages=None, include_confidence=False,
             native_threshold=100, dpi=300, language="eng", min_confidence=0)
```
- Engines: "native" (fast), "auto" (native→OCR), "smart" (per-page), "ocr"/"tesseract", "force_ocr"
- Set `include_confidence=True` for word-level confidence scores

**Unified Page Splitting:**
```python
split_pdf(pdf_path, output_dir, mode="pages", pages_per_split=1)
```
- Modes: "pages" (by count), "bookmarks" (by table of contents)

**Unified Export:**
```python
export_pdf(pdf_path, output_path, format="markdown", engine="auto", ...)
```
- Formats: "markdown", "json"

**Extended Metadata:**
```python
get_pdf_metadata(pdf_path, full=False)  # Set full=True for document info
```

### Agentic AI (v0.8.0+) with Local VLM Support (v0.9.0+)

LLM-powered tools for intelligent PDF processing. **Uses local VLM by default (free, no API costs!)** Falls back to Ollama or OpenAI.

- `get_llm_backend_info()`: Check which LLM backends are available (local, ollama, openai).
- `auto_fill_pdf_form(pdf_path, output_path, source_data, backend=None)`: Intelligently fill form fields with LLM-powered field mapping.
- `extract_structured_data(pdf_path, data_type=None, schema=None, ocr_engine="auto", ocr_language="eng", backend=None)`: Extract structured data using pattern matching or LLM (includes non-LLM `passport` extraction via MRZ + labels). Use `ocr_language` for non-English OCR.
- `analyze_pdf_content(pdf_path, include_summary=True, detect_entities=True, backend=None)`: Analyze PDF for document type classification, entity extraction, and summarization.

**Backend Priority (v0.9.0+):**
1. **local** (free): Local model server at `localhost:8100` using Qwen3-VL
2. **ollama** (free): Local Ollama models
3. **openai** (paid): OpenAI API (requires `OPENAI_API_KEY`)

**Start local model server (recommended):**
```bash
# Self-contained setup (one-time)
./scripts/setup_environment.sh

# Start local VLM (auto-detects best GPU)
./scripts/run_local_vlm.sh
```

**Or use Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
# Install model (skips download if already present)
make install-llm-models
```
Optional override:
```bash
export PDF_MCP_OLLAMA_MODEL="qwen3-vl:8b"
```

**Or use OpenAI (paid):**
```bash
pip install -e ".[llm]"
export OPENAI_API_KEY="your-api-key"
export PDF_MCP_LLM_BACKEND="openai"  # Force OpenAI
```

**Example:**
```python
# Check available backends
info = get_llm_backend_info()
print(info["current_backend"])  # "local", "ollama", or "openai"

# Intelligent form filling (uses local VLM by default)
result = auto_fill_pdf_form("form.pdf", "filled.pdf", {
    "name": "John Smith",  # Maps to "Full Name" field
    "email_address": "john@example.com"  # Maps to "Email" field
})
print(result["backend"])  # Which backend was used

# Extract invoice data
result = extract_structured_data("invoice.pdf", data_type="invoice")
print(result["data"]["total"])  # "$162.00"

# Analyze document
result = analyze_pdf_content("document.pdf")
print(result["document_type"])  # "invoice"
print(result["summary"])  # "Invoice #12345 for $162.00..."
```

## Conventions
- Paths should be absolute; outputs are created with parent directories if missing.
- Inputs must exist and be files; errors return `{ "error": "..." }`.
- Form flattening prefers fillpdf+poppler; falls back to a pypdf-only flatten (removes form structures).
 - Text insert/edit/remove is implemented via managed FreeText annotations, not by editing PDF content streams.

## Smoke tests (manual)
```bash
python - <<'PY'
from pdf_mcp import pdf_tools
sample = "/path/to/sample.pdf"
out = "/tmp/out.pdf"
print(pdf_tools.get_pdf_form_fields(sample))
print(pdf_tools.fill_pdf_form(sample, out, {"Name": "Test"}, flatten=True))
PY
```

## Automated Tests

```bash
cd /path/to/pdf-mcp-server

# Run all tests
make test

# Run OCR-specific tests (requires Tesseract)
make test-ocr

# Quick test run
make test-quick

# Check Tesseract installation status
make check-tesseract

# Install OCR dependencies
make install-ocr

# Pre-push checks (lint + format + test + smoke)
make prepush
```

### LLM/Agentic AI Tests (v0.9.2+)

```bash
# Run all LLM-related tests (mocked)
make test-llm

# Run E2E tests with real LLM backends
# Requires: local model server at localhost:8100, or Ollama, or OPENAI_API_KEY
make test-e2e

# Check LLM backend status
make check-llm

# Install LLM dependencies
make install-llm

# Ensure Ollama model is present (no duplicate installs)
make install-llm-models
```

**E2E Test Requirements:**
- **Local VLM**: Run `./scripts/run_local_vlm.sh` (auto-detects best GPU)
- **Ollama**: Install with `curl -fsSL https://ollama.ai/install.sh | sh`, then `make install-llm-models`
- **OpenAI**: Optional remote fallback. Set `OPENAI_API_KEY` plus
  `PDF_MCP_ENABLE_REMOTE_LLM=1` to opt in. Sensitive flows such as
  passport extraction still require
  `PDF_MCP_ALLOW_REMOTE_LLM_FOR_SENSITIVE=1`.

**Full environment setup (one-time):**
```bash
./scripts/setup_environment.sh
```
This handles Python venv, system packages, Ollama, GPU detection, and VLM runner generation for both macOS and WSL/Linux.

### Test coverage
- **526 collected tests** (496 passing, 30 optional-backend skips on the
  current macOS dev lane) covering Tier 1/2, agentic features,
  multi-backend behavior, e2e paths, v1.3.0 CLI surface, and verb-group
  mounts.
- 75% line coverage gate enforced in CI via `pytest --cov-fail-under=75` (see `pyproject.toml`).
- `pytest --cov` ships in dev extras (`make install-dev`) and writes
  `coverage.xml` for upload to coverage tools.
- All tests pass with Tesseract installed; a small number skip
  depending on which optional dependencies / backends are available.

## Privacy and remote LLMs

`pdf-mcp` is local-first. PDF text is processed on your machine by
default, and LLM-backed tools prefer local model servers or Ollama.

OpenAI support is optional and explicit:

```bash
export OPENAI_API_KEY=...
export PDF_MCP_ENABLE_REMOTE_LLM=1
```

Sensitive workflows, including passport extraction and form-field
mapping, are blocked from remote LLM backends by default. To use a
remote model for those flows, set:

```bash
export PDF_MCP_ALLOW_REMOTE_LLM_FOR_SENSITIVE=1
```

This fail-closed behaviour avoids accidental provider policy blocks and
prevents identity-document text from leaving the host unless the user
deliberately opts in.

## Development Workflow
- Use feature branches off `main` and open a PR for review.
- Keep each PR focused on a single tool or capability with tests.
- For larger features, split into small PRs (tool surface, core implementation, tests, docs).
- After merging a PR, delete the feature branch and run `git fetch --prune` locally to keep branch state clean.

## License

`pdf-mcp` itself is **Apache-2.0** (since v1.3.0; v1.2.x and earlier were AGPL-3.0).
See [`LICENSE`](LICENSE) for full text and [`NOTICE`](NOTICE) for the relicense
notice and attribution.

> **Important:** `pdf-mcp` depends on [`pymupdf`](https://github.com/pymupdf/PyMuPDF),
> which is **AGPL-3.0** (with an optional commercial license from Artifex). When
> you distribute `pdf-mcp` together with `pymupdf`, the combined distribution is
> subject to AGPL-3.0 terms because `pymupdf` is the copyleft component. The
> Apache-2.0 license on `pdf-mcp`'s own code remains valid for use, modification,
> and redistribution of `pdf-mcp` code in isolation or with permissive-licensed
> dependencies. Closed-source/proprietary distributors should contact Artifex
> for a commercial `pymupdf` license, or wait for a future release that
> replaces `pymupdf` with permissive-licensed alternatives. See [`NOTICE`](NOTICE)
> for the full third-party dependency breakdown.

## Changelog
See `CHANGELOG.md`.
