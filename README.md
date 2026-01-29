# PDF MCP Server

**Version 0.9.8** | MCP server for PDF form filling, editing, OCR text extraction, table extraction, image extraction, link extraction, and batch processing.

Built with Python, `pypdf`, `fillpdf`, and `pymupdf` (AGPL).

**Goal**: Extract 99% of information from any PDF file, including scanned/image-based documents, and fill any PDF forms.

## Status
[![CI](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml)
[![CodeQL](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/nfsarch33/pdf-mcp-server.git
cd pdf-mcp-server
uv pip install -r requirements.txt

# Run tests
make test

# Start the MCP server
python -m pdf_mcp.server
```

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
    "pdf-handler": {
      "command": "/path/to/pdf-mcp-server/.venv/bin/python",
      "args": ["-m", "pdf_mcp.server"],
      "description": "Local PDF form filling and editing (stdio)"
    }
  }
}
```
Restart Cursor after saving.

## Features Overview (51 tools)

| Category | Tools | Description |
|----------|-------|-------------|
| **Form Handling** | 6 tools | Fill, clear, flatten, and create PDF forms |
| **Page Operations** | 5 tools | Merge, extract, rotate, reorder, insert, remove pages |
| **Annotations** | 11 tools | Text annotations, comments, watermarks, redaction, numbering, highlights |
| **Signatures & Security** | 7 tools | Digital signing, verification, encryption |
| **Metadata** | 3 tools | Get, set, and sanitize PDF metadata |
| **Export & Text** | 5 tools | Unified text extraction, export to Markdown/JSON |
| **Table Extraction** | 1 tool | Extract tables as structured data |
| **Image Extraction** | 2 tools | Extract/analyze embedded images |
| **Form Detection** | 1 tool | Auto-detect form fields in non-AcroForm PDFs |
| **PII Detection** | 1 tool | Detect common PII patterns (email, phone, SSN, credit cards) |
| **Link Extraction** | 1 tool | Extract URLs, hyperlinks, internal references |
| **PDF Optimization** | 1 tool | Compress/reduce PDF file size |
| **Barcode/QR Detection** | 1 tool | Detect and decode barcodes and QR codes |
| **Page Splitting** | 1 tool | Split by bookmarks or page count (unified API) |
| **PDF Comparison** | 1 tool | Compare two PDFs and find differences |
| **Batch Processing** | 1 tool | Process multiple PDFs at once |
| **Agentic AI** | 4 tools | LLM-powered form filling, entity extraction, document analysis + local VLM (v0.9.0+) |

## Available Tools

### Form Handling
- `get_pdf_form_fields(pdf_path)`: list fields and count.
- `fill_pdf_form(input_path, output_path, data, flatten=False)`: fill fields; optional flatten.
- `fill_pdf_form_any(input_path, output_path, data, flatten=False)`: fill standard or non-standard forms using label detection when needed.
- `clear_pdf_form_fields(input_path, output_path, fields=None)`: clear values while keeping fields fillable.
- `flatten_pdf(input_path, output_path)`: flatten forms/annotations.
- `create_pdf_form(output_path, fields, page_size=None, pages=1)`: create a new PDF with AcroForm fields.

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
- `extract_structured_data(pdf_path, data_type=None, schema=None, backend=None)`: Extract structured data using pattern matching or LLM.
- `analyze_pdf_content(pdf_path, include_summary=True, detect_entities=True, backend=None)`: Analyze PDF for document type classification, entity extraction, and summarization.

**Backend Priority (v0.9.0+):**
1. **local** (free): Local model server at `localhost:8100` using Qwen3-VL
2. **ollama** (free): Local Ollama models
3. **openai** (paid): OpenAI API (requires `OPENAI_API_KEY`)

**Start local model server (recommended):**
```bash
# From agentic-ai-research project
cd ~/agentic-ai-research
uv run python -m services.model_server.cli serve --port 8100
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
export PDF_MCP_OLLAMA_MODEL="qwen2.5:7b"
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
- **Local VLM**: Start server at `~/agentic-ai-research` with `uv run python -m services.model_server.cli serve --port 8100`
- **Ollama**: Install with `curl -fsSL https://ollama.ai/install.sh | sh`, then `make install-llm-models`
- **OpenAI**: Set `OPENAI_API_KEY` environment variable (costs money!)

### Test Coverage
- **268 tests** total (includes Tier 1/2 coverage + agentic features + multi-backend + e2e tests)
- All tests pass with Tesseract installed
- 12-18 tests skip depending on which optional dependencies/backends are available

## Development Workflow
- Use feature branches off `main` and open a PR for review.
- Keep each PR focused on a single tool or capability with tests.
- For larger features, split into small PRs (tool surface, core implementation, tests, docs).
- After merging a PR, delete the feature branch and run `git fetch --prune` locally to keep branch state clean.
- Portability/migration notes: see `PROJECT_MEMO/`.

## License
GNU AGPL-3.0, see `LICENSE`.

## Changelog
See `CHANGELOG.md`.

