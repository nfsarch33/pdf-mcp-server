# PDF MCP Server

MCP server for PDF form filling, basic editing (merge, extract, rotate, flatten), and **OCR text extraction**. Built with Python, `pypdf`, `fillpdf`, and `pymupdf` (AGPL).

**Goal**: Extract 99% of information from any PDF file, including scanned/image-based documents, and fill any PDF forms.

## Status
[![CI](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/ci.yml)
[![CodeQL](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml/badge.svg)](https://github.com/nfsarch33/pdf-mcp-server/actions/workflows/codeql.yml)

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

## Available tools (initial)
- `get_pdf_form_fields(pdf_path)`: list fields and count.
- `fill_pdf_form(input_path, output_path, data, flatten=False)`: fill fields; optional flatten (uses fillpdf if available, else pypdf fallback).
- `clear_pdf_form_fields(input_path, output_path, fields=None)`: clear (delete) values for selected form fields while keeping fields fillable.
- `flatten_pdf(input_path, output_path)`: flatten forms/annotations.
- `merge_pdfs(pdf_list, output_path)`: merge multiple PDFs.
- `extract_pages(input_path, pages, output_path)`: 1-based pages, supports negatives (e.g., -1 = last).
- `rotate_pages(input_path, pages, degrees, output_path)`: degrees must be multiple of 90.
- `add_text_annotation(input_path, page, text, output_path, rect=None, annotation_id=None)`: add a FreeText annotation (managed text insertion).
- `update_text_annotation(input_path, output_path, annotation_id, text, pages=None)`: update an annotation by id.
- `remove_text_annotation(input_path, output_path, annotation_id, pages=None)`: remove an annotation by id.
- `remove_annotations(input_path, output_path, pages, subtype=None)`: remove annotations on pages, optionally filtered by subtype (example FreeText).
- `insert_pages(input_path, insert_from_path, at_page, output_path)`: insert all pages from another PDF before at_page (1-based).
- `remove_pages(input_path, pages, output_path)`: remove specific 1-based pages.
- `insert_text(input_path, page, text, output_path, rect=None, text_id=None)`: insert text via a managed FreeText annotation.
- `edit_text(input_path, output_path, text_id, text, pages=None)`: edit managed inserted text.
- `remove_text(input_path, output_path, text_id, pages=None)`: remove managed inserted text.
- `get_pdf_metadata(pdf_path)`: return basic PDF document metadata.
- `set_pdf_metadata(input_path, output_path, title=None, author=None, subject=None, keywords=None)`: set basic metadata fields.
- `add_text_watermark(input_path, output_path, text, pages=None, rect=None, annotation_id=None)`: add a simple text watermark or stamp via FreeText annotations.
- `add_comment(input_path, output_path, page, text, pos, comment_id=None)`: add a PDF comment (Text annotation, sticky note).
- `update_comment(input_path, output_path, comment_id, text, pages=None)`: update a PDF comment by id.
- `remove_comment(input_path, output_path, comment_id, pages=None)`: remove a PDF comment by id.
- `add_signature_image(input_path, output_path, page, image_path, rect)`: add a signature image to a page (returns `signature_xref`).
- `update_signature_image(input_path, output_path, page, signature_xref, image_path=None, rect=None)`: update or resize a signature image.
- `remove_signature_image(input_path, output_path, page, signature_xref)`: remove a signature image.
- `encrypt_pdf(input_path, output_path, user_password, owner_password=None, ...)`: encrypt (password-protect) a PDF (use after `add_signature_image` to protect a signed PDF).

### OCR and Text Extraction Tools
- `detect_pdf_type(pdf_path)`: analyze PDF to classify as "searchable", "image_based", or "hybrid"; returns page-by-page metrics and OCR recommendation.
- `extract_text_native(pdf_path, pages=None)`: extract text using native PDF text layer only (fast, no OCR).
- `extract_text_ocr(pdf_path, pages=None, engine="auto", dpi=300, language="eng")`: extract text with OCR fallback; engine options: "auto" (native→OCR), "native", "tesseract", "force_ocr".
- `get_pdf_text_blocks(pdf_path, pages=None)`: extract text blocks with bounding box positions (useful for form field detection).

### OCR Phase 2: Enhanced OCR Tools
- `get_ocr_languages()`: get available Tesseract languages and installation status.
- `extract_text_with_confidence(pdf_path, pages=None, language="eng", dpi=300, min_confidence=0)`: OCR with word-level confidence scores; supports multi-language (e.g., "eng+fra").
- `extract_text_smart(pdf_path, pages=None, native_threshold=100, ocr_dpi=300, language="eng")`: smart per-page method selection (native vs OCR) for hybrid documents.

### Table Extraction
- `extract_tables(pdf_path, pages=None, output_format="list")`: extract tables as structured data; format "list" or "dict" (with headers).

### Image Extraction
- `extract_images(pdf_path, output_dir, pages=None, min_width=50, min_height=50, image_format="png")`: extract embedded images to files.
- `get_image_info(pdf_path, pages=None)`: get image metadata (dimensions, format, positions) without extracting.

### Form Auto-Detection
- `detect_form_fields(pdf_path, pages=None)`: detect potential form fields in non-AcroForm PDFs using text pattern analysis (labels, checkboxes, underlines).

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

## Automated tests
```bash
cd /path/to/pdf-mcp-server
make test
```

## Development workflow
- Use feature branches off `main` and open a PR for review.
- Keep each PR focused on a single tool or capability with tests.
- For larger features, split into small PRs (tool surface, core implementation, tests, docs).
- After merging a PR, delete the feature branch and run `git fetch --prune` locally to keep branch state clean.
- Portability/migration notes: see `PROJECT_MEMO/`.

## License
GNU AGPL-3.0, see `LICENSE`.

## Changelog
See `CHANGELOG.md`.

