# Changelog

All notable changes to this project will be documented in this file.

This project follows Keep a Changelog and Semantic Versioning.

## Unreleased

## 0.9.2 - 2026-01-29

### Added
- **E2E Tests with Real LLM Backends** (not just mocks!)
  - `TestE2ELocalVLM`: 5 tests that call actual local model server at localhost:8100
  - `TestE2EOllama`: 2 tests that call actual Ollama service
  - `TestE2EOpenAI`: 2 tests that call actual OpenAI API
  - `TestBackendComparison`: 1 test comparing outputs across backends
- **Makefile LLM targets**:
  - `make test-llm`: Run all LLM-related tests (mocked)
  - `make test-e2e`: Run E2E tests with real LLM backends (requires running servers)
  - `make check-llm`: Check LLM backend status
  - `make install-llm`: Install LLM dependencies
- Registered `pytest.mark.slow` marker for E2E tests
- Documentation for LLM test targets in README

### Changed
- Total tests increased from 237 to 255 (+18 new tests)
- Skipped tests increased from 8 to 18 (E2E tests skip when servers not available)
- Updated README test coverage section with LLM test documentation

### Technical Notes
- E2E tests automatically skip if corresponding backend is not available
- Local VLM tests require model server at `http://localhost:8100`
- Ollama tests require Ollama service running locally
- OpenAI tests require `OPENAI_API_KEY` (incurs actual API costs!)

## 0.9.1 - 2026-01-29

### Added
- 20 new comprehensive integration tests for v0.9.0 multi-backend LLM support
  - Local VLM integration tests
  - Ollama integration tests
  - Backend field verification tests
  - Backend fallback chain tests
  - Environment configuration tests
  - Unified `_call_llm` routing tests
  - MCP tool registration tests for v0.9.0

### Fixed
- Test compatibility with optional Ollama dependency (proper skip handling)
- Test compatibility with pypdf form filling edge cases

### Test Coverage
- Total tests increased from 217 to 237
- 8 tests skip when optional dependencies unavailable

## 0.9.0 - 2026-01-29

### Added
- **Local VLM Support**: Cost-free local model integration for agentic AI features.
  - Multi-backend support: `local` (localhost:8100), `ollama`, and `openai`
  - Backend auto-detection with priority: local > ollama > openai
  - `get_llm_backend_info()`: Check available backends and current selection
  - Environment variable `PDF_MCP_LLM_BACKEND` for backend override
  - Environment variable `LOCAL_MODEL_SERVER_URL` for custom server URL
- New helper functions:
  - `_check_local_model_server()`: Health check for local server
  - `_call_local_llm()`: Call local model server
  - `_call_ollama_llm()`: Call Ollama models
  - `_get_llm_backend()`: Auto-select best available backend
- 18 new tests for multi-backend support (local VLM, Ollama, OpenAI)

### Changed
- All agentic functions now support `backend` parameter to force specific backend
- Default model changed from `gpt-4o-mini` to `auto` (auto-selects based on backend)
- Agentic functions return `backend` field indicating which LLM was used
- Total test count increased from 199 to 217
- Tool count increased from 50 to 51

### Technical Notes
- **Zero cost option**: Local VLM using Qwen3-VL-30B-A3B at localhost:8100
- **Best VLM for Mac**: Qwen3-VL-30B-A3B (MoE, 95.7% DocVQA, 16.5GB memory)
- **Cross-platform**: Ollama support for easy deployment anywhere
- All backends gracefully degrade - pattern matching works without any LLM

## 0.8.0 - 2026-01-28

### Added
- **Agentic AI Integration**: LLM-powered PDF processing capabilities.
  - `auto_fill_pdf_form`: Intelligent form filling with LLM-powered field mapping. Maps source data to form fields even when names don't exactly match.
  - `extract_structured_data`: Extract structured data from PDFs using pattern matching or LLM. Supports invoice, receipt, contract types and custom schemas.
  - `analyze_pdf_content`: Document analysis including type classification, entity extraction (dates, amounts, names), and summarization.
- `[llm]` optional dependency group for OpenAI integration.
- 22 new tests for agentic features (unit tests with mocks + integration tests).
- LLM helper functions: `_call_llm`, `_HAS_OPENAI` flag for optional OpenAI support.

### Changed
- Total test count increased from 180 to 199 (202 collected, 3 skipped for optional deps).
- Tool count increased from 47 to 50.
- Module docstrings updated to reflect new agentic capabilities.

### Technical Notes
- Agentic features gracefully degrade without OpenAI:
  - `auto_fill_pdf_form`: Falls back to direct field name matching
  - `extract_structured_data`: Uses pattern-based extraction
  - `analyze_pdf_content`: Uses keyword-based classification
- Pattern matching supports common document types without LLM dependency.
- LLM integration uses `gpt-4o-mini` by default for cost efficiency.

## 0.7.0 - 2026-01-26

### Removed
Deprecated tools have been removed as announced in v0.6.0:
- `insert_text`, `edit_text`, `remove_text` - Use `add_text_annotation`, `update_text_annotation`, `remove_text_annotation` instead
- `extract_text_native`, `extract_text_ocr`, `extract_text_smart`, `extract_text_with_confidence` - Use `extract_text` with appropriate `engine` parameter instead
- `split_pdf_by_bookmarks`, `split_pdf_by_pages` - Use `split_pdf` with `mode` parameter instead
- `export_to_markdown`, `export_to_json` - Use `export_pdf` with `format` parameter instead
- `get_full_metadata` - Use `get_pdf_metadata(pdf_path, full=True)` instead

### Changed
- **API Cleanup**: 12 deprecated functions removed, consolidating into 4 unified tools
- Internal implementations refactored with `_impl` helper functions for cleaner code
- All tests updated to use the new consolidated API
- Tool count reduced from 59 to 47 (cleaner API surface)

### Fixed
- Server module docstring had duplicate "PII detection" line

## 0.6.0 - 2026-01-28

### Added
- **Consolidated API**: Unified tools for cleaner, more maintainable API surface.
  - `extract_text`: Unified text extraction with engine selection (native, auto, smart, ocr, force_ocr) and optional confidence scores.
  - `split_pdf`: Unified PDF splitting with mode selection (pages, bookmarks).
  - `export_pdf`: Unified export with format selection (markdown, json).
  - `get_pdf_metadata(full=True)`: Extended metadata including document info.
- 12 new integration tests for consolidated API tools.

### Changed
- Total test count increased from 168 to 180.

### Deprecated
The following tools are deprecated and will be removed in v0.7.0:
- `insert_text`, `edit_text`, `remove_text` → Use `add_text_annotation`, `update_text_annotation`, `remove_text_annotation`
- `extract_text_native`, `extract_text_ocr`, `extract_text_smart`, `extract_text_with_confidence` → Use `extract_text`
- `split_pdf_by_bookmarks`, `split_pdf_by_pages` → Use `split_pdf`
- `export_to_markdown`, `export_to_json` → Use `export_pdf`
- `get_full_metadata` → Use `get_pdf_metadata(full=True)`

## 0.5.2 - 2026-01-29

### Added
- Signature timestamping via `timestamp_url` for `sign_pdf` and `sign_pdf_pem`.
- Revocation checks and validation embedding controls for signing.
- DocMDP permission selection (`no_changes`, `fill_forms`, `annotate`) for signed PDFs.
- Integration test coverage for timestamped and DocMDP-signed PDFs.

## 0.5.1 - 2026-01-28

### Added
- `sign_pdf`: digitally sign PDFs using PKCS#12/PFX certificates.
- `sign_pdf_pem`: digitally sign PDFs using PEM key + cert chain.
- Integration tests for certificate-based signing.
- `cryptography` dependency for test certificate generation.

## 0.5.0 - 2026-01-28

### Added
- `create_pdf_form`: create PDF files with standard AcroForm fields.
- `fill_pdf_form_any`: fill standard and non-standard forms using label detection.
- `add_highlight`: add highlight annotations by text or rectangle.
- `add_date_stamp`: add date stamps as FreeText annotations.
- `detect_pii_patterns`: detect common PII patterns (email, phone, SSN, credit cards).
- Release runbook: added PR/branch hygiene SOP.

## 0.4.1 - 2026-01-27

### Added
- `PROJECT_MEMO/PROJECT_STATUS_PROMPT.md`: status prompt for v0.3.0 planning.
- `PROJECT_MEMO/README.md`: reference to the new status prompt.

## 0.4.0 - 2026-01-27

### Added
- `reorder_pages`: reorder PDF pages with an explicit 1-based page list.
- `redact_text_regex`: redact text using a regex pattern.
- `sanitize_pdf_metadata`: remove standard and custom metadata keys.
- `export_to_markdown`: export PDF text to Markdown.
- `export_to_json`: export PDF text and metadata to JSON.
- `add_page_numbers`: add page numbers as annotations.
- `add_bates_numbering`: add Bates numbering as annotations.
- `verify_digital_signatures`: validate digital signatures in PDFs.
- `get_full_metadata`: return full metadata and document info.

## 0.3.0 - 2026-01-27

### Added
- **Link Extraction**
  - `extract_links`: Extract URLs, hyperlinks, and internal references from PDFs
  - Link categorization by type (uri, goto, external_goto, launch, named)
  - Page-level link filtering

- **PDF Optimization**
  - `optimize_pdf`: Compress/reduce PDF file size
  - Quality settings: low (max compression), medium, high (min compression)
  - Reports original/optimized size and compression ratio

- **Barcode/QR Code Detection**
  - `detect_barcodes`: Detect and decode barcodes and QR codes in PDFs
  - Supports QR codes, Code128, Code39, EAN13, EAN8, UPC-A, etc.
  - Requires optional pyzbar library

- **Page Splitting**
  - `split_pdf_by_bookmarks`: Split PDFs by table of contents/bookmarks
  - `split_pdf_by_pages`: Split PDFs by page count
  - Configurable pages per split

- **PDF Comparison**
  - `compare_pdfs`: Diff two PDFs and identify differences
  - Compares page count, text content, and optionally images
  - Generates human-readable summary

- **Batch Processing**
  - `batch_process`: Process multiple PDFs with a single operation
  - Supports: get_info, extract_text, extract_links, optimize
  - Reports individual success/failure for each file

- 42 new integration tests for Phase 3 features
- Module docstrings in pdf_tools.py and server.py

### Changed
- Total test count increased to 149

## 0.2.0 - 2026-01-26

### Added
- **OCR Phase 2: Enhanced OCR with Multi-language Support**
  - `get_ocr_languages`: Get available OCR languages and Tesseract status
  - `extract_text_with_confidence`: OCR with word-level confidence scores (0-100)
  - Multi-language support using Tesseract language codes (e.g., "eng+fra")
  - Confidence filtering with `min_confidence` parameter

- **Table Extraction**
  - `extract_tables`: Extract tables from PDF pages as structured data
  - Output formats: "list" (list of lists) or "dict" (list of dicts with headers)
  - Table bounding box and cell data extraction

- **Image Extraction**
  - `extract_images`: Extract embedded images to files (png/jpeg/ppm)
  - `get_image_info`: Get image metadata without extracting
  - Configurable minimum dimensions filter
  - Image position and format information

- **Smart/Hybrid Text Extraction**
  - `extract_text_smart`: Per-page method selection (native vs OCR)
  - Configurable native text threshold for OCR fallback
  - Optimal handling of hybrid documents with mixed page types

- **Form Auto-Detection**
  - `detect_form_fields`: Detect potential form fields using text analysis
  - Label pattern detection (Name:, Date:, Address:, etc.)
  - Checkbox/selection pattern detection
  - Suggestions for non-AcroForm PDF forms
  - Field type guessing based on label text

- Comprehensive integration tests for all Phase 2 features (38 new test cases)

### Changed
- Project goal clarified: "Extract 99% of information from any PDF file"

## 0.1.3 - 2026-01-06

### Added
- **OCR Support (Phase 1)**: New tools for text extraction from scanned/image-based PDFs.
  - `detect_pdf_type`: Classify PDFs as "searchable", "image_based", or "hybrid" with detailed metrics.
  - `extract_text_native`: Fast native text layer extraction (no OCR).
  - `extract_text_ocr`: Text extraction with OCR fallback; supports auto/native/tesseract/force_ocr engines.
  - `get_pdf_text_blocks`: Extract text blocks with bounding box positions for layout analysis.
- Optional `[ocr]` dependency group: `pytesseract` and `pillow` for Tesseract integration.
- Comprehensive OCR test suite (`tests/test_ocr.py`) covering 9 PDF fixtures with 33+ test methods.
- New PDF test fixtures for OCR testing: scanned documents, image-based PDFs, hybrid documents.

### Changed
- Updated project description to reflect OCR capabilities.
- README now includes OCR setup instructions and tool documentation.

## 0.1.2 - 2025-12-17

### Fixed
- `fill_pdf_form`: if `fillpdf/pdfrw` cannot parse PDFs with compressed object streams (common in some Adobe InDesign exports), we fall back to the `pypdf` fill path so the operation succeeds.
- `flatten_pdf`: same robustness as above; falls back to `pypdf` when `fillpdf/pdfrw` cannot parse the input.
- `flatten_pdf` internal behavior: handle PDFs where `/AcroForm` is an indirect object and ensure `/Annots` updates use proper PDF object keys.

### Added
- Real-world regression coverage using `tests/1006.pdf` (InDesign-style form PDF) that runs every MCP tool end-to-end with two scenarios.

## 0.1.1 - 2025-12-16

### Added
- `clear_pdf_form_fields`: clear (delete) values for selected form fields while keeping fields fillable.
- `encrypt_pdf`: password-protect PDFs (intended after `add_signature_image` to protect a signed PDF).
- Cursor post-push smoke test: `scripts/cursor_smoke.py` and `docs/CURSOR_SMOKE_TEST.md`.

### Changed
- Form filling is more robust on non-standard AcroForms; values are persisted in `/V` and `encrypt_pdf` normalizes trailer IDs for compatibility.
- Memory/rules hygiene: repo includes `.cursor/rules/template.rules` and documented SOP to keep academic/personal content untracked.

## 0.1.0

### Added
- MCP server over stdio with PDF tools for form fields, form fill, flatten, merge, extract, rotate.
- Annotation and page editing tools.
- Managed text insert, edit, remove via FreeText annotations.
- Metadata get and set tools.
- GitHub Actions workflows: CI, CodeQL, dependency review, optional AI review.


