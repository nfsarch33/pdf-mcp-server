# Changelog

All notable changes to this project will be documented in this file.

This project follows Keep a Changelog and Semantic Versioning.

## Unreleased

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


