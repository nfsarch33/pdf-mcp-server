# Changelog

All notable changes to this project will be documented in this file.

This project follows Keep a Changelog and Semantic Versioning.

## Unreleased

### Planned for v0.3.0 (Phase 3)
- **PDF Optimization**: Compress/reduce PDF file size
- **Link Extraction**: Extract URLs, hyperlinks, and internal references
- **Barcode/QR Code**: Detect and decode barcodes and QR codes in PDFs
- **Page Splitting**: Split PDFs by bookmarks or content markers
- **PDF Comparison**: Diff two PDFs and highlight changes
- **Batch Processing**: Process multiple PDFs in a single call

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


