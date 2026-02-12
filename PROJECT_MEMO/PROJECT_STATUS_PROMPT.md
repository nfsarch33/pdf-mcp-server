# Project Status Prompt

Use this prompt to sync status and plan next moves for `pdf-mcp-server`.

Version is derived from `pyproject.toml` (single source of truth). Check with:
`python -c "from pdf_mcp import __version__; print(__version__)"`

---

Review the status and plan next moves. Version centralized in pyproject.toml; tag-based release CI creates GitHub Releases automatically.

Context
- Repo: https://github.com/nfsarch33/pdf-mcp-server
- Cursor has pdf-handler MCP enabled; verify tool behaviors through MCP calls.

Immediate directive
- Only bump pyproject.toml when preparing the next release tag (SemVer).
- Prepare tag + GitHub Release notes only when release-ready.

Hard requirements (must be fully working)
- Enhanced PDF editing beyond basic manipulation.
- PDF forms: fill, update, and delete values in fillable fields.
- Signing + encryption: sign, then password-protect signed PDFs.
- Digital certificate signing (`sign_pdf`, `sign_pdf_pem`) with timestamping, validation embedding, and DocMDP permissions.

Expanded capabilities (current main)
- Agentic AI integration (v0.8.0+): `auto_fill_pdf_form`, `extract_structured_data`, `analyze_pdf_content`.
- Local VLM integration (v0.9.0+): zero-cost local backend with priority local > ollama > openai.
- OCR Phase 1/2: detect_pdf_type, unified `extract_text` with engine selection.
- Table extraction, image extraction, form auto-detection.
- Link extraction, optimization, barcode detection (optional pyzbar).
- Unified `split_pdf` (modes: pages, bookmarks), PDF comparison, batch processing.
- Unified `export_pdf` (formats: markdown, json).
- Form creation (`create_pdf_form`), label-based fill (`fill_pdf_form_any`).
- Highlight annotations, date stamps, PII pattern detection.
- Certificate-based signing with PKCS#12/PFX and PEM support.
- Advanced signing options: timestamp_url, embed_validation_info, allow_fetching, docmdp_permissions.
- Consolidated API (v0.6.0): `extract_text`, `split_pdf`, `export_pdf`, `get_pdf_metadata(full=True)`.

Quality bar
- Feature-ready with unit + E2E coverage before pushing.
- Test every MCP tool end-to-end with dummy PDFs (multiple fixtures if needed).
- Always verify outputs by re-opening PDFs and re-reading metadata/fields.
- 388 passed, 5 skipped, 0 failures (depends on optional backends and OCR availability).

Workflow expectations
- Use safe git workflow (branching + PRs); never change main directly.
- Keep README + CHANGELOG current per version.
- Clean up stale branches before releases.

Memory / rules / global KB system
- Canonical onboarding: PROJECT_MEMO/GLOBAL_CURSOR_INSTRUCTIONS.md
- Pepper sync source: PROJECT_MEMO/PEPPER_SYNC/memories-and-kb/
- When asked "add to memory/rules", route automatically:
  - repo docs (versioned),
  - Cursor memory graph,
  - Pepper memory bank,
  - global KB archive.

MCP tool overload SOP
- Maintain MCP index + selection SOP (auto-refresh).
- Keep routing lightweight and automatable.

Deliverables checklist
- All features implemented and verified.
- Unit tests + E2E tests passing.
- Dummy PDF suite that exercises every tool.
- Updated README + CHANGELOG (SemVer).
- Tag + Release notes only when ready.

Suggested tests (current main):
- make test
- make test-ocr (if tesseract installed)
- make test-quick (fast check)
- make prepush (lint + format + test + smoke)
- make check-llm
- make install-llm-models
- make test-llm
- make test-e2e  # requires local server at localhost:8100
