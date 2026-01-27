# Project Status Prompt (v0.3.0)

Use this prompt to sync status and plan next moves for `pdf-mcp-server`.

---

Review the status and plan next moves (main branch is now v0.3.0).

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

Expanded capabilities (current main)
- OCR Phase 1/2: detect_pdf_type, extract_text_native/ocr/smart, confidence scoring.
- Table extraction, image extraction, form auto-detection.
- Link extraction, optimization, barcode detection (optional pyzbar).
- Splitting by bookmarks/pages, PDF comparison, batch processing.

Quality bar
- Feature-ready with unit + E2E coverage before pushing.
- Test every MCP tool end-to-end with dummy PDFs (multiple fixtures if needed).
- Always verify outputs by re-opening PDFs and re-reading metadata/fields.

Workflow expectations
- Use safe git workflow (branching + PRs); never change main directly.
- Keep README + CHANGELOG current per version.

Memory / rules / global KB system
- Canonical onboarding: PROJECT_MEMO/GLOBAL_CURSOR_INSTRUCTIONS.md
- Pepper sync source: PROJECT_MEMO/PEPPER_SYNC/memories-and-kb/
- When asked “add to memory/rules”, route automatically:
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

