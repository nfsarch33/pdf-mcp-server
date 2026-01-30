# Troubleshooting (repo-local)

## Cursor MCP server didn’t pick up changes

Symptom: you updated code/tests but Cursor still throws old errors.

Fix:
- restart Cursor, or
- toggle the `pdf-handler` MCP server off/on

## Form filling issues

Recommended verification:
- write an output PDF
- re-open with `pypdf.PdfReader` and inspect `get_fields()` and `/V`

Hard requirement tools:
- `fill_pdf_form`
- `clear_pdf_form_fields`

## OCR language packs

If OCR output is noisy for passports or non-English documents:
- Run `make check-tesseract` to confirm Tesseract availability.
- Use `get_ocr_languages()` to list installed languages.
- Install additional language packs (Linux):
  - `tesseract-ocr-chi-tra`, `tesseract-ocr-jpn`, `tesseract-ocr-kor`

## Signature + encryption workflow

Use “sign then encrypt”:

1) `add_signature_image` (produces a new PDF)  
2) `encrypt_pdf` on the signed output

Verify:
- `PdfReader(...).is_encrypted is True`
- `decrypt(password)` returns 1 or 2

## Where to find deeper SOPs (optional)

If you have the Pepper memory bank repo on this machine, canonical docs include:
- `pdf-handler-troubleshooting.md`
- `mcp-index-and-selection-sop.md`
- `release-sop-tag-based.md`


