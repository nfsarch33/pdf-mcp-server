---
name: pdf-form-filling
description: Fills PDF forms with client data using pdf-mcp-server MCP tools, extracts passport and document data, and verifies filled output. Use when filling government or business PDF forms, extracting passport data, or performing form-filling QA.
---
# PDF Form Filling

## When to Use
- Filling government or business PDF forms with client data
- Extracting structured data from passports or identity documents
- Verifying filled forms before submission
- Mapping source document fields to target form fields

## Workflow

1. **Inspect the form**: `get_pdf_form_fields(pdf_path="<form.pdf>")` to list all fields, types, and current values
2. **Extract source data** (if from passport/document): `extract_structured_data(pdf_path="<scan.pdf>", data_type="passport")`
3. **Review confidence scores**: Fields with confidence < 0.7 must be manually verified against source documents
4. **Map data to form fields**: Create a dict mapping form field names to values
5. **Fill the form**: `fill_pdf_form(input_path="<form.pdf>", output_path="<filled.pdf>", data={...})`
6. **Verify**: `get_pdf_form_fields(pdf_path="<filled.pdf>")` to confirm all values
7. **Check diagnostics**: Review `filled_fields_count`, `total_form_fields`, `unmatched_fields` in fill result
8. **(Optional) Flatten**: `flatten_pdf(input_path="<filled.pdf>", output_path="<final.pdf>")`

## Key MCP Tools

| Tool | Purpose |
|------|---------|
| `get_pdf_form_fields` | List fields, types (/Tx=text, /Btn=checkbox), current values |
| `fill_pdf_form` | Fill AcroForm fields; checkboxes: "Yes"/"No" as NameObjects |
| `fill_pdf_form_any` | Fill non-standard forms via label detection heuristics |
| `extract_structured_data` | Extract passport/invoice/receipt data with VLM; use `consensus_runs=3` for critical documents |
| `extract_text` | Extract text (native, OCR, auto modes) |
| `analyze_pdf_content` | Document analysis, entity extraction, summarization |
| `flatten_pdf` | Make filled form non-editable for final submission |

## Critical Rules

1. **Verify after filling**: Always re-read fields to confirm values are correct
2. **Checkbox handling**: Use "Yes" for checked, "No" for unchecked
3. **Passport dates**: If `derived_fields` key is present, those dates were computed, not read directly -- verify against physical passport
4. **Never commit PII**: Do not push client names, passport numbers, or personal data to any repository
5. **Non-standard forms**: Use `fill_pdf_form_any` which uses geometric and text label heuristics
6. **XFA forms**: Not supported; convert to AcroForm first

## VLM Backend

- Model auto-detected via `get_llm_backend_info` tool
- Must be running before VLM-dependent tools are called
- Default port: localhost:8100

## Known Issues

- Some PDFs with compressed object streams may fail with fillpdf; server auto-falls back to pypdf
- MRZ-extracted names may contain OCR noise; server sanitizes and adjusts confidence
- VLM may return placeholder strings ("NULL", "None"); server filters these automatically
