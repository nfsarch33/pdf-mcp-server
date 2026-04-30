# pdf-mcp CLI Usage Reference

Auto-generated from `pdf_mcp.registry` by `scripts/generate_usage_doc.py`. Do not edit by hand; re-run the generator after adding or removing a tool.

## Invocation

```
pdf-mcp <verb> <tool-name> [--json '{...}'] [--json-file PATH]
                            [--pretty] [--output PATH]
```

* `--json` and `--json-file` are mutually exclusive (`--json` wins if both are passed).
* `--pretty` indents the JSON output for human reading.
* `--output PATH` writes the JSON result to a file instead of stdout.
* Tool exceptions exit non-zero with `error: <tool> failed: <msg>` on stderr.

Run `pdf-mcp --help` for the top-level surface and `pdf-mcp <verb> --help` for the per-verb tool list.

## Verb groups

| Verb | Tools | Description |
| ---- | ----- | ----------- |
| `form` | 9 | PDF form discovery, filling, templates, and flattening. |
| `security` | 2 | Encryption and PII detection. |
| `pages` | 8 | Page-level mutation: merge, split, extract, rotate, reorder, insert, remove. |
| `text` | 13 | Text annotations, redaction, watermarks, comments, page numbers, Bates stamps. |
| `metadata` | 4 | Metadata read/write/sanitisation and PDF type/feature detection. |
| `sign` | 6 | Digital and visual signatures. |
| `extract` | 7 | Read-only extraction: text blocks, tables, images, links, structured data. |
| `ocr` | 2 | OCR helpers and image inspection. |
| `batch` | 2 | Multi-file processing and comparisons. |
| `ai` | 4 | LLM-backed extraction, auto-fill, and analysis. |

## `pdf-mcp form`

PDF form discovery, filling, templates, and flattening.

| Tool | Description |
| ---- | ----------- |
| `get-pdf-form-fields` | Return available form fields in the PDF. |
| `fill-pdf-form` | Fill a PDF form with provided data. Optionally flatten to make non-editable. |
| `fill-pdf-form-any` | Fill standard or non-standard forms using label detection when needed. |
| `create-pdf-form` | Create a new PDF with AcroForm fields. |
| `get-form-templates` | List built-in form templates for common workflows. |
| `create-pdf-form-from-template` | Create a PDF form using a built-in template. |
| `flatten-pdf` | Flatten a PDF (remove form fields/annotations). |
| `clear-pdf-form-fields` | Clear (delete) values for PDF form fields while keeping fields fillable. |
| `detect-form-fields` | Detect potential form fields in a PDF using text analysis. |

## `pdf-mcp security`

Encryption and PII detection.

| Tool | Description |
| ---- | ----------- |
| `encrypt-pdf` | Encrypt (password-protect) a PDF using pypdf. |
| `detect-pii-patterns` | Detect common PII patterns (email, phone, SSN, credit card) in a PDF. |

## `pdf-mcp pages`

Page-level mutation: merge, split, extract, rotate, reorder, insert, remove.

| Tool | Description |
| ---- | ----------- |
| `merge-pdfs` | Merge multiple PDFs into a single file. |
| `extract-pages` | Extract specific 1-based pages into a new PDF. |
| `rotate-pages` | Rotate specified 1-based pages by degrees (must be multiple of 90). |
| `reorder-pages` | Reorder pages in a PDF using a 1-based page list. |
| `insert-pages` | Insert pages from another PDF before at_page (1-based). |
| `remove-pages` | Remove specified 1-based pages from a PDF. |
| `optimize-pdf` | Optimize/compress a PDF to reduce file size. |
| `split-pdf` | Split a PDF into multiple files. |

## `pdf-mcp text`

Text annotations, redaction, watermarks, comments, page numbers, Bates stamps.

| Tool | Description |
| ---- | ----------- |
| `add-text-annotation` | Add a FreeText annotation to a page (managed text insertion). |
| `update-text-annotation` | Update an existing annotation by annotation_id. |
| `remove-text-annotation` | Remove an existing annotation by annotation_id. |
| `remove-annotations` | Remove annotations from given pages. Optionally filter by subtype (e.g., FreeText). |
| `redact-text-regex` | Redact text using a regex pattern. |
| `add-page-numbers` | Add page numbers as FreeText annotations. |
| `add-bates-numbering` | Add Bates numbering as FreeText annotations. |
| `add-text-watermark` | Add a simple text watermark or stamp via FreeText annotations. |
| `add-highlight` | Add highlight annotations by text search or rectangle. |
| `add-date-stamp` | Add a date stamp as a FreeText annotation. |
| `add-comment` | Add a PDF comment (sticky note) using PyMuPDF. |
| `update-comment` | Update a PDF comment by id using PyMuPDF. |
| `remove-comment` | Remove a PDF comment by id using PyMuPDF. |

## `pdf-mcp metadata`

Metadata read/write/sanitisation and PDF type/feature detection.

| Tool | Description |
| ---- | ----------- |
| `get-pdf-metadata` | Get PDF document metadata. |
| `set-pdf-metadata` | Set basic PDF document metadata (title, author, subject, keywords). |
| `sanitize-pdf-metadata` | Remove metadata keys from a PDF. |
| `detect-pdf-type` | Analyze a PDF to classify its content type. |

## `pdf-mcp sign`

Digital and visual signatures.

| Tool | Description |
| ---- | ----------- |
| `verify-digital-signatures` | Verify digital signatures in a PDF. |
| `sign-pdf` | Digitally sign a PDF using a PKCS#12/PFX certificate. |
| `sign-pdf-pem` | Digitally sign a PDF using PEM key + cert chain. |
| `add-signature-image` | Add a signature image by inserting it on a page (PyMuPDF). |
| `update-signature-image` | Update or resize a signature image (PyMuPDF). |
| `remove-signature-image` | Remove a signature image by xref (PyMuPDF). |

## `pdf-mcp extract`

Read-only extraction: text blocks, tables, images, links, structured data.

| Tool | Description |
| ---- | ----------- |
| `get-pdf-text-blocks` | Extract text blocks with position information from PDF. |
| `extract-tables` | Extract tables from PDF pages. |
| `extract-images` | Extract embedded images from PDF pages. |
| `extract-links` | Extract links (URLs, hyperlinks, internal references) from a PDF. |
| `detect-barcodes` | Detect and decode barcodes/QR codes in a PDF. |
| `extract-text` | Unified text extraction with multiple engine options and optional confidence scores. |
| `export-pdf` | Export PDF content to different formats. |

## `pdf-mcp ocr`

OCR helpers and image inspection.

| Tool | Description |
| ---- | ----------- |
| `get-ocr-languages` | Get available OCR languages and Tesseract installation status. |
| `get-image-info` | Get information about images in a PDF without extracting them. |

## `pdf-mcp batch`

Multi-file processing and comparisons.

| Tool | Description |
| ---- | ----------- |
| `compare-pdfs` | Compare two PDFs and identify differences. |
| `batch-process` | Process multiple PDFs with a single operation. |

## `pdf-mcp ai`

LLM-backed extraction, auto-fill, and analysis.

| Tool | Description |
| ---- | ----------- |
| `get-llm-backend-info` | Get information about available LLM backends. |
| `auto-fill-pdf-form` | Intelligently fill PDF form fields using LLM-powered field mapping. |
| `extract-structured-data` | Extract structured data from PDF using pattern matching or LLM. |
| `analyze-pdf-content` | Analyze PDF content for document type, key entities, and summary. |

## Backwards compatibility

* `pdf-mcp serve` runs the MCP server over stdio (drop-in replacement for `python -m pdf_mcp.server`).
* All tools remain reachable via the MCP protocol with their original `snake_case` names.
