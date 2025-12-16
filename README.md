# PDF MCP Server

MCP server for PDF form filling and basic editing (merge, extract, rotate, flatten). Built with Python, `pypdf`, and `fillpdf`.

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

## License
MIT, see `LICENSE`.

## Changelog
See `CHANGELOG.md`.

