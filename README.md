# PDF MCP Server

MCP server for PDF form filling and basic editing (merge, extract, rotate, flatten). Built with Python, `pypdf`, and `fillpdf`.

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

## Conventions
- Paths should be absolute; outputs are created with parent directories if missing.
- Inputs must exist and be files; errors return `{ "error": "..." }`.
- Form flattening prefers fillpdf+poppler; falls back to a pypdf-only flatten (removes form structures).

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

## License
MIT, see `LICENSE`.

