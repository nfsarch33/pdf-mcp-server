# Cursor smoke test (post-push)

Purpose: a fast local check that complements `pytest` and mimics real user flows in Cursor.

## When to run
- After pulling or merging changes
- After restarting or toggling the `pdf-handler` MCP server
- Before cutting a release

## Automated smoke (recommended)

Self-contained (generates its own inputs):

```bash
cd /mnt/f/onedrive/repo/mcp-pdf-server
./.venv/bin/python scripts/cursor_smoke.py
```

With your private fixtures (expects `blank.pdf`, `form.pdf`, `sig.png`):

```bash
cd /mnt/f/onedrive/repo/mcp-pdf-server
OUTDIR=/home/jason/Code/pdf-handler-private-tests/outputs/cursor-smoke-manual
./.venv/bin/python scripts/cursor_smoke.py \\
  --inputs-dir /home/jason/Code/pdf-handler-private-tests/inputs \\
  --out-dir \"$OUTDIR\"
ls -la \"$OUTDIR\"
```

What it verifies (hard requirements):
- Fill, update, and clear (delete values) of form fields
- Sign using signature image stamp
- Encrypt (password protect) the signed PDF and decrypt to confirm it is readable

## Cursor manual spot-check (5 minutes)

Use absolute paths and write outputs to a temp folder. Recommended order:

- `get_pdf_form_fields(form.pdf)`
- `fill_pdf_form(form.pdf -> filled.pdf, data={Name: \"Test\"}, flatten=false)`
- `clear_pdf_form_fields(filled.pdf -> cleared.pdf, fields=[\"Name\"])`
- `add_signature_image(cleared.pdf -> signed.pdf, page=1, image_path=sig.png, rect=[...])`
- `encrypt_pdf(signed.pdf -> signed-encrypted.pdf, user_password=\"pw\")`

# Cursor smoke test (post-push)

Purpose: a fast local check that complements `pytest` and mimics real user flows in Cursor.

## When to run
- After pulling or merging changes
- After restarting or toggling the `pdf-handler` MCP server
- Before cutting a release

## Automated smoke (recommended)

Self-contained (generates its own inputs):

```bash
cd /path/to/pdf-mcp-server
./.venv/bin/python scripts/cursor_smoke.py
```

With your private fixtures (expects `blank.pdf`, `form.pdf`, `sig.png`):

```bash
cd /path/to/pdf-mcp-server
OUTDIR=/tmp/pdf-handler-cursor-smoke
./.venv/bin/python scripts/cursor_smoke.py --inputs-dir /path/to/inputs --out-dir "$OUTDIR"
ls -la "$OUTDIR"
```

What it verifies (hard requirements):
- Fill, update, and clear (delete values) of form fields
- Sign using signature image stamp
- Encrypt (password protect) the signed PDF and decrypt to confirm it is readable

## Cursor manual spot-check (5 minutes)

Use absolute paths and write outputs to a temp folder. Recommended order:

- `get_pdf_form_fields(form.pdf)`
- `fill_pdf_form(form.pdf -> filled.pdf, data={Name: "Test"}, flatten=false)`
- `clear_pdf_form_fields(filled.pdf -> cleared.pdf, fields=["Name"])`
- `add_signature_image(cleared.pdf -> signed.pdf, page=1, image_path=sig.png, rect=[...])`
- `encrypt_pdf(signed.pdf -> signed-encrypted.pdf, user_password="pw")`

Then verify:
- Re-open `signed-encrypted.pdf` using a PDF viewer that supports password prompts
- Confirm it prompts for a password and opens with `"pw"`


