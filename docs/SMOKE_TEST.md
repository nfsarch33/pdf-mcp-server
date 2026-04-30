# Smoke Test

This smoke test complements `pytest` by exercising the same PDF flows a
user or MCP host would invoke: form filling, clearing, visual signing,
encryption, and decryption.

## When to Run

- After pulling or merging changes.
- Before cutting a release.
- After changing form, signature, encryption, or MCP registration code.

## Automated Smoke

Self-contained run that generates its own sample inputs:

```bash
cd /path/to/pdf-mcp-server
./.venv/bin/python scripts/cursor_smoke.py
```

With custom fixtures (`blank.pdf`, `form.pdf`, and `sig.png`):

```bash
cd /path/to/pdf-mcp-server
OUTDIR=/tmp/pdf-mcp-smoke
./.venv/bin/python scripts/cursor_smoke.py \
  --inputs-dir /path/to/inputs \
  --out-dir "$OUTDIR"
```

## What It Verifies

- Fill, update, and clear PDF form fields.
- Add a signature image stamp.
- Encrypt a signed PDF.
- Decrypt the encrypted PDF and confirm it is readable.

## Manual Spot Check

Use absolute paths and write outputs to a temp folder. Recommended order:

1. `get_pdf_form_fields(form.pdf)`
2. `fill_pdf_form(form.pdf -> filled.pdf, data={Name: "Test"}, flatten=false)`
3. `clear_pdf_form_fields(filled.pdf -> cleared.pdf, fields=["Name"])`
4. `add_signature_image(cleared.pdf -> signed.pdf, page=1, image_path=sig.png, rect=[...])`
5. `encrypt_pdf(signed.pdf -> signed-encrypted.pdf, user_password="pw")`

Then verify the encrypted PDF opens in a viewer that supports password
prompts.
