# Migration runbook (new computer)

Goal: a developer can clone this repo, register the `pdf-handler` MCP server in Cursor, run tests/smoke checks, and continue development without digging through chat history.

## 0) Preconditions

- Python >= 3.10
- `uv` installed (recommended) OR any Python environment tool you prefer
- Cursor installed

## 1) Clone + install

```bash
git clone git@github.com:nfsarch33/pdf-mcp-server.git
cd pdf-mcp-server
make install
make test
```

## 2) Register `pdf-handler` MCP server in Cursor

In `~/.cursor/mcp.json`, add (adjust paths as needed):

```json
{
  "mcpServers": {
    "pdf-handler": {
      "command": "/ABS/PATH/TO/pdf-mcp-server/.venv/bin/python",
      "args": ["-m", "pdf_mcp.server"],
      "description": "Local PDF form filling and editing (stdio)"
    }
  }
}
```

Restart Cursor (or toggle the MCP server off/on) after editing.

## 3) Verify hard requirements (fast)

These are the non-negotiables:

- Form support: fill/update, clear/delete values: `fill_pdf_form`, `clear_pdf_form_fields`
- “Sign then protect”: `add_signature_image` then `encrypt_pdf`

Run the repo smoke script (self-contained fixture generation):

```bash
OUTDIR=/tmp/pdf-handler-cursor-smoke
./.venv/bin/python scripts/cursor_smoke.py --out-dir "$OUTDIR"
ls -la "$OUTDIR"
```

Also see `docs/CURSOR_SMOKE_TEST.md` for a Cursor manual spot-check order.

## 4) Optional: connect the multi-layer memory system

This repo is open-source and stays secret-free. The memory system’s canonical artifacts are intended to live in separate, private git repos:

- **Pepper memory bank** (procedures/SOPs/checklists): `~/memo/` (private repo)
- **Global KB** (historical/RCA/archive): `~/Code/global-kb/` (private repo)

Repo-local routing reference:
- `.cursor/rules/template.rules`
- `MEMORY_AND_RULES.md`

If those private repos exist on the new machine, follow their README/SOPs to:
- install the daily refresh automation (MCP index + repo memories sync)
- keep `pdf-handler` troubleshooting + release SOPs up to date

### WSL note (systemd user bus)

Some WSL setups have `systemd` but no `systemctl --user` bus. If the Pepper SOP installer fails with “Failed to connect to bus”, use the cron-based installer described in Pepper’s `memory-sop.md`.

## 5) What NOT to migrate

- `cursor-memory-complete-setup-guide.md` may contain personal/academic content; keep it **untracked**.
- Never copy secrets into this repo; keep them in machine-local configs.


