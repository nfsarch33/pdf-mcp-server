# Security Policy

We take the security of `pdf-mcp` seriously. PDF parsers and OCR
pipelines have a long history of CVE-class issues, so reports about
crashes, denial-of-service vectors, sandbox escapes, or dependency
chain weaknesses are very welcome.

## Supported versions

| Version  | Supported |
| -------- | --------- |
| `1.3.x`  | ✅ Latest. Security fixes ship as point releases. |
| `1.2.x`  | ✅ Until `1.4.0`. Critical fixes only. |
| `< 1.2`  | ❌ Please upgrade. |

The release cadence is documented in
[`docs/release-sop.md`](docs/release-sop.md) and the version timeline
is in [`CHANGELOG.md`](CHANGELOG.md).

## Reporting a vulnerability

**Please do not open public GitHub issues for security reports.** Use
one of the private channels below.

1. **GitHub private vulnerability reporting** (preferred): visit
   <https://github.com/nfsarch33/pdf-mcp-server/security/advisories/new>
   and file a private security advisory. This routes directly to the
   maintainer and lets us coordinate a fix + GHSA before public
   disclosure.
2. **Email**: if private vulnerability reporting is unavailable, use the
   maintainer contact listed on the GitHub repository profile. Include a
   description, reproduction steps, and any relevant logs or sample PDFs.
   We aim to acknowledge within **3 business days** and provide a triage
   update within **7 days**.

When reporting, please include:

- The `pdf-mcp` version (`pdf-mcp --version` or the `__version__` in
  `pdf_mcp/__init__.py`).
- The Python version, OS, and any relevant native deps (Tesseract,
  Poppler, libgs, etc.).
- A minimal PDF sample if the issue is parser- or input-driven (we'll
  treat samples as confidential and delete them once the advisory is
  resolved).
- Whether you'd like to be credited in the advisory.

## Out of scope

The following do not constitute security vulnerabilities for this
project:

- Resource exhaustion when running on user-supplied input without
  documented limits (`extract_text`, `ocr_pdf`, etc.). Please raise
  hardening requests as normal feature issues.
- Behavior of optional LLM backends (OpenAI, vLLM, Ollama). Report
  those upstream.
- Misconfiguration of the embedding application (Cursor, Claude
  Desktop, your own MCP host) when invoking `pdf-mcp`.

## LLM privacy boundary

`pdf-mcp` is local-first. LLM-backed tools prefer a local OpenAI-compatible
server or Ollama and do not auto-select hosted LLM backends just because an
API key exists.

Remote LLM use requires `PDF_MCP_ENABLE_REMOTE_LLM=1`. Sensitive flows
such as passport extraction and form-field mapping require the additional
`PDF_MCP_ALLOW_REMOTE_LLM_FOR_SENSITIVE=1` opt-in. Reports that this
boundary is bypassed are security-relevant.

## Disclosure timeline

- **Day 0** — Report received via private channel.
- **Day 0–7** — Triage; we confirm reproduction and assign a severity.
- **Day 7–30** — Fix developed, reviewed, and released as a patch
  version. CVE / GHSA filed.
- **Day 30+** — Public disclosure via the GitHub advisory and the
  release CHANGELOG entry.

We are happy to coordinate longer embargoes for downstream packagers
(distros, container vendors, hosted MCP gateways).

Thanks for helping keep `pdf-mcp` and its users safe.
