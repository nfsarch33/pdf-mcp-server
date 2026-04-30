# pdf-mcp Roadmap — v1.3.0 to v1.7.0

This roadmap keeps `pdf-mcp` focused as a general-purpose open-source
PDF CLI and MCP server. It is intentionally independent of any private
agent stack, internal knowledge base, or organisation-specific provider.

## Principles

- **CLI and MCP parity**: every public tool is declared once in
  `pdf_mcp.registry` and appears in both `pdf-mcp <verb> <tool>` and
  the MCP server.
- **Local-first privacy**: PDF content stays on the user's machine by
  default. Hosted LLMs require explicit opt-in.
- **Evidence-first delivery**: every release needs tests, coverage,
  docs, release notes, and a short quality scorecard.
- **Stable automation surface**: commands must be scriptable, return
  machine-readable JSON when requested, and use predictable exit codes.
- **Open-source portability**: no private paths, private runbooks,
  personal memory artifacts, or IDE-specific operating rules in the
  release branch.

## Current State

| Surface | State |
| --- | --- |
| MCP server | Registry-driven server loop; tools are mounted from `pdf_mcp.registry`. |
| CLI | 10 verb groups and 57 tools exposed as `pdf-mcp <verb> <tool>`. |
| Docs | `USAGE.md` generated from the registry; README is CLI-first. |
| Tests | 526 collected tests, including registry parity and lazy-import checks. |
| Coverage | CI gate at `--cov-fail-under=75`; baseline is above the gate. |
| Privacy | Remote LLM auto-selection is disabled unless explicitly opted in. |

## Five-Release Plan

| Release | Theme | Target Quality Bar | End-to-End Outcome |
| --- | --- | ---: | --- |
| **v1.3.0** | CLI-ready release | 95/100 | `uv tool install pdf-mcp==1.3.0` works, all 57 tools are visible from CLI help, MCP parity tests pass, release notes published. |
| **v1.4.0** | CLI ergonomics | 88/100 | Root `--format text/json/md`, predictable exit codes, `pdf-mcp doctor`, and packaging smoke tests. |
| **v1.5.0** | Local AI extraction | 88/100 | Schema-bound extraction through local/Ollama-compatible backends, with hosted backends fail-closed unless opted in. |
| **v1.6.0** | Supply-chain hardening | 90/100 | SBOM, signed artifacts, reproducible-build checks, and release provenance. |
| **v1.7.0** | Extension and distribution | 92/100 | Plugin discovery, Homebrew formula, GHCR image, and documented plugin contract. |

## v1.3.0 — CLI-Ready Release

### Goals

- Cut the first CLI-first release tag.
- Publish the package so users can install it as a tool.
- Keep the MCP server as a first-class surface.
- Preserve lazy imports so help/version paths stay fast.

### Backlog

| Ticket | Priority | Acceptance |
| --- | --- | --- |
| T-09 PyPI publish workflow | P0 | Tag push runs tests, validates version and changelog, builds distributions, and publishes only from release tags. |
| T-10 Tool-install smoke | P0 | A clean environment can run `uv tool install pdf-mcp==1.3.0` and `pdf-mcp --help`. |
| T-11 Release notes | P0 | Changelog and GitHub Release describe CLI, MCP parity, coverage, and privacy defaults. |

### Quality Rubric

| Dimension | Weight | Evidence |
| --- | ---: | --- |
| CLI readiness | 20 | All verb groups visible and at least one command per group smoke-tested. |
| MCP parity | 15 | Registry-to-MCP parity tests pass. |
| Tests and coverage | 20 | Full suite green; coverage gate above 75%. |
| Docs | 15 | README, USAGE, CONTRIBUTING, SECURITY, and release notes are current. |
| Privacy and safety | 15 | Hosted LLM use is opt-in and sensitive flows fail closed. |
| Release hygiene | 15 | Tag, package build, and install smoke all green. |

## v1.4.0 — CLI Ergonomics

### Goals

- Make the CLI comfortable for humans and reliable for scripts.
- Improve errors, output formats, diagnostics, and packaging breadth.

### Backlog

| Ticket | Priority | Acceptance |
| --- | --- | --- |
| T-14 Output formats | P0 | `--format text/json/md` works at root and is covered by golden tests. |
| T-15 Exit code taxonomy | P0 | Invalid usage, tool failure, missing dependency, and unsafe remote-LLM request have distinct exit codes. |
| T-16 `pdf-mcp doctor` | P1 | Reports Python, PyMuPDF, Tesseract, Poppler, optional LLM, and write-permission health. |
| T-17 Metrics output | P1 | Optional Prometheus textfile output for command duration and failure counts. |
| T-18 Single-file packaging smoke | P1 | A packaged artifact can run `pdf-mcp --help` on macOS and Linux. |
| T-21 Coverage floor lift | P0 | Raise the gate from 75 to 80 once the release branch is stable. |

## v1.5.0 — Local AI Extraction

### Goals

- Add safe schema-bound AI extraction for users who run local models.
- Keep hosted model traffic explicit and visible.

### Backlog

| Ticket | Priority | Acceptance |
| --- | --- | --- |
| T-30 LLM router shim | P0 | Central helper owns all hosted/local model routing and rejects unknown provider URLs. |
| T-31 `pdf-mcp ai extract` | P0 | Extracts a schema-bound JSON object from text PDFs with deterministic tests. |
| T-32 Sensitive-flow policy | P0 | Passport, identity, financial, and form-mapping flows block remote LLMs unless explicitly opted in. |
| T-33 Streaming progress | P1 | Long-running extraction emits progress that scripts can consume. |
| T-35 Coverage floor lift | P0 | Raise the gate from 80 to 82. |

## v1.6.0 — Supply-Chain Hardening

### Goals

- Make release artifacts auditable.
- Give downstream users a clear security and provenance story.

### Backlog

| Ticket | Priority | Acceptance |
| --- | --- | --- |
| T-40 SBOM | P0 | Release attaches a CycloneDX SBOM. |
| T-41 Artifact signing | P0 | Wheels, sdists, and packaged artifacts are signed. |
| T-42 Provenance | P1 | Release includes a provenance attestation. |
| T-43 Reproducible build check | P1 | CI compares build outputs across clean runners where feasible. |
| T-46 Coverage floor lift | P0 | Raise the gate from 82 to 84. |

## v1.7.0 — Extension and Distribution

### Goals

- Let the community extend the tool surface without forking.
- Broaden install options.

### Backlog

| Ticket | Priority | Acceptance |
| --- | --- | --- |
| T-50 Plugin discovery | P0 | Plugins can register tools via a documented Python entry point. |
| T-51 Plugin trust model | P0 | Unsigned plugins are blocked by default unless explicitly trusted. |
| T-52 Plugin CLI | P1 | `pdf-mcp plugins list` shows installed plugin metadata. |
| T-54 Homebrew formula | P0 | macOS users can install with Homebrew and run `pdf-mcp --help`. |
| T-55 Container image | P0 | GHCR image runs `pdf-mcp serve` and CLI smoke tests. |
| T-58 Coverage floor lift | P0 | Raise the gate from 84 to 85. |

## Cross-Release Invariants

- Write a failing test before changing behavior.
- Keep `pdf_mcp.registry` as the source of truth.
- Keep `pdf-mcp --help` lazy: no heavyweight PDF or LLM imports.
- Regenerate `USAGE.md` after tool-surface changes.
- Keep hosted LLMs opt-in and sensitive flows fail-closed.
- Run full tests before tagging.
- Update `CHANGELOG.md` for user-visible changes.
- Keep public docs free of private paths, private runbooks, secrets, and
  machine-specific instructions.

## Release Acceptance Checklist

- [ ] All planned tickets merged.
- [ ] Full CI green on the tag commit.
- [ ] Coverage gate green at the release threshold.
- [ ] `pdf-mcp --help` and `pdf-mcp serve --help` snapshots current.
- [ ] `USAGE.md` generated and docs-sync check green.
- [ ] Changelog has a release entry.
- [ ] Security and privacy docs match actual behavior.
- [ ] Install smoke passes from a clean environment.
- [ ] GitHub Release created with notes and artifacts.
