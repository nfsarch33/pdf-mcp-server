# pdf-mcp-server — 2-Sprint CLI-Readiness Roadmap (v1.3.0 → v1.4.0)

- **Owner**: nfsarch33 / jaslian@gmail.com
- **Drafted**: 2026-04-30 (v257 W1 D2 — wave-3 close)
- **Companion**: `~/Code/global-kb/backlog/roadmap-v257-v261.md` (master fleet roadmap)
- **Methodology**: gstack + autoresearch + harness engineering — research first, TDD first, container first, automation first, Golang-where-justified, evidence first
- **Grading**: every sprint closes with an HD rubric scorecard (target ≥ 85/100). Sub-HD outcomes carry the failing dimension into the next sprint as P0
- **Hard contract**: pdf-mcp-server is a **dual-surface package**. The MCP server (`python -m pdf_mcp.server`) and the CLI (`pdf-mcp ...`) are mounted from the same single-source registry (`pdf_mcp/registry.py`). Adding a tool is one `register_tool(...)` call; both surfaces gain it automatically. **No surface drift, ever.**

> **Boundary rule (v257 W0 / D28 ratification, MacBook-only ZD AI gateway)**:
> pdf-mcp-server **NEVER** binds to `ai-gateway.zende.sk/{bedrock,v1}`.
> Any LLM use inside pdf-mcp-server (currently zero, kept for safety) must
> route through the local `llm-cluster-router` only. See
> `~/Code/global-kb/sop/zd-ai-gateway-tier-a.md` for the canonical wiring map.

---

## State at v257 W1 D2 close (anchor for the plan)

| Surface | State |
|---|---|
| MCP server | Functional, 57 tools across 10 verbs (`form / pages / text / extract / sign / metadata / ocr / ai / batch / security`) |
| CLI | Bootstrapped — `pdf-mcp --version` / `--help` / `serve` (TICKET-01 in PR #44 merged 2026-04-30) |
| Logging | Structured JSON / text formatter + `--verbose / --quiet / --log-format` (TICKET-02 in PR #45 merged 2026-04-30) |
| Golden snapshots | `tests/golden_helpers.py` + 3 baselines (TICKET-03 in PR #45) |
| Registry | `pdf_mcp/registry.py` + 9 contract tests (TICKET-05 c1 — PR #46 OPEN, awaiting CI) |
| Test count | 459 → 446 on this venv after registry merge (CI baseline = 459 + 9 = 468 expected); 0 regressions |
| Coverage gate | **NOT YET ENFORCED** — TICKET-08 follow-up is the gating work for HD architecture/coverage dims |
| Release-gate | `scripts/check_release_ready.py` regex pinned by 4 tests (TICKET-08 c1) |
| ZD AI gateway exposure | **None** — fleet boundary holds |

This document plans the next two sprints (v1.3.0 close → v1.4.0 cut).

---

## Cross-cutting invariants (apply to every ticket below)

1. **Test-first** — RED test committed before the implementation commit.
2. **Single-source registry** — every new tool added via `register_tool(...)`. Server and CLI consume the registry; no parallel surface lists.
3. **Lazy import contract** — `pdf-mcp --help` must NOT import `pdf_mcp.pdf_tools` (PyMuPDF / Pillow / cryptography stay cold until a verb runs). Enforced by `tests/test_registry.py::test_registry_does_not_eagerly_import_pdf_tools`.
4. **Surface parity** — automated parity test (TICKET-V13) compares `server.list_tools()` names+order to `registry.iter_all()` names+order. PR cannot merge if drift.
5. **Container parity** — every new CLI-mode demo runnable from `docker compose up` (existing `Dockerfile`).
6. **No ZD AI Gateway** — `pdf-mcp-server` LLM config (when introduced) accepts `llm-cluster-router` only. RED test rejects any `ai-gateway.zende.sk` URL.
7. **Coverage floor** — `pytest --cov=pdf_mcp` gate at the package level (≥ 75% in v1.3.0, ≥ 80% in v1.4.0).
8. **Identity gate** — every push validated by `cursor-tools doctor identity --strict` (the wave-1 gate). Tokens must not leak across personal/work boundary.
9. **Outcome capsules** — every PR open + every merge emits an `agent_outcome` capsule via `cursor-tools outcome emit` (NDJSON authoritative; Mem0 best-effort while quota throttled until 2026-05-11).
10. **Evidence dirs** — every ticket closes with proof under `~/Code/global-kb/session-handoffs/evidence/v257-w?-pdf-mcp-<ticket>/`.

---

## Sprint v1.3.0 — CLI foundation + registry-driven dual surface

- **Window**: 2026-04-30 → 2026-05-13 (overlaps fleet v257 W1)
- **Theme**: complete the registry → server + CLI dual mount, ship coverage gate, sync docs, cut v1.3.0
- **HD bar**: ≥ 85 / 100 on the per-sprint rubric below
- **End-to-end deliverable**: `uv tool install pdf-mcp-server==1.3.0` works from PyPI; `pdf-mcp --help` lists all 10 verb groups; `pdf-mcp form fill --in input.pdf --out out.pdf --field-data fields.json` runs end-to-end; the same tool is callable from `python -m pdf_mcp.server`. Lazy-import contract still proven. CI coverage gate green.

### Sprint v1.3.0 backlog (12 tickets, atomic, TDD-first)

| Ticket | Title | Pri | Status | PR | Coverage rubric dim |
|---|---|---|---|---|---|
| T-08 c1 | Release-gate regex fix + 4 contract tests | P0 | DONE | #44 (merged) | Release-gate |
| T-01 | Typer CLI bootstrap (`--version`, `--help`, `serve`) | P0 | DONE | #44 (merged) | CLI-readiness |
| T-11 c1 | CHANGELOG + README CLI groundwork sync | P0 | DONE | #44 (merged) | Docs |
| T-02 | Structured logging (text + JSON, `--verbose / --quiet / --log-format`) | P0 | DONE | #45 (merged) | Observability |
| T-03 | Golden CLI snapshot harness + 3 baselines | P0 | DONE | #45 (merged) | Release-gate |
| T-05 c1 | Single-source registry + LazyCallable + 9 tests | P0 | OPEN | #46 (CI running) | Architecture |
| T-05 c2 | `pdf_mcp/server.py` registry-driven loop (replace 57 hand-written `@mcp.tool()`) | P0 | NEXT | #47 | Architecture |
| T-05 c3 | Mount CLI verb groups from registry (`pdf-mcp form fill`, etc.) | P0 | NEXT | #48 | CLI-readiness |
| T-05 c4 | Registry contributor docs + `--help` regen + hardening | P0 | NEXT | #49 | Docs |
| T-13 | Server↔registry parity test (canonical drift detector) | P0 | NEXT | #50 | Tests |
| T-08 c2 | `pytest --cov` CI gate at 75% package threshold | P0 | NEXT | #51 | Coverage |
| T-09 | Package for PyPI + `uv tool install pdf-mcp-server` smoke | P0 | NEXT | #52 | Distribution |

#### Sprint v1.3.0 day skeleton

- **D2 (2026-04-30)**: T-05 c1 PR open ✅ (this session, wave 3).
- **D3-D4**: T-05 c2 + T-13 (parity test). PR #47 + PR #50.
- **D5-D6**: T-05 c3 (CLI verb groups). PR #48. Refresh golden snapshots.
- **D7**: T-05 c4 (docs + hardening). PR #49.
- **D8**: T-08 c2 (coverage gate). PR #51.
- **D9**: T-09 (PyPI + uv smoke). PR #52. Tag `v1.3.0`.
- **D10**: Sprint retro + HD scorecard committed to `session-handoffs/evidence/v257-w2-pdf-mcp-v130-close/SCORECARD.md`.

#### Sprint v1.3.0 rubric (target ≥ 85/100)

| Dim | Weight | DoD evidence |
|---|---:|---|
| TDD | 10 | Each PR shows RED before GREEN; the 9 registry tests + parity test + coverage gate land |
| Tests | 10 | ≥ 470 passing; 0 regressions vs prior baseline |
| Coverage | 10 | `pytest --cov=pdf_mcp --cov-fail-under=75` green in CI |
| Architecture | 10 | All 57 tools sourced from `pdf_mcp/registry.py`; `server.py` decorator wrappers gone; lazy contract still verified |
| Docs | 10 | README + CHANGELOG + `docs/USAGE.md` + `docs/CLI.md` regenerated; contributor doc shows `register_tool(...)` flow |
| CLI-readiness | 15 | `pdf-mcp --help` lists 10 verb groups; one E2E call per group passes (golden snapshots + smoke) |
| Release-gate | 10 | `scripts/check_release_ready.py` green; release notes via `gh release create v1.3.0` |
| Observability | 5 | structured logs default-on; `--log-format json` shipped |
| Identity discipline | 10 | every push past `cursor-tools doctor identity --strict`; no token leakage |
| Memory discipline | 10 | every PR + merge emits `agent_outcome` capsule (NDJSON authoritative); evidence dir per ticket |

---

## Sprint v1.4.0 — CLI maturity, packaging breadth, observability uplift

- **Window**: 2026-05-14 → 2026-05-27 (overlaps fleet v257 W2)
- **Theme**: turn `pdf-mcp` into a first-class operator CLI (verb-group breadth, output formatting, exit-code taxonomy, CI hooks, multi-distribution)
- **HD bar**: ≥ 88 / 100 on the per-sprint rubric below
- **End-to-end deliverable**: `pdf-mcp` runs in CI as a release-gate (`pdf-mcp scan --in dist/foo.pdf --format json` returns clean JSON for downstream tooling); `pex` single-file build smokes green on macOS + Linux; observability lands `pdf-mcp metrics push` for Prometheus textfile collector; `pdf-mcp doctor` self-diagnoses missing native deps; v1.4.0 tagged.

### Sprint v1.4.0 backlog (10 tickets, atomic)

| Ticket | Title | Pri | TDD must-fail-first |
|---|---|---|---|
| T-14 | `--format text\|json\|md` flag at root level + per-verb golden snapshots | P0 | `tests/test_cli_format.py::test_format_json_outputs_valid_json` |
| T-15 | Exit-code taxonomy (`0/1/2/3/4`) + error class → exit-code mapping | P0 | `tests/test_cli_exit_codes.py::test_invalid_args_exit_2` |
| T-16 | `pdf-mcp doctor` (native-deps health: PyMuPDF, OCR, ImageMagick) | P1 | `tests/test_cli_doctor.py::test_doctor_reports_missing_tesseract` |
| T-17 | `pdf-mcp metrics push --pushgateway http://...` for Prometheus textfile collector | P1 | `tests/test_cli_metrics.py::test_push_emits_textfile_format` |
| T-18 | `pex` single-file build + smoke matrix (macOS + Ubuntu CI) | P1 | `tests/packaging/test_pex_smoke.py::test_pex_runs_help` |
| T-19 | Homebrew tap formula + smoke (`brew install nfsarch33/tap/pdf-mcp`) | P2 | `tests/packaging/test_brew_formula.py::test_brew_audit` |
| T-20 | `pdf-mcp` becomes a release-gate citizen for downstream client repos | P0 | `tests/test_cli_release_gate.py::test_scan_returns_zero_on_clean_pdf` |
| T-21 | Coverage floor lift: 75 → 80 (T-08 c2 dial-up) | P0 | `tests/quality/test_coverage_floor.py::test_cov_floor_at_80` |
| T-22 | CLI tutorial in `docs/USAGE.md` + screencast (gif via `vhs`) | P1 | `tests/docs/test_examples_match_snapshots.py::test_usage_examples_run` |
| T-23 | v1.4.0 release retro + HD scorecard | P0 | n/a — gating doc |

#### Sprint v1.4.0 day skeleton

- **D1-D2**: T-14 + T-15 (format flag + exit-code taxonomy). PRs.
- **D3-D4**: T-16 + T-17 (doctor + metrics-push).
- **D5-D6**: T-18 (pex) + T-19 (brew tap).
- **D7**: T-20 (release-gate citizen for downstream).
- **D8**: T-21 (coverage 80%).
- **D9**: T-22 (tutorial + vhs screencast).
- **D10**: T-23 retro + tag `v1.4.0` + HD scorecard.

#### Sprint v1.4.0 rubric (target ≥ 88/100)

| Dim | Weight | DoD evidence |
|---|---:|---|
| TDD | 10 | RED→GREEN on every ticket; format / exit-code / doctor / metrics tests |
| Tests | 10 | ≥ 510 passing; 0 regressions |
| Coverage | 10 | `--cov-fail-under=80` green in CI (lifted from 75) |
| Architecture | 10 | Verb-group breadth without bloating `pdf_tools.py`; LLM-free; lazy contract intact |
| Docs | 10 | `docs/USAGE.md` rewritten; `vhs` screencast linked from README; CHANGELOG complete |
| CLI-readiness | 15 | `pdf-mcp` ergonomic at the CLI prompt; structured exits; usable in CI shell scripts |
| Release-gate | 10 | release-gate citizen E2E green; `pdf-mcp scan` adopted by at least one downstream sample |
| Observability | 10 | metrics-push working against the local Pushgateway; structured logs include request-id |
| Identity discipline | 5 | token-leak gate clean across all 10 PRs |
| Memory discipline | 5 | outcome capsules per PR + merge; rubric scorecard committed |
| Distribution | 5 | PyPI + pex + brew tap all green |

---

## Subagent offload pattern (proven across this sprint window)

| Lane | CLI | Model | When to use | Evidence template |
|---|---|---|---|---|
| Cursor in-session | Composer (Sonnet 4.6) | sonnet-4.6 | TDD-tight loops, multi-file edits, PR open + descriptive body | session transcript |
| Claude Code | `claude --print` / `claude --bare` | Opus 4.7 / Sonnet 4.6 / Haiku 3.5 | Single-file artifact gen with strict acceptance test (e.g., regex fix) | `evidence/v257-w?-<ticket>/claude-bare-output.txt` |
| Codex CLI | `codex exec --skip-git-repo-check --sandbox read-only` | gpt-5.2 (ZD AI gateway, MacBook-only) | Doc-style design briefs, large-context audits, throw-away analysis | `evidence/v257-w?-<ticket>/codex-design-doc.md` |
| Cursor explore subagent (read-only) | (in-session) | sonnet-4.6 | Multi-target codebase exploration before edits | child agent output (cite parent uuid only) |

**Hard rule**: Codex CLI background mode (`codex exec --full-auto`) is **gated** for sandboxed venv work — outbound `pypi.org` fetches fail. Use `codex exec` for design briefs / artifacts that don't require a venv install.

---

## EvoLoop-DRL feedback contract

Every PR open + merge in this roadmap MUST emit an `agent_outcome` capsule. Examples already on the local NDJSON tape today:

```text
pdf-mcp-server:cli-pr44-merged                 (sprint=v257, kpi-delta=+0.04)
pdf-mcp-server:logging-and-golden-pr45-open    (sprint=v257, kpi-delta=+0.05)
pdf-mcp-server:logging-and-golden-pr45-merged  (sprint=v257, kpi-delta=+0.04)
pdf-mcp-server:ticket-05-commit-1:registry-shipped (sprint=v257, kpi-delta=+0.04)
fleet:v257-w1-d2:wave-3-close-out              (sprint=v257, kpi-delta=+0.06)
```

These NDJSON capsules are authoritative until the Mem0 PAYG window opens
2026-05-11. The EvoLoop daemon (macOS replica + WSL1 hub) ingests them
and feeds the cross-node KPI loop. Daily KPI delta target for this
package is **+0.20 across the v1.3.0 sprint** (cumulative).

---

## Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Mem0 quota stays throttled past 2026-05-11 | Med | Med | NDJSON authority remains canonical; outbox flush is best-effort |
| `server.py` registry-driven loop drifts from MCP SDK ergonomics (TICKET-05 c2) | Med | High | parity test (T-13) + golden tool-list snapshot pin order before refactor |
| Coverage gate (T-08 c2) reveals untested branches | Med | Med | budget 0.5 day to write coverage-gap tests in the same PR |
| pex single-file build (T-18) trips on PyMuPDF native bindings | Med | Med | fall back to `--platform manylinux_2_17_x86_64` matrix; skip mac if blocking |
| Codex CLI quota / latency interruption | Med | Low | always have Claude Code + in-session as the redundant lane |

---

## Acceptance + exit criteria

A sprint is **accepted** only if every row below is green by the close session:

- [ ] All sprint tickets closed (PR merged + evidence committed)
- [ ] HD rubric ≥ target band (≥ 85 for v1.3.0; ≥ 88 for v1.4.0)
- [ ] CI on `main` is fully green at the tag commit
- [ ] `pytest --cov=pdf_mcp --cov-fail-under=<threshold>` green
- [ ] `pdf-mcp --help` regenerated golden snapshot committed
- [ ] CHANGELOG entry for the released version committed
- [ ] `gh release create v<x.y.z>` ran with release notes
- [ ] Outcome capsules emitted for every PR + merge + tag
- [ ] HD scorecard file committed under `session-handoffs/evidence/v257-w?-pdf-mcp-v<x.y.z>-close/SCORECARD.md`

If any row is amber/red, the failing dimension carries into the next sprint as P0.
