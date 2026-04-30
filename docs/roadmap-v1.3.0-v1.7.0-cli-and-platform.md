# pdf-mcp-server — 5-Sprint Roadmap (v1.3.0 → v1.7.0)

- **Owner**: nfsarch33 / jaslian@gmail.com
- **Drafted**: 2026-04-30 (v257 W1 D2 — wave-4 close, sprint v1.3.0 backlog exhausted at HD top-band 100/100)
- **Companions**:
  - `~/Code/pdf-mcp-server/docs/roadmap-v1.3.0-v1.4.0-cli-readiness.md` (the canonical 2-sprint plan; this doc supersedes for the look-ahead)
  - `~/Code/global-kb/backlog/roadmap-v257-v261.md` (master fleet roadmap; sprint windows align)
  - `~/Code/global-kb/session-handoffs/evidence/v257-w1-d2-wave4-close/SCORECARD.md` (anchor scorecard)
- **Methodology**: gstack + autoresearch + harness engineering — research first, TDD first, container first, automation first, evidence first
- **Grading**: every release closes with an HD rubric scorecard. HD band ≥ 85 / 100; HD top-band ≥ 95 / 100. Sub-HD outcomes carry the failing dimension into the next sprint as P0
- **Hard contract** (carried unchanged from the 2-sprint plan):
  - Single-source registry (`pdf_mcp/registry.py`) drives BOTH `server.py` AND `cli.py`. Surface drift fails CI.
  - Lazy import contract: `pdf-mcp --help` must NOT import `pdf_mcp.pdf_tools`. Enforced by 31 subprocess `sys.modules` tests.
  - No ZD AI Gateway exposure ever. Any LLM use routes through the local `llm-cluster-router` only.
  - Identity gate: `nfsarch33` strict on every push.

---

## Anchor state (v1.3.0 ready-to-tag, v257 W1 D2 close)

| Surface | State | Evidence |
|---|---|---|
| MCP server | Registry-driven, 94 LOC (1175 → 94, 92% reduction) | `pdf_mcp/server.py` |
| CLI | 10 verb groups, 57 tools, lazy-import preserved | `pdf-mcp --help`, `tests/test_cli_verb_groups.py` |
| Registry | 57 tools / 10 verbs, 9 contract tests + 6 parity tests | `pdf_mcp/registry.py`, `tests/test_registry.py` |
| Coverage gate | `--cov-fail-under=75` (baseline 80.69%) | `pyproject.toml`, `.github/workflows/ci.yml` |
| Docs | USAGE.md generator + `--check` CI gate, README CLI-first, CONTRIBUTING.md | `scripts/generate_usage_doc.py`, `USAGE.md`, `README.md`, `CONTRIBUTING.md` |
| Test count | 487 passing (+50 from v1.2.x baseline 437); 0 regressions | CI run on `main@6e5de9e` |
| Release-gate | `scripts/check_release_ready.py` regex pinned by 4 tests | `scripts/check_release_ready.py`, `tests/test_release_gate.py` |
| Scorecard | 100/100 HD top-band realised | `~/Code/global-kb/session-handoffs/evidence/v257-w1-d2-wave4-close/SCORECARD.md` |
| ZD AI gateway exposure | None — fleet boundary holds | `sop/zd-ai-gateway-tier-a.md` |

The remaining v1.3.0 gates are administrative: bump `pyproject.toml` version, write release notes citing #44–#50 and #48, sign the tag, attach the scorecard.

---

## Five-sprint look-ahead

| Release | Sprint window | Theme | HD bar | End-to-end deliverable |
|---|---|---|---:|---|
| **v1.3.0** | 2026-04-30 → 2026-05-13 (overlaps fleet v257 W1) | **Tag v1.3.0 + PyPI publish** (sprint backlog already exhausted; this window closes the release) | 95 | `uv tool install pdf-mcp==1.3.0` works from PyPI; `pdf-mcp --help` lists 10 verbs; `pdf-mcp form fill --in input.pdf --out out.pdf --field-data fields.json` runs end-to-end via the CLI; the same tool reachable via `python -m pdf_mcp.server` (registry parity proven). |
| **v1.4.0** | 2026-05-14 → 2026-05-27 (overlaps fleet v257 W2) | **CLI maturity + packaging breadth** | 88 | `pdf-mcp scan --in dist/foo.pdf --format json` returns clean JSON; `pex` single-file build smokes green on macOS + Ubuntu; `pdf-mcp metrics push --pushgateway http://...` lands; `pdf-mcp doctor` self-diagnoses missing native deps; v1.4.0 tagged. |
| **v1.5.0** | 2026-05-28 → 2026-06-10 (overlaps fleet v258 W1) | **Local-LLM extraction + safety sidecar** (gated through `llm-cluster-router`, NEVER ZD gateway) | 88 | `pdf-mcp ai extract --in <pdf> --schema <json-schema>` returns a structured result via Qwen 3.5 (router-routed); safety sidecar fronts every LLM call; coverage floor lifts 80 → 82. |
| **v1.6.0** | 2026-06-11 → 2026-06-24 (overlaps fleet v258 W2 / v259 W1) | **Operator UX + SBOM / supply-chain hardening** | 90 | `pdf-mcp` ships SBOM (CycloneDX); reproducible builds verified across macOS + Linux CI lanes; `pdf-mcp` adopted as a release-gate citizen by at least one downstream client repo; vhs-recorded screencast linked from README. |
| **v1.7.0** | 2026-06-25 → 2026-07-08 (overlaps fleet v259 W2) | **Plugin / extension surface + multi-distribution** | 92 | Third-party tool plugins discoverable via `pdf-mcp plugins list`; Homebrew tap formula smokes green; `pdf-mcp` Docker image published to GHCR with multi-arch manifest; coverage floor reaches 85. |

---

## Sprint v1.3.0 — close-out window (2026-04-30 → 2026-05-13)

- **Backlog status**: exhausted, all twelve tickets shipped (T-08 c1, T-01, T-11 c1, T-02, T-03, T-05 c1-c4, T-13, T-08 c2). T-09 (PyPI publish) is the only outstanding gate.
- **Remaining work**: tag + publish.
- **HD bar**: 95 (top-band) — driven by the realised 100/100 wave-4 scorecard plus the PyPI smoke.

### Day skeleton (close-out)

- **D3-D5 (2026-05-01 → 2026-05-05)**: TICKET-09 (PyPI publish workflow). One PR with `release.yml` (`twine check` dry-run, `pypi-publish` action gated on `v1.3.0` tag), `uv tool install pdf-mcp` smoke in CI matrix.
- **D6**: bump `pyproject.toml` 1.2.x → 1.3.0; write release notes; squash-land via PR.
- **D7 (≈ 2026-05-12)**: cut `v1.3.0` tag with `gh release create`. Smoke `uv tool install pdf-mcp==1.3.0` from a clean venv; capture evidence under `~/Code/global-kb/session-handoffs/evidence/v257-w2-pdf-mcp-v130-close/`.
- **D8**: sprint retro + scorecard committed.

### Sprint v1.3.0 rubric (target ≥ 95/100, top-band)

| Dim | Weight | DoD evidence |
|---:|---:|---|
| TDD | 10 | RED-first on every PR (already satisfied; see PR #46 / #47 / #48 / #49 / #50). PyPI workflow ships with a `--dry-run` test harness. |
| Tests | 10 | ≥ 487 passing on tag commit; PyPI smoke adds at least 2 install-path tests. |
| Coverage | 10 | `--cov-fail-under=75` green at tag (already achieved on main). |
| Architecture | 10 | Registry single-source, no surface drift, no orphan `@mcp.tool()` decorators (all checked by parity tests). |
| Docs | 10 | USAGE.md generated and `--check` green at tag; release notes link to USAGE.md and CHANGELOG.md. |
| CLI-readiness | 15 | All 57 tools reachable via `pdf-mcp <verb> <tool>`; `uv tool install` smoke green. |
| Release-gate | 10 | `scripts/check_release_ready.py` green; `release.yml` proves dry-run + real publish on tag. |
| Observability | 5 | Structured logs default-on; release-notes link the metrics shape. |
| Identity discipline | 10 | `cursor-tools doctor identity --strict` green on every push during the close-out. |
| Memory discipline | 10 | Outcome capsules emitted at tag-cut + at PyPI publish; HD scorecard committed. |

---

## Sprint v1.4.0 — CLI maturity + packaging breadth (2026-05-14 → 2026-05-27)

The 2-sprint plan in `roadmap-v1.3.0-v1.4.0-cli-readiness.md` already canonicalises this sprint. Reproducing the backlog inline only for compactness.

### Backlog (10 tickets)

| Ticket | Title | Pri | TDD must-fail-first |
|---|---|---|---|
| T-14 | `--format text\|json\|md` flag at root level + per-verb golden snapshots | P0 | `tests/test_cli_format.py::test_format_json_outputs_valid_json` |
| T-15 | Exit-code taxonomy (`0/1/2/3/4`) + error class → exit-code mapping | P0 | `tests/test_cli_exit_codes.py::test_invalid_args_exit_2` |
| T-16 | `pdf-mcp doctor` (native-deps health: PyMuPDF, OCR, ImageMagick) | P1 | `tests/test_cli_doctor.py::test_doctor_reports_missing_tesseract` |
| T-17 | `pdf-mcp metrics push --pushgateway http://...` | P1 | `tests/test_cli_metrics.py::test_push_emits_textfile_format` |
| T-18 | `pex` single-file build + smoke matrix (macOS + Ubuntu CI) | P1 | `tests/packaging/test_pex_smoke.py::test_pex_runs_help` |
| T-19 | Homebrew tap formula + smoke (`brew install nfsarch33/tap/pdf-mcp`) | P2 | `tests/packaging/test_brew_formula.py::test_brew_audit` |
| T-20 | `pdf-mcp` becomes a release-gate citizen for downstream client repos | P0 | `tests/test_cli_release_gate.py::test_scan_returns_zero_on_clean_pdf` |
| T-21 | Coverage floor lift: 75 → 80 (T-08 c2 dial-up) | P0 | `tests/quality/test_coverage_floor.py::test_cov_floor_at_80` |
| T-22 | CLI tutorial in `docs/USAGE.md` + screencast (gif via `vhs`) | P1 | `tests/docs/test_examples_match_snapshots.py::test_usage_examples_run` |
| T-23 | v1.4.0 release retro + HD scorecard | P0 | n/a — gating doc |

### Tech-debt sweep (carried from v1.3.0 close)

- **TD-01**: `Makefile` lint/format target + `.github/workflows/lint.yml` use `grep -E '\.py$$'` which over-escapes the backslash and silently never matches Python files in PR diffs (effective lint bypass on diff-only mode). Pre-commit covers in practice but the workflow needs a one-line fix.

---

## Sprint v1.5.0 — local-LLM extraction + safety sidecar (2026-05-28 → 2026-06-10)

- **Theme**: introduce LLM-driven structured extraction WITHOUT crossing the ZD AI gateway boundary. All LLM traffic routes through `llm-cluster-router` (Qwen 3.5 / Qwen 3.6 lanes), and a safety-api-sidecar fronts every call.
- **HD bar**: 88 / 100.
- **End-to-end deliverable**: `pdf-mcp ai extract --in <pdf> --schema <json-schema>` returns a typed JSON object validated against the schema; the same surface is callable as a registry tool from the MCP server. No prompt content reaches Mem0; only metric counters do.

### Research-first checklist (W1 D1)

1. Read `~/Code/global-kb/sop/zd-ai-gateway-tier-a.md` and re-confirm the boundary rule (pdf-mcp → router only). Capture the wiring diagram in `docs/research/llm-routing-2026-05-28.md`.
2. Survey JSON-schema-constrained generation patterns supported by Qwen 3.5 / 3.6 instruct lanes in `llm-cluster-router`; pick the lowest-OOM-risk lane (likely 4070TiS for Qwen 3.5 9B, with 2070 fallback at Qwen 3.5 4B).
3. Decide format: `pydantic` schemas or raw JSON Schema. Lock the answer in `docs/decisions/0006-llm-extraction-schema-format.md`.

### Backlog (8 tickets)

| Ticket | Title | Pri | TDD must-fail-first |
|---|---|---|---|
| T-30 | `LLMRouter` shim with mandatory `LLM_CLUSTER_ROUTER_URL`; fail-closed if unset; reject any `ai-gateway.zende.sk` URL | P0 | `tests/test_llm_router.py::test_rejects_zd_gateway_url` |
| T-31 | `pdf-mcp ai extract` MVP (single-page, schema-bound) | P0 | `tests/test_ai_extract.py::test_returns_schema_compliant_json` |
| T-32 | Safety sidecar: PII redaction + prompt-injection ruleset; `--safety on/off` flag (default on) | P0 | `tests/test_safety_sidecar.py::test_redacts_email_before_prompt` |
| T-33 | Streaming progress for long extractions (SSE → CLI progress bar) | P1 | `tests/test_ai_extract.py::test_streams_progress_lines` |
| T-34 | Prompt cache hit/miss telemetry counter (Prometheus textfile) | P1 | `tests/test_metrics.py::test_prompt_cache_counter_increments` |
| T-35 | Coverage floor lift 80 → 82 | P0 | `tests/quality/test_coverage_floor.py::test_cov_floor_at_82` |
| T-36 | `pdf-mcp ai redact-pii` companion verb (no LLM, regex + heuristics; sidecar reuse) | P1 | `tests/test_ai_redact_pii.py::test_emails_masked` |
| T-37 | v1.5.0 retro + HD scorecard | P0 | n/a |

### Sprint v1.5.0 rubric (target ≥ 88/100)

| Dim | Weight | DoD evidence |
|---:|---:|---|
| TDD | 10 | All 8 tickets RED-first; safety sidecar bypass attempts also rejected by tests. |
| Tests | 10 | ≥ 540 passing; 0 regressions. |
| Coverage | 10 | `--cov-fail-under=82` green. |
| Architecture | 10 | LLM access centralised in `LLMRouter`; no direct `httpx.post` to anthropic/openai/anything else; lazy contract intact. |
| Docs | 10 | `docs/USAGE.md` regenerated; safety + LLM-routing doc landed. |
| CLI-readiness | 10 | `pdf-mcp ai extract` ergonomic; streaming progress; sane errors when router unavailable. |
| Release-gate | 10 | Tag `v1.5.0`; smoke `uv tool install pdf-mcp==1.5.0` green. |
| Safety / boundary | 15 | Zero ZD-gateway calls in trace; safety sidecar rejects PII before prompt. |
| Identity discipline | 5 | All pushes pass `cursor-tools doctor identity --strict`. |
| Memory discipline | 10 | Outcome capsules per ticket close; HD scorecard committed. |

---

## Sprint v1.6.0 — operator UX + SBOM / supply-chain hardening (2026-06-11 → 2026-06-24)

- **Theme**: turn `pdf-mcp` into a credible operator-grade artifact in regulated environments. SBOM, reproducible builds, signed releases, downstream-citizen evidence.
- **HD bar**: 90 / 100.
- **End-to-end deliverable**: `gh release v1.6.0` ships with attached CycloneDX SBOM + cosign signature; `slsa-verifier` proves provenance; vhs screencast in README; downstream client repo opens a PR adopting `pdf-mcp scan` as a release-gate citizen.

### Backlog (8 tickets)

| Ticket | Title | Pri | TDD must-fail-first |
|---|---|---|---|
| T-40 | CycloneDX SBOM emitted by release workflow | P0 | `tests/release/test_sbom.py::test_release_yaml_emits_cyclonedx` |
| T-41 | Cosign-sign release artefacts (wheel + tarball + pex) | P0 | `tests/release/test_cosign.py::test_artefacts_signed` |
| T-42 | SLSA provenance attestation via `slsa-github-generator` | P1 | `tests/release/test_slsa.py::test_provenance_attestation_attached` |
| T-43 | Reproducible builds verified across macOS + Linux CI lanes | P1 | `tests/release/test_repro.py::test_builds_match_across_runners` |
| T-44 | `pdf-mcp` adopted as release-gate citizen by `~/agentic-ai-research` (or alt downstream) | P0 | `tests/integration/test_downstream_adoption.py::test_scan_blocks_dirty_pdf` |
| T-45 | vhs-recorded screencast linked from README | P1 | `tests/docs/test_screencast_present.py::test_vhs_tape_compiles` |
| T-46 | Coverage floor 82 → 84 | P0 | `tests/quality/test_coverage_floor.py::test_cov_floor_at_84` |
| T-47 | v1.6.0 retro + HD scorecard | P0 | n/a |

---

## Sprint v1.7.0 — plugin surface + multi-distribution (2026-06-25 → 2026-07-08)

- **Theme**: open `pdf-mcp` to community contribution and broaden distribution to Homebrew + GHCR Docker.
- **HD bar**: 92 / 100.
- **End-to-end deliverable**: `pdf-mcp plugins list` discovers third-party tool plugins via Python entry-points; Homebrew tap install smokes green on macOS; `ghcr.io/nfsarch33/pdf-mcp:1.7.0` published with multi-arch manifest (`linux/amd64`, `linux/arm64`, `darwin/amd64`, `darwin/arm64`); coverage floor at 85.

### Backlog (10 tickets)

| Ticket | Title | Pri | TDD must-fail-first |
|---|---|---|---|
| T-50 | Plugin discovery via Python entry-points (group `pdf_mcp.tools`) | P0 | `tests/plugins/test_discovery.py::test_loads_third_party_plugin` |
| T-51 | Plugin trust model: only signed plugins activate by default; `--trust unsigned` opt-in | P0 | `tests/plugins/test_trust.py::test_unsigned_blocked_by_default` |
| T-52 | `pdf-mcp plugins list` / `install` / `remove` verbs | P1 | `tests/plugins/test_plugins_verb.py::test_list_shows_installed` |
| T-53 | Sample plugin scaffold + cookiecutter template | P1 | `tests/plugins/test_scaffold.py::test_template_creates_runnable_plugin` |
| T-54 | Homebrew tap formula at `nfsarch33/homebrew-tap`; CI smoke | P0 | `tests/packaging/test_brew_smoke.py::test_brew_install_runs_help` |
| T-55 | GHCR multi-arch image build + push | P0 | `tests/packaging/test_docker_multiarch.py::test_image_runs_on_arm64` |
| T-56 | `pdf-mcp` works behind a corporate proxy (HTTPS_PROXY, NO_PROXY honoured) | P1 | `tests/test_proxy_support.py::test_honours_https_proxy` |
| T-57 | Plugin contract test harness — third parties can run `pytest` against a stub registry | P1 | `tests/plugins/test_contract_harness.py::test_third_party_can_validate` |
| T-58 | Coverage floor 84 → 85 | P0 | `tests/quality/test_coverage_floor.py::test_cov_floor_at_85` |
| T-59 | v1.7.0 retro + HD scorecard | P0 | n/a |

---

## Cross-sprint invariants (apply to every release v1.3.0 → v1.7.0)

1. **Test-first**: RED test committed before implementation commit. Every PR shows the RED commit in its history.
2. **Single-source registry**: every new tool added via `register_tool(...)`. Server, CLI, USAGE.md, and (from v1.7) plugins consume the registry.
3. **Lazy import contract**: `pdf-mcp --help` must not import any heavyweight optional dep. Subprocess `sys.modules` test must remain green.
4. **Surface parity**: registry → server.list_tools(), registry → CLI verb-groups, registry → USAGE.md. Three parity tests must remain green.
5. **No ZD AI Gateway**: `pdf-mcp` LLM access (from v1.5.0 onward) accepts `LLM_CLUSTER_ROUTER_URL` only; RED test rejects any `ai-gateway.zende.sk` URL.
6. **Coverage floor**: progressively tightened (75 → 80 → 82 → 84 → 85). Regressions block merge.
7. **Identity gate**: every push validated by `cursor-tools doctor identity --strict`. Tokens must not leak across personal/work boundary.
8. **Outcome capsules**: every PR open + every merge emits an `agent_outcome` capsule via `cursor-tools outcome emit` (NDJSON authoritative; Mem0 best-effort under quota).
9. **Evidence dirs**: every release closes with proof under `~/Code/global-kb/session-handoffs/evidence/v25?-w?-pdf-mcp-v<x.y.z>-close/SCORECARD.md`.

---

## Subagent offload pattern (carried + extended)

| Lane | CLI | Model | When to use |
|---|---|---|---|
| Cursor in-session (Composer) | n/a | Sonnet 4.6 | TDD-tight loops, multi-file edits, PR open + descriptive body |
| Claude Code | `claude --print` | Opus 4.7 / Sonnet 4.6 / Haiku 3.5 | Single-file artifact gen with strict acceptance test |
| Codex CLI | `codex exec --skip-git-repo-check --sandbox read-only` | gpt-5.2 (Tier A, MacBook-only) | Doc-style design briefs, large-context audits |
| Cursor explore subagent (read-only) | (in-session) | Sonnet 4.6 | Multi-target codebase exploration before edits |
| Async background subagent | Task tool, `subagent_type=generalPurpose` | Composer 2 fast | Long-running search / synthesis when context window pressure is high |

**Hard rule**: Codex CLI background mode (`codex exec --full-auto`) remains gated for sandboxed venv work — outbound `pypi.org` fetches fail. Use Codex for design briefs / artifacts that don't require a venv install.

---

## EvoLoop-DRL feedback contract

Every PR open + merge in this roadmap MUST emit an `agent_outcome` capsule. Daily KPI delta target across the v1.3.0 → v1.7.0 window: **+1.00 cumulative** (averaged ≥ +0.20 per release). Mem0 outbox absorbs throttle windows (HTTP 429); NDJSON authority remains canonical per `sop/evoloop-mem0-degraded-authority.md`.

---

## Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Mem0 quota stays throttled past 2026-05-11 | Med | Med | NDJSON authority remains canonical; outbox flush is best-effort |
| LLM-router lane OOM on Qwen 3.5 9B (v1.5.0) | Med | High | 4070TiS lane fallback to Qwen 3.5 4B; 2070 lane reserved for short contexts |
| pex single-file build trips on PyMuPDF native bindings (v1.4.0 T-18) | Med | Med | fall back to `--platform manylinux_2_17_x86_64` matrix; skip mac if blocking |
| Cosign / SLSA tooling churn (v1.6.0) | Med | Low | pin actions to SHA; mirror to a local GHCR cache |
| Plugin trust model attack surface (v1.7.0) | Low | High | TUF metadata + signed plugin manifest; default-deny-unsigned |
| Codex CLI quota / latency interruption | Med | Low | always have Claude Code + in-session as the redundant lane |

---

## Acceptance + exit criteria (per release)

A release is **accepted** only if every row below is green by the close session:

- [ ] All sprint tickets closed (PR merged + evidence committed)
- [ ] HD rubric ≥ target band (95 v1.3.0; 88 v1.4.0; 88 v1.5.0; 90 v1.6.0; 92 v1.7.0)
- [ ] CI on `main` is fully green at the tag commit
- [ ] `pytest --cov=pdf_mcp --cov-fail-under=<threshold>` green
- [ ] `pdf-mcp --help` regenerated golden snapshot committed
- [ ] `USAGE.md` regenerated and `--check` green at tag
- [ ] CHANGELOG entry for the released version committed
- [ ] `gh release create v<x.y.z>` ran with release notes
- [ ] PyPI publish succeeds (from v1.3.0 onward)
- [ ] Outcome capsules emitted for every PR + merge + tag
- [ ] HD scorecard file committed under `~/Code/global-kb/session-handoffs/evidence/v25?-w?-pdf-mcp-v<x.y.z>-close/SCORECARD.md`

If any row is amber/red, the failing dimension carries into the next sprint as P0.
