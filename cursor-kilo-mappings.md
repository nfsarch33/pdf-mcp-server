# Cursor ↔ Kilo Code Mappings — Gap Report

Repository: `pdf-mcp-server`
Generated: 2026-08-12T23:09:55+0000

> This file is operator-owned. Phase 7 review per plan will decide disposition
> for each gap (reimplement / drop / accept).

| Capability | Cursor | Kilo | Gap | Remediation |
|---|---|---|---|---|
| beforeShellExecution hooks | Each shell command runs pre-tool hooks (guard-shell, lowfat-rewrite, rtk-rewrite, semble-discipline, no-raw-docker). | Kilo Code exposes `permission.bash` (allow/deny/ask) but no shell-rewriting hook layer. | HIGH — must rely on operator discipline + pre-commit hooks (Phase 6) to enforce shell guards. | Run pre-tool guards via wrapper scripts (e.g. `kilo-shim`) until Kilo exposes hook APIs. |
| afterShellExecution hooks (doctor, eval) | post-shell hook chain runs helix-dev-tools doctor and rtk session-savings. | Kilo Code does not have an afterShellExecution hook; session savings can be a wrapper-script on top of `kilo`. | MEDIUM — session-end automation is partial. | Wrap `kilo` invocation with a session-savings hook (separate from IDE). |
| sessionStart hook chain (sync, env scrub, planning init) | Eight sessionStart hooks run before first tool call. | Kilo Code does not have a sessionStart hook; can be replaced by a wrapper. | MEDIUM — first-session-of-day sync may be missed. | Documented in kilo-mappings.md; operator runs `runx sync kb` manually before kilo sessions. |
| subagent-watchdog (cap concurrent Task) | Pre-Task hook enforces ≤2 concurrent subagents + depth cap. | Kilo has `subagent_depth` (1 default) but no global concurrent Task cap. | MEDIUM — fan-out safety net is partial. | Set `subagent_depth: 1` in kilo.jsonc; operator manually queues dispatches for fan-out >1. |
| MCP guardrails (no-raw-docker, sanitize-read) | beforeReadFile + beforeMCPExecution hooks enforce secret redaction and docker denial. | Kilo has no equivalent pre-tool hooks for Read/MCP. | HIGH — secrets may leak via Read or MCP tool calls. | Run Read/MCP inside podman wrappers that filter secrets; or accept the regression and rely on .gitignore + pre-commit secrets scan. |
| Per-file globs in rules | `globs: **/*.go` makes a rule apply only when matching files are touched. | AGENTS.md files are per-directory only; no glob patterns. | MEDIUM — Cursor's 183 rules split into ~50 alwaysApply + ~133 scoped; Kilo loads only the alwaysApply block at session start. | AGENTS.md includes a 'Per-globs / scoped rules' appendix listing scoped rules with their globs; agent must Read the relevant .mdc file before editing matching paths. |
| Custom slash-commands | cursor-config/commands/*.md are invoked via /command-name. | Kilo has `instructions` array (paths) but no native slash-command equivalent. | MEDIUM — operator must re-create 7 commands (check-tools, create-zbt-test, flutter-task, go-task, memory-task, security-audit, ...). | Phase 2 of plan: symlink `~/.config/kilo/command` → `cursor-config/commands`. Verify parity in both IDEs. |
| Plan-mode-must-open | Hook refuses to close plan mode without a written plan file. | Kilo has a plan agent mode; not the same as plan-mode-must-open gate. | LOW — Kilo's plan agent covers most use cases. | Document expectation in AGENTS.md; rely on operator discipline. |

## Operator review checklist (Phase 7)

- [ ] Decide per-gap disposition: reimplement / drop / accept.
- [ ] Update AGENTS.md 'operator notes' section with accepted gaps.
- [ ] If a gap is accepted, file a CF entry referencing this row.
- [ ] Promote resolved gaps into rules (`00-p1-*.mdc` style).
