# Next Release Plan (v0.9.5)

## Goal
Harden LLM setup and manual QA flows with clear diagnostics, while keeping
all installations idempotent and cost-safe.

## Scope (Proposed)
1. LLM diagnostics
   - Add a lightweight `/models` check for local model server (if available).
   - Report which local model is loaded when the endpoint supports it.
2. Ollama guardrails
   - Skip E2E Ollama tests if the configured model is missing (clear reason).
   - Add a short FAQ entry for common Ollama service errors.
3. Manual QA coverage
   - Expand LLM manual QA steps in release SOP (local, ollama, openai).
   - Add a small PDF fixture dedicated to LLM extraction edge cases.
4. Docs and SOP
   - Align README + Pepper SOP with any new steps or env vars.

## Acceptance Criteria
- `make check-llm` shows clear local/Ollama/OpenAI state and model presence.
- E2E tests skip with explicit reasons when models are missing.
- Manual QA steps validated on local backend with real server running.
- No duplicate model downloads during setup.

## Test Plan
- `make check-llm`
- `make test-llm`
- `make test-e2e` (local server running)
- Manual QA: run local backend through `auto_fill_pdf_form`,
  `extract_structured_data`, and `analyze_pdf_content` using real fixtures.
