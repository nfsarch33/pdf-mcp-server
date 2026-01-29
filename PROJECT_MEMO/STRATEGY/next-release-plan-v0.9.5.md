# Release Plan v0.9.5 (COMPLETE)

## Goal
Harden LLM setup and manual QA flows with clear diagnostics, while keeping
all installations idempotent and cost-safe.

## Implemented
1. LLM diagnostics
   - Added `get_local_server_health()` and `get_local_server_models()` functions
   - `make check-llm` now reports loaded models when available
   - Recommended model info displayed (Qwen3-VL-30B-A3B)

2. Ollama guardrails
   - E2E Ollama tests now skip with clear reasons
   - Model presence check before running tests
   - Specific skip messages for missing CLI, service, or model

3. Manual QA coverage
   - LLM manual QA steps in release SOP
   - Enhanced test skips with actionable messages

4. Docs and SOP
   - README updated with 267 tests count
   - Pepper memory updated with research findings
   - CHANGELOG includes technical notes on Qwen3-VL

## Acceptance Criteria (MET)
- `make check-llm` shows clear local/Ollama/OpenAI state and model presence
- E2E tests skip with explicit reasons when models are missing
- Manual QA steps validated on local backend with real server running
- No duplicate model downloads during setup

## Test Results
- 267 tests collected
- 249 passed, 18 skipped (without LLM servers)
- All prepush checks pass

## Verified
- Local VLM: `make check-llm` reports model info
- E2E: `make test-e2e` shows clear skip reasons
- Smoke: `make smoke` passes

*Completed: 2026-01-29*
