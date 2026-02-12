---
name: llm-e2e-qa
description: Runs and verifies LLM E2E tests for local VLM (vLLM on NVIDIA GPU) and Ollama, and documents manual QA steps. Use when validating LLM integrations before release or reproducing E2E results.
---
# LLM E2E QA

## When to Use
- Validating LLM integrations before release
- Reproducing E2E results with real backends
- Performing manual QA for local VLM and Ollama

## Instructions
1. Confirm local model server (vLLM with Qwen2.5-VL-7B-Instruct on RTX 3090):
   - `curl -s http://localhost:8100/v1/models`
2. Confirm Ollama service and model:
   - `ollama list`
3. Run LLM status check:
   - `make check-llm`
4. Run E2E tests:
   - `python -m pytest tests/test_agentic_features.py -m slow`
5. Record results:
   - Local VLM: 5/5 passing
   - Ollama: 2/2 passing (requires model)
   - OpenAI: skipped if no API key

## Manual QA Checklist
- `auto_fill_pdf_form` returns results with a real local model
- `extract_structured_data` works on sample PDF (passport, invoice)
- `analyze_pdf_content` returns summary and entities
- Ollama backend responds to a simple prompt

## Output Format
- E2E test summary
- Manual QA checklist status
- Any skips with reasons
