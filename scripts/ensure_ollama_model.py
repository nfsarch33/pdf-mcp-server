#!/usr/bin/env python3
"""
Ensure the configured Ollama model is installed without duplicating downloads.

Usage:
  python scripts/ensure_ollama_model.py
"""

import subprocess
import sys

from pdf_mcp import llm_setup


def main() -> int:
    model = llm_setup.get_ollama_model_name()

    print("OLLAMA MODEL CHECK")
    print("-" * 60)

    if not llm_setup.ollama_is_installed():
        print("status: ollama not installed")
        print("install: curl -fsSL https://ollama.ai/install.sh | sh")
        return 0

    models = llm_setup.ollama_list_models()
    if not models:
        print("status: unable to list models (is the ollama service running?)")
        print(f"next: start ollama, then run: ollama pull {model}")
        return 0

    if model in models:
        print(f"status: model already installed ({model})")
        print("action: skip pull (no duplicate install)")
        return 0

    print(f"status: model missing ({model})")
    print(f"action: ollama pull {model}")
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        print("error: ollama pull failed")
        return result.returncode
    print("done: model installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
