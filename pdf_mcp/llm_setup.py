from __future__ import annotations

import os
import shutil
import subprocess
from typing import Set

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_MODEL_ENV = "PDF_MCP_OLLAMA_MODEL"


def get_ollama_model_name() -> str:
    return os.environ.get(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)


def ollama_is_installed() -> bool:
    return shutil.which("ollama") is not None


def parse_ollama_list_output(output: str) -> Set[str]:
    models: Set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("name"):
            continue
        parts = line.split()
        if parts:
            models.add(parts[0])
    return models


def ollama_list_models(run=subprocess.run) -> Set[str]:
    if not ollama_is_installed():
        return set()

    result = run(["ollama", "list"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return set()
    return parse_ollama_list_output(result.stdout)


def ollama_model_installed(model: str, run=subprocess.run) -> bool:
    return model in ollama_list_models(run=run)
