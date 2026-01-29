from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Set

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_MODEL_ENV = "PDF_MCP_OLLAMA_MODEL"
LOCAL_MODEL_SERVER_URL = os.environ.get("LOCAL_MODEL_SERVER_URL", "http://localhost:8100")


def get_ollama_model_name() -> str:
    return os.environ.get(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)


def get_local_server_models() -> Optional[Dict[str, Any]]:
    """
    Query /models endpoint on local model server to get loaded models.
    
    Returns:
        Dict with model info if available, None if server unavailable or endpoint missing.
    """
    try:
        import requests
        response = requests.get(f"{LOCAL_MODEL_SERVER_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_local_server_health() -> Optional[Dict[str, Any]]:
    """
    Query /health endpoint on local model server.
    
    Returns:
        Dict with health info if available, None if server unavailable.
    """
    try:
        import requests
        response = requests.get(f"{LOCAL_MODEL_SERVER_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                return data
            return {"status": "ok"}
    except Exception:
        pass
    return None


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
