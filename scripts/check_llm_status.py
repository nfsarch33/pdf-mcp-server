#!/usr/bin/env python3
"""
Check LLM backend status and print formatted output.

Usage: python scripts/check_llm_status.py
"""

from pdf_mcp import llm_setup, pdf_tools


def _status_label(available: bool) -> str:
    return "OK" if available else "NO"


def main():
    info = pdf_tools.get_llm_backend_info()
    current = info["current_backend"] or "none"

    print("LLM BACKEND STATUS")
    print("-" * 60)
    print(f"current_backend: {current}")
    print()
    print("backends:")

    ollama_model = llm_setup.get_ollama_model_name()
    ollama_installed = llm_setup.ollama_is_installed()
    ollama_models = llm_setup.ollama_list_models() if ollama_installed else set()

    # Query local server for model info
    llm_setup.get_local_server_health()  # verify reachability
    local_models = llm_setup.get_local_server_models()

    for name, data in info["backends"].items():
        available = data.get("available", False)
        cost = data.get("cost", "unknown")
        print(f"- {name:8} status={_status_label(available)} cost={cost}")

        if name == "local":
            print(f"  url={data.get('url')} config_model={data.get('model')}")
            if available:
                if local_models:
                    models_list = local_models.get("models", local_models.get("available", []))
                    if models_list:
                        # Handle both list of strings and list of dicts
                        if isinstance(models_list, list):
                            names = [m.get("name", str(m)) if isinstance(m, dict) else str(m) for m in models_list]
                            print(f"  loaded_models: {', '.join(names)}")
                        else:
                            print(f"  loaded_models: {models_list}")
                    else:
                        print("  loaded_models: (endpoint returned no model list)")
                else:
                    print("  loaded_models: (no /models endpoint)")
            else:
                print("  start: ./scripts/run_local_vlm.sh")

        if name == "ollama":
            if not ollama_installed:
                print("  install: curl -fsSL https://ollama.ai/install.sh | sh")
            elif not ollama_models:
                print("  note: ollama list failed or no models found (is the service running?)")
            elif ollama_model in ollama_models:
                print(f"  model: {ollama_model} (installed)")
            else:
                print(f"  model: {ollama_model} (missing)")
                print(f"  install: ollama pull {ollama_model}")

        if name == "openai" and not available:
            print("  set: export OPENAI_API_KEY=your-key")

    print()
    print("recommended model: Qwen3-VL-30B-A3B (95.7% DocVQA, MoE architecture)")
    print()
    print("override backend: export PDF_MCP_LLM_BACKEND=local|ollama|openai")
    print(f"override ollama model: export {llm_setup.OLLAMA_MODEL_ENV}=model:tag")
    print()


if __name__ == "__main__":
    main()
