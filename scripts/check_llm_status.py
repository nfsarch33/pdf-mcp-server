#!/usr/bin/env python3
"""
Check LLM backend status and print formatted output.

Usage: python scripts/check_llm_status.py
"""

from pdf_mcp import pdf_tools

# ANSI color codes
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def main():
    info = pdf_tools.get_llm_backend_info()
    
    current = info["current_backend"] or "None"
    current_color = GREEN if info["current_backend"] else RED
    print(f"Current Backend: {current_color}{current}{RESET}")
    print()
    print("Available Backends:")
    
    for name, data in info["backends"].items():
        available = data.get("available", False)
        status = f"{GREEN}✓ Available{RESET}" if available else f"{RED}✗ Not Available{RESET}"
        cost = f"{YELLOW}{data.get('cost', 'unknown')}{RESET}"
        print(f"  • {name:8} {status:30} ({cost})")
        
        # Show setup instructions if not available
        if not available:
            if name == "local":
                print("             └─ Start: cd ~/agentic-ai-research && uv run python -m services.model_server.cli serve")
            elif name == "ollama":
                print("             └─ Install: curl -fsSL https://ollama.ai/install.sh | sh")
            elif name == "openai":
                print("             └─ Set: export OPENAI_API_KEY=your-key")
    
    print()
    print("Override: export PDF_MCP_LLM_BACKEND=local|ollama|openai")
    print()


if __name__ == "__main__":
    main()
