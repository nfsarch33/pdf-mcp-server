---
name: cross-platform-setup
description: Set up pdf-mcp-server on a new machine (macOS, Linux, or Windows WSL). Use when migrating to a new environment or onboarding a new workstation.
---
# Cross-Platform Setup

## When to Use
- Setting up pdf-mcp-server on a new machine
- Migrating from macOS to Windows WSL (or vice versa)
- Onboarding a new development workstation

## Prerequisites

Ensure these are installed:
- Python 3.10+
- Git with SSH key configured for GitHub
- Cursor IDE

## Step 1: Clone Repositories

```bash
# Clone pdf-mcp-server
git clone git@github.com:nfsarch33/pdf-mcp-server.git ~/Code/pdf-mcp-server

# Clone memory repos (private)
git clone git@github.com:nfsarch33/cursor-memory-bank.git ~/memo
git clone git@github.com:nfsarch33/cursor-global-kb.git ~/Code/global-kb
```

## Step 2: Install Dependencies

```bash
cd ~/Code/pdf-mcp-server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install -e ".[dev,ocr,llm]"
```

## Step 3: Platform-Specific Setup

### macOS
```bash
# Install Tesseract OCR
brew install tesseract

# Install zbar for barcode detection
brew install zbar

# Install Ollama (optional)
curl -fsSL https://ollama.ai/install.sh | sh
```

### Linux/WSL
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install tesseract-ocr

# Install zbar
sudo apt-get install libzbar0

# Install Ollama (optional)
curl -fsSL https://ollama.ai/install.sh | sh
```

## Step 4: Configure MCP Server

Edit `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "pdf-handler": {
      "command": "$HOME/Code/pdf-mcp-server/.venv/bin/python",
      "args": ["-m", "pdf_mcp.server"],
      "description": "Local PDF form filling and editing"
    }
  }
}
```

**Note for WSL**: Use the full Linux path (e.g., `/home/username/...`), not Windows paths.

## Step 5: Configure Memory System

```bash
# Set global-kb path
echo "$HOME/Code/global-kb" > ~/memo/tools/global-kb-path.txt

# Configure MCP for memory bank
# Add to ~/.cursor/mcp.json:
# "allPepper-memory-bank": { "command": "...", "env": {"MEMORY_BANK_ROOT": "$HOME/memo"} }
```

## Step 6: Install Daily Automation

### macOS
```bash
~/memo/tools/install-launchd-automation.sh
```

### Linux
```bash
~/memo/tools/install-daily-automation.sh
```

### WSL (Windows Task Scheduler)
```powershell
# Run from PowerShell as Administrator
~/memo/tools/install-wsl-task-scheduler.ps1
```

## Step 7: Verify Installation

```bash
cd ~/Code/pdf-mcp-server
source .venv/bin/activate

# Run tests
make test

# Check LLM status
make check-llm

# Verify in Cursor
# Say "health check" in Agent chat
```

## Quick Reference

| Platform | Tesseract | zbar | Ollama |
|----------|-----------|------|--------|
| macOS | `brew install tesseract` | `brew install zbar` | curl script |
| Linux/WSL | `apt install tesseract-ocr` | `apt install libzbar0` | curl script |

## Troubleshooting

### WSL Path Issues
- Use Linux paths inside WSL, not Windows paths
- Ensure SSH key is configured in WSL, not just Windows

### MCP Server Not Found
- Restart Cursor after editing mcp.json
- Verify Python path in mcp.json matches your venv

### Ollama Not Working
- Check service is running: `ollama serve`
- Pull required model: `ollama pull qwen2.5:1.5b`
