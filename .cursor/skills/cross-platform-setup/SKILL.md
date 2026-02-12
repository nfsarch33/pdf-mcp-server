---
name: cross-platform-setup
description: Sets up pdf-mcp-server on a new machine (macOS, Linux, or Windows WSL) with GPU-aware VLM support. Use when migrating to a new environment, onboarding a new workstation, or configuring local VLM serving.
---
# Cross-Platform Setup

## When to Use
- Setting up pdf-mcp-server on a new machine
- Migrating from macOS to Windows WSL (or vice versa)
- Onboarding a new development workstation
- Setting up local VLM on a multi-GPU machine

## Quick Setup (Recommended)

```bash
# Clone pdf-mcp-server
git clone git@github.com:nfsarch33/pdf-mcp-server.git ~/Code/pdf-mcp-server
cd ~/Code/pdf-mcp-server

# One-command setup: venv, system packages, Ollama, GPU detection, VLM
./scripts/setup_environment.sh

# Start local VLM server (auto-detects best GPU)
./scripts/run_local_vlm.sh
```

The setup script handles:
- Python venv creation and all dependency installation
- System packages (Tesseract OCR, zbar)
- Ollama installation and model pull
- GPU detection and optimal card selection
- VLM backend installation (vLLM for NVIDIA, MLX for Apple Silicon)
- Ensures `run_local_vlm.sh` (committed in repo) is executable

## Prerequisites

- Python 3.10+
- Git with SSH key configured for GitHub
- Cursor IDE

## Manual Setup (Step by Step)

### Step 1: Clone Repository

```bash
git clone git@github.com:nfsarch33/pdf-mcp-server.git ~/Code/pdf-mcp-server
cd ~/Code/pdf-mcp-server
```

### Step 2: Install Dependencies

```bash
# Create venv + install
make install

# Install optional dependencies
uv pip install -e ".[dev,ocr,llm]"
```

### Step 3: Platform-Specific Packages

**macOS:**
```bash
brew install tesseract zbar
```

**Linux/WSL:**
```bash
sudo apt-get install -y tesseract-ocr libzbar0
```

### Step 4: Ollama Setup

```bash
curl -fsSL https://ollama.ai/install.sh | sh
make install-llm-models
```

### Step 5: Configure MCP Server

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

**WSL note**: Use Linux paths (e.g., `/home/username/...`), not Windows paths.

## GPU-Aware VLM Serving

### How It Works

`run_local_vlm.sh` detects GPUs at **runtime** and pins the VLM to the best card:

| Platform | Detection | Backend | Notes |
|----------|-----------|---------|-------|
| macOS Apple Silicon | `sysctl hw.memsize` | MLX | Unified memory |
| Linux/WSL + NVIDIA | `nvidia-smi` query | vLLM | Selects highest VRAM card |
| No GPU | fallback | Ollama | CPU-based |

### Multi-GPU Example (RTX 4070 Ti Super + RTX 3090)

```
[INFO] Detected NVIDIA GPUs:
  GPU 0: NVIDIA GeForce RTX 4070 Ti Super (16384MB VRAM)
  GPU 1: NVIDIA GeForce RTX 3090 (24576MB VRAM)
[OK]   Auto-selected GPU 1 (24576MB VRAM)
[INFO] Starting vLLM server on port 8100 with model Qwen/Qwen2.5-VL-7B-Instruct
```

The RTX 3090 (24GB) is automatically selected for the heavy VLM model.

### Override GPU Selection

```bash
# Force specific GPU
CUDA_VISIBLE_DEVICES=0 ./scripts/run_local_vlm.sh

# Custom port
./scripts/run_local_vlm.sh --port 9000

# Custom model
./scripts/run_local_vlm.sh --model Qwen/Qwen2.5-7B-Instruct
```

## Memory System Setup (Optional)

```bash
# Clone memory repos (private)
git clone git@github.com:nfsarch33/cursor-memory-bank.git ~/memo

# Install daily automation
~/memo/tools/install-daily-automation.sh  # Linux
~/memo/tools/install-launchd-automation.sh  # macOS
```

## Verify Installation

```bash
cd ~/Code/pdf-mcp-server
source .venv/bin/activate

make test         # Run tests
make check-llm    # Check LLM status
make test-e2e     # E2E tests (requires VLM running)
```

## Quick Reference

| Task | Command |
|------|---------|
| Full setup | `./scripts/setup_environment.sh` |
| Start VLM | `./scripts/run_local_vlm.sh` |
| Check status | `make check-llm` |
| Run tests | `make test` |
| E2E tests | `make test-e2e` |

## Troubleshooting

### WSL Path Issues
- Use Linux paths, not Windows paths
- Ensure SSH key is configured in WSL

### MCP Server Not Found
- Restart Cursor after editing mcp.json
- Verify Python path matches your venv

### GPU Not Detected in WSL
- Ensure NVIDIA drivers are installed on Windows host
- WSL needs `nvidia-smi` accessible (comes with NVIDIA Container Toolkit)
- Run `nvidia-smi` in WSL terminal to verify

### VLM Out of Memory
- Use a smaller model: `./scripts/run_local_vlm.sh --model Qwen/Qwen2.5-7B-Instruct`
- For 16GB GPUs, 4-bit quantised models work well via Ollama
