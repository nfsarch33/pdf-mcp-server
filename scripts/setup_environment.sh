#!/usr/bin/env bash
# setup_environment.sh - Cross-platform setup for pdf-mcp-server
#
# Supported platforms:
#   - macOS (Apple Silicon / Intel)
#   - Linux / WSL Ubuntu
#
# What it does:
#   1. Creates Python venv and installs all dependencies
#   2. Installs system packages (Tesseract, zbar)
#   3. Installs Ollama + default model
#   4. On NVIDIA GPU machines: installs vLLM for local Qwen3-VL serving
#   5. Configures environment for optimal GPU utilisation
#
# Usage:
#   chmod +x scripts/setup_environment.sh
#   ./scripts/setup_environment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

# -- Colours (safe for non-interactive too) --
if [ -t 1 ]; then
    GREEN='\033[1;32m'; RED='\033[1;31m'; YELLOW='\033[1;33m'
    CYAN='\033[1;36m'; RESET='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; CYAN=''; RESET=''
fi

info()  { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()   { echo -e "${RED}[ERR]${RESET}  $*"; }

# ============================================================
# Detect platform
# ============================================================
detect_platform() {
    local uname_s
    uname_s="$(uname -s)"
    case "$uname_s" in
        Darwin) PLATFORM="macos" ;;
        Linux)
            if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
                PLATFORM="wsl"
            else
                PLATFORM="linux"
            fi
            ;;
        *) err "Unsupported OS: $uname_s"; exit 1 ;;
    esac

    local uname_m
    uname_m="$(uname -m)"
    case "$uname_m" in
        arm64|aarch64) ARCH="arm64" ;;
        x86_64|amd64)  ARCH="x86_64" ;;
        *) ARCH="$uname_m" ;;
    esac

    info "Platform: $PLATFORM  Arch: $ARCH"
}

# ============================================================
# Step 1: Python venv + dependencies
# ============================================================
setup_python() {
    info "Setting up Python environment..."

    if ! command -v python3 &>/dev/null; then
        err "Python 3 not found. Install Python 3.10+ first."
        exit 1
    fi

    PYTHON_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    info "Python version: $PYTHON_VER"

    # Create venv if missing
    if [ ! -d "$VENV_DIR" ]; then
        if command -v uv &>/dev/null; then
            info "Creating venv with uv..."
            uv venv "$VENV_DIR"
        else
            info "Creating venv with python3..."
            python3 -m venv "$VENV_DIR"
        fi
    fi
    ok "Virtual environment ready: $VENV_DIR"

    # Install dependencies
    info "Installing dependencies..."
    if command -v uv &>/dev/null; then
        (cd "$PROJECT_DIR" && uv pip install -r requirements.txt)
        (cd "$PROJECT_DIR" && uv pip install -e ".[dev,ocr,llm]")
    else
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
        "$VENV_DIR/bin/pip" install -e "$PROJECT_DIR[dev,ocr,llm]"
    fi
    ok "Python dependencies installed"
}

# ============================================================
# Step 2: System packages
# ============================================================
install_system_packages() {
    info "Installing system packages..."

    case "$PLATFORM" in
        macos)
            if ! command -v brew &>/dev/null; then
                warn "Homebrew not installed. Skipping system packages."
                return
            fi
            # Tesseract
            if ! command -v tesseract &>/dev/null; then
                info "Installing Tesseract..."
                brew install tesseract
            else
                ok "Tesseract already installed"
            fi
            # zbar
            if ! brew list zbar &>/dev/null 2>&1; then
                info "Installing zbar..."
                brew install zbar
            else
                ok "zbar already installed"
            fi
            ;;
        linux|wsl)
            info "Installing apt packages (may need sudo)..."
            sudo apt-get update -qq
            sudo apt-get install -y -qq tesseract-ocr libzbar0
            ok "System packages installed"
            ;;
    esac
}

# ============================================================
# Step 3: Ollama
# ============================================================
install_ollama() {
    info "Checking Ollama..."

    if command -v ollama &>/dev/null; then
        ok "Ollama already installed"
    else
        info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        ok "Ollama installed"
    fi

    # Ensure default model
    info "Checking Ollama model..."
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/ensure_ollama_model.py"
}

# ============================================================
# Step 4: GPU detection + local VLM setup
# ============================================================
detect_gpus() {
    GPU_COUNT=0
    BEST_GPU_ID=""
    BEST_GPU_VRAM=0
    HAS_NVIDIA=false
    HAS_APPLE_SILICON=false

    if [ "$PLATFORM" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        HAS_APPLE_SILICON=true
        # Unified memory - get total system RAM as proxy
        local total_mem
        total_mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        BEST_GPU_VRAM=$((total_mem / 1024 / 1024))  # MB
        info "Apple Silicon detected: unified memory ${BEST_GPU_VRAM}MB"
        return
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. No NVIDIA GPU detected."
        return
    fi

    HAS_NVIDIA=true

    # Query all GPUs: index, name, memory (MiB)
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)

    if [ -z "$gpu_info" ]; then
        warn "nvidia-smi returned no data."
        return
    fi

    info "Detected NVIDIA GPUs:"
    while IFS=', ' read -r idx name vram_mb; do
        # Trim whitespace
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        vram_mb=$(echo "$vram_mb" | xargs)

        GPU_COUNT=$((GPU_COUNT + 1))
        info "  GPU $idx: $name (${vram_mb}MB VRAM)"

        if [ "$vram_mb" -gt "$BEST_GPU_VRAM" ]; then
            BEST_GPU_ID="$idx"
            BEST_GPU_VRAM="$vram_mb"
        fi
    done <<< "$gpu_info"

    if [ -n "$BEST_GPU_ID" ]; then
        ok "Best GPU for VLM: GPU $BEST_GPU_ID (${BEST_GPU_VRAM}MB VRAM)"
    fi
}

setup_nvidia_vlm() {
    if [ "$HAS_NVIDIA" != "true" ]; then
        return
    fi

    info "Setting up NVIDIA local VLM support..."

    # Install vLLM for CUDA-based serving
    if command -v uv &>/dev/null; then
        uv pip install vllm
    else
        "$VENV_DIR/bin/pip" install vllm
    fi
    ok "vLLM installed for NVIDIA GPU serving"

    # Write GPU config
    local config_file="$PROJECT_DIR/.gpu_config"
    cat > "$config_file" <<EOF
# Auto-generated by setup_environment.sh
# GPU configuration for local VLM serving
GPU_COUNT=$GPU_COUNT
BEST_GPU_ID=$BEST_GPU_ID
BEST_GPU_VRAM=$BEST_GPU_VRAM
CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
EOF
    ok "GPU config written to $config_file"
}

setup_macos_vlm() {
    if [ "$HAS_APPLE_SILICON" != "true" ]; then
        return
    fi

    info "Setting up Apple Silicon local VLM support..."

    # MLX is the recommended backend for Apple Silicon
    if command -v uv &>/dev/null; then
        uv pip install mlx mlx-lm
    else
        "$VENV_DIR/bin/pip" install mlx mlx-lm
    fi
    ok "MLX installed for Apple Silicon serving"
}

# ============================================================
# Step 5: Generate runner script
# ============================================================
generate_runner_script() {
    local runner="$SCRIPT_DIR/run_local_vlm.sh"

    info "Generating local VLM runner script..."

    cat > "$runner" <<'RUNNER_HEADER'
#!/usr/bin/env bash
# run_local_vlm.sh - Start the local VLM server on the best available hardware
#
# Auto-generated by setup_environment.sh
# Detects GPUs at runtime and pins the VLM to the card with the most VRAM.
#
# Usage:
#   ./scripts/run_local_vlm.sh [--port PORT] [--model MODEL]
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES  - Force specific GPU(s)
#   LOCAL_VLM_PORT        - Override default port (8100)
#   LOCAL_VLM_MODEL       - Override default model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
PY="$VENV_DIR/bin/python"

PORT="${LOCAL_VLM_PORT:-8100}"
MODEL="${LOCAL_VLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"

# Parse CLI args
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Colours
if [ -t 1 ]; then
    GREEN='\033[1;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; RESET='\033[0m'
else
    GREEN=''; YELLOW=''; CYAN=''; RESET=''
fi

info()  { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }

# ---- Detect best GPU at runtime ----
select_best_gpu() {
    local uname_s uname_m
    uname_s="$(uname -s)"
    uname_m="$(uname -m)"

    if [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
        BACKEND="mlx"
        info "Apple Silicon detected -> using MLX backend"
        return
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        warn "No NVIDIA GPU found. Falling back to Ollama."
        BACKEND="ollama"
        return
    fi

    BACKEND="vllm"

    # Already set by user?
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        info "CUDA_VISIBLE_DEVICES already set: $CUDA_VISIBLE_DEVICES"
        return
    fi

    # Auto-select GPU with most VRAM
    local best_id="" best_vram=0
    while IFS=', ' read -r idx vram_mb; do
        idx=$(echo "$idx" | xargs)
        vram_mb=$(echo "$vram_mb" | xargs)
        if [ "$vram_mb" -gt "$best_vram" ]; then
            best_id="$idx"
            best_vram="$vram_mb"
        fi
    done <<< "$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null)"

    if [ -n "$best_id" ]; then
        export CUDA_VISIBLE_DEVICES="$best_id"
        ok "Auto-selected GPU $best_id (${best_vram}MB VRAM)"
    fi
}

# ---- Serve with detected backend ----
serve_vllm() {
    info "Starting vLLM server on port $PORT with model $MODEL"
    info "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

    # Determine GPU memory utilisation based on VRAM
    local gpu_util="0.90"
    exec "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --trust-remote-code \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len 4096
}

serve_mlx() {
    info "Starting MLX server on port $PORT"
    # Use mlx_lm for Apple Silicon
    local mlx_model="${MODEL}"
    # Map common names to MLX-compatible model IDs
    case "$mlx_model" in
        *Qwen3-VL-30B*) mlx_model="Qwen/Qwen3-VL-30B-A3B-Instruct" ;;
        *qwen2.5-7b*)   mlx_model="mlx-community/Qwen2.5-7B-Instruct-4bit" ;;
    esac

    exec "$PY" -m mlx_lm.server \
        --model "$mlx_model" \
        --port "$PORT"
}

serve_ollama() {
    info "Starting Ollama serve (model will be loaded on first request)"
    warn "For Qwen3-VL, ensure the model is pulled: ollama pull qwen3-vl:30b-a3b"
    exec ollama serve
}

# ---- Main ----
select_best_gpu

info "Backend: $BACKEND | Port: $PORT | Model: $MODEL"

case "$BACKEND" in
    vllm)   serve_vllm ;;
    mlx)    serve_mlx ;;
    ollama) serve_ollama ;;
    *)      echo "Unknown backend: $BACKEND"; exit 1 ;;
esac
RUNNER_HEADER

    chmod +x "$runner"
    ok "Runner script created: $runner"
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    echo -e "${GREEN}============================================================${RESET}"
    echo -e "${GREEN}  Setup Complete!${RESET}"
    echo -e "${GREEN}============================================================${RESET}"
    echo ""
    echo "Platform: $PLATFORM ($ARCH)"
    echo ""

    if [ "$HAS_NVIDIA" = "true" ]; then
        echo "NVIDIA GPUs: $GPU_COUNT detected"
        echo "Best GPU:    #$BEST_GPU_ID (${BEST_GPU_VRAM}MB VRAM)"
        echo "Backend:     vLLM (CUDA)"
    elif [ "$HAS_APPLE_SILICON" = "true" ]; then
        echo "Apple Silicon: unified memory (~${BEST_GPU_VRAM}MB)"
        echo "Backend:     MLX"
    else
        echo "GPU:         none detected"
        echo "Backend:     Ollama (CPU fallback)"
    fi

    echo ""
    echo "Quick start:"
    echo "  source .venv/bin/activate"
    echo "  make test               # run tests"
    echo "  make check-llm          # check LLM status"
    echo "  ./scripts/run_local_vlm.sh  # start local VLM server"
    echo ""
    echo "Override GPU:  CUDA_VISIBLE_DEVICES=1 ./scripts/run_local_vlm.sh"
    echo "Override port: ./scripts/run_local_vlm.sh --port 9000"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    info "pdf-mcp-server environment setup"
    info "================================"
    echo ""

    detect_platform
    setup_python
    install_system_packages
    install_ollama
    detect_gpus

    if [ "$HAS_NVIDIA" = "true" ]; then
        setup_nvidia_vlm
    elif [ "$HAS_APPLE_SILICON" = "true" ]; then
        setup_macos_vlm
    fi

    generate_runner_script
    print_summary
}

main "$@"
