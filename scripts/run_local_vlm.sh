#!/usr/bin/env bash
# run_local_vlm.sh - Start the local VLM server on the best available hardware
#
# Detects GPUs at runtime and pins the VLM to the card with the most VRAM.
# Works on macOS (Apple Silicon / Intel) and Linux / WSL.
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
