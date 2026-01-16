#!/usr/bin/env bash
#
# Song-to-Video Installation Script
# Installs all dependencies and sets up the environment automatically.
#
# Usage:
#   ./install.sh                       # Interactive installation
#   INSTALL_OLLAMA=auto ./install.sh   # Auto-install Ollama without prompting
#   INSTALL_OLLAMA=skip ./install.sh   # Skip Ollama installation
#   HF_LOGIN=skip ./install.sh         # Skip HuggingFace login prompt
#
set -euo pipefail

# Configuration
PYTHON_MIN_VERSION="3.11"
VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_DISK_SPACE_GB=50

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Logging functions
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step()  { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }

# State variables
GPU_AVAILABLE=false
HARDWARE_TIER="cpu"
OLLAMA_INSTALLED=false
OLLAMA_RUNNING=false
PYTHON_CMD="python3"

print_banner() {
    echo ""
    echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║${NC}           ${GREEN}Song-to-Video${NC} ${DIM}Installer${NC}                            ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}     ${DIM}Transform songs into AI-generated music videos${NC}            ${BOLD}║${NC}"
    echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

#───────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
#───────────────────────────────────────────────────────────────────────────────

check_internet() {
    log_info "Checking internet connection..."
    if curl -s --connect-timeout 5 https://github.com > /dev/null 2>&1; then
        log_ok "Internet connection available"
        return 0
    else
        log_error "No internet connection detected"
        log_info "This installer requires internet access to download dependencies"
        exit 1
    fi
}

check_disk_space() {
    log_info "Checking disk space..."
    local available_gb
    available_gb=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')

    if [[ $available_gb -lt $MIN_DISK_SPACE_GB ]]; then
        log_warn "Low disk space: ${available_gb}GB available (${MIN_DISK_SPACE_GB}GB recommended)"
        log_info "AI models require significant storage space"
    else
        log_ok "Disk space: ${available_gb}GB available"
    fi
}

check_wsl() {
    log_info "Checking environment..."
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log_ok "Running in WSL2"
    else
        log_warn "Not running in WSL (installation may still work on native Linux)"
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# System dependencies
#───────────────────────────────────────────────────────────────────────────────

install_system_deps() {
    log_step "Installing System Dependencies"

    local packages_to_install=()

    # Check each required package
    log_info "Checking required packages..."

    if ! command -v curl &>/dev/null; then
        packages_to_install+=("curl")
    fi

    if ! command -v git &>/dev/null; then
        packages_to_install+=("git")
    fi

    if ! command -v ffmpeg &>/dev/null; then
        packages_to_install+=("ffmpeg")
    fi

    if ! command -v zstd &>/dev/null; then
        packages_to_install+=("zstd")
    fi

    # Check for Python 3.11+
    local python_ok=false
    local python_version=""
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            if $cmd -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                PYTHON_CMD="$cmd"
                python_version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
                python_ok=true
                break
            fi
        fi
    done

    if [[ "$python_ok" == "false" ]]; then
        log_warn "Python 3.11+ not found"
        packages_to_install+=("python3.11" "python3.11-venv")
        PYTHON_CMD="python3.11"
        python_version="3.11"
    else
        # Python found, check if venv module is available
        if ! $PYTHON_CMD -c "import venv" &>/dev/null 2>&1; then
            log_info "Python venv module not found, will install python${python_version}-venv"
            packages_to_install+=("python${python_version}-venv")
        fi
    fi

    # Install missing packages
    if [[ ${#packages_to_install[@]} -gt 0 ]]; then
        log_info "Installing: ${packages_to_install[*]}"

        # Add deadsnakes PPA if we need Python 3.11+
        if [[ " ${packages_to_install[*]} " =~ " python3.11 " ]] || [[ " ${packages_to_install[*]} " =~ " python3.12 " ]]; then
            log_info "Adding Python repository..."
            sudo apt-get update -qq
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
        fi

        sudo apt-get update -qq
        if sudo apt-get install -y "${packages_to_install[@]}"; then
            log_ok "System packages installed"
        else
            log_error "Failed to install system packages"
            log_info "Try running: sudo apt-get install ${packages_to_install[*]}"
            exit 1
        fi
    else
        log_ok "All system packages already installed"
    fi

    # Verify Python
    local python_version
    python_version=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_ok "Python $python_version ($PYTHON_CMD)"

    # Verify FFmpeg
    local ffmpeg_version
    ffmpeg_version=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}' || echo "unknown")
    log_ok "FFmpeg $ffmpeg_version"
}

#───────────────────────────────────────────────────────────────────────────────
# GPU Detection
#───────────────────────────────────────────────────────────────────────────────

check_gpu() {
    log_step "Detecting Hardware"

    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "NVIDIA GPU not detected (nvidia-smi not found)"
        log_info "Video generation will be very slow on CPU"
        GPU_AVAILABLE=false
        HARDWARE_TIER="cpu"
        return 0
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)

    if [[ -z "$gpu_info" ]]; then
        log_warn "No NVIDIA GPU detected"
        GPU_AVAILABLE=false
        HARDWARE_TIER="cpu"
        return 0
    fi

    local gpu_name gpu_vram
    gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
    gpu_vram=$(echo "$gpu_info" | cut -d',' -f2 | grep -oE '[0-9]+' || echo "0")

    GPU_AVAILABLE=true
    log_ok "GPU: $gpu_name (${gpu_vram} MiB VRAM)"

    if [[ $gpu_vram -lt 10000 ]]; then
        HARDWARE_TIER="low"
        log_info "Tier: LOW - Will use memory optimization (8GB VRAM)"
    elif [[ $gpu_vram -lt 20000 ]]; then
        HARDWARE_TIER="mid"
        log_info "Tier: MID - Balanced quality/performance (12-16GB VRAM)"
    else
        HARDWARE_TIER="high"
        log_info "Tier: HIGH - Full quality enabled (24GB+ VRAM)"
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# Ollama Installation
#───────────────────────────────────────────────────────────────────────────────

check_ollama() {
    log_step "Setting Up Ollama (LLM Server)"

    if command -v ollama &>/dev/null; then
        OLLAMA_INSTALLED=true
        local ollama_version
        ollama_version=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
        log_ok "Ollama $ollama_version installed"

        # Start service if not running
        if ! curl -s --connect-timeout 2 http://localhost:11434/api/version &>/dev/null; then
            start_ollama_service
        else
            OLLAMA_RUNNING=true
            log_ok "Ollama server is running"
        fi

        # Check/pull model
        if [[ "$OLLAMA_RUNNING" == "true" ]]; then
            local models
            models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | head -5 | tr '\n' ' ' || true)
            if [[ -n "$models" ]]; then
                log_info "Available models: $models"
            fi

            # Ensure we have the default model
            if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
                pull_ollama_model
            fi
        fi
        return 0
    fi

    log_warn "Ollama not found (required for AI scene planning)"

    case "${INSTALL_OLLAMA:-prompt}" in
        auto)
            install_ollama
            ;;
        prompt)
            echo ""
            read -p "Install Ollama now? [Y/n] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                install_ollama
            else
                log_info "Skipping Ollama installation"
                log_info "Install later with: curl -fsSL https://ollama.com/install.sh | sh"
            fi
            ;;
        skip)
            log_info "Skipping Ollama (INSTALL_OLLAMA=skip)"
            ;;
    esac
}

install_ollama() {
    log_info "Installing Ollama..."

    # Download and run Ollama installer
    if curl -fsSL https://ollama.com/install.sh | sh; then
        OLLAMA_INSTALLED=true
        log_ok "Ollama installed successfully"

        # Start Ollama service
        start_ollama_service

        # Pull the default model
        pull_ollama_model
    else
        log_error "Ollama installation failed"
        log_info "Try installing manually:"
        log_info "  curl -fsSL https://ollama.com/install.sh | sh"
        return 1
    fi
}

start_ollama_service() {
    log_info "Starting Ollama service..."

    # Check if already running
    if curl -s --connect-timeout 2 http://localhost:11434/api/version &>/dev/null; then
        OLLAMA_RUNNING=true
        log_ok "Ollama is already running"
        return 0
    fi

    # Try systemd first (Ollama installs as a service)
    if command -v systemctl &>/dev/null && systemctl is-enabled ollama &>/dev/null 2>&1; then
        sudo systemctl start ollama
        sleep 2
    else
        # Start in background manually
        log_info "Starting Ollama in background..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
    fi

    # Verify it started
    if curl -s --connect-timeout 5 http://localhost:11434/api/version &>/dev/null; then
        OLLAMA_RUNNING=true
        log_ok "Ollama service started"
    else
        log_warn "Could not start Ollama automatically"
        log_info "Start manually with: ollama serve"
    fi
}

pull_ollama_model() {
    if [[ "$OLLAMA_RUNNING" != "true" ]]; then
        log_warn "Ollama not running, skipping model download"
        log_info "Run later: ollama pull llama3.1:8b"
        return 1
    fi

    # Check if model already exists
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        log_ok "Model llama3.1:8b already installed"
        return 0
    fi

    log_info "Downloading LLM model (llama3.1:8b)..."
    log_info "This may take a few minutes..."

    if ollama pull llama3.1:8b; then
        log_ok "Model llama3.1:8b installed"
    else
        log_warn "Model download failed"
        log_info "Run later: ollama pull llama3.1:8b"
        return 1
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# Python Environment
#───────────────────────────────────────────────────────────────────────────────

setup_venv() {
    log_step "Setting Up Python Environment"

    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment already exists"
        read -p "Recreate it? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing old environment..."
            rm -rf "$VENV_DIR"
        fi
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    log_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools -q

    log_ok "Virtual environment ready"
}

install_packages() {
    log_step "Installing Python Packages"

    log_info "This may take several minutes..."
    echo ""

    # Install PyTorch nightly (supports latest GPUs like RTX 5090/Blackwell)
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log_info "Installing PyTorch nightly with CUDA support..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 -q
    else
        log_info "Installing PyTorch (CPU only)..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu -q
    fi
    log_ok "PyTorch nightly installed"

    # Install song-to-video
    if [[ "${EDITABLE_INSTALL:-true}" == "true" ]]; then
        log_info "Installing song-to-video..."
        pip install -e ".[dev]" -q
    else
        pip install ".[dev]" -q
    fi
    log_ok "song-to-video installed"
}

#───────────────────────────────────────────────────────────────────────────────
# HuggingFace Login (optional, enables FLUX models)
#───────────────────────────────────────────────────────────────────────────────

setup_huggingface() {
    log_step "HuggingFace Setup (Optional)"

    # Check if already logged in
    if python -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null 2>&1; then
        local hf_user
        hf_user=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null || echo "unknown")
        log_ok "Already logged in as: $hf_user"
        log_info "FLUX models (higher quality) are available"
        return 0
    fi

    log_info "HuggingFace login enables access to FLUX models (higher quality)"
    log_info "Without login, SDXL will be used (still good quality)"

    case "${HF_LOGIN:-prompt}" in
        skip)
            log_info "Skipping HuggingFace login (HF_LOGIN=skip)"
            return 0
            ;;
        prompt)
            echo ""
            read -p "Login to HuggingFace? [y/N] " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo ""
                log_info "Get your token at: https://huggingface.co/settings/tokens"
                log_info "After login, accept FLUX license at: https://huggingface.co/black-forest-labs/FLUX.1-schnell"
                echo ""
                if huggingface-cli login; then
                    log_ok "HuggingFace login successful"
                else
                    log_warn "Login failed - SDXL will be used as default"
                fi
            else
                log_info "Skipping HuggingFace login (SDXL will be used)"
            fi
            ;;
    esac
}

#───────────────────────────────────────────────────────────────────────────────
# Verification
#───────────────────────────────────────────────────────────────────────────────

verify_installation() {
    log_step "Verifying Installation"

    # Ensure venv bin is in PATH
    export PATH="$SCRIPT_DIR/$VENV_DIR/bin:$PATH"
    hash -r 2>/dev/null || true

    # Try the command directly, or via the venv bin path
    local song_cmd="$SCRIPT_DIR/$VENV_DIR/bin/song-to-video"

    if [[ ! -x "$song_cmd" ]]; then
        log_error "song-to-video not found in venv"
        log_info "Installation may have failed. Try:"
        log_info "  source .venv/bin/activate && pip install -e ."
        return 1
    fi

    local version
    version=$("$song_cmd" version 2>&1 || echo "unknown")
    log_ok "$version"

    # Run verification script if available
    if [[ -f "$SCRIPT_DIR/scripts/verify_install.py" ]]; then
        echo ""
        "$SCRIPT_DIR/$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/verify_install.py"
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# Summary
#───────────────────────────────────────────────────────────────────────────────

print_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║               Installation Complete!                          ║${NC}"
    echo -e "${GREEN}${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Hardware: ${BOLD}$HARDWARE_TIER${NC} tier"
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo ""
    echo "  1. Activate the environment:"
    echo -e "     ${DIM}source .venv/bin/activate${NC}"
    echo ""
    echo "  2. Generate a video:"
    echo -e "     ${DIM}song-to-video generate song.mp3 -o video.mp4${NC}"
    echo ""

    if [[ "$OLLAMA_RUNNING" != "true" ]]; then
        echo -e "${YELLOW}Note:${NC} Ollama needs to be running before generating videos."
        echo -e "      Start it with: ${DIM}ollama serve${NC}"
        echo ""
    fi

    if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        echo -e "${YELLOW}Note:${NC} LLM model not yet installed."
        echo -e "      Run: ${DIM}ollama pull llama3.1:8b${NC}"
        echo ""
    fi

    echo -e "${DIM}AI models (~30-50GB) download automatically on first run.${NC}"
    echo ""
}

#───────────────────────────────────────────────────────────────────────────────
# Main
#───────────────────────────────────────────────────────────────────────────────

main() {
    cd "$SCRIPT_DIR"

    print_banner
    check_internet
    check_disk_space
    check_wsl
    install_system_deps
    check_gpu
    check_ollama
    setup_venv
    install_packages
    setup_huggingface
    verify_installation
    print_summary
}

main "$@"
