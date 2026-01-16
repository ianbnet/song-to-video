#!/usr/bin/env bash
#
# Song-to-Video Installation Script
# Installs dependencies and sets up the development environment
#
# Usage:
#   ./install.sh                       # Interactive installation
#   INSTALL_OLLAMA=auto ./install.sh   # Auto-install Ollama
#   INSTALL_OLLAMA=skip ./install.sh   # Skip Ollama check
#
set -euo pipefail

# Configuration
PYTHON_MIN_VERSION="3.11"
VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# State variables
GPU_AVAILABLE=false
HARDWARE_TIER="cpu"
OLLAMA_INSTALLED=false
OLLAMA_RUNNING=false

print_banner() {
    echo -e "${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    Song-to-Video Installer                    ║"
    echo "║           Transform songs into AI-generated videos            ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_wsl() {
    log_info "Checking environment..."
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log_ok "Running in WSL2"
    else
        log_warn "Not running in WSL (installation may still work)"
    fi
}

check_python() {
    log_info "Checking Python..."

    if ! command -v python3 &>/dev/null; then
        log_error "Python 3 not found"
        log_info "Install with: sudo apt install python3 python3-venv"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        log_error "Python >= 3.11 required (found $PYTHON_VERSION)"
        exit 1
    fi

    if ! python3 -c "import venv" &>/dev/null; then
        log_error "Python venv module not found"
        log_info "Install with: sudo apt install python3-venv"
        exit 1
    fi

    log_ok "Python $PYTHON_VERSION"
}

check_ffmpeg() {
    log_info "Checking FFmpeg..."

    if command -v ffmpeg &>/dev/null; then
        FFMPEG_VERSION=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}' || echo "unknown")
        log_ok "FFmpeg $FFMPEG_VERSION"
        return 0
    fi

    log_warn "FFmpeg not found"

    if [[ "${SKIP_SYSTEM_DEPS:-false}" == "true" ]]; then
        log_warn "Skipping FFmpeg install (SKIP_SYSTEM_DEPS=true)"
        return 1
    fi

    log_info "Installing FFmpeg (requires sudo)..."
    if sudo apt-get update && sudo apt-get install -y ffmpeg; then
        log_ok "FFmpeg installed"
        return 0
    else
        log_error "FFmpeg installation failed"
        log_info "Install manually: sudo apt install ffmpeg"
        return 1
    fi
}

check_gpu() {
    log_info "Checking GPU..."

    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found - GPU not available"
        GPU_AVAILABLE=false
        HARDWARE_TIER="cpu"
        return 0
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)

    if [[ -z "$GPU_INFO" ]]; then
        log_warn "No NVIDIA GPU detected"
        GPU_AVAILABLE=false
        HARDWARE_TIER="cpu"
        return 0
    fi

    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_VRAM=$(echo "$GPU_INFO" | cut -d',' -f2 | grep -oE '[0-9]+' || echo "0")

    GPU_AVAILABLE=true
    log_ok "GPU: $GPU_NAME (${GPU_VRAM} MiB)"

    if [[ $GPU_VRAM -lt 10000 ]]; then
        HARDWARE_TIER="low"
        log_info "Hardware tier: LOW (aggressive offloading)"
    elif [[ $GPU_VRAM -lt 20000 ]]; then
        HARDWARE_TIER="mid"
        log_info "Hardware tier: MID (moderate offloading)"
    else
        HARDWARE_TIER="high"
        log_info "Hardware tier: HIGH (full precision)"
    fi
}

check_ollama() {
    log_info "Checking Ollama..."

    if command -v ollama &>/dev/null; then
        OLLAMA_INSTALLED=true
        OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
        log_ok "Ollama $OLLAMA_VERSION installed"

        if curl -s --connect-timeout 2 http://localhost:11434/api/version &>/dev/null; then
            OLLAMA_RUNNING=true
            log_ok "Ollama server running"

            MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | head -5 | tr '\n' ' ' || true)
            if [[ -n "$MODELS" ]]; then
                log_info "Models: $MODELS"
            fi
        else
            log_warn "Ollama not running - start with: ollama serve"
        fi
        return 0
    fi

    log_warn "Ollama not found (required for scene planning)"

    case "${INSTALL_OLLAMA:-prompt}" in
        auto)
            install_ollama
            ;;
        prompt)
            echo ""
            read -p "Install Ollama now? [y/N] " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_ollama
            else
                log_info "Skipping - install later: curl -fsSL https://ollama.com/install.sh | sh"
            fi
            ;;
        skip)
            log_info "Skipping (INSTALL_OLLAMA=skip)"
            ;;
    esac
}

install_ollama() {
    log_info "Installing Ollama..."

    # Check for zstd (required for Ollama extraction)
    if ! command -v zstd &>/dev/null; then
        log_info "Installing zstd (required for Ollama)..."
        if ! sudo apt-get install -y zstd; then
            log_error "Failed to install zstd"
            log_info "Install manually: sudo apt install zstd"
            return 1
        fi
        log_ok "zstd installed"
    fi

    # Install Ollama
    if curl -fsSL https://ollama.com/install.sh | sh; then
        OLLAMA_INSTALLED=true
        log_ok "Ollama installed"
        log_info "Start with: ollama serve"
        log_info "Then pull a model: ollama pull llama3.1:8b"
    else
        log_error "Ollama installation failed"
        log_info "Try installing manually: curl -fsSL https://ollama.com/install.sh | sh"
        return 1
    fi
}

setup_venv() {
    log_info "Setting up virtual environment..."

    if [[ -d "$VENV_DIR" ]]; then
        log_warn "Virtual environment exists"
        read -p "Recreate? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        fi
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    log_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools -q

    log_ok "Virtual environment ready"
}

install_packages() {
    log_info "Installing Python packages (this may take a few minutes)..."

    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log_info "Installing PyTorch with CUDA..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    else
        log_info "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi

    if [[ "${EDITABLE_INSTALL:-true}" == "true" ]]; then
        log_info "Installing song-to-video (editable)..."
        pip install -e ".[dev]" -q
    else
        log_info "Installing song-to-video..."
        pip install ".[dev]" -q
    fi

    log_ok "Packages installed"
}

verify_installation() {
    log_info "Verifying installation..."

    if ! command -v song-to-video &>/dev/null; then
        log_error "song-to-video command not found"
        return 1
    fi

    song-to-video version
    log_ok "Installation verified"

    if [[ -f "$SCRIPT_DIR/scripts/verify_install.py" ]]; then
        log_info "Running verification script..."
        python3 "$SCRIPT_DIR/scripts/verify_install.py"
    fi
}

print_summary() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}Installation Complete!${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Hardware Tier: $HARDWARE_TIER"
    echo ""
    echo -e "${BOLD}NEXT STEPS:${NC}"
    echo ""
    echo "1. Activate the environment:"
    echo "   source .venv/bin/activate"
    echo ""

    if [[ "$OLLAMA_RUNNING" != "true" ]]; then
        echo "2. Start Ollama (in another terminal):"
        echo "   ollama serve"
        echo ""
        echo "3. Pull an LLM model:"
        echo "   ollama pull llama3.1:8b"
        echo ""
    fi

    echo "4. Generate your first video:"
    echo "   song-to-video generate song.mp3 --output video.mp4"
    echo ""
    echo -e "${YELLOW}NOTE:${NC} AI models download automatically on first use:"
    echo "  - Whisper: ~1-3 GB"
    echo "  - Flux/SDXL: ~10-15 GB"
    echo "  - LTX-Video: ~15-20 GB"
    echo ""
    echo "For help: song-to-video --help"
    echo ""
}

# Main execution
main() {
    cd "$SCRIPT_DIR"

    print_banner
    check_wsl
    check_python
    check_ffmpeg
    check_gpu
    check_ollama
    setup_venv
    install_packages
    verify_installation
    print_summary
}

main "$@"
