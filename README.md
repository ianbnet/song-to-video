# Song-to-Video

Transform songs into AI-generated music videos using local AI models.

## Features

- **Fully Local**: All AI processing runs on your machine - no cloud APIs required
- **Automatic Lyrics**: Transcribes vocals using Whisper AI
- **Smart Scene Planning**: LLM analyzes lyrics to create visual narrative and prompts
- **Consistent Style**: Master seed ensures visual coherence across all scenes
- **Hardware Adaptive**: Automatically optimizes for your GPU (8GB to 24GB+ VRAM)
- **Preview Mode**: Review storyboard before committing to video generation

## Requirements

- **OS**: WSL2 on Windows 10/11 (or native Linux)
- **Python**: 3.11 or newer
- **GPU**: NVIDIA RTX 3060+ recommended (8GB+ VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 50-100GB for AI models (downloaded on first use)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ianbnet/song-to-video.git
cd song-to-video
./install.sh
```

### 2. Start Ollama

```bash
# In a separate terminal
ollama serve

# Pull the LLM model (one time)
ollama pull llama3.1:8b
```

### 3. Generate a Video

```bash
source .venv/bin/activate
song-to-video generate song.mp3 --output video.mp4
```

## Installation Options

The installer supports environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTALL_OLLAMA` | `prompt` | `auto`, `prompt`, or `skip` |
| `EDITABLE_INSTALL` | `true` | Use editable pip install for development |
| `SKIP_SYSTEM_DEPS` | `false` | Skip apt package installation |

Example:
```bash
INSTALL_OLLAMA=auto ./install.sh
```

## Commands

### Generate Video (Full Pipeline)

```bash
# Basic usage
song-to-video generate song.mp3 --output video.mp4

# With options
song-to-video generate song.mp3 \
    --output video.mp4 \
    --quality standard \
    --lyrics lyrics.srt \
    --seed 42
```

### Preview Mode

Stop after frame generation to review the storyboard before creating video:

```bash
song-to-video generate song.mp3 --preview
# Opens storyboard.html for review
# Run again without --preview to complete
```

### Individual Pipeline Steps

```bash
# Check hardware capabilities
song-to-video hardware

# Check memory status
song-to-video memory --diagnostics

# Transcribe lyrics only
song-to-video transcribe song.mp3 --output lyrics.json

# Analyze audio features
song-to-video analyze song.mp3

# Create scene plan
song-to-video plan song.mp3 --output plan.json

# Generate reference frames
song-to-video frames plan.json --output ./frames

# Generate video clips
song-to-video clips plan.json --frames ./frames --output ./clips

# Compose final video
song-to-video compose plan.json ./clips song.mp3 --output video.mp4

# Generate storyboard preview
song-to-video preview plan.json --frames ./frames --output storyboard.html
```

## Hardware Tiers

The application automatically detects your hardware and optimizes settings:

| Tier | VRAM | Configuration |
|------|------|---------------|
| LOW | 8GB | 480p, aggressive offloading, 4-bit quantization |
| MID | 12-16GB | 720p, moderate offloading, 8-bit quantization |
| HIGH | 24GB+ | 1080p, full precision |
| CPU | None | CPU-only mode (very slow) |

## AI Models

Models download automatically on first use to `~/.cache/huggingface/`:

| Model | Size | Purpose |
|-------|------|---------|
| Whisper | 0.5-3GB | Audio transcription |
| Flux Schnell / SDXL | 10-15GB | Reference image generation |
| LTX-Video | 15-20GB | Video clip generation |
| Llama 3.1 8B | ~5GB | Scene planning (via Ollama) |

## Pipeline Overview

```
MP3 Audio
    │
    ├─► Transcribe (Whisper) ─► Lyrics with timestamps
    │
    ├─► Analyze (Librosa) ─► Tempo, sections, energy, mood
    │
    └─► Plan (Ollama LLM) ─► Scene descriptions + prompts
            │
            ├─► Generate Frames (Flux/SDXL) ─► Reference images
            │
            ├─► [Optional] Preview storyboard.html
            │
            ├─► Generate Clips (LTX-Video) ─► Video segments
            │
            └─► Compose (FFmpeg) ─► Final video with audio
```

## Troubleshooting

### "Ollama is not running"

```bash
# Start Ollama in a separate terminal
ollama serve

# Verify it's running
curl http://localhost:11434/api/version
```

### Out of VRAM

```bash
# Use lower quality preset
song-to-video generate song.mp3 --quality draft

# Check memory status
song-to-video memory --diagnostics

# Flush VRAM
song-to-video memory --flush
```

### GPU Not Detected in WSL2

Ensure you have the Windows NVIDIA driver installed (not the Linux driver).
The driver passes through to WSL2 automatically.

```bash
# Verify GPU access
nvidia-smi
```

### Models Won't Download

Check your internet connection and HuggingFace access:

```bash
# Test HuggingFace connection
curl -I https://huggingface.co
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/
```

## Project Structure

```
song-to-video/
├── src/song_to_video/
│   ├── audio/        # Transcription and analysis
│   ├── planning/     # LLM scene planning
│   ├── image/        # Reference frame generation
│   ├── video/        # Video generation and composition
│   ├── memory/       # GPU/VRAM management
│   └── cli.py        # Command-line interface
├── tests/            # Test suite
├── install.sh        # Installation script
└── pyproject.toml    # Package configuration
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built with:
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for transcription
- [Diffusers](https://github.com/huggingface/diffusers) for image/video generation
- [Ollama](https://ollama.ai/) for local LLM inference
- [FFmpeg](https://ffmpeg.org/) for video encoding
