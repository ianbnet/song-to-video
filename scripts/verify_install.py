#!/usr/bin/env python3
"""Post-installation verification for song-to-video."""

import sys
import shutil


def check_python_version():
    """Verify Python version."""
    if sys.version_info < (3, 11):
        print(f"FAIL: Python {sys.version_info.major}.{sys.version_info.minor} < 3.11")
        return False
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_core_imports():
    """Verify core packages can be imported."""
    packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("faster_whisper", "Faster Whisper"),
        ("librosa", "Librosa"),
        ("typer", "Typer"),
        ("rich", "Rich"),
    ]

    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"OK: {name}")
        except ImportError as e:
            print(f"FAIL: {name} - {e}")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"OK: CUDA available - {device_name} ({vram:.1f}GB)")
            return True
        else:
            print("WARN: CUDA not available (CPU mode)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"WARN: CUDA check failed - {e}")
        return True


def check_ffmpeg():
    """Check FFmpeg is available."""
    if shutil.which("ffmpeg"):
        print("OK: FFmpeg found")
        return True
    else:
        print("FAIL: FFmpeg not found")
        return False


def check_ollama():
    """Check Ollama availability."""
    import urllib.request
    import urllib.error

    if not shutil.which("ollama"):
        print("WARN: Ollama not installed")
        return True  # Warning only

    try:
        req = urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        print("OK: Ollama server running")
        return True
    except (urllib.error.URLError, TimeoutError):
        print("WARN: Ollama installed but not running")
        return True


def check_song_to_video():
    """Check song-to-video package."""
    try:
        from song_to_video import __version__

        print(f"OK: song-to-video v{__version__}")

        from song_to_video.memory import detect_system_info, get_hardware_tier

        info = detect_system_info()
        tier = get_hardware_tier(info.gpu)
        print(f"OK: Hardware tier: {tier.value}")

        return True
    except Exception as e:
        print(f"FAIL: song-to-video - {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Song-to-Video Installation Verification")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Core Packages", check_core_imports),
        ("CUDA/GPU", check_cuda),
        ("FFmpeg", check_ffmpeg),
        ("Ollama", check_ollama),
        ("Song-to-Video", check_song_to_video),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n[{name}]")
        if not check_fn():
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("All checks passed!")
        return 0
    else:
        print("Some checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
