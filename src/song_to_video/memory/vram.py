"""VRAM flush and management utilities."""

import gc
from dataclasses import dataclass
from typing import Optional


@dataclass
class VRAMUsage:
    """Current VRAM usage statistics."""

    total_gb: float
    used_gb: float
    free_gb: float
    allocated_gb: float  # PyTorch allocated
    reserved_gb: float  # PyTorch reserved (cached)

    @property
    def used_percent(self) -> float:
        """Percentage of VRAM currently used."""
        if self.total_gb == 0:
            return 0.0
        return (self.used_gb / self.total_gb) * 100


# Known model VRAM requirements (approximate, in GB)
MODEL_VRAM_ESTIMATES: dict[str, float] = {
    # Whisper models
    "whisper-tiny": 1.0,
    "whisper-base": 1.5,
    "whisper-small": 2.5,
    "whisper-medium": 5.0,
    "whisper-large": 10.0,
    "whisper-large-v3": 10.0,
    # LLMs (quantized)
    "llama-3.1-8b-q4": 5.0,
    "llama-3.1-14b-q4": 8.0,
    "llama-3.1-14b-q8": 14.0,
    "mistral-nemo-12b-q4": 7.0,
    "gemma-3-12b-q4": 7.0,
    # Image generation
    "flux-schnell": 12.0,
    "flux-dev": 16.0,
    "sdxl": 8.0,
    # Video generation
    "ltx-2-distilled": 8.0,
    "ltx-2-full": 16.0,
    "wan-2.2-5b": 8.0,
    "wan-2.2-14b": 16.0,
    "animatediff-lightning": 6.0,
}


def _bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_val / (1024**3)


def get_vram_usage() -> Optional[VRAMUsage]:
    """
    Get current VRAM usage statistics.

    Returns None if no GPU is available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()

        # Get total and free memory from CUDA
        free_mem, total_mem = torch.cuda.mem_get_info(device)

        # Get PyTorch-specific allocations
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        total_gb = _bytes_to_gb(total_mem)
        free_gb = _bytes_to_gb(free_mem)

        return VRAMUsage(
            total_gb=total_gb,
            used_gb=total_gb - free_gb,
            free_gb=free_gb,
            allocated_gb=_bytes_to_gb(allocated),
            reserved_gb=_bytes_to_gb(reserved),
        )

    except ImportError:
        return None
    except Exception:
        return None


def flush_vram(force_gc: bool = True) -> dict[str, float]:
    """
    Perform a complete VRAM flush between pipeline phases.

    This should be called after unloading models to ensure all GPU memory
    is properly freed. The caller is responsible for deleting model references
    before calling this function.

    Args:
        force_gc: Whether to force garbage collection (recommended)

    Returns:
        Dictionary with before/after memory stats in GB
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"status": "no_gpu"}

        # Get memory before flush
        before = get_vram_usage()
        before_used = before.used_gb if before else 0.0

        # Force garbage collection first
        if force_gc:
            gc.collect()

        # Clear PyTorch CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()

        # Force another GC pass
        if force_gc:
            gc.collect()

        # Get memory after flush
        after = get_vram_usage()
        after_used = after.used_gb if after else 0.0

        return {
            "before_gb": before_used,
            "after_gb": after_used,
            "freed_gb": before_used - after_used,
            "status": "success",
        }

    except ImportError:
        return {"status": "torch_not_available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def require_vram(gb: float, buffer: float = 0.5) -> bool:
    """
    Check if the required VRAM is available.

    Args:
        gb: Required VRAM in gigabytes
        buffer: Extra buffer to maintain (default 0.5 GB)

    Returns:
        True if enough VRAM is available, False otherwise
    """
    usage = get_vram_usage()
    if usage is None:
        return False

    return usage.free_gb >= (gb + buffer)


def estimate_model_vram(model_name: str) -> Optional[float]:
    """
    Estimate VRAM requirement for a known model.

    Args:
        model_name: Name of the model (see MODEL_VRAM_ESTIMATES)

    Returns:
        Estimated VRAM in GB, or None if model is unknown
    """
    # Normalize model name
    normalized = model_name.lower().replace("_", "-").replace(" ", "-")

    # Check exact match first
    if normalized in MODEL_VRAM_ESTIMATES:
        return MODEL_VRAM_ESTIMATES[normalized]

    # Check partial matches
    for key, value in MODEL_VRAM_ESTIMATES.items():
        if key in normalized or normalized in key:
            return value

    return None


def can_load_model(model_name: str, buffer: float = 0.5) -> tuple[bool, str]:
    """
    Check if a model can be loaded with available VRAM.

    Args:
        model_name: Name of the model to check
        buffer: Extra buffer to maintain

    Returns:
        Tuple of (can_load, message)
    """
    estimate = estimate_model_vram(model_name)
    if estimate is None:
        return False, f"Unknown model: {model_name}"

    usage = get_vram_usage()
    if usage is None:
        return False, "No GPU available"

    if usage.free_gb >= (estimate + buffer):
        return True, f"OK: {estimate:.1f} GB required, {usage.free_gb:.1f} GB available"
    else:
        return False, (
            f"Insufficient VRAM: {estimate:.1f} GB required, "
            f"{usage.free_gb:.1f} GB available"
        )
