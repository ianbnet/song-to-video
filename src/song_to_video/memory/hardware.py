"""Hardware detection and tier classification."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil


@dataclass
class GPUInfo:
    """Information about the detected GPU."""

    name: str
    vram_total_gb: float
    vram_available_gb: float
    compute_capability: tuple[int, int]
    driver_version: str
    cuda_version: str


@dataclass
class SystemInfo:
    """Complete system hardware information."""

    ram_total_gb: float
    ram_available_gb: float
    gpu: Optional[GPUInfo]


class HardwareTier(Enum):
    """Hardware tier classification based on available VRAM."""

    LOW = "low"  # 8GB VRAM - aggressive offloading
    MID = "mid"  # 12-16GB VRAM - moderate offloading
    HIGH = "high"  # 24GB+ VRAM - minimal offloading
    CPU_ONLY = "cpu"  # No compatible GPU


def _bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_val / (1024**3)


def detect_gpu() -> Optional[GPUInfo]:
    """
    Detect NVIDIA GPU and return its specifications.

    Returns None if no compatible GPU is found.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        # Get device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        # Get memory info
        total_mem = props.total_memory
        # Get current available memory
        free_mem, _ = torch.cuda.mem_get_info(device)

        # Get driver/CUDA versions
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception:
            driver_version = "unknown"

        cuda_version = torch.version.cuda or "unknown"

        return GPUInfo(
            name=props.name,
            vram_total_gb=_bytes_to_gb(total_mem),
            vram_available_gb=_bytes_to_gb(free_mem),
            compute_capability=(props.major, props.minor),
            driver_version=driver_version,
            cuda_version=cuda_version,
        )

    except ImportError:
        return None
    except Exception:
        return None


def detect_system_info() -> SystemInfo:
    """Detect complete system hardware information."""
    mem = psutil.virtual_memory()

    return SystemInfo(
        ram_total_gb=_bytes_to_gb(mem.total),
        ram_available_gb=_bytes_to_gb(mem.available),
        gpu=detect_gpu(),
    )


def get_hardware_tier(gpu: Optional[GPUInfo] = None) -> HardwareTier:
    """
    Classify hardware into a tier based on GPU VRAM.

    Tier thresholds:
    - LOW: VRAM < 10GB (e.g., RTX 4060 8GB)
    - MID: 10GB <= VRAM < 20GB (e.g., RTX 4070 12GB, RTX 4080 16GB)
    - HIGH: VRAM >= 20GB (e.g., RTX 4090 24GB, RTX 5090 32GB)
    - CPU_ONLY: No compatible GPU
    """
    if gpu is None:
        gpu = detect_gpu()

    if gpu is None:
        return HardwareTier.CPU_ONLY

    vram = gpu.vram_total_gb

    if vram < 10:
        return HardwareTier.LOW
    elif vram < 20:
        return HardwareTier.MID
    else:
        return HardwareTier.HIGH


def is_gpu_compatible(gpu: Optional[GPUInfo] = None) -> bool:
    """
    Check if the GPU is compatible with song-to-video.

    Requires:
    - CUDA compute capability >= 7.0 (Volta or newer)
    - At least 8GB VRAM
    """
    if gpu is None:
        gpu = detect_gpu()

    if gpu is None:
        return False

    # Check compute capability (7.0+ for Tensor Cores)
    major, minor = gpu.compute_capability
    if major < 7:
        return False

    # Check minimum VRAM
    if gpu.vram_total_gb < 7.5:  # Allow some headroom below 8GB
        return False

    return True
