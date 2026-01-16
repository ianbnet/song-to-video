"""RAM offloading configuration for different hardware tiers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hardware import HardwareTier


@dataclass
class OffloadConfig:
    """Configuration for model offloading to CPU RAM or disk."""

    enable_cpu_offload: bool
    enable_disk_offload: bool
    max_gpu_memory_gb: Optional[float]
    max_cpu_memory_gb: Optional[float]
    offload_folder: Optional[Path]

    # Quantization settings
    use_8bit: bool = False
    use_4bit: bool = False

    # Attention slicing for memory efficiency
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False

    # Sequential CPU offload (slower but uses less VRAM)
    enable_sequential_offload: bool = False


# Default offload folder
DEFAULT_OFFLOAD_FOLDER = Path.home() / ".cache" / "song-to-video" / "offload"


def get_offload_config(
    tier: HardwareTier,
    system_ram_gb: float = 64.0,
) -> OffloadConfig:
    """
    Get the appropriate offload configuration for a hardware tier.

    Args:
        tier: The detected hardware tier
        system_ram_gb: Available system RAM in GB

    Returns:
        OffloadConfig with appropriate settings
    """
    if tier == HardwareTier.LOW:
        # 8GB VRAM: Aggressive offloading required
        return OffloadConfig(
            enable_cpu_offload=True,
            enable_disk_offload=system_ram_gb < 32,  # Use disk if low RAM
            max_gpu_memory_gb=6.0,  # Leave 2GB headroom
            max_cpu_memory_gb=min(32.0, system_ram_gb * 0.5),
            offload_folder=DEFAULT_OFFLOAD_FOLDER,
            use_4bit=True,  # Use 4-bit quantization
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_sequential_offload=True,
        )

    elif tier == HardwareTier.MID:
        # 12-16GB VRAM: Moderate offloading
        return OffloadConfig(
            enable_cpu_offload=True,
            enable_disk_offload=False,
            max_gpu_memory_gb=12.0,  # Use most of GPU
            max_cpu_memory_gb=min(32.0, system_ram_gb * 0.4),
            offload_folder=DEFAULT_OFFLOAD_FOLDER,
            use_8bit=True,  # Use 8-bit quantization
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            enable_vae_tiling=False,
            enable_sequential_offload=False,
        )

    elif tier == HardwareTier.HIGH:
        # 24GB+ VRAM: Minimal offloading
        return OffloadConfig(
            enable_cpu_offload=False,
            enable_disk_offload=False,
            max_gpu_memory_gb=None,  # Use full GPU
            max_cpu_memory_gb=None,
            offload_folder=None,
            use_8bit=False,  # Full precision
            use_4bit=False,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            enable_sequential_offload=False,
        )

    else:  # CPU_ONLY
        # No GPU: Full CPU mode
        return OffloadConfig(
            enable_cpu_offload=False,  # Already on CPU
            enable_disk_offload=system_ram_gb < 32,
            max_gpu_memory_gb=0,
            max_cpu_memory_gb=min(48.0, system_ram_gb * 0.7),
            offload_folder=DEFAULT_OFFLOAD_FOLDER,
            use_4bit=True,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_sequential_offload=True,
        )


def ensure_offload_folder(config: OffloadConfig) -> None:
    """Create the offload folder if it doesn't exist."""
    if config.offload_folder is not None:
        config.offload_folder.mkdir(parents=True, exist_ok=True)


def get_accelerate_config(config: OffloadConfig) -> dict:
    """
    Generate configuration dict for HuggingFace Accelerate.

    This can be passed to model loading functions that support Accelerate.
    """
    accelerate_config = {}

    if config.max_gpu_memory_gb is not None:
        accelerate_config["max_memory"] = {
            0: f"{config.max_gpu_memory_gb}GB",
        }

    if config.enable_cpu_offload and config.max_cpu_memory_gb is not None:
        accelerate_config["max_memory"]["cpu"] = f"{config.max_cpu_memory_gb}GB"

    if config.enable_disk_offload and config.offload_folder is not None:
        accelerate_config["offload_folder"] = str(config.offload_folder)

    if config.use_4bit:
        accelerate_config["load_in_4bit"] = True
    elif config.use_8bit:
        accelerate_config["load_in_8bit"] = True

    return accelerate_config


def get_diffusers_config(config: OffloadConfig) -> dict:
    """
    Generate configuration dict for Diffusers pipelines.

    This can be used when loading Stable Diffusion, Flux, or video models.
    """
    diffusers_config = {}

    if config.enable_sequential_offload:
        diffusers_config["enable_sequential_cpu_offload"] = True
    elif config.enable_cpu_offload:
        diffusers_config["enable_model_cpu_offload"] = True

    if config.enable_attention_slicing:
        diffusers_config["enable_attention_slicing"] = True

    if config.enable_vae_slicing:
        diffusers_config["enable_vae_slicing"] = True

    if config.enable_vae_tiling:
        diffusers_config["enable_vae_tiling"] = True

    return diffusers_config
