"""Memory management module for song-to-video."""

from .hardware import (
    GPUInfo,
    SystemInfo,
    HardwareTier,
    detect_gpu,
    detect_system_info,
    get_hardware_tier,
    is_gpu_compatible,
)
from .vram import (
    flush_vram,
    get_vram_usage,
    require_vram,
)
from .offload import (
    OffloadConfig,
    get_offload_config,
)
from .monitor import (
    MemoryStatus,
    MemoryWarning,
    MemoryMonitor,
)
from .phase import (
    PipelinePhase,
    PhaseViolationError,
    InsufficientVRAMError,
    phase,
    get_current_phase,
    is_phase_locked,
    force_unlock,
)

__all__ = [
    # hardware
    "GPUInfo",
    "SystemInfo",
    "HardwareTier",
    "detect_gpu",
    "detect_system_info",
    "get_hardware_tier",
    "is_gpu_compatible",
    # vram
    "flush_vram",
    "get_vram_usage",
    "require_vram",
    # offload
    "OffloadConfig",
    "get_offload_config",
    # monitor
    "MemoryStatus",
    "MemoryWarning",
    "MemoryMonitor",
    # phase
    "PipelinePhase",
    "PhaseViolationError",
    "InsufficientVRAMError",
    "phase",
    "get_current_phase",
    "is_phase_locked",
    "force_unlock",
]
