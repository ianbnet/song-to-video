"""Memory monitoring and alerting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import psutil

from .vram import get_vram_usage, VRAMUsage

logger = logging.getLogger(__name__)


class WarningSeverity(Enum):
    """Severity levels for memory warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MemoryStatus:
    """Current memory status snapshot."""

    timestamp: datetime
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    vram: Optional[VRAMUsage]

    @property
    def vram_percent(self) -> Optional[float]:
        """VRAM usage percentage, if GPU available."""
        if self.vram is None:
            return None
        return self.vram.used_percent


@dataclass
class MemoryWarning:
    """A memory warning or alert."""

    severity: WarningSeverity
    message: str
    component: str  # "ram" or "vram"
    current_percent: float
    threshold_percent: float


@dataclass
class MemorySnapshot:
    """A recorded memory snapshot for history tracking."""

    timestamp: datetime
    phase: str
    ram_used_gb: float
    vram_used_gb: Optional[float]
    note: str = ""


class MemoryMonitor:
    """
    Monitor memory usage and provide warnings.

    Usage:
        monitor = MemoryMonitor()

        # Check current status
        status = monitor.get_status()
        print(f"RAM: {status.ram_percent:.1f}%")

        # Check for warnings
        warnings = monitor.check_health()
        for w in warnings:
            print(f"[{w.severity.value}] {w.message}")

        # Log snapshots during processing
        monitor.log_snapshot("transcription", "Whisper model loaded")
    """

    def __init__(
        self,
        ram_warning_threshold: float = 0.80,
        ram_critical_threshold: float = 0.95,
        vram_warning_threshold: float = 0.85,
        vram_critical_threshold: float = 0.95,
        max_history: int = 100,
    ):
        """
        Initialize the memory monitor.

        Args:
            ram_warning_threshold: RAM usage % to trigger warning (0-1)
            ram_critical_threshold: RAM usage % to trigger critical (0-1)
            vram_warning_threshold: VRAM usage % to trigger warning (0-1)
            vram_critical_threshold: VRAM usage % to trigger critical (0-1)
            max_history: Maximum number of snapshots to retain
        """
        self.ram_warning_threshold = ram_warning_threshold
        self.ram_critical_threshold = ram_critical_threshold
        self.vram_warning_threshold = vram_warning_threshold
        self.vram_critical_threshold = vram_critical_threshold
        self.max_history = max_history
        self._history: list[MemorySnapshot] = []

    def _bytes_to_gb(self, bytes_val: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_val / (1024**3)

    def get_status(self) -> MemoryStatus:
        """Get current memory status."""
        mem = psutil.virtual_memory()
        vram = get_vram_usage()

        return MemoryStatus(
            timestamp=datetime.now(),
            ram_total_gb=self._bytes_to_gb(mem.total),
            ram_used_gb=self._bytes_to_gb(mem.used),
            ram_available_gb=self._bytes_to_gb(mem.available),
            ram_percent=mem.percent,
            vram=vram,
        )

    def check_health(self) -> list[MemoryWarning]:
        """
        Check memory health and return any warnings.

        Returns:
            List of MemoryWarning objects (empty if all healthy)
        """
        warnings: list[MemoryWarning] = []
        status = self.get_status()

        # Check RAM
        ram_ratio = status.ram_percent / 100.0
        if ram_ratio >= self.ram_critical_threshold:
            warnings.append(
                MemoryWarning(
                    severity=WarningSeverity.CRITICAL,
                    message=f"System RAM critically low: {status.ram_percent:.1f}% used",
                    component="ram",
                    current_percent=status.ram_percent,
                    threshold_percent=self.ram_critical_threshold * 100,
                )
            )
        elif ram_ratio >= self.ram_warning_threshold:
            warnings.append(
                MemoryWarning(
                    severity=WarningSeverity.WARNING,
                    message=f"System RAM usage high: {status.ram_percent:.1f}% used",
                    component="ram",
                    current_percent=status.ram_percent,
                    threshold_percent=self.ram_warning_threshold * 100,
                )
            )

        # Check VRAM
        if status.vram is not None:
            vram_ratio = status.vram.used_percent / 100.0
            if vram_ratio >= self.vram_critical_threshold:
                warnings.append(
                    MemoryWarning(
                        severity=WarningSeverity.CRITICAL,
                        message=f"VRAM critically low: {status.vram.used_percent:.1f}% used",
                        component="vram",
                        current_percent=status.vram.used_percent,
                        threshold_percent=self.vram_critical_threshold * 100,
                    )
                )
            elif vram_ratio >= self.vram_warning_threshold:
                warnings.append(
                    MemoryWarning(
                        severity=WarningSeverity.WARNING,
                        message=f"VRAM usage high: {status.vram.used_percent:.1f}% used",
                        component="vram",
                        current_percent=status.vram.used_percent,
                        threshold_percent=self.vram_warning_threshold * 100,
                    )
                )

        return warnings

    def log_snapshot(self, phase: str, note: str = "") -> MemorySnapshot:
        """
        Log a memory snapshot for the current phase.

        Args:
            phase: Name of the current pipeline phase
            note: Optional note about what's happening

        Returns:
            The created MemorySnapshot
        """
        status = self.get_status()

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            phase=phase,
            ram_used_gb=status.ram_used_gb,
            vram_used_gb=status.vram.used_gb if status.vram else None,
            note=note,
        )

        self._history.append(snapshot)

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        # Log to standard logging
        vram_str = f", VRAM: {snapshot.vram_used_gb:.2f}GB" if snapshot.vram_used_gb else ""
        logger.debug(
            f"[{phase}] RAM: {snapshot.ram_used_gb:.2f}GB{vram_str}"
            + (f" - {note}" if note else "")
        )

        return snapshot

    def get_history(self, phase: Optional[str] = None) -> list[MemorySnapshot]:
        """
        Get memory history, optionally filtered by phase.

        Args:
            phase: If provided, only return snapshots for this phase

        Returns:
            List of MemorySnapshot objects
        """
        if phase is None:
            return list(self._history)
        return [s for s in self._history if s.phase == phase]

    def clear_history(self) -> None:
        """Clear all recorded snapshots."""
        self._history.clear()

    def get_peak_usage(self, phase: Optional[str] = None) -> dict[str, float]:
        """
        Get peak memory usage from history.

        Args:
            phase: If provided, only consider this phase

        Returns:
            Dictionary with peak RAM and VRAM usage in GB
        """
        history = self.get_history(phase)

        if not history:
            return {"ram_peak_gb": 0.0, "vram_peak_gb": 0.0}

        ram_peak = max(s.ram_used_gb for s in history)
        vram_values = [s.vram_used_gb for s in history if s.vram_used_gb is not None]
        vram_peak = max(vram_values) if vram_values else 0.0

        return {"ram_peak_gb": ram_peak, "vram_peak_gb": vram_peak}


# Global monitor instance for convenience
_global_monitor: Optional[MemoryMonitor] = None


def get_monitor() -> MemoryMonitor:
    """Get or create the global memory monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor
