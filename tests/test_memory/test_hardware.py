"""Tests for hardware detection and tier classification."""

import pytest
from unittest.mock import patch, MagicMock

from song_to_video.memory.hardware import (
    GPUInfo,
    SystemInfo,
    HardwareTier,
    get_hardware_tier,
    is_gpu_compatible,
    detect_system_info,
)


class TestHardwareTier:
    """Tests for hardware tier classification."""

    def test_low_tier_8gb(self):
        """8GB GPU should be LOW tier."""
        gpu = GPUInfo(
            name="RTX 4060",
            vram_total_gb=8.0,
            vram_available_gb=7.5,
            compute_capability=(8, 9),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert get_hardware_tier(gpu) == HardwareTier.LOW

    def test_mid_tier_12gb(self):
        """12GB GPU should be MID tier."""
        gpu = GPUInfo(
            name="RTX 4070",
            vram_total_gb=12.0,
            vram_available_gb=11.0,
            compute_capability=(8, 9),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert get_hardware_tier(gpu) == HardwareTier.MID

    def test_mid_tier_16gb(self):
        """16GB GPU should be MID tier."""
        gpu = GPUInfo(
            name="RTX 4080",
            vram_total_gb=16.0,
            vram_available_gb=15.0,
            compute_capability=(8, 9),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert get_hardware_tier(gpu) == HardwareTier.MID

    def test_high_tier_24gb(self):
        """24GB GPU should be HIGH tier."""
        gpu = GPUInfo(
            name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=23.0,
            compute_capability=(8, 9),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert get_hardware_tier(gpu) == HardwareTier.HIGH

    def test_high_tier_32gb(self):
        """32GB GPU should be HIGH tier."""
        gpu = GPUInfo(
            name="RTX 5090",
            vram_total_gb=32.0,
            vram_available_gb=31.0,
            compute_capability=(10, 0),
            driver_version="560.0",
            cuda_version="12.4",
        )
        assert get_hardware_tier(gpu) == HardwareTier.HIGH

    @patch("song_to_video.memory.hardware.detect_gpu")
    def test_cpu_only_no_gpu(self, mock_detect):
        """No GPU should be CPU_ONLY tier."""
        mock_detect.return_value = None
        assert get_hardware_tier() == HardwareTier.CPU_ONLY


class TestGPUCompatibility:
    """Tests for GPU compatibility checks."""

    def test_compatible_rtx_4090(self):
        """RTX 4090 should be compatible."""
        gpu = GPUInfo(
            name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=23.0,
            compute_capability=(8, 9),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert is_gpu_compatible(gpu) is True

    def test_incompatible_old_gpu(self):
        """Old GPU with compute < 7.0 should not be compatible."""
        gpu = GPUInfo(
            name="GTX 1050",
            vram_total_gb=4.0,
            vram_available_gb=3.5,
            compute_capability=(6, 1),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert is_gpu_compatible(gpu) is False

    def test_incompatible_low_vram(self):
        """GPU with < 8GB VRAM should not be compatible."""
        gpu = GPUInfo(
            name="RTX 3050",
            vram_total_gb=4.0,
            vram_available_gb=3.5,
            compute_capability=(8, 6),
            driver_version="535.0",
            cuda_version="12.1",
        )
        assert is_gpu_compatible(gpu) is False

    @patch("song_to_video.memory.hardware.detect_gpu")
    def test_no_gpu(self, mock_detect):
        """No GPU should not be compatible."""
        mock_detect.return_value = None
        assert is_gpu_compatible() is False


class TestSystemDetection:
    """Tests for system info detection."""

    def test_detect_system_info_returns_info(self):
        """detect_system_info should return SystemInfo."""
        info = detect_system_info()
        assert isinstance(info, SystemInfo)
        assert info.ram_total_gb > 0
        assert info.ram_available_gb > 0
        assert info.ram_available_gb <= info.ram_total_gb
