"""Tests for phase isolation."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from song_to_video.memory.phase import (
    PipelinePhase,
    PhaseViolationError,
    InsufficientVRAMError,
    phase,
    get_current_phase,
    is_phase_locked,
    force_unlock,
    LOCK_FILE,
)


class TestPipelinePhase:
    """Tests for PipelinePhase context manager."""

    def setup_method(self):
        """Clean up lock file before each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def teardown_method(self):
        """Clean up lock file after each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def test_basic_phase_context(self):
        """Basic phase context manager should work."""
        assert not is_phase_locked()

        with PipelinePhase("test_phase"):
            assert is_phase_locked()
            assert get_current_phase() == "test_phase"

        assert not is_phase_locked()

    def test_phase_locks_file(self):
        """Phase should create and lock file."""
        with PipelinePhase("lock_test"):
            assert LOCK_FILE.exists()
            content = LOCK_FILE.read_text()
            assert "lock_test" in content
            assert f"pid: {os.getpid()}" in content

    def test_phase_cleans_up_on_exit(self):
        """Phase should release lock on normal exit."""
        with PipelinePhase("cleanup_test"):
            pass

        # Lock should be released (another phase should be able to start)
        with PipelinePhase("second_phase"):
            pass

    def test_phase_cleans_up_on_exception(self):
        """Phase should release lock even if exception occurs."""
        with pytest.raises(ValueError):
            with PipelinePhase("exception_test"):
                raise ValueError("Test error")

        # Lock should be released
        assert not is_phase_locked()

    @patch("song_to_video.memory.phase.require_vram")
    def test_insufficient_vram_raises(self, mock_require):
        """Should raise if VRAM requirement not met."""
        mock_require.return_value = False

        with pytest.raises(InsufficientVRAMError):
            with PipelinePhase("vram_test", required_vram_gb=100.0):
                pass

        # Lock should be released after error
        assert not is_phase_locked()


class TestPhaseDecorator:
    """Tests for @phase decorator."""

    def setup_method(self):
        """Clean up lock file before each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def teardown_method(self):
        """Clean up lock file after each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def test_decorator_wraps_function(self):
        """@phase decorator should wrap function in phase context."""

        @phase("decorated_test")
        def my_function():
            assert is_phase_locked()
            assert get_current_phase() == "decorated_test"
            return "result"

        assert not is_phase_locked()
        result = my_function()
        assert result == "result"
        assert not is_phase_locked()

    def test_decorator_preserves_function_name(self):
        """Decorator should preserve function metadata."""

        @phase("meta_test")
        def original_name():
            """Original docstring."""
            pass

        assert original_name.__name__ == "original_name"
        assert original_name.__doc__ == "Original docstring."


class TestForceUnlock:
    """Tests for force_unlock utility."""

    def setup_method(self):
        """Clean up lock file before each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def teardown_method(self):
        """Clean up lock file after each test."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def test_force_unlock_removes_file(self):
        """force_unlock should remove the lock file."""
        # Create a stale lock file
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text("stale_phase\npid: 99999\n")

        assert LOCK_FILE.exists()
        result = force_unlock()
        assert result is True
        assert not LOCK_FILE.exists()

    def test_force_unlock_no_lock(self):
        """force_unlock should return False if no lock exists."""
        assert not LOCK_FILE.exists()
        result = force_unlock()
        assert result is False
