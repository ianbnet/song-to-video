"""Pipeline phase isolation with process-level locking."""

import fcntl
import functools
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .monitor import get_monitor
from .vram import flush_vram, get_vram_usage, require_vram

logger = logging.getLogger(__name__)

# Lock file location
LOCK_DIR = Path.home() / ".cache" / "song-to-video"
LOCK_FILE = LOCK_DIR / "phase.lock"


class PhaseViolationError(Exception):
    """Raised when phase isolation rules are violated."""

    pass


class InsufficientVRAMError(Exception):
    """Raised when there isn't enough VRAM for a phase."""

    pass


class PipelinePhase:
    """
    Context manager for pipeline phase isolation.

    Ensures that:
    1. Only one phase runs at a time (process-level lock)
    2. VRAM is flushed on entry and exit
    3. Memory is monitored throughout

    Usage:
        with PipelinePhase("transcription", required_vram_gb=5.0) as phase:
            model = load_whisper()
            result = transcribe(audio)
        # VRAM automatically flushed on exit

    The lock file at ~/.cache/song-to-video/phase.lock prevents multiple
    song-to-video processes from running GPU-intensive phases simultaneously.
    """

    def __init__(
        self,
        name: str,
        required_vram_gb: float = 0.0,
        flush_on_enter: bool = True,
        flush_on_exit: bool = True,
    ):
        """
        Initialize a pipeline phase.

        Args:
            name: Human-readable name for this phase
            required_vram_gb: Minimum VRAM required (0 to skip check)
            flush_on_enter: Whether to flush VRAM when entering
            flush_on_exit: Whether to flush VRAM when exiting
        """
        self.name = name
        self.required_vram_gb = required_vram_gb
        self.flush_on_enter = flush_on_enter
        self.flush_on_exit = flush_on_exit
        self._lock_file: Optional[Any] = None
        self._start_time: Optional[datetime] = None

    def __enter__(self) -> "PipelinePhase":
        """Enter the phase, acquiring the lock."""
        # Ensure lock directory exists
        LOCK_DIR.mkdir(parents=True, exist_ok=True)

        # Open and lock the file
        self._lock_file = open(LOCK_FILE, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None

            # Try to read what phase is running
            try:
                current_phase = LOCK_FILE.read_text().strip().split("\n")[0]
                raise PhaseViolationError(
                    f"Cannot start phase '{self.name}': "
                    f"another process is running phase '{current_phase}'"
                )
            except (FileNotFoundError, IndexError):
                raise PhaseViolationError(
                    f"Cannot start phase '{self.name}': "
                    "another song-to-video process is already running"
                )

        # Write current phase info to lock file
        self._lock_file.write(f"{self.name}\n")
        self._lock_file.write(f"pid: {os.getpid()}\n")
        self._lock_file.write(f"started: {datetime.now().isoformat()}\n")
        self._lock_file.flush()

        self._start_time = datetime.now()
        logger.info(f"Entering phase: {self.name}")

        # Flush VRAM on entry
        if self.flush_on_enter:
            result = flush_vram()
            if result.get("status") == "success":
                logger.debug(f"VRAM flushed: freed {result.get('freed_gb', 0):.2f} GB")

        # Check VRAM requirement
        if self.required_vram_gb > 0:
            if not require_vram(self.required_vram_gb):
                usage = get_vram_usage()
                available = usage.free_gb if usage else 0
                self._release_lock()
                raise InsufficientVRAMError(
                    f"Phase '{self.name}' requires {self.required_vram_gb:.1f} GB VRAM, "
                    f"but only {available:.1f} GB available"
                )

        # Log memory snapshot
        get_monitor().log_snapshot(self.name, "phase started")

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the phase, releasing the lock."""
        # Log memory before flush
        get_monitor().log_snapshot(self.name, "phase ending")

        # Flush VRAM on exit
        if self.flush_on_exit:
            result = flush_vram()
            if result.get("status") == "success":
                logger.debug(f"VRAM flushed: freed {result.get('freed_gb', 0):.2f} GB")

        # Calculate duration
        if self._start_time:
            duration = datetime.now() - self._start_time
            logger.info(f"Exiting phase: {self.name} (duration: {duration})")

        # Release the lock
        self._release_lock()

        # Don't suppress exceptions
        return False

    def _release_lock(self) -> None:
        """Release the file lock."""
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def phase(
    name: str,
    required_vram_gb: float = 0.0,
    flush_on_enter: bool = True,
    flush_on_exit: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to wrap a function in a pipeline phase.

    Usage:
        @phase("video_generation", required_vram_gb=12.0)
        def generate_video(prompt: str) -> Video:
            model = load_video_model()
            return model.generate(prompt)

    Args:
        name: Name of the phase
        required_vram_gb: Minimum VRAM required
        flush_on_enter: Whether to flush VRAM when entering
        flush_on_exit: Whether to flush VRAM when exiting
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with PipelinePhase(
                name=name,
                required_vram_gb=required_vram_gb,
                flush_on_enter=flush_on_enter,
                flush_on_exit=flush_on_exit,
            ):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def no_phase_lock():
    """
    Context manager to temporarily bypass phase locking.

    Use with caution - this is mainly for testing or when you know
    what you're doing.
    """
    # Just yield without doing anything special
    yield


def get_current_phase() -> Optional[str]:
    """
    Get the name of the currently running phase, if any.

    Returns:
        Phase name, or None if no phase is locked
    """
    try:
        if not LOCK_FILE.exists():
            return None

        content = LOCK_FILE.read_text().strip()
        if content:
            return content.split("\n")[0]
        return None
    except Exception:
        return None


def is_phase_locked() -> bool:
    """Check if any phase is currently locked by trying to acquire the lock."""
    if not LOCK_FILE.exists():
        return False

    try:
        # Try to acquire the lock non-blocking
        lock_file = open(LOCK_FILE, "r")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we got here, no one else holds the lock
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            return False
        except BlockingIOError:
            # Lock is held by another process
            lock_file.close()
            return True
    except Exception:
        return False


def force_unlock() -> bool:
    """
    Force unlock the phase lock.

    Use with caution - only for recovery from crashes.

    Returns:
        True if lock was released, False if no lock existed
    """
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            logger.warning("Force unlocked phase lock")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to force unlock: {e}")
        return False
