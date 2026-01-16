"""Audio file loading and validation."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from .models import AudioInfo, AudioValidationError

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma"}

# Maximum song duration in seconds (6 minutes per PRD)
MAX_DURATION_SECONDS = 6 * 60


def load_audio(path: str | Path) -> AudioInfo:
    """
    Load and validate an audio file.

    Args:
        path: Path to the audio file

    Returns:
        AudioInfo with file metadata

    Raises:
        AudioValidationError: If the file is invalid or unsupported
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        raise AudioValidationError(f"File not found: {path}")

    if not path.is_file():
        raise AudioValidationError(f"Not a file: {path}")

    # Check format
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # Get audio metadata using mutagen
    try:
        import mutagen

        audio = mutagen.File(path)
        if audio is None:
            raise AudioValidationError(f"Could not read audio file: {path}")

        duration = audio.info.length
        sample_rate = getattr(audio.info, "sample_rate", 44100)
        channels = getattr(audio.info, "channels", 2)
        bitrate = getattr(audio.info, "bitrate", None)
        if bitrate:
            bitrate = int(bitrate / 1000)  # Convert to kbps

    except ImportError:
        raise AudioValidationError("mutagen library not installed")
    except Exception as e:
        raise AudioValidationError(f"Failed to read audio metadata: {e}")

    # Validate duration
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Song too long: {duration / 60:.1f} minutes. "
            f"Maximum: {MAX_DURATION_SECONDS / 60:.0f} minutes"
        )

    if duration < 1:
        raise AudioValidationError(f"Song too short: {duration:.1f} seconds")

    logger.info(f"Loaded audio: {path.name} ({duration:.1f}s, {sample_rate}Hz)")

    return AudioInfo(
        path=path,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        bitrate=bitrate,
        format=suffix.lstrip("."),
    )


def convert_to_wav(
    audio_path: str | Path,
    output_dir: Optional[Path] = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """
    Convert audio to WAV format for Whisper processing.

    Whisper works best with 16kHz mono WAV files.

    Args:
        audio_path: Path to source audio file
        output_dir: Directory for output file (uses temp dir if None)
        sample_rate: Target sample rate (default 16000 for Whisper)
        mono: Convert to mono (default True for Whisper)

    Returns:
        Path to the converted WAV file

    Raises:
        AudioValidationError: If conversion fails
    """
    audio_path = Path(audio_path)

    # Determine output path
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "song-to-video"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{audio_path.stem}.wav"

    # If already a WAV with correct settings, just return it
    if audio_path.suffix.lower() == ".wav":
        try:
            info = load_audio(audio_path)
            if info.sample_rate == sample_rate and (not mono or info.channels == 1):
                logger.debug(f"Audio already in correct format: {audio_path}")
                return audio_path
        except AudioValidationError:
            pass  # Continue with conversion

    try:
        from pydub import AudioSegment

        logger.info(f"Converting {audio_path.name} to WAV ({sample_rate}Hz, {'mono' if mono else 'stereo'})")

        # Load audio
        audio = AudioSegment.from_file(str(audio_path))

        # Convert to mono if requested
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample if needed
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export as WAV
        audio.export(str(output_path), format="wav")

        logger.info(f"Converted to: {output_path}")
        return output_path

    except ImportError:
        raise AudioValidationError("pydub library not installed")
    except Exception as e:
        raise AudioValidationError(f"Failed to convert audio: {e}")


def get_audio_duration(path: str | Path) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    info = load_audio(path)
    return info.duration_seconds


def validate_audio_for_transcription(path: str | Path) -> AudioInfo:
    """
    Validate that an audio file is suitable for transcription.

    This performs additional checks beyond basic loading.

    Args:
        path: Path to audio file

    Returns:
        AudioInfo if valid

    Raises:
        AudioValidationError: If file is not suitable for transcription
    """
    info = load_audio(path)

    # Check sample rate (Whisper works best with 16kHz, but can handle others)
    if info.sample_rate < 8000:
        raise AudioValidationError(
            f"Sample rate too low: {info.sample_rate}Hz. Minimum: 8000Hz"
        )

    # Warn about very high sample rates (will be resampled)
    if info.sample_rate > 48000:
        logger.warning(
            f"High sample rate ({info.sample_rate}Hz) will be resampled to 16kHz"
        )

    return info
