"""Audio transcription using Whisper."""

import logging
from pathlib import Path
from typing import Optional

from ..memory import (
    PipelinePhase,
    get_hardware_tier,
    detect_gpu,
    HardwareTier,
)
from ..downloads import get_model_manager, WHISPER_MODEL_BY_TIER
from .models import Lyrics, LyricLine, Word, LyricsSource, TranscriptionError

logger = logging.getLogger(__name__)

# VRAM requirements by model size (GB)
VRAM_REQUIREMENTS = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.0,
}


class Transcriber:
    """
    Transcribes audio to text with timestamps using faster-whisper.

    Uses the memory management system to ensure proper VRAM handling.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "auto",
        compute_type: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
                       If None, auto-selects based on hardware tier
            device: Device to use ("cuda", "cpu", or "auto")
            compute_type: Compute type ("float16", "int8", "int8_float16", "auto")
        """
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._model_size = None

        # Auto-select model size if not specified
        if model_size is None:
            tier = get_hardware_tier()
            model_size = WHISPER_MODEL_BY_TIER.get(tier.value, "medium")
            logger.info(f"Auto-selected Whisper model '{model_size}' for {tier.value} tier")

        self.model_size = model_size

    def _get_device_and_compute_type(self) -> tuple[str, str]:
        """Determine device and compute type based on hardware."""
        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            gpu = detect_gpu()
            device = "cuda" if gpu is not None else "cpu"

        if compute_type is None:
            if device == "cuda":
                tier = get_hardware_tier()
                if tier == HardwareTier.LOW:
                    compute_type = "int8"
                elif tier == HardwareTier.MID:
                    compute_type = "float16"
                else:
                    compute_type = "float16"
            else:
                compute_type = "int8"

        return device, compute_type

    def _load_model(self):
        """Load the Whisper model."""
        if self._model is not None and self._model_size == self.model_size:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

        device, compute_type = self._get_device_and_compute_type()

        logger.info(
            f"Loading Whisper model '{self.model_size}' "
            f"(device={device}, compute_type={compute_type})"
        )

        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )
            self._model_size = self.model_size
            logger.info("Whisper model loaded successfully")
            return self._model

        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model: {e}")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
        word_timestamps: bool = True,
    ) -> Lyrics:
        """
        Transcribe audio file to lyrics.

        Args:
            audio_path: Path to audio file
            language: Language code (default "en" for English)
            word_timestamps: Whether to include word-level timestamps

        Returns:
            Lyrics object with transcribed text and timestamps

        Raises:
            TranscriptionError: If transcription fails
        """
        audio_path = Path(audio_path)

        # Get VRAM requirement for this model
        vram_required = VRAM_REQUIREMENTS.get(self.model_size, 5.0)

        # Use pipeline phase for memory management
        with PipelinePhase("transcription", required_vram_gb=vram_required):
            return self._transcribe_internal(audio_path, language, word_timestamps)

    def _transcribe_internal(
        self,
        audio_path: Path,
        language: str,
        word_timestamps: bool,
    ) -> Lyrics:
        """Internal transcription logic (called within PipelinePhase)."""
        # faster-whisper can read most audio formats directly via av/ffmpeg
        # Only convert if absolutely necessary
        input_path = audio_path

        # Load model
        model = self._load_model()

        logger.info(f"Transcribing: {audio_path.name}")

        try:
            segments, info = model.transcribe(
                str(input_path),
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=True,  # Voice activity detection
            )

            # Convert segments to lyrics
            lines = []
            total_confidence = 0.0
            segment_count = 0

            for segment in segments:
                words = []

                if word_timestamps and segment.words:
                    for word_info in segment.words:
                        words.append(
                            Word(
                                text=word_info.word.strip(),
                                start=word_info.start,
                                end=word_info.end,
                            )
                        )

                lines.append(
                    LyricLine(
                        text=segment.text.strip(),
                        start=segment.start,
                        end=segment.end,
                        words=words,
                    )
                )

                # Track confidence
                if hasattr(segment, "avg_logprob"):
                    # Convert log probability to rough confidence
                    # avg_logprob typically ranges from -1 (good) to -2+ (poor)
                    confidence = min(1.0, max(0.0, 1.0 + segment.avg_logprob))
                    total_confidence += confidence
                    segment_count += 1

            # Calculate average confidence
            avg_confidence = (
                total_confidence / segment_count if segment_count > 0 else 0.5
            )

            logger.info(
                f"Transcription complete: {len(lines)} lines, "
                f"confidence={avg_confidence:.2f}"
            )

            return Lyrics(
                lines=lines,
                source=LyricsSource.TRANSCRIBED,
                language=language,
                confidence=avg_confidence,
            )

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_size = None
            logger.debug("Whisper model unloaded")


def transcribe_audio(
    audio_path: str | Path,
    model_size: Optional[str] = None,
    language: str = "en",
) -> Lyrics:
    """
    Convenience function to transcribe audio.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (auto-selects if None)
        language: Language code

    Returns:
        Lyrics object with transcription
    """
    transcriber = Transcriber(model_size=model_size)
    return transcriber.transcribe(audio_path, language=language)


def get_recommended_model() -> str:
    """Get the recommended Whisper model for current hardware."""
    tier = get_hardware_tier()
    return WHISPER_MODEL_BY_TIER.get(tier.value, "medium")
