"""Audio processing module for song-to-video."""

from .models import (
    Word,
    LyricLine,
    Lyrics,
    LyricsSource,
    AudioInfo,
    AudioValidationError,
    LyricsParseError,
    TranscriptionError,
    # Audio analysis models
    Beat,
    Section,
    SectionType,
    EnergyPoint,
    AudioFeatures,
    Mood,
)
from .loader import (
    load_audio,
    convert_to_wav,
    get_audio_duration,
    validate_audio_for_transcription,
    SUPPORTED_FORMATS,
    MAX_DURATION_SECONDS,
)
from .lyrics import (
    extract_embedded_lyrics,
    parse_srt,
    parse_lrc,
    parse_txt,
    parse_lyrics_file,
    needs_alignment,
)
from .transcribe import (
    Transcriber,
    transcribe_audio,
    get_recommended_model,
)
from .analysis import (
    AudioAnalyzer,
    analyze_audio,
)

__all__ = [
    # Models
    "Word",
    "LyricLine",
    "Lyrics",
    "LyricsSource",
    "AudioInfo",
    "AudioValidationError",
    "LyricsParseError",
    "TranscriptionError",
    # Audio analysis models
    "Beat",
    "Section",
    "SectionType",
    "EnergyPoint",
    "AudioFeatures",
    "Mood",
    # Loader
    "load_audio",
    "convert_to_wav",
    "get_audio_duration",
    "validate_audio_for_transcription",
    "SUPPORTED_FORMATS",
    "MAX_DURATION_SECONDS",
    # Lyrics
    "extract_embedded_lyrics",
    "parse_srt",
    "parse_lrc",
    "parse_txt",
    "parse_lyrics_file",
    "needs_alignment",
    # Transcribe
    "Transcriber",
    "transcribe_audio",
    "get_recommended_model",
    # Analysis
    "AudioAnalyzer",
    "analyze_audio",
]
