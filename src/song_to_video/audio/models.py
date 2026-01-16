"""Data models for audio and lyrics processing."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class LyricsSource(Enum):
    """Source of lyrics data."""

    MANUAL = "manual"  # User-provided file (SRT/LRC/TXT)
    EMBEDDED = "embedded"  # Extracted from audio file ID3 tags
    TRANSCRIBED = "transcribed"  # Generated via Whisper


@dataclass
class Word:
    """A single word with timing information."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] {self.text}"


@dataclass
class LyricLine:
    """A line of lyrics with timing and optional word-level detail."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of the line in seconds."""
        return self.end - self.start

    @property
    def has_word_timing(self) -> bool:
        """Whether word-level timing is available."""
        return len(self.words) > 0

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] {self.text}"


@dataclass
class Lyrics:
    """Complete lyrics with metadata."""

    lines: list[LyricLine]
    source: LyricsSource
    language: str = "en"
    confidence: float = 1.0  # Confidence score (0-1), relevant for transcribed

    @property
    def duration(self) -> float:
        """Total duration of lyrics in seconds."""
        if not self.lines:
            return 0.0
        return self.lines[-1].end - self.lines[0].start

    @property
    def text(self) -> str:
        """Full lyrics as plain text."""
        return "\n".join(line.text for line in self.lines)

    @property
    def word_count(self) -> int:
        """Total number of words."""
        return sum(len(line.text.split()) for line in self.lines)

    @property
    def has_word_timing(self) -> bool:
        """Whether any lines have word-level timing."""
        return any(line.has_word_timing for line in self.lines)

    def get_line_at(self, time: float) -> Optional[LyricLine]:
        """Get the lyric line active at the given time."""
        for line in self.lines:
            if line.start <= time <= line.end:
                return line
        return None

    def get_word_at(self, time: float) -> Optional[Word]:
        """Get the word active at the given time."""
        line = self.get_line_at(time)
        if line and line.has_word_timing:
            for word in line.words:
                if word.start <= time <= word.end:
                    return word
        return None

    def to_srt(self) -> str:
        """Export lyrics as SRT format."""
        lines = []
        for i, line in enumerate(self.lines, 1):
            start = _format_srt_time(line.start)
            end = _format_srt_time(line.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(line.text)
            lines.append("")
        return "\n".join(lines)

    def to_lrc(self) -> str:
        """Export lyrics as LRC format."""
        lines = []
        for line in self.lines:
            timestamp = _format_lrc_time(line.start)
            lines.append(f"{timestamp}{line.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export lyrics as dictionary for JSON serialization."""
        return {
            "source": self.source.value,
            "language": self.language,
            "confidence": self.confidence,
            "duration": self.duration,
            "word_count": self.word_count,
            "lines": [
                {
                    "text": line.text,
                    "start": line.start,
                    "end": line.end,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end}
                        for w in line.words
                    ],
                }
                for line in self.lines
            ],
        }


@dataclass
class AudioInfo:
    """Metadata about an audio file."""

    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    bitrate: Optional[int] = None  # kbps, may not be available
    format: str = "mp3"

    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration_seconds / 60

    @property
    def is_stereo(self) -> bool:
        """Whether audio is stereo."""
        return self.channels == 2

    def __str__(self) -> str:
        return (
            f"{self.path.name}: {self.duration_minutes:.1f}min, "
            f"{self.sample_rate}Hz, {self.channels}ch"
        )


class SectionType(Enum):
    """Type of song section."""

    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    INSTRUMENTAL = "instrumental"
    BREAKDOWN = "breakdown"
    BUILDUP = "buildup"
    DROP = "drop"
    UNKNOWN = "unknown"


class Mood(Enum):
    """Musical mood classification."""

    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    AGGRESSIVE = "aggressive"
    ROMANTIC = "romantic"
    MELANCHOLIC = "melancholic"
    UPLIFTING = "uplifting"
    DARK = "dark"
    NEUTRAL = "neutral"


@dataclass
class Beat:
    """A single beat in the audio."""

    time: float  # Time in seconds
    strength: float = 1.0  # Beat strength (0-1)


@dataclass
class Section:
    """A section of the song (verse, chorus, etc.)."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    type: SectionType
    confidence: float = 1.0  # Confidence score (0-1)
    label: str = ""  # Optional label like "Verse 1", "Chorus"

    @property
    def duration(self) -> float:
        """Duration of the section in seconds."""
        return self.end - self.start

    def __str__(self) -> str:
        label = self.label or self.type.value
        return f"[{self.start:.2f}-{self.end:.2f}] {label}"


@dataclass
class EnergyPoint:
    """Energy level at a point in time."""

    time: float  # Time in seconds
    energy: float  # Energy level (0-1)
    rms: float = 0.0  # Raw RMS value


@dataclass
class AudioFeatures:
    """Extracted audio features for a song."""

    # Basic properties
    duration: float  # Total duration in seconds
    sample_rate: int  # Sample rate in Hz

    # Tempo and rhythm
    tempo: float  # BPM (beats per minute)
    tempo_confidence: float  # Confidence in tempo detection (0-1)
    beats: list[Beat] = field(default_factory=list)  # Beat positions
    downbeats: list[float] = field(default_factory=list)  # Downbeat positions

    # Energy
    energy_curve: list[EnergyPoint] = field(default_factory=list)
    average_energy: float = 0.5  # Average energy (0-1)
    energy_variance: float = 0.0  # How much energy varies

    # Structure
    sections: list[Section] = field(default_factory=list)

    # Vocal detection
    has_vocals: bool = True
    vocal_ratio: float = 1.0  # Ratio of vocal to instrumental segments

    # Mood/genre
    mood: Mood = Mood.NEUTRAL
    mood_confidence: float = 0.5
    genre_tags: list[str] = field(default_factory=list)

    # Spectral features (for advanced analysis)
    key: str = ""  # Musical key (e.g., "C major", "A minor")
    key_confidence: float = 0.0

    def get_section_at(self, time: float) -> Optional[Section]:
        """Get the section at the given time."""
        for section in self.sections:
            if section.start <= time <= section.end:
                return section
        return None

    def get_energy_at(self, time: float) -> float:
        """Get interpolated energy at the given time."""
        if not self.energy_curve:
            return self.average_energy

        # Find surrounding points
        prev_point = None
        next_point = None

        for point in self.energy_curve:
            if point.time <= time:
                prev_point = point
            elif next_point is None:
                next_point = point
                break

        if prev_point is None:
            return self.energy_curve[0].energy if self.energy_curve else self.average_energy
        if next_point is None:
            return prev_point.energy

        # Linear interpolation
        t = (time - prev_point.time) / (next_point.time - prev_point.time)
        return prev_point.energy + t * (next_point.energy - prev_point.energy)

    def get_beats_in_range(self, start: float, end: float) -> list[Beat]:
        """Get beats within a time range."""
        return [b for b in self.beats if start <= b.time <= end]

    def to_dict(self) -> dict:
        """Export as dictionary for JSON serialization."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "tempo": self.tempo,
            "tempo_confidence": self.tempo_confidence,
            "average_energy": self.average_energy,
            "energy_variance": self.energy_variance,
            "has_vocals": self.has_vocals,
            "vocal_ratio": self.vocal_ratio,
            "mood": self.mood.value,
            "mood_confidence": self.mood_confidence,
            "genre_tags": self.genre_tags,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "beat_count": len(self.beats),
            "section_count": len(self.sections),
            "sections": [
                {
                    "start": s.start,
                    "end": s.end,
                    "type": s.type.value,
                    "label": s.label,
                    "confidence": s.confidence,
                }
                for s in self.sections
            ],
        }


class AudioValidationError(Exception):
    """Raised when audio file validation fails."""

    pass


class LyricsParseError(Exception):
    """Raised when lyrics parsing fails."""

    pass


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


# Helper functions for time formatting


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_lrc_time(seconds: float) -> str:
    """Format seconds as LRC timestamp ([mm:ss.xx])."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"
