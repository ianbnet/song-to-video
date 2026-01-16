"""Lyrics extraction and parsing from various sources."""

import logging
import re
from pathlib import Path
from typing import Optional

from .models import Lyrics, LyricLine, Word, LyricsSource, LyricsParseError

logger = logging.getLogger(__name__)


def extract_embedded_lyrics(audio_path: str | Path) -> Optional[Lyrics]:
    """
    Extract lyrics embedded in audio file ID3 tags.

    Supports:
    - USLT (Unsynchronized Lyrics) - plain text lyrics
    - SYLT (Synchronized Lyrics) - lyrics with timestamps

    Args:
        audio_path: Path to audio file

    Returns:
        Lyrics object if found, None otherwise
    """
    audio_path = Path(audio_path)

    try:
        import mutagen
        from mutagen.id3 import ID3

        # Try to read ID3 tags
        try:
            tags = ID3(audio_path)
        except mutagen.id3.ID3NoHeaderError:
            logger.debug(f"No ID3 tags found in {audio_path.name}")
            return None

        # Try SYLT (synchronized lyrics) first
        sylt_frames = tags.getall("SYLT")
        if sylt_frames:
            for frame in sylt_frames:
                lyrics = _parse_sylt_frame(frame)
                if lyrics:
                    logger.info(f"Found synchronized lyrics in {audio_path.name}")
                    return lyrics

        # Fall back to USLT (unsynchronized lyrics)
        uslt_frames = tags.getall("USLT")
        if uslt_frames:
            for frame in uslt_frames:
                if frame.text:
                    logger.info(f"Found unsynchronized lyrics in {audio_path.name}")
                    return _parse_plain_text(frame.text, source=LyricsSource.EMBEDDED)

        logger.debug(f"No lyrics found in {audio_path.name}")
        return None

    except ImportError:
        logger.warning("mutagen not installed, cannot extract embedded lyrics")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract embedded lyrics: {e}")
        return None


def _parse_sylt_frame(frame) -> Optional[Lyrics]:
    """Parse a SYLT (synchronized lyrics) frame."""
    try:
        lines = []
        current_words = []
        current_text = []
        line_start = None

        for text, timestamp in frame.text:
            # timestamp is in milliseconds
            time_sec = timestamp / 1000.0

            if line_start is None:
                line_start = time_sec

            # Check if this is a line break
            if text.strip() == "" or text == "\n":
                if current_text:
                    lines.append(
                        LyricLine(
                            text=" ".join(current_text),
                            start=line_start,
                            end=time_sec,
                            words=current_words,
                        )
                    )
                    current_words = []
                    current_text = []
                    line_start = None
            else:
                current_text.append(text.strip())
                current_words.append(
                    Word(text=text.strip(), start=time_sec, end=time_sec)
                )

        # Don't forget the last line
        if current_text:
            lines.append(
                LyricLine(
                    text=" ".join(current_text),
                    start=line_start or 0,
                    end=current_words[-1].end if current_words else 0,
                    words=current_words,
                )
            )

        if lines:
            return Lyrics(lines=lines, source=LyricsSource.EMBEDDED)
        return None

    except Exception as e:
        logger.warning(f"Failed to parse SYLT frame: {e}")
        return None


def parse_srt(path: str | Path) -> Lyrics:
    """
    Parse SubRip (SRT) subtitle format.

    Format:
    ```
    1
    00:00:01,000 --> 00:00:04,000
    First line of lyrics

    2
    00:00:04,500 --> 00:00:08,000
    Second line
    ```

    Args:
        path: Path to SRT file

    Returns:
        Parsed Lyrics object

    Raises:
        LyricsParseError: If parsing fails
    """
    path = Path(path)

    if not path.exists():
        raise LyricsParseError(f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            raise LyricsParseError(f"Failed to read file: {e}")

    lines = []
    # SRT pattern: index, timestamp --> timestamp, text
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*\n"
        r"(.*?)(?=\n\n|\n\d+\s*\n|\Z)",
        re.DOTALL,
    )

    for match in pattern.finditer(content):
        # Parse start time
        start = (
            int(match.group(2)) * 3600
            + int(match.group(3)) * 60
            + int(match.group(4))
            + int(match.group(5)) / 1000
        )

        # Parse end time
        end = (
            int(match.group(6)) * 3600
            + int(match.group(7)) * 60
            + int(match.group(8))
            + int(match.group(9)) / 1000
        )

        # Get text (may span multiple lines)
        text = match.group(10).strip().replace("\n", " ")

        if text:
            lines.append(LyricLine(text=text, start=start, end=end))

    if not lines:
        raise LyricsParseError(f"No valid entries found in SRT file: {path}")

    logger.info(f"Parsed {len(lines)} lines from SRT: {path.name}")
    return Lyrics(lines=lines, source=LyricsSource.MANUAL)


def parse_lrc(path: str | Path) -> Lyrics:
    """
    Parse LRC lyrics format.

    Format:
    ```
    [00:01.00]First line of lyrics
    [00:04.50]Second line
    ```

    Also supports metadata tags like [ar:Artist], [ti:Title], etc.

    Args:
        path: Path to LRC file

    Returns:
        Parsed Lyrics object

    Raises:
        LyricsParseError: If parsing fails
    """
    path = Path(path)

    if not path.exists():
        raise LyricsParseError(f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            raise LyricsParseError(f"Failed to read file: {e}")

    lines = []
    # LRC timestamp pattern: [mm:ss.xx] or [mm:ss:xx]
    pattern = re.compile(r"\[(\d{2}):(\d{2})[\.:]\d{2}\](.+)")

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # Skip metadata tags
        if re.match(r"\[[a-z]{2}:", line):
            continue

        match = pattern.match(line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            text = match.group(3).strip()

            if text:
                start = minutes * 60 + seconds
                lines.append(LyricLine(text=text, start=start, end=start))

    if not lines:
        raise LyricsParseError(f"No valid entries found in LRC file: {path}")

    # Calculate end times (start of next line or +3 seconds for last line)
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            line.end = lines[i + 1].start
        else:
            line.end = line.start + 3.0

    logger.info(f"Parsed {len(lines)} lines from LRC: {path.name}")
    return Lyrics(lines=lines, source=LyricsSource.MANUAL)


def parse_txt(path: str | Path) -> Lyrics:
    """
    Parse plain text lyrics file (no timestamps).

    Each non-empty line becomes a lyric line with no timing information.
    Timing must be added later via alignment.

    Args:
        path: Path to TXT file

    Returns:
        Parsed Lyrics object (with zero timestamps)

    Raises:
        LyricsParseError: If parsing fails
    """
    path = Path(path)

    if not path.exists():
        raise LyricsParseError(f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            raise LyricsParseError(f"Failed to read file: {e}")

    lines = []
    for line in content.splitlines():
        text = line.strip()
        if text:
            lines.append(LyricLine(text=text, start=0.0, end=0.0))

    if not lines:
        raise LyricsParseError(f"No text found in file: {path}")

    logger.info(f"Parsed {len(lines)} lines from TXT: {path.name} (no timestamps)")
    return Lyrics(lines=lines, source=LyricsSource.MANUAL)


def _parse_plain_text(text: str, source: LyricsSource = LyricsSource.MANUAL) -> Lyrics:
    """Parse plain text into lyrics (no timestamps)."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(LyricLine(text=line, start=0.0, end=0.0))

    return Lyrics(lines=lines, source=source)


def parse_lyrics_file(path: str | Path) -> Lyrics:
    """
    Parse a lyrics file, auto-detecting format from extension.

    Args:
        path: Path to lyrics file (.srt, .lrc, or .txt)

    Returns:
        Parsed Lyrics object

    Raises:
        LyricsParseError: If format is unsupported or parsing fails
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".srt":
        return parse_srt(path)
    elif suffix == ".lrc":
        return parse_lrc(path)
    elif suffix == ".txt":
        return parse_txt(path)
    else:
        raise LyricsParseError(
            f"Unsupported lyrics format: {suffix}. "
            "Supported: .srt, .lrc, .txt"
        )


def needs_alignment(lyrics: Lyrics) -> bool:
    """
    Check if lyrics need timing alignment.

    Returns True if:
    - All timestamps are 0 (plain text import)
    - No word-level timing exists

    Args:
        lyrics: Lyrics to check

    Returns:
        True if alignment is needed
    """
    if not lyrics.lines:
        return False

    # Check if all timestamps are 0
    all_zero = all(line.start == 0 and line.end == 0 for line in lyrics.lines)
    if all_zero:
        return True

    return False
