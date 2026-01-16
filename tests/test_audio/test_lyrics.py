"""Tests for lyrics parsing."""

import pytest
import tempfile
from pathlib import Path

from song_to_video.audio.lyrics import (
    parse_srt,
    parse_lrc,
    parse_txt,
    parse_lyrics_file,
    needs_alignment,
)
from song_to_video.audio.models import Lyrics, LyricsSource, LyricsParseError


class TestParseSRT:
    """Tests for SRT parsing."""

    def test_parse_valid_srt(self, tmp_path):
        """Parse a valid SRT file."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
First line of lyrics

2
00:00:04,500 --> 00:00:08,000
Second line of lyrics

3
00:00:08,500 --> 00:00:12,000
Third line
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        lyrics = parse_srt(srt_file)

        assert len(lyrics.lines) == 3
        assert lyrics.lines[0].text == "First line of lyrics"
        assert lyrics.lines[0].start == 1.0
        assert lyrics.lines[0].end == 4.0
        assert lyrics.source == LyricsSource.MANUAL

    def test_parse_srt_multiline(self, tmp_path):
        """Parse SRT with multi-line entries."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
First line
Second part of first

2
00:00:05,000 --> 00:00:08,000
Another line
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        lyrics = parse_srt(srt_file)

        assert len(lyrics.lines) == 2
        # Multi-line becomes single line with space
        assert "First line" in lyrics.lines[0].text

    def test_parse_srt_file_not_found(self):
        """Raise error for missing file."""
        with pytest.raises(LyricsParseError):
            parse_srt(Path("/nonexistent/file.srt"))


class TestParseLRC:
    """Tests for LRC parsing."""

    def test_parse_valid_lrc(self, tmp_path):
        """Parse a valid LRC file."""
        lrc_content = """[ar:Artist Name]
[ti:Song Title]
[00:01.00]First line of lyrics
[00:04.50]Second line of lyrics
[00:08.00]Third line
"""
        lrc_file = tmp_path / "test.lrc"
        lrc_file.write_text(lrc_content)

        lyrics = parse_lrc(lrc_file)

        assert len(lyrics.lines) == 3
        assert lyrics.lines[0].text == "First line of lyrics"
        assert lyrics.lines[0].start == 1.0
        assert lyrics.source == LyricsSource.MANUAL

    def test_parse_lrc_skips_metadata(self, tmp_path):
        """Skip metadata tags in LRC."""
        lrc_content = """[ar:Artist]
[ti:Title]
[al:Album]
[00:01.00]Actual lyrics
"""
        lrc_file = tmp_path / "test.lrc"
        lrc_file.write_text(lrc_content)

        lyrics = parse_lrc(lrc_file)

        assert len(lyrics.lines) == 1
        assert lyrics.lines[0].text == "Actual lyrics"

    def test_parse_lrc_calculates_end_times(self, tmp_path):
        """End times calculated from next line start."""
        lrc_content = """[00:01.00]First line
[00:04.00]Second line
[00:08.00]Third line
"""
        lrc_file = tmp_path / "test.lrc"
        lrc_file.write_text(lrc_content)

        lyrics = parse_lrc(lrc_file)

        assert lyrics.lines[0].end == 4.0  # Start of next line
        assert lyrics.lines[1].end == 8.0  # Start of next line
        assert lyrics.lines[2].end == 11.0  # Last line gets +3 seconds


class TestParseTXT:
    """Tests for plain text parsing."""

    def test_parse_valid_txt(self, tmp_path):
        """Parse plain text file."""
        txt_content = """First line
Second line

Third line after blank
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        lyrics = parse_txt(txt_file)

        assert len(lyrics.lines) == 3
        assert lyrics.lines[0].text == "First line"
        assert lyrics.lines[0].start == 0.0  # No timestamps
        assert lyrics.lines[0].end == 0.0

    def test_parse_txt_needs_alignment(self, tmp_path):
        """Plain text lyrics need alignment."""
        txt_content = """First line
Second line
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        lyrics = parse_txt(txt_file)

        assert needs_alignment(lyrics) is True


class TestParseLyricsFile:
    """Tests for auto-detecting lyrics format."""

    def test_auto_detect_srt(self, tmp_path):
        """Auto-detect SRT format."""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n")

        lyrics = parse_lyrics_file(srt_file)
        assert lyrics.source == LyricsSource.MANUAL

    def test_auto_detect_lrc(self, tmp_path):
        """Auto-detect LRC format."""
        lrc_file = tmp_path / "test.lrc"
        lrc_file.write_text("[00:01.00]Test\n")

        lyrics = parse_lyrics_file(lrc_file)
        assert lyrics.source == LyricsSource.MANUAL

    def test_auto_detect_txt(self, tmp_path):
        """Auto-detect TXT format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test line\n")

        lyrics = parse_lyrics_file(txt_file)
        assert lyrics.source == LyricsSource.MANUAL

    def test_unsupported_format(self, tmp_path):
        """Raise error for unsupported format."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")

        with pytest.raises(LyricsParseError):
            parse_lyrics_file(bad_file)


class TestLyricsExport:
    """Tests for lyrics export functionality."""

    def test_to_srt(self, tmp_path):
        """Export lyrics as SRT."""
        from song_to_video.audio.models import LyricLine

        lyrics = Lyrics(
            lines=[
                LyricLine(text="First line", start=1.0, end=4.0),
                LyricLine(text="Second line", start=4.5, end=8.0),
            ],
            source=LyricsSource.MANUAL,
        )

        srt = lyrics.to_srt()

        assert "00:00:01,000 --> 00:00:04,000" in srt
        assert "First line" in srt
        assert "00:00:04,500 --> 00:00:08,000" in srt

    def test_to_lrc(self, tmp_path):
        """Export lyrics as LRC."""
        from song_to_video.audio.models import LyricLine

        lyrics = Lyrics(
            lines=[
                LyricLine(text="First line", start=1.0, end=4.0),
                LyricLine(text="Second line", start=65.5, end=70.0),
            ],
            source=LyricsSource.MANUAL,
        )

        lrc = lyrics.to_lrc()

        assert "[00:01.00]First line" in lrc
        assert "[01:05.50]Second line" in lrc

    def test_to_dict(self):
        """Export lyrics as dictionary."""
        from song_to_video.audio.models import LyricLine, Word

        lyrics = Lyrics(
            lines=[
                LyricLine(
                    text="Hello world",
                    start=1.0,
                    end=3.0,
                    words=[
                        Word(text="Hello", start=1.0, end=1.5),
                        Word(text="world", start=1.6, end=3.0),
                    ],
                ),
            ],
            source=LyricsSource.TRANSCRIBED,
            confidence=0.95,
        )

        data = lyrics.to_dict()

        assert data["source"] == "transcribed"
        assert data["confidence"] == 0.95
        assert len(data["lines"]) == 1
        assert len(data["lines"][0]["words"]) == 2
