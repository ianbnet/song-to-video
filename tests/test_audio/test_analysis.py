"""Tests for audio analysis."""

import pytest
import numpy as np
from pathlib import Path

from song_to_video.audio.models import (
    AudioFeatures,
    Beat,
    Section,
    SectionType,
    EnergyPoint,
    Mood,
)
from song_to_video.audio.analysis import AudioAnalyzer


class TestAudioFeatures:
    """Tests for AudioFeatures data class."""

    def test_get_energy_at_interpolation(self):
        """Test energy interpolation between points."""
        features = AudioFeatures(
            duration=10.0,
            sample_rate=44100,
            tempo=120.0,
            tempo_confidence=0.9,
            energy_curve=[
                EnergyPoint(time=0.0, energy=0.2, rms=0.1),
                EnergyPoint(time=5.0, energy=0.8, rms=0.4),
                EnergyPoint(time=10.0, energy=0.4, rms=0.2),
            ],
            average_energy=0.5,
        )

        # Test at exact points
        assert features.get_energy_at(0.0) == pytest.approx(0.2)
        assert features.get_energy_at(5.0) == pytest.approx(0.8)

        # Test interpolation at midpoint
        assert features.get_energy_at(2.5) == pytest.approx(0.5)

    def test_get_section_at(self):
        """Test getting section at a specific time."""
        features = AudioFeatures(
            duration=60.0,
            sample_rate=44100,
            tempo=120.0,
            tempo_confidence=0.9,
            sections=[
                Section(start=0.0, end=10.0, type=SectionType.INTRO),
                Section(start=10.0, end=30.0, type=SectionType.VERSE),
                Section(start=30.0, end=50.0, type=SectionType.CHORUS),
                Section(start=50.0, end=60.0, type=SectionType.OUTRO),
            ],
        )

        assert features.get_section_at(5.0).type == SectionType.INTRO
        assert features.get_section_at(20.0).type == SectionType.VERSE
        assert features.get_section_at(40.0).type == SectionType.CHORUS
        assert features.get_section_at(55.0).type == SectionType.OUTRO
        assert features.get_section_at(100.0) is None

    def test_get_beats_in_range(self):
        """Test getting beats within a time range."""
        features = AudioFeatures(
            duration=10.0,
            sample_rate=44100,
            tempo=120.0,
            tempo_confidence=0.9,
            beats=[
                Beat(time=0.5, strength=0.8),
                Beat(time=1.0, strength=0.6),
                Beat(time=1.5, strength=0.8),
                Beat(time=2.0, strength=0.6),
                Beat(time=2.5, strength=0.8),
            ],
        )

        beats = features.get_beats_in_range(1.0, 2.0)
        assert len(beats) == 3
        assert beats[0].time == 1.0
        assert beats[-1].time == 2.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        features = AudioFeatures(
            duration=60.0,
            sample_rate=44100,
            tempo=120.0,
            tempo_confidence=0.9,
            has_vocals=True,
            vocal_ratio=0.8,
            mood=Mood.HAPPY,
            mood_confidence=0.7,
            key="C major",
            key_confidence=0.6,
            genre_tags=["upbeat", "pop"],
            sections=[
                Section(start=0.0, end=30.0, type=SectionType.VERSE, label="Verse 1"),
            ],
        )

        data = features.to_dict()

        assert data["duration"] == 60.0
        assert data["tempo"] == 120.0
        assert data["has_vocals"] is True
        assert data["mood"] == "happy"
        assert data["key"] == "C major"
        assert "upbeat" in data["genre_tags"]
        assert len(data["sections"]) == 1
        assert data["sections"][0]["type"] == "verse"


class TestSection:
    """Tests for Section data class."""

    def test_section_duration(self):
        """Test section duration calculation."""
        section = Section(start=10.0, end=25.0, type=SectionType.CHORUS)
        assert section.duration == 15.0

    def test_section_str(self):
        """Test section string representation."""
        section = Section(
            start=10.0,
            end=25.0,
            type=SectionType.CHORUS,
            label="Chorus 1",
        )
        assert "Chorus 1" in str(section)
        assert "10.00" in str(section)


class TestMood:
    """Tests for Mood enum."""

    def test_mood_values(self):
        """Test mood enum has expected values."""
        assert Mood.HAPPY.value == "happy"
        assert Mood.SAD.value == "sad"
        assert Mood.ENERGETIC.value == "energetic"
        assert Mood.CALM.value == "calm"


class TestSectionType:
    """Tests for SectionType enum."""

    def test_section_type_values(self):
        """Test section type enum has expected values."""
        assert SectionType.VERSE.value == "verse"
        assert SectionType.CHORUS.value == "chorus"
        assert SectionType.INTRO.value == "intro"
        assert SectionType.OUTRO.value == "outro"
        assert SectionType.BRIDGE.value == "bridge"
