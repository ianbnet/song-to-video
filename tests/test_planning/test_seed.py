"""Tests for seed generation."""

import pytest
import tempfile
from pathlib import Path

from song_to_video.planning.seed import (
    generate_audio_fingerprint,
    generate_style_hash,
    generate_master_seed,
    get_scene_seed,
    get_frame_seed,
)


class TestAudioFingerprint:
    """Tests for audio fingerprint generation."""

    def test_fingerprint_deterministic(self, tmp_path):
        """Same file produces same fingerprint."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test audio content" * 100)

        fp1 = generate_audio_fingerprint(test_file)
        fp2 = generate_audio_fingerprint(test_file)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex length

    def test_different_files_different_fingerprints(self, tmp_path):
        """Different files produce different fingerprints."""
        file1 = tmp_path / "test1.bin"
        file2 = tmp_path / "test2.bin"
        file1.write_bytes(b"content one")
        file2.write_bytes(b"content two")

        fp1 = generate_audio_fingerprint(file1)
        fp2 = generate_audio_fingerprint(file2)

        assert fp1 != fp2


class TestStyleHash:
    """Tests for style hash generation."""

    def test_style_hash_deterministic(self):
        """Same description produces same hash."""
        desc = "cinematic noir style"

        hash1 = generate_style_hash(desc)
        hash2 = generate_style_hash(desc)

        assert hash1 == hash2

    def test_different_styles_different_hashes(self):
        """Different descriptions produce different hashes."""
        hash1 = generate_style_hash("bright pop")
        hash2 = generate_style_hash("dark noir")

        assert hash1 != hash2


class TestMasterSeed:
    """Tests for master seed generation."""

    def test_master_seed_deterministic(self, tmp_path):
        """Same inputs produce same seed."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test audio")

        seed1 = generate_master_seed(test_file, "style A")
        seed2 = generate_master_seed(test_file, "style A")

        assert seed1 == seed2

    def test_different_audio_different_seed(self, tmp_path):
        """Different audio produces different seed."""
        file1 = tmp_path / "test1.bin"
        file2 = tmp_path / "test2.bin"
        file1.write_bytes(b"audio one")
        file2.write_bytes(b"audio two")

        seed1 = generate_master_seed(file1, "same style")
        seed2 = generate_master_seed(file2, "same style")

        assert seed1 != seed2

    def test_different_style_different_seed(self, tmp_path):
        """Different style produces different seed."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test audio")

        seed1 = generate_master_seed(test_file, "style A")
        seed2 = generate_master_seed(test_file, "style B")

        assert seed1 != seed2

    def test_user_seed_override(self, tmp_path):
        """User seed overrides generated seed."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test audio")

        seed = generate_master_seed(test_file, "style", user_seed=42)

        assert seed == 42

    def test_seed_in_valid_range(self, tmp_path):
        """Seed is in valid range (0 to 2^31-1)."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test audio content")

        seed = generate_master_seed(test_file, "any style")

        assert 0 <= seed < 2**31


class TestSceneSeed:
    """Tests for scene seed generation."""

    def test_scene_seed_deterministic(self):
        """Same inputs produce same scene seed."""
        seed1 = get_scene_seed(12345, 0)
        seed2 = get_scene_seed(12345, 0)

        assert seed1 == seed2

    def test_different_scenes_different_seeds(self):
        """Different scene indices produce different seeds."""
        master = 12345

        seed0 = get_scene_seed(master, 0)
        seed1 = get_scene_seed(master, 1)
        seed2 = get_scene_seed(master, 2)

        assert len({seed0, seed1, seed2}) == 3  # All unique

    def test_different_masters_different_seeds(self):
        """Different master seeds produce different scene seeds."""
        seed1 = get_scene_seed(12345, 0)
        seed2 = get_scene_seed(54321, 0)

        assert seed1 != seed2


class TestFrameSeed:
    """Tests for frame seed generation."""

    def test_frame_seed_deterministic(self):
        """Same inputs produce same frame seed."""
        seed1 = get_frame_seed(12345, 0)
        seed2 = get_frame_seed(12345, 0)

        assert seed1 == seed2

    def test_different_frames_different_seeds(self):
        """Different frame indices produce different seeds."""
        scene_seed = 12345

        seeds = [get_frame_seed(scene_seed, i) for i in range(5)]

        assert len(set(seeds)) == 5  # All unique
