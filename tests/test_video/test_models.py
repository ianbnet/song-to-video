"""Tests for video generation models."""

import pytest
import numpy as np
from pathlib import Path

from song_to_video.video.models import (
    VideoModel,
    VideoQuality,
    VideoWorkflow,
    VideoConfig,
    GeneratedClip,
    VideoClipSet,
    QUALITY_PRESETS,
    VRAM_REQUIREMENTS,
)


class TestVideoModel:
    """Tests for VideoModel enum."""

    def test_model_values(self):
        """Test model enum has expected values."""
        assert VideoModel.LTX_VIDEO.value == "ltx-video"
        assert VideoModel.WAN_I2V.value == "wan-i2v"
        assert VideoModel.WAN_T2V.value == "wan-t2v"


class TestVideoQuality:
    """Tests for VideoQuality enum."""

    def test_quality_values(self):
        """Test quality enum has expected values."""
        assert VideoQuality.DRAFT.value == "draft"
        assert VideoQuality.STANDARD.value == "standard"
        assert VideoQuality.HIGH.value == "high"


class TestVideoWorkflow:
    """Tests for VideoWorkflow enum."""

    def test_workflow_values(self):
        """Test workflow enum has expected values."""
        assert VideoWorkflow.IMG2VID.value == "img2vid"
        assert VideoWorkflow.TXT2VID.value == "txt2vid"


class TestVideoConfig:
    """Tests for VideoConfig data class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VideoConfig()

        assert config.model == VideoModel.LTX_VIDEO
        assert config.quality == VideoQuality.STANDARD
        assert config.workflow == VideoWorkflow.IMG2VID
        assert config.width == 768
        assert config.height == 512
        assert config.num_frames == 49
        assert config.frame_rate == 24.0

    def test_config_for_quality(self):
        """Test creating config from quality preset."""
        config = VideoConfig.for_quality(VideoQuality.DRAFT)

        assert config.quality == VideoQuality.DRAFT
        assert config.width == QUALITY_PRESETS[VideoQuality.DRAFT]["width"]
        assert config.height == QUALITY_PRESETS[VideoQuality.DRAFT]["height"]
        assert config.num_frames == QUALITY_PRESETS[VideoQuality.DRAFT]["num_frames"]

    def test_config_for_high_quality(self):
        """Test high quality preset."""
        config = VideoConfig.for_quality(VideoQuality.HIGH)

        assert config.quality == VideoQuality.HIGH
        assert config.width == 1920
        assert config.height == 1080
        assert config.num_frames == 97

    def test_config_for_duration(self):
        """Test creating config for specific duration."""
        config = VideoConfig.for_duration(5.0, VideoQuality.STANDARD)

        # 5 seconds at 24fps = 120 frames, round down to 8n+1 = 113 (14*8+1)
        # Formula: ((raw_frames - 1) // 8) * 8 + 1
        assert config.num_frames == 113
        assert config.quality == VideoQuality.STANDARD

    def test_config_for_duration_minimum(self):
        """Test minimum frame count."""
        config = VideoConfig.for_duration(0.1)  # Very short

        # Minimum is 9 frames
        assert config.num_frames == 9

    def test_config_for_duration_maximum(self):
        """Test maximum frame count capping."""
        config = VideoConfig.for_duration(30.0)  # 30 seconds

        # Maximum is 481 frames (~20 seconds)
        assert config.num_frames == 481

    def test_duration_seconds(self):
        """Test duration calculation."""
        config = VideoConfig(num_frames=49, frame_rate=24.0)

        assert abs(config.duration_seconds - (49 / 24.0)) < 0.01

    def test_vram_required(self):
        """Test VRAM requirement lookup."""
        config = VideoConfig(model=VideoModel.LTX_VIDEO, quality=VideoQuality.STANDARD)

        expected = VRAM_REQUIREMENTS[VideoModel.LTX_VIDEO][VideoQuality.STANDARD]
        assert config.vram_required == expected

    def test_config_to_dict(self):
        """Test config serialization."""
        config = VideoConfig(
            model=VideoModel.WAN_I2V,
            quality=VideoQuality.DRAFT,
            width=640,
            height=480,
        )

        data = config.to_dict()

        assert data["model"] == "wan-i2v"
        assert data["quality"] == "draft"
        assert data["width"] == 640
        assert data["height"] == 480


class TestGeneratedClip:
    """Tests for GeneratedClip data class."""

    def test_generated_clip_properties(self):
        """Test generated clip properties."""
        frames = np.zeros((25, 480, 640, 3), dtype=np.uint8)
        config = VideoConfig(frame_rate=24.0)

        clip = GeneratedClip(
            frames=frames,
            prompt="test prompt",
            negative_prompt="bad quality",
            seed=12345,
            scene_id=0,
            config=config,
            generation_time_ms=5000,
        )

        assert clip.num_frames == 25
        assert clip.width == 640
        assert clip.height == 480
        assert abs(clip.duration_seconds - (25 / 24.0)) < 0.01

    def test_generated_clip_to_dict(self):
        """Test generated clip serialization."""
        frames = np.zeros((25, 480, 640, 3), dtype=np.uint8)
        config = VideoConfig()

        clip = GeneratedClip(
            frames=frames,
            prompt="test",
            negative_prompt="",
            seed=1,
            scene_id=5,
            config=config,
            generation_time_ms=1000,
        )

        data = clip.to_dict()

        assert data["prompt"] == "test"
        assert data["seed"] == 1
        assert data["scene_id"] == 5
        assert data["num_frames"] == 25
        assert data["generation_time_ms"] == 1000


class TestVideoClipSet:
    """Tests for VideoClipSet data class."""

    def test_clip_count(self, tmp_path):
        """Test clip count property."""
        frames = np.zeros((25, 100, 100, 3), dtype=np.uint8)
        config = VideoConfig()

        clips = [
            GeneratedClip(frames, "p1", "", 1, 0, config),
            GeneratedClip(frames, "p2", "", 2, 1, config),
            GeneratedClip(frames, "p3", "", 3, 2, config),
        ]

        clip_set = VideoClipSet(
            clips=clips,
            master_seed=12345,
            output_dir=tmp_path,
        )

        assert clip_set.clip_count == 3

    def test_total_duration(self, tmp_path):
        """Test total duration calculation."""
        config = VideoConfig(frame_rate=24.0)
        frames1 = np.zeros((24, 100, 100, 3), dtype=np.uint8)  # 1 second
        frames2 = np.zeros((48, 100, 100, 3), dtype=np.uint8)  # 2 seconds

        clips = [
            GeneratedClip(frames1, "p1", "", 1, 0, config),
            GeneratedClip(frames2, "p2", "", 2, 1, config),
        ]

        clip_set = VideoClipSet(
            clips=clips,
            master_seed=12345,
            output_dir=tmp_path,
        )

        # Total should be 3 seconds
        assert abs(clip_set.total_duration - 3.0) < 0.1

    def test_get_clip_for_scene(self, tmp_path):
        """Test getting clip by scene ID."""
        frames = np.zeros((25, 100, 100, 3), dtype=np.uint8)
        config = VideoConfig()

        clips = [
            GeneratedClip(frames, "p1", "", 1, 0, config),
            GeneratedClip(frames, "p2", "", 2, 5, config),  # scene 5
            GeneratedClip(frames, "p3", "", 3, 10, config),
        ]

        clip_set = VideoClipSet(
            clips=clips,
            master_seed=12345,
            output_dir=tmp_path,
        )

        clip = clip_set.get_clip_for_scene(5)
        assert clip is not None
        assert clip.prompt == "p2"

        clip = clip_set.get_clip_for_scene(99)
        assert clip is None

    def test_to_dict(self, tmp_path):
        """Test clip set serialization."""
        frames = np.zeros((25, 100, 100, 3), dtype=np.uint8)
        config = VideoConfig()

        clips = [
            GeneratedClip(frames, "p1", "", 1, 0, config, generation_time_ms=500),
        ]

        clip_set = VideoClipSet(
            clips=clips,
            master_seed=99999,
            output_dir=tmp_path,
            total_generation_time_ms=500,
        )

        data = clip_set.to_dict()

        assert data["master_seed"] == 99999
        assert data["clip_count"] == 1
        assert data["total_generation_time_ms"] == 500
        assert len(data["clips"]) == 1
