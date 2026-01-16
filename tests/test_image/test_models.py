"""Tests for image generation models."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from song_to_video.image.models import (
    ImageModel,
    ImageQuality,
    ImageConfig,
    GeneratedImage,
    ReferenceFrameSet,
)


class TestImageModel:
    """Tests for ImageModel enum."""

    def test_model_values(self):
        """Test model enum has expected values."""
        assert ImageModel.FLUX_DEV.value == "flux-dev"
        assert ImageModel.FLUX_SCHNELL.value == "flux-schnell"
        assert ImageModel.SDXL.value == "sdxl"
        assert ImageModel.SD15.value == "sd-1.5"


class TestImageQuality:
    """Tests for ImageQuality enum."""

    def test_quality_values(self):
        """Test quality enum has expected values."""
        assert ImageQuality.DRAFT.value == "draft"
        assert ImageQuality.STANDARD.value == "standard"
        assert ImageQuality.HIGH.value == "high"


class TestImageConfig:
    """Tests for ImageConfig data class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ImageConfig()

        assert config.model == ImageModel.FLUX_SCHNELL
        assert config.width == 1024
        assert config.height == 576
        assert config.num_inference_steps == 28
        assert config.use_fp16 is True

    def test_config_for_draft_quality(self):
        """Test draft quality preset."""
        config = ImageConfig.for_quality(ImageQuality.DRAFT)

        assert config.quality == ImageQuality.DRAFT
        assert config.width == 768
        assert config.height == 432
        assert config.num_inference_steps == 4  # Schnell default

    def test_config_for_high_quality(self):
        """Test high quality preset."""
        config = ImageConfig.for_quality(ImageQuality.HIGH)

        assert config.quality == ImageQuality.HIGH
        assert config.width == 1280
        assert config.height == 720
        assert config.num_inference_steps == 28

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ImageConfig(
            model=ImageModel.SDXL,
            width=512,
            height=512,
        )

        data = config.to_dict()

        assert data["model"] == "sdxl"
        assert data["width"] == 512
        assert data["height"] == 512


class TestGeneratedImage:
    """Tests for GeneratedImage data class."""

    def test_generated_image_to_dict(self):
        """Test generated image serialization."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")

        config = ImageConfig()
        gen_image = GeneratedImage(
            image=img,
            prompt="test prompt",
            negative_prompt="bad quality",
            seed=12345,
            scene_id=0,
            config=config,
            generation_time_ms=1000,
        )

        data = gen_image.to_dict()

        assert data["prompt"] == "test prompt"
        assert data["seed"] == 12345
        assert data["scene_id"] == 0
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["generation_time_ms"] == 1000

    def test_generated_image_save(self, tmp_path):
        """Test saving generated image."""
        img = Image.new("RGB", (100, 100), color="blue")
        config = ImageConfig()

        gen_image = GeneratedImage(
            image=img,
            prompt="test",
            negative_prompt="",
            seed=1,
            scene_id=0,
            config=config,
        )

        save_path = tmp_path / "subdir" / "test.png"
        gen_image.save(save_path)

        assert save_path.exists()
        loaded = Image.open(save_path)
        assert loaded.size == (100, 100)


class TestReferenceFrameSet:
    """Tests for ReferenceFrameSet data class."""

    def test_frame_count(self, tmp_path):
        """Test frame count property."""
        img = Image.new("RGB", (100, 100))
        config = ImageConfig()

        frames = [
            GeneratedImage(img, "p1", "", 1, 0, config),
            GeneratedImage(img, "p2", "", 2, 1, config),
            GeneratedImage(img, "p3", "", 3, 2, config),
        ]

        frame_set = ReferenceFrameSet(
            frames=frames,
            master_seed=12345,
            output_dir=tmp_path,
        )

        assert frame_set.frame_count == 3

    def test_get_frame_for_scene(self, tmp_path):
        """Test getting frame by scene ID."""
        img = Image.new("RGB", (100, 100))
        config = ImageConfig()

        frames = [
            GeneratedImage(img, "p1", "", 1, 0, config),
            GeneratedImage(img, "p2", "", 2, 5, config),  # scene 5
            GeneratedImage(img, "p3", "", 3, 10, config),
        ]

        frame_set = ReferenceFrameSet(
            frames=frames,
            master_seed=12345,
            output_dir=tmp_path,
        )

        frame = frame_set.get_frame_for_scene(5)
        assert frame is not None
        assert frame.prompt == "p2"

        frame = frame_set.get_frame_for_scene(99)
        assert frame is None

    def test_save_all(self, tmp_path):
        """Test saving all frames."""
        config = ImageConfig()

        frames = [
            GeneratedImage(Image.new("RGB", (50, 50), "red"), "p1", "", 1, 0, config),
            GeneratedImage(Image.new("RGB", (50, 50), "green"), "p2", "", 2, 1, config),
        ]

        frame_set = ReferenceFrameSet(
            frames=frames,
            master_seed=12345,
            output_dir=tmp_path,
        )

        paths = frame_set.save_all()

        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert paths[0].name == "scene_000.png"
        assert paths[1].name == "scene_001.png"

    def test_to_dict(self, tmp_path):
        """Test frame set serialization."""
        img = Image.new("RGB", (100, 100))
        config = ImageConfig()

        frames = [
            GeneratedImage(img, "p1", "", 1, 0, config, generation_time_ms=500),
        ]

        frame_set = ReferenceFrameSet(
            frames=frames,
            master_seed=99999,
            output_dir=tmp_path,
            total_generation_time_ms=500,
        )

        data = frame_set.to_dict()

        assert data["master_seed"] == 99999
        assert data["frame_count"] == 1
        assert data["total_generation_time_ms"] == 500
        assert len(data["frames"]) == 1
