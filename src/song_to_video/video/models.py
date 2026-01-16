"""Data models for video generation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np


class VideoModel(Enum):
    """Available video generation models."""

    LTX_VIDEO = "ltx-video"  # LTX-Video (primary)
    WAN_I2V = "wan-i2v"  # Wan Image-to-Video (fallback)
    WAN_T2V = "wan-t2v"  # Wan Text-to-Video


class VideoQuality(Enum):
    """Video quality presets."""

    DRAFT = "draft"  # 480p, fewer steps, fast
    STANDARD = "standard"  # 720p, balanced
    HIGH = "high"  # 1080p, more steps


class VideoWorkflow(Enum):
    """Video generation workflow types."""

    IMG2VID = "img2vid"  # Image-to-video (animate reference frames)
    TXT2VID = "txt2vid"  # Text-to-video direct


# VRAM requirements by model (in GB)
VRAM_REQUIREMENTS = {
    VideoModel.LTX_VIDEO: {
        VideoQuality.DRAFT: 10.0,
        VideoQuality.STANDARD: 14.0,
        VideoQuality.HIGH: 20.0,
    },
    VideoModel.WAN_I2V: {
        VideoQuality.DRAFT: 8.0,
        VideoQuality.STANDARD: 12.0,
        VideoQuality.HIGH: 16.0,
    },
    VideoModel.WAN_T2V: {
        VideoQuality.DRAFT: 8.0,
        VideoQuality.STANDARD: 12.0,
        VideoQuality.HIGH: 16.0,
    },
}

# Model IDs on Hugging Face
MODEL_IDS = {
    VideoModel.LTX_VIDEO: "Lightricks/LTX-Video",
    VideoModel.WAN_I2V: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    VideoModel.WAN_T2V: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

# Quality presets
QUALITY_PRESETS = {
    VideoQuality.DRAFT: {
        "width": 640,
        "height": 480,
        "num_frames": 25,  # ~1 second at 24fps
        "num_inference_steps": 20,
        "frame_rate": 24.0,
    },
    VideoQuality.STANDARD: {
        "width": 1280,
        "height": 720,
        "num_frames": 49,  # ~2 seconds at 24fps
        "num_inference_steps": 30,
        "frame_rate": 24.0,
    },
    VideoQuality.HIGH: {
        "width": 1920,
        "height": 1080,
        "num_frames": 97,  # ~4 seconds at 24fps
        "num_inference_steps": 40,
        "frame_rate": 24.0,
    },
}


@dataclass
class VideoConfig:
    """Configuration for video generation."""

    model: VideoModel = VideoModel.LTX_VIDEO
    quality: VideoQuality = VideoQuality.STANDARD
    workflow: VideoWorkflow = VideoWorkflow.IMG2VID

    # Video dimensions (will be adjusted to model requirements)
    width: int = 768
    height: int = 512

    # Frame settings
    num_frames: int = 49  # Must be 8n+1 for LTX
    frame_rate: float = 24.0

    # Generation settings
    num_inference_steps: int = 30
    guidance_scale: float = 4.0

    # Hardware settings
    use_cpu_offload: bool = True
    use_fp16: bool = True

    @classmethod
    def for_quality(cls, quality: VideoQuality) -> "VideoConfig":
        """Create config for a quality preset."""
        preset = QUALITY_PRESETS[quality]
        return cls(
            quality=quality,
            width=preset["width"],
            height=preset["height"],
            num_frames=preset["num_frames"],
            num_inference_steps=preset["num_inference_steps"],
            frame_rate=preset["frame_rate"],
        )

    @classmethod
    def for_duration(
        cls,
        duration_seconds: float,
        quality: VideoQuality = VideoQuality.STANDARD,
        frame_rate: float = 24.0,
    ) -> "VideoConfig":
        """Create config for a specific duration."""
        # Calculate frames needed (must be 8n+1 for LTX)
        raw_frames = int(duration_seconds * frame_rate)
        # Round to nearest valid value (8n+1)
        num_frames = ((raw_frames - 1) // 8) * 8 + 1
        # Minimum 9 frames (1 + 8)
        num_frames = max(9, num_frames)
        # Maximum ~20 seconds
        num_frames = min(num_frames, 481)

        preset = QUALITY_PRESETS[quality]
        return cls(
            quality=quality,
            width=preset["width"],
            height=preset["height"],
            num_frames=num_frames,
            num_inference_steps=preset["num_inference_steps"],
            frame_rate=frame_rate,
        )

    @property
    def duration_seconds(self) -> float:
        """Calculate video duration in seconds."""
        return self.num_frames / self.frame_rate

    @property
    def vram_required(self) -> float:
        """Get VRAM requirement for this config."""
        return VRAM_REQUIREMENTS.get(self.model, {}).get(self.quality, 12.0)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "quality": self.quality.value,
            "workflow": self.workflow.value,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class GeneratedClip:
    """A generated video clip."""

    frames: np.ndarray  # Shape: (num_frames, height, width, channels)
    prompt: str
    negative_prompt: str
    seed: int
    scene_id: int
    config: VideoConfig
    generation_time_ms: int = 0

    @property
    def num_frames(self) -> int:
        """Number of frames in the clip."""
        return self.frames.shape[0]

    @property
    def width(self) -> int:
        """Frame width."""
        return self.frames.shape[2]

    @property
    def height(self) -> int:
        """Frame height."""
        return self.frames.shape[1]

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.num_frames / self.config.frame_rate

    def save(self, output_path: Path) -> Path:
        """Save clip as video file using ffmpeg."""
        import subprocess
        import tempfile

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write frames to temp directory, then use ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save frames as images
            from PIL import Image

            for i, frame in enumerate(self.frames):
                img = Image.fromarray(frame.astype(np.uint8))
                img.save(tmpdir / f"frame_{i:05d}.png")

            # Use ffmpeg to create video
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(self.config.frame_rate),
                "-i",
                str(tmpdir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(output_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)

        return output_path

    def to_dict(self) -> dict:
        """Convert to dictionary (without frame data)."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "scene_id": self.scene_id,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "generation_time_ms": self.generation_time_ms,
            "config": self.config.to_dict(),
        }


@dataclass
class VideoClipSet:
    """Set of generated video clips for a song."""

    clips: list[GeneratedClip]
    master_seed: int
    output_dir: Path
    total_generation_time_ms: int = 0

    @property
    def clip_count(self) -> int:
        """Number of clips in the set."""
        return len(self.clips)

    @property
    def total_duration(self) -> float:
        """Total duration of all clips in seconds."""
        return sum(clip.duration_seconds for clip in self.clips)

    def get_clip_for_scene(self, scene_id: int) -> Optional[GeneratedClip]:
        """Get clip by scene ID."""
        for clip in self.clips:
            if clip.scene_id == scene_id:
                return clip
        return None

    def save_all(self, filename_pattern: str = "scene_{scene_id:03d}.mp4") -> list[Path]:
        """Save all clips to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for clip in self.clips:
            filename = filename_pattern.format(scene_id=clip.scene_id)
            path = self.output_dir / filename
            clip.save(path)
            paths.append(path)

        return paths

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "master_seed": self.master_seed,
            "clip_count": self.clip_count,
            "total_duration": self.total_duration,
            "total_generation_time_ms": self.total_generation_time_ms,
            "output_dir": str(self.output_dir),
            "clips": [clip.to_dict() for clip in self.clips],
        }


class VideoGenerationError(Exception):
    """Error during video generation."""

    pass
