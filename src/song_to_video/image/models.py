"""Data models for image generation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from PIL import Image


class ImageModel(Enum):
    """Supported image generation models."""

    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    SDXL = "sdxl"
    SD15 = "sd-1.5"


class ImageQuality(Enum):
    """Image generation quality presets."""

    DRAFT = "draft"  # Fast, lower quality
    STANDARD = "standard"  # Balanced
    HIGH = "high"  # Best quality, slower


@dataclass
class ImageConfig:
    """Configuration for image generation."""

    model: ImageModel = ImageModel.FLUX_SCHNELL
    width: int = 1024
    height: int = 576  # 16:9 aspect ratio for video
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    quality: ImageQuality = ImageQuality.STANDARD

    # Hardware-specific settings
    enable_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    use_fp16: bool = True

    @classmethod
    def for_quality(cls, quality: ImageQuality, model: ImageModel = ImageModel.FLUX_SCHNELL) -> "ImageConfig":
        """Create config for a specific quality level."""
        if quality == ImageQuality.DRAFT:
            return cls(
                model=model,
                width=768,
                height=432,
                num_inference_steps=4 if model == ImageModel.FLUX_SCHNELL else 15,
                guidance_scale=0.0 if model == ImageModel.FLUX_SCHNELL else 3.5,
                quality=quality,
            )
        elif quality == ImageQuality.HIGH:
            return cls(
                model=model,
                width=1280,
                height=720,
                num_inference_steps=50 if model == ImageModel.FLUX_DEV else 28,
                guidance_scale=3.5,
                quality=quality,
            )
        else:  # STANDARD
            return cls(
                model=model,
                width=1024,
                height=576,
                num_inference_steps=28,
                guidance_scale=3.5,
                quality=quality,
            )

    def to_dict(self) -> dict:
        return {
            "model": self.model.value,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "quality": self.quality.value,
            "enable_cpu_offload": self.enable_cpu_offload,
            "use_fp16": self.use_fp16,
        }


@dataclass
class GeneratedImage:
    """A generated image with metadata."""

    image: Image.Image
    prompt: str
    negative_prompt: str
    seed: int
    scene_id: int
    config: ImageConfig
    generation_time_ms: int = 0

    def save(self, path: Path) -> None:
        """Save the image to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "scene_id": self.scene_id,
            "width": self.image.width,
            "height": self.image.height,
            "generation_time_ms": self.generation_time_ms,
            "config": self.config.to_dict(),
        }


@dataclass
class ReferenceFrameSet:
    """A set of reference frames for all scenes."""

    frames: list[GeneratedImage]
    master_seed: int
    output_dir: Path
    total_generation_time_ms: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def get_frame_for_scene(self, scene_id: int) -> Optional[GeneratedImage]:
        """Get the reference frame for a specific scene."""
        for frame in self.frames:
            if frame.scene_id == scene_id:
                return frame
        return None

    def save_all(self) -> list[Path]:
        """Save all frames to disk and return paths."""
        paths = []
        for frame in self.frames:
            path = self.output_dir / f"scene_{frame.scene_id:03d}.png"
            frame.save(path)
            paths.append(path)
        return paths

    def to_dict(self) -> dict:
        return {
            "master_seed": self.master_seed,
            "frame_count": self.frame_count,
            "total_generation_time_ms": self.total_generation_time_ms,
            "output_dir": str(self.output_dir),
            "frames": [f.to_dict() for f in self.frames],
        }


class ImageGenerationError(Exception):
    """Raised when image generation fails."""

    pass
