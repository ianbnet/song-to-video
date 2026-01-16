"""Video generation module for animating reference frames."""

from .models import (
    VideoModel,
    VideoQuality,
    VideoWorkflow,
    VideoConfig,
    GeneratedClip,
    VideoClipSet,
    VideoGenerationError,
    MODEL_IDS,
    VRAM_REQUIREMENTS,
    QUALITY_PRESETS,
)
from .generator import (
    VideoGenerator,
    generate_scene_clips,
    DEFAULT_NEGATIVE_PROMPT,
)
from .compose import (
    VideoComposer,
    CompositionConfig,
    CompositionResult,
    CompositionError,
    FFmpegError,
    ClipMismatchError,
    compose_video,
    COMPOSITION_QUALITY,
)

__all__ = [
    # Models
    "VideoModel",
    "VideoQuality",
    "VideoWorkflow",
    "VideoConfig",
    "GeneratedClip",
    "VideoClipSet",
    "VideoGenerationError",
    # Constants
    "MODEL_IDS",
    "VRAM_REQUIREMENTS",
    "QUALITY_PRESETS",
    "DEFAULT_NEGATIVE_PROMPT",
    # Generator
    "VideoGenerator",
    "generate_scene_clips",
    # Composition
    "VideoComposer",
    "CompositionConfig",
    "CompositionResult",
    "CompositionError",
    "FFmpegError",
    "ClipMismatchError",
    "compose_video",
    "COMPOSITION_QUALITY",
]
