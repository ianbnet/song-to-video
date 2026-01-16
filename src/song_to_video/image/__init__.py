"""Image generation module for reference frames."""

from .models import (
    ImageModel,
    ImageQuality,
    ImageConfig,
    GeneratedImage,
    ReferenceFrameSet,
    ImageGenerationError,
)
from .prompts import (
    PromptTemplate,
    build_scene_prompt,
    get_template_for_style,
    CINEMATIC_TEMPLATE,
    ANIMATED_TEMPLATE,
    ABSTRACT_TEMPLATE,
    DREAMY_TEMPLATE,
)
from .flux import (
    FluxGenerator,
    generate_reference_frames,
    MODEL_IDS,
    VRAM_REQUIREMENTS,
)

__all__ = [
    # Models
    "ImageModel",
    "ImageQuality",
    "ImageConfig",
    "GeneratedImage",
    "ReferenceFrameSet",
    "ImageGenerationError",
    # Prompts
    "PromptTemplate",
    "build_scene_prompt",
    "get_template_for_style",
    "CINEMATIC_TEMPLATE",
    "ANIMATED_TEMPLATE",
    "ABSTRACT_TEMPLATE",
    "DREAMY_TEMPLATE",
    # Generator
    "FluxGenerator",
    "generate_reference_frames",
    "MODEL_IDS",
    "VRAM_REQUIREMENTS",
]
