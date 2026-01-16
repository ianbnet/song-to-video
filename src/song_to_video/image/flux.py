"""Flux model integration for image generation."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ..memory import (
    PipelinePhase,
    get_hardware_tier,
    HardwareTier,
    flush_vram,
)
from ..planning.models import Scene, StyleGuide, ScenePlan
from ..planning.seed import get_scene_seed
from .models import (
    ImageConfig,
    ImageModel,
    ImageQuality,
    GeneratedImage,
    ReferenceFrameSet,
    ImageGenerationError,
)
from .prompts import build_scene_prompt

logger = logging.getLogger(__name__)


def _is_huggingface_authenticated() -> bool:
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import HfApi
        HfApi().whoami()
        return True
    except Exception:
        return False


# Model IDs for Hugging Face
MODEL_IDS = {
    ImageModel.FLUX_DEV: "black-forest-labs/FLUX.1-dev",
    ImageModel.FLUX_SCHNELL: "black-forest-labs/FLUX.1-schnell",
    ImageModel.SDXL: "stabilityai/stable-diffusion-xl-base-1.0",
    ImageModel.SD15: "stable-diffusion-v1-5/stable-diffusion-v1-5",
}

# VRAM requirements by model (GB)
VRAM_REQUIREMENTS = {
    ImageModel.FLUX_DEV: 24.0,  # Full precision
    ImageModel.FLUX_SCHNELL: 12.0,  # Optimized
    ImageModel.SDXL: 8.0,
    ImageModel.SD15: 4.0,
}


class FluxGenerator:
    """
    Generates images using Flux or fallback models.

    Automatically configures for available hardware.
    """

    def __init__(
        self,
        model: Optional[ImageModel] = None,
        config: Optional[ImageConfig] = None,
    ):
        """
        Initialize the generator.

        Args:
            model: Model to use (auto-selects if None)
            config: Image config (uses defaults if None)
        """
        self._pipeline = None
        self._current_model = None

        # Auto-select model based on hardware
        if model is None:
            model = self._select_model_for_hardware()

        self.model = model
        self.config = config or self._get_config_for_hardware(model)

    def _select_model_for_hardware(self) -> ImageModel:
        """Select best model for current hardware."""
        tier = get_hardware_tier()
        hf_auth = _is_huggingface_authenticated()

        # If authenticated with HuggingFace, use FLUX (higher quality)
        # Otherwise fall back to SDXL (no auth required)
        if tier == HardwareTier.HIGH:
            if hf_auth:
                logger.info("HuggingFace authenticated - using FLUX.1-schnell")
                return ImageModel.FLUX_SCHNELL
            return ImageModel.SDXL
        elif tier == HardwareTier.MID:
            if hf_auth:
                logger.info("HuggingFace authenticated - using FLUX.1-schnell")
                return ImageModel.FLUX_SCHNELL
            return ImageModel.SDXL
        elif tier == HardwareTier.LOW:
            return ImageModel.SDXL
        else:
            return ImageModel.SD15

    def _get_config_for_hardware(self, model: ImageModel) -> ImageConfig:
        """Get optimized config for hardware tier."""
        tier = get_hardware_tier()
        config = ImageConfig.for_quality(ImageQuality.STANDARD, model)

        if tier == HardwareTier.LOW:
            # Aggressive optimizations for 8GB VRAM
            config.enable_cpu_offload = True
            config.enable_attention_slicing = True
            config.enable_vae_slicing = True
            config.enable_vae_tiling = True  # Additional VRAM savings
            config.width = 768
            config.height = 432
        elif tier == HardwareTier.MID:
            config.enable_vae_slicing = True
        elif tier == HardwareTier.CPU_ONLY:
            config.enable_cpu_offload = True
            config.enable_attention_slicing = True
            config.enable_vae_slicing = True
            config.enable_vae_tiling = True
            config.use_fp16 = False
            config.width = 512
            config.height = 288
            config.num_inference_steps = 10

        return config

    def _load_pipeline(self):
        """Load the image generation pipeline."""
        if self._pipeline is not None and self._current_model == self.model:
            return self._pipeline

        try:
            from diffusers import (
                FluxPipeline,
                StableDiffusionXLPipeline,
                StableDiffusionPipeline,
            )
        except ImportError:
            raise ImageGenerationError(
                "diffusers not installed. Run: pip install diffusers"
            )

        model_id = MODEL_IDS[self.model]
        logger.info(f"Loading image model: {model_id}")

        try:
            # Determine dtype
            dtype = torch.float16 if self.config.use_fp16 else torch.float32

            if self.model in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL):
                self._pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                )
            elif self.model == ImageModel.SDXL:
                self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if self.config.use_fp16 else None,
                )
            else:
                self._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                )

            # Apply optimizations
            if self.config.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline = self._pipeline.to("cuda")

            if self.config.enable_attention_slicing:
                self._pipeline.enable_attention_slicing()

            if self.config.enable_vae_slicing:
                self._pipeline.enable_vae_slicing()

            if self.config.enable_vae_tiling:
                self._pipeline.enable_vae_tiling()

            self._current_model = self.model
            logger.info("Image model loaded successfully")

            return self._pipeline

        except Exception as e:
            raise ImageGenerationError(f"Failed to load model: {e}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        scene_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> GeneratedImage:
        """
        Generate a single image.

        Args:
            prompt: Generation prompt
            negative_prompt: Things to avoid
            seed: Random seed for reproducibility
            scene_id: Scene this image is for
            width: Image width (uses config default if None)
            height: Image height (uses config default if None)
            num_inference_steps: Steps (uses config default if None)

        Returns:
            GeneratedImage with the result
        """
        width = width or self.config.width
        height = height or self.config.height
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        # Get VRAM requirement
        vram_required = VRAM_REQUIREMENTS.get(self.model, 8.0)
        if self.config.enable_cpu_offload:
            vram_required = min(4.0, vram_required)

        logger.info(f"Generating image: {width}x{height}, steps={num_inference_steps}, seed={seed}")

        start_time = time.time()

        with PipelinePhase("image_generation", required_vram_gb=vram_required):
            pipeline = self._load_pipeline()

            # Set up generator for reproducibility
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

            # Generate
            try:
                if self.model in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL):
                    # Flux doesn't use negative prompts in the same way
                    result = pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                    )
                else:
                    # SD/SDXL use negative prompts
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                    )

                image = result.images[0]

            except Exception as e:
                raise ImageGenerationError(f"Generation failed: {e}")

        generation_time = int((time.time() - start_time) * 1000)

        return GeneratedImage(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            scene_id=scene_id,
            config=self.config,
            generation_time_ms=generation_time,
        )

    def generate_for_scene(
        self,
        scene: Scene,
        style_guide: StyleGuide,
        master_seed: int,
    ) -> GeneratedImage:
        """
        Generate a reference frame for a scene.

        Args:
            scene: Scene to generate for
            style_guide: Visual style guide
            master_seed: Master seed for consistency

        Returns:
            GeneratedImage for the scene
        """
        # Build prompt
        prompt, negative_prompt = build_scene_prompt(scene, style_guide)

        # Get scene-specific seed
        scene_seed = get_scene_seed(master_seed, scene.id)

        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=scene_seed,
            scene_id=scene.id,
        )

    def unload(self):
        """Unload the model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._current_model = None
            flush_vram()
            logger.debug("Image model unloaded")


def generate_reference_frames(
    scene_plan: ScenePlan,
    output_dir: Path,
    model: Optional[ImageModel] = None,
    quality: ImageQuality = ImageQuality.STANDARD,
    skip_existing: bool = True,
) -> ReferenceFrameSet:
    """
    Generate reference frames for all scenes in a plan.

    Args:
        scene_plan: Scene plan with scenes and style guide
        output_dir: Directory to save frames
        model: Model to use (auto-selects if None)
        quality: Quality preset
        skip_existing: Skip scenes that already have frames

    Returns:
        ReferenceFrameSet with all generated frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = FluxGenerator(model=model)
    if quality != ImageQuality.STANDARD:
        generator.config = ImageConfig.for_quality(quality, generator.model)

    frames = []
    total_time = 0

    logger.info(f"Generating {len(scene_plan.scenes)} reference frames")

    for scene in scene_plan.scenes:
        # Check if frame already exists
        frame_path = output_dir / f"scene_{scene.id:03d}.png"
        if skip_existing and frame_path.exists():
            logger.info(f"Skipping scene {scene.id} (frame exists)")
            # Load existing frame
            existing_image = Image.open(frame_path)
            frames.append(GeneratedImage(
                image=existing_image,
                prompt="(loaded from disk)",
                negative_prompt="",
                seed=get_scene_seed(scene_plan.master_seed, scene.id),
                scene_id=scene.id,
                config=generator.config,
            ))
            continue

        logger.info(f"Generating frame for scene {scene.id}: {scene.section_type}")

        try:
            frame = generator.generate_for_scene(
                scene=scene,
                style_guide=scene_plan.style_guide,
                master_seed=scene_plan.master_seed,
            )
            frames.append(frame)
            total_time += frame.generation_time_ms

            # Save immediately
            frame.save(frame_path)
            logger.info(f"Saved frame: {frame_path}")

        except ImageGenerationError as e:
            logger.error(f"Failed to generate frame for scene {scene.id}: {e}")
            # Continue with other scenes

    # Cleanup
    generator.unload()

    return ReferenceFrameSet(
        frames=frames,
        master_seed=scene_plan.master_seed,
        output_dir=output_dir,
        total_generation_time_ms=total_time,
    )
