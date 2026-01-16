"""Video generation using LTX-Video and Wan models."""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from ..memory.hardware import get_hardware_tier, HardwareTier, detect_gpu
from ..memory.phase import PipelinePhase
from ..memory import flush_vram
from ..planning.models import Scene, StyleGuide
from ..planning.seed import get_scene_seed
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
)

logger = logging.getLogger(__name__)

# Negative prompts for video generation
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, logo, static, frozen"
)


class VideoGenerator:
    """
    Video generator supporting multiple backends.

    Automatically selects appropriate model based on hardware tier:
    - HIGH (24GB+): LTX-Video at high quality
    - MID (12-16GB): LTX-Video at standard quality with offloading
    - LOW (8GB): Wan at draft quality with aggressive offloading
    """

    def __init__(
        self,
        model: Optional[VideoModel] = None,
        config: Optional[VideoConfig] = None,
    ):
        """
        Initialize video generator.

        Args:
            model: Video model to use (auto-selected if None)
            config: Video configuration (auto-created if None)
        """
        self.hardware_tier = get_hardware_tier(detect_gpu())
        self.config = config or VideoConfig()

        # Auto-select model based on hardware if not specified
        if model is None:
            model = self._select_model_for_hardware()
        self.model = model
        self.config.model = model

        self._pipeline = None
        logger.info(
            f"VideoGenerator initialized: model={model.value}, "
            f"hardware_tier={self.hardware_tier.name}"
        )

    def _select_model_for_hardware(self) -> VideoModel:
        """Select appropriate model for detected hardware."""
        if self.hardware_tier == HardwareTier.HIGH:
            return VideoModel.LTX_VIDEO
        elif self.hardware_tier == HardwareTier.MID:
            return VideoModel.LTX_VIDEO
        else:
            # LOW tier - use Wan which has lower VRAM requirements
            return VideoModel.WAN_I2V

    def _select_quality_for_hardware(self) -> VideoQuality:
        """Select appropriate quality for detected hardware."""
        if self.hardware_tier == HardwareTier.HIGH:
            return VideoQuality.HIGH
        elif self.hardware_tier == HardwareTier.MID:
            return VideoQuality.STANDARD
        else:
            return VideoQuality.DRAFT

    def _load_pipeline(self):
        """Load the video generation pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        model_id = MODEL_IDS[self.model]
        dtype = torch.float16 if self.config.use_fp16 else torch.float32

        logger.info(f"Loading video model: {model_id}")

        if self.model == VideoModel.LTX_VIDEO:
            from diffusers import LTXImageToVideoPipeline, LTXPipeline

            if self.config.workflow == VideoWorkflow.IMG2VID:
                self._pipeline = LTXImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                )
            else:
                self._pipeline = LTXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                )
        elif self.model in (VideoModel.WAN_I2V, VideoModel.WAN_T2V):
            from diffusers import WanImageToVideoPipeline, WanPipeline

            if self.config.workflow == VideoWorkflow.IMG2VID:
                self._pipeline = WanImageToVideoPipeline.from_pretrained(
                    MODEL_IDS[VideoModel.WAN_I2V],
                    torch_dtype=dtype,
                )
            else:
                self._pipeline = WanPipeline.from_pretrained(
                    MODEL_IDS[VideoModel.WAN_T2V],
                    torch_dtype=dtype,
                )
        else:
            raise VideoGenerationError(f"Unsupported model: {self.model}")

        # Enable memory optimizations
        if self.config.use_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
            logger.info("Enabled CPU offloading for video model")

        return self._pipeline

    def _unload_pipeline(self):
        """Unload the pipeline to free VRAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            flush_vram()

    def generate_from_image(
        self,
        image: Union[Image.Image, np.ndarray, Path],
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        scene_id: int = 0,
        num_frames: Optional[int] = None,
    ) -> GeneratedClip:
        """
        Generate video from a reference image (image-to-video).

        Args:
            image: Input reference image
            prompt: Text prompt describing desired motion/action
            negative_prompt: Things to avoid
            seed: Random seed for reproducibility
            scene_id: Scene identifier
            num_frames: Number of frames to generate (uses config default if None)

        Returns:
            GeneratedClip with video frames
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Resize image to match config dimensions
        image = image.resize((self.config.width, self.config.height), Image.LANCZOS)

        # Set up generation parameters
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        num_frames = num_frames or self.config.num_frames
        vram_required = self.config.vram_required

        logger.info(
            f"Generating video: {self.config.width}x{self.config.height}, "
            f"{num_frames} frames, seed={seed}"
        )

        start_time = time.time()

        with PipelinePhase("video_generation", required_vram_gb=vram_required):
            pipeline = self._load_pipeline()

            # Set up generator for reproducibility
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Generate video
            if self.model == VideoModel.LTX_VIDEO:
                output = pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.config.width,
                    height=self.config.height,
                    num_frames=num_frames,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                    output_type="np",
                )
                frames = output.frames[0]  # Shape: (num_frames, H, W, C)
            elif self.model == VideoModel.WAN_I2V:
                output = pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.config.width,
                    height=self.config.height,
                    num_frames=num_frames,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                    output_type="np",
                )
                frames = output.frames[0]
            else:
                raise VideoGenerationError(f"Image-to-video not supported for {self.model}")

            # Convert to uint8 if needed
            if frames.dtype != np.uint8:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)

        generation_time_ms = int((time.time() - start_time) * 1000)

        return GeneratedClip(
            frames=frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            scene_id=scene_id,
            config=self.config,
            generation_time_ms=generation_time_ms,
        )

    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        scene_id: int = 0,
        num_frames: Optional[int] = None,
    ) -> GeneratedClip:
        """
        Generate video directly from text (text-to-video).

        Args:
            prompt: Text prompt describing the scene
            negative_prompt: Things to avoid
            seed: Random seed for reproducibility
            scene_id: Scene identifier
            num_frames: Number of frames to generate (uses config default if None)

        Returns:
            GeneratedClip with video frames
        """
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        num_frames = num_frames or self.config.num_frames
        vram_required = self.config.vram_required

        logger.info(
            f"Generating video from text: {self.config.width}x{self.config.height}, "
            f"{num_frames} frames, seed={seed}"
        )

        start_time = time.time()

        # Temporarily switch to text-to-video workflow
        original_workflow = self.config.workflow
        self.config.workflow = VideoWorkflow.TXT2VID
        self._unload_pipeline()  # Reload with new workflow

        try:
            with PipelinePhase("video_generation", required_vram_gb=vram_required):
                pipeline = self._load_pipeline()

                generator = torch.Generator(device="cpu").manual_seed(seed)

                if self.model == VideoModel.LTX_VIDEO:
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=self.config.width,
                        height=self.config.height,
                        num_frames=num_frames,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                        output_type="np",
                    )
                    frames = output.frames[0]
                elif self.model == VideoModel.WAN_T2V:
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=self.config.width,
                        height=self.config.height,
                        num_frames=num_frames,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                        output_type="np",
                    )
                    frames = output.frames[0]
                else:
                    raise VideoGenerationError(f"Text-to-video not supported for {self.model}")

                if frames.dtype != np.uint8:
                    frames = (frames * 255).clip(0, 255).astype(np.uint8)
        finally:
            # Restore original workflow
            self.config.workflow = original_workflow
            self._unload_pipeline()

        generation_time_ms = int((time.time() - start_time) * 1000)

        return GeneratedClip(
            frames=frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            scene_id=scene_id,
            config=self.config,
            generation_time_ms=generation_time_ms,
        )

    def generate_for_scene(
        self,
        scene: Scene,
        style_guide: StyleGuide,
        master_seed: int,
        reference_image: Optional[Union[Image.Image, Path]] = None,
    ) -> GeneratedClip:
        """
        Generate video for a scene using its prompt and style.

        Args:
            scene: Scene with prompt and timing info
            style_guide: Visual style guide
            master_seed: Master seed for consistency
            reference_image: Optional reference frame (for img2vid workflow)

        Returns:
            GeneratedClip for the scene
        """
        # Build motion-focused prompt for video
        motion_prompt = self._build_video_prompt(scene, style_guide)

        # Get scene-specific seed
        seed = get_scene_seed(master_seed, scene.id)

        # Calculate frames for scene duration
        duration = scene.duration
        num_frames = self._frames_for_duration(duration)

        # Build negative prompt
        negative_parts = [DEFAULT_NEGATIVE_PROMPT]
        if style_guide.negative_prompts:
            negative_parts.extend(style_guide.negative_prompts)
        negative_prompt = ", ".join(negative_parts)

        if reference_image is not None and self.config.workflow == VideoWorkflow.IMG2VID:
            return self.generate_from_image(
                image=reference_image,
                prompt=motion_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                scene_id=scene.id,
                num_frames=num_frames,
            )
        else:
            return self.generate_from_text(
                prompt=motion_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                scene_id=scene.id,
                num_frames=num_frames,
            )

    def _build_video_prompt(self, scene: Scene, style_guide: StyleGuide) -> str:
        """Build a motion-focused prompt for video generation."""
        parts = []

        # Scene description (may include motion)
        parts.append(scene.prompt)

        # Add camera movement if specified
        if scene.camera_movement:
            parts.append(scene.camera_movement)

        # Add style elements
        parts.append(style_guide.aesthetic)
        parts.append(f"{style_guide.lighting} lighting")

        # Add energy-based motion hint
        if scene.energy > 0.7:
            parts.append("dynamic movement, energetic motion")
        elif scene.energy > 0.4:
            parts.append("smooth motion, natural movement")
        else:
            parts.append("subtle motion, gentle movement")

        return ", ".join(parts)

    def _frames_for_duration(self, duration_seconds: float) -> int:
        """Calculate number of frames for a duration (must be 8n+1 for LTX)."""
        raw_frames = int(duration_seconds * self.config.frame_rate)
        # Round to nearest valid value (8n+1)
        num_frames = ((raw_frames - 1) // 8) * 8 + 1
        # Minimum 9 frames, maximum 481 frames (~20 seconds)
        num_frames = max(9, min(num_frames, 481))
        return num_frames


def generate_scene_clips(
    scenes: list[Scene],
    style_guide: StyleGuide,
    master_seed: int,
    output_dir: Path,
    reference_frames: Optional[dict[int, Path]] = None,
    config: Optional[VideoConfig] = None,
    skip_existing: bool = True,
) -> VideoClipSet:
    """
    Generate video clips for all scenes.

    Args:
        scenes: List of scenes to generate
        style_guide: Visual style guide
        master_seed: Master seed for consistency
        output_dir: Directory to save clips
        reference_frames: Optional dict mapping scene_id to reference frame path
        config: Video configuration
        skip_existing: Skip scenes that already have clips

    Returns:
        VideoClipSet with all generated clips
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator(config=config)
    clips = []
    total_time = 0

    for scene in scenes:
        output_path = output_dir / f"scene_{scene.id:03d}.mp4"

        if skip_existing and output_path.exists():
            logger.info(f"Skipping scene {scene.id} (exists)")
            continue

        logger.info(f"Generating clip for scene {scene.id}")

        # Get reference frame if available
        ref_frame = None
        if reference_frames and scene.id in reference_frames:
            ref_frame = reference_frames[scene.id]

        clip = generator.generate_for_scene(
            scene=scene,
            style_guide=style_guide,
            master_seed=master_seed,
            reference_image=ref_frame,
        )

        clips.append(clip)
        total_time += clip.generation_time_ms

        # Save clip
        clip.save(output_path)
        logger.info(f"Saved clip: {output_path}")

    return VideoClipSet(
        clips=clips,
        master_seed=master_seed,
        output_dir=output_dir,
        total_generation_time_ms=total_time,
    )
