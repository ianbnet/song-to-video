"""Video composition - assembles clips into final music video."""

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..planning.models import Scene, ScenePlan, SceneTransition

logger = logging.getLogger(__name__)


class CompositionError(Exception):
    """Error during video composition."""

    pass


class FFmpegError(CompositionError):
    """FFmpeg command failed."""

    def __init__(self, message: str, returncode: int, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class ClipMismatchError(CompositionError):
    """Scene plan doesn't match available clips."""

    pass


@dataclass
class CompositionConfig:
    """Configuration for video composition."""

    # Output encoding
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"
    crf: int = 18  # Quality (lower = better, 18 = visually lossless)
    preset: str = "medium"  # Encoding speed/quality tradeoff

    # Audio settings
    audio_bitrate: str = "192k"

    # Transition settings
    transition_duration_ms: int = 500  # Duration for FADE/DISSOLVE

    # Output resolution (None = use input resolution)
    output_width: Optional[int] = None
    output_height: Optional[int] = None


# Quality presets for composition
COMPOSITION_QUALITY = {
    "draft": CompositionConfig(crf=28, preset="ultrafast"),
    "standard": CompositionConfig(crf=18, preset="medium"),
    "high": CompositionConfig(crf=15, preset="slow"),
}


@dataclass
class CompositionResult:
    """Result of video composition."""

    output_path: Path
    duration_seconds: float
    clip_count: int
    transitions_applied: dict = field(default_factory=dict)
    composition_time_ms: int = 0
    file_size_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "output_path": str(self.output_path),
            "duration_seconds": self.duration_seconds,
            "clip_count": self.clip_count,
            "transitions_applied": self.transitions_applied,
            "composition_time_ms": self.composition_time_ms,
            "file_size_bytes": self.file_size_bytes,
        }


class VideoComposer:
    """
    Composes video clips into a final synchronized music video.

    Uses FFmpeg for all video operations to maintain consistency
    with existing clip generation and avoid additional dependencies.
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        """
        Initialize the video composer.

        Args:
            config: Composition configuration (uses defaults if None)
        """
        self.config = config or CompositionConfig()
        self._validate_ffmpeg()

    def _validate_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"FFmpeg found: {result.stdout.split(chr(10))[0]}")
        except FileNotFoundError:
            raise CompositionError(
                "FFmpeg not found. Please install FFmpeg to use video composition."
            )
        except subprocess.CalledProcessError as e:
            raise CompositionError(f"FFmpeg check failed: {e.stderr}")

    def compose(
        self,
        scene_plan: ScenePlan,
        clips_dir: Path,
        audio_path: Path,
        output_path: Path,
    ) -> CompositionResult:
        """
        Compose clips into final video with audio.

        Args:
            scene_plan: Scene plan with timing and transitions
            clips_dir: Directory containing scene_XXX.mp4 files
            audio_path: Path to original audio file
            output_path: Path for output video

        Returns:
            CompositionResult with metadata
        """
        clips_dir = Path(clips_dir)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        start_time = time.time()

        # Discover and validate clips
        clips = self._discover_clips(clips_dir, scene_plan)
        logger.info(f"Found {len(clips)} clips for {len(scene_plan.scenes)} scenes")

        # Get audio duration for sync
        audio_duration = self._get_audio_duration(audio_path)
        logger.info(f"Audio duration: {audio_duration:.2f}s")

        # Count transitions
        transitions_count = self._count_transitions(clips)

        # Check if we need complex filter (has non-CUT transitions)
        has_complex_transitions = any(
            scene.transition_out != SceneTransition.CUT
            for scene, _ in clips[:-1]  # Last clip's transition_out doesn't matter
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Note: xfade transitions have compatibility issues with varying inputs
        # For now, always use simple concat which is more reliable
        # TODO: Re-enable complex transitions after fixing frame rate/timebase issues
        if has_complex_transitions:
            logger.warning("Complex transitions detected but using simple concat for reliability")

        self._compose_with_concat(clips, audio_path, output_path)

        composition_time_ms = int((time.time() - start_time) * 1000)

        # Get output file size
        file_size = output_path.stat().st_size if output_path.exists() else 0

        # Get output duration
        output_duration = self._get_video_duration(output_path)

        result = CompositionResult(
            output_path=output_path,
            duration_seconds=output_duration,
            clip_count=len(clips),
            transitions_applied=transitions_count,
            composition_time_ms=composition_time_ms,
            file_size_bytes=file_size,
        )

        logger.info(
            f"Composition complete: {output_path} "
            f"({output_duration:.2f}s, {file_size / 1024 / 1024:.1f}MB)"
        )

        return result

    def _discover_clips(
        self,
        clips_dir: Path,
        scene_plan: ScenePlan,
    ) -> list[tuple[Scene, Path]]:
        """Match scenes to clip files in order."""
        clips = []
        missing = []

        for scene in scene_plan.scenes:
            clip_path = clips_dir / f"scene_{scene.id:03d}.mp4"
            if clip_path.exists():
                clips.append((scene, clip_path))
            else:
                missing.append(scene.id)

        if missing:
            raise ClipMismatchError(
                f"Missing clips for scenes: {missing}. "
                f"Expected files like scene_000.mp4 in {clips_dir}"
            )

        # Sort by scene ID
        clips.sort(key=lambda x: x[0].id)
        return clips

    def _count_transitions(
        self, clips: list[tuple[Scene, Path]]
    ) -> dict[str, int]:
        """Count transitions by type."""
        counts: dict[str, int] = {}
        for scene, _ in clips[:-1]:  # Last clip's transition_out not applied
            t = scene.transition_out.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffmpeg."""
        # Use ffmpeg to get duration (more portable than ffprobe)
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-f", "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Parse duration from stderr (ffmpeg outputs info to stderr)
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            hours, mins, secs, frac = match.groups()
            return int(hours) * 3600 + int(mins) * 60 + int(secs) + int(frac) / 100
        raise CompositionError(f"Failed to get audio duration from {audio_path}")

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-f", "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            hours, mins, secs, frac = match.groups()
            return int(hours) * 3600 + int(mins) * 60 + int(secs) + int(frac) / 100
        return 0.0

    def _compose_with_concat(
        self,
        clips: list[tuple[Scene, Path]],
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Compose using simple concat demuxer (all CUT transitions)."""
        logger.info("Using simple concat (all CUT transitions)")

        # Create concat file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for _, clip_path in clips:
                # FFmpeg concat requires absolute paths and escaped quotes
                abs_path = str(clip_path.resolve())
                escaped_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            concat_file = f.name

        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-i", str(audio_path),
                "-c:v", self.config.video_codec,
                "-pix_fmt", self.config.pixel_format,
                "-crf", str(self.config.crf),
                "-preset", self.config.preset,
                "-c:a", self.config.audio_codec,
                "-b:a", self.config.audio_bitrate,
                "-map", "0:v",
                "-map", "1:a",
                "-shortest",
                str(output_path),
            ]

            self._run_ffmpeg(cmd, "concat composition")

        finally:
            os.unlink(concat_file)

    def _compose_with_filter_complex(
        self,
        clips: list[tuple[Scene, Path]],
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Compose using filter_complex for transitions."""
        logger.info("Using filter_complex for transitions")

        # Build input list
        inputs = []
        for _, clip_path in clips:
            inputs.extend(["-i", str(clip_path)])
        inputs.extend(["-i", str(audio_path)])

        # Build filter complex
        filter_complex = self._build_filter_complex(clips)

        # Audio input is the last one
        audio_index = len(clips)

        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", f"{audio_index}:a",
            "-c:v", self.config.video_codec,
            "-pix_fmt", self.config.pixel_format,
            "-crf", str(self.config.crf),
            "-preset", self.config.preset,
            "-c:a", self.config.audio_codec,
            "-b:a", self.config.audio_bitrate,
            "-shortest",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "filter_complex composition")

    def _build_filter_complex(
        self,
        clips: list[tuple[Scene, Path]],
    ) -> str:
        """
        Build FFmpeg filter_complex string for transitions.

        For N clips with transitions, we chain them together:
        [0:v][1:v]xfade=...[v01]; [v01][2:v]xfade=...[v012]; etc.
        """
        if len(clips) == 1:
            return "[0:v]copy[outv]"

        filter_parts = []
        transition_duration = self.config.transition_duration_ms / 1000.0

        # First, normalize all clips to same format/timebase for xfade compatibility
        # This prevents timebase mismatch errors
        normalized_inputs = []
        for i in range(len(clips)):
            normalized_inputs.append(
                f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[n{i}]"
            )
        filter_parts.extend(normalized_inputs)

        # Track cumulative offset for transition timing
        cumulative_duration = 0.0

        for i in range(len(clips) - 1):
            scene, clip_path = clips[i]
            next_scene, _ = clips[i + 1]

            # Get clip duration
            clip_duration = self._get_video_duration(clip_path)
            transition = scene.transition_out

            # Input labels (use normalized inputs)
            if i == 0:
                input_a = f"[n{i}]"
            else:
                input_a = f"[v{i-1}]"

            input_b = f"[n{i+1}]"

            # Output label
            if i == len(clips) - 2:
                output = "[outv]"
            else:
                output = f"[v{i}]"

            # Calculate offset for transition (when transition starts)
            offset = cumulative_duration + clip_duration - transition_duration
            offset = max(0, offset)  # Ensure non-negative

            if transition == SceneTransition.CUT:
                # Simple concat
                filter_parts.append(
                    f"{input_a}{input_b}concat=n=2:v=1:a=0{output}"
                )
            elif transition == SceneTransition.FADE:
                # Fade out then fade in (through black)
                # We use xfade with 'fade' transition type
                filter_parts.append(
                    f"{input_a}{input_b}xfade=transition=fade:"
                    f"duration={transition_duration}:offset={offset:.3f}{output}"
                )
            elif transition == SceneTransition.DISSOLVE:
                # Crossfade/dissolve
                filter_parts.append(
                    f"{input_a}{input_b}xfade=transition=dissolve:"
                    f"duration={transition_duration}:offset={offset:.3f}{output}"
                )
            else:
                # Default to cut for unsupported transitions
                logger.warning(f"Unsupported transition {transition}, using CUT")
                filter_parts.append(
                    f"{input_a}{input_b}concat=n=2:v=1:a=0{output}"
                )

            # Update cumulative duration (subtract transition overlap)
            if transition in (SceneTransition.FADE, SceneTransition.DISSOLVE):
                cumulative_duration += clip_duration - transition_duration
            else:
                cumulative_duration += clip_duration

        return ";".join(filter_parts)

    def _run_ffmpeg(self, cmd: list[str], description: str) -> None:
        """Execute FFmpeg command with error handling."""
        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise FFmpegError(
                f"FFmpeg {description} failed",
                result.returncode,
                result.stderr,
            )

        logger.debug(f"FFmpeg {description} completed successfully")


def compose_video(
    scene_plan: ScenePlan,
    clips_dir: Path,
    audio_path: Path,
    output_path: Path,
    quality: str = "standard",
) -> CompositionResult:
    """
    Convenience function to compose video with a quality preset.

    Args:
        scene_plan: Scene plan with timing and transitions
        clips_dir: Directory containing video clips
        audio_path: Path to audio file
        output_path: Output video path
        quality: Quality preset (draft, standard, high)

    Returns:
        CompositionResult with metadata
    """
    config = COMPOSITION_QUALITY.get(quality, CompositionConfig())
    composer = VideoComposer(config=config)
    return composer.compose(scene_plan, clips_dir, audio_path, output_path)
