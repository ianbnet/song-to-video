"""Tests for video composition module."""

import pytest
import subprocess
import shutil
from pathlib import Path
from dataclasses import field

from song_to_video.video.compose import (
    VideoComposer,
    CompositionConfig,
    CompositionResult,
    CompositionError,
    ClipMismatchError,
    COMPOSITION_QUALITY,
)
from song_to_video.planning.models import (
    Scene,
    ScenePlan,
    StyleGuide,
    ColorPalette,
    VisualStyle,
    SceneTransition,
    NarrativeAnalysis,
)


# Check if FFmpeg is available
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
SKIP_FFMPEG = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not installed")


class TestCompositionConfig:
    """Tests for CompositionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompositionConfig()

        assert config.video_codec == "libx264"
        assert config.audio_codec == "aac"
        assert config.crf == 18
        assert config.preset == "medium"
        assert config.transition_duration_ms == 500

    def test_quality_presets(self):
        """Test quality preset configurations."""
        assert "draft" in COMPOSITION_QUALITY
        assert "standard" in COMPOSITION_QUALITY
        assert "high" in COMPOSITION_QUALITY

        draft = COMPOSITION_QUALITY["draft"]
        assert draft.crf > COMPOSITION_QUALITY["standard"].crf  # Lower quality = higher CRF

        high = COMPOSITION_QUALITY["high"]
        assert high.crf < COMPOSITION_QUALITY["standard"].crf  # Higher quality = lower CRF


class TestCompositionResult:
    """Tests for CompositionResult dataclass."""

    def test_result_to_dict(self, tmp_path):
        """Test result serialization."""
        result = CompositionResult(
            output_path=tmp_path / "output.mp4",
            duration_seconds=120.5,
            clip_count=10,
            transitions_applied={"cut": 5, "fade": 4},
            composition_time_ms=5000,
            file_size_bytes=10485760,
        )

        data = result.to_dict()

        assert "output.mp4" in data["output_path"]
        assert data["duration_seconds"] == 120.5
        assert data["clip_count"] == 10
        assert data["transitions_applied"]["cut"] == 5
        assert data["composition_time_ms"] == 5000


class TestVideoComposer:
    """Tests for VideoComposer class."""

    @SKIP_FFMPEG
    def test_init_default_config(self):
        """Test composer initialization with default config."""
        composer = VideoComposer()
        assert composer.config is not None
        assert composer.config.crf == 18

    @SKIP_FFMPEG
    def test_init_custom_config(self):
        """Test composer initialization with custom config."""
        config = CompositionConfig(crf=23, preset="fast")
        composer = VideoComposer(config=config)

        assert composer.config.crf == 23
        assert composer.config.preset == "fast"

    @SKIP_FFMPEG
    def test_ffmpeg_validation(self):
        """Test FFmpeg availability check."""
        # Should not raise if FFmpeg is installed
        composer = VideoComposer()
        # If we get here, FFmpeg was found

    @SKIP_FFMPEG
    def test_discover_clips_success(self, tmp_path):
        """Test clip discovery with matching files."""
        # Create mock clip files
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "scene_000.mp4").touch()
        (clips_dir / "scene_001.mp4").touch()
        (clips_dir / "scene_002.mp4").touch()

        # Create minimal scene plan
        scenes = [
            _create_scene(id=0, start=0, end=10),
            _create_scene(id=1, start=10, end=20),
            _create_scene(id=2, start=20, end=30),
        ]
        scene_plan = _create_scene_plan(scenes)

        composer = VideoComposer()
        clips = composer._discover_clips(clips_dir, scene_plan)

        assert len(clips) == 3
        assert clips[0][0].id == 0
        assert clips[1][0].id == 1
        assert clips[2][0].id == 2

    @SKIP_FFMPEG
    def test_discover_clips_missing(self, tmp_path):
        """Test clip discovery with missing files."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "scene_000.mp4").touch()
        # scene_001.mp4 missing
        (clips_dir / "scene_002.mp4").touch()

        scenes = [
            _create_scene(id=0, start=0, end=10),
            _create_scene(id=1, start=10, end=20),
            _create_scene(id=2, start=20, end=30),
        ]
        scene_plan = _create_scene_plan(scenes)

        composer = VideoComposer()
        with pytest.raises(ClipMismatchError) as exc_info:
            composer._discover_clips(clips_dir, scene_plan)

        assert "1" in str(exc_info.value)

    @SKIP_FFMPEG
    def test_count_transitions(self, tmp_path):
        """Test transition counting."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        scenes = [
            _create_scene(id=0, transition_out=SceneTransition.CUT),
            _create_scene(id=1, transition_out=SceneTransition.FADE),
            _create_scene(id=2, transition_out=SceneTransition.DISSOLVE),
            _create_scene(id=3, transition_out=SceneTransition.CUT),
        ]

        # Create clip files
        for s in scenes:
            (clips_dir / f"scene_{s.id:03d}.mp4").touch()

        scene_plan = _create_scene_plan(scenes)
        composer = VideoComposer()
        clips = composer._discover_clips(clips_dir, scene_plan)

        counts = composer._count_transitions(clips)

        # Last scene's transition_out doesn't count (only 3 transitions for 4 clips)
        assert counts.get("cut", 0) == 1  # Scene 0 and 3 have CUT, but scene 3 is last
        assert counts.get("fade", 0) == 1  # Scene 1
        assert counts.get("dissolve", 0) == 1  # Scene 2

    @SKIP_FFMPEG
    def test_build_filter_single_clip(self):
        """Test filter complex with single clip."""
        composer = VideoComposer()

        scenes = [_create_scene(id=0)]
        clips = [(scenes[0], Path("scene_000.mp4"))]

        filter_str = composer._build_filter_complex(clips)

        assert "[0:v]copy[outv]" == filter_str

    @SKIP_FFMPEG
    def test_build_filter_all_cuts(self, tmp_path):
        """Test filter complex with all CUT transitions."""
        # Create real video files for duration check
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        scenes = [
            _create_scene(id=0, transition_out=SceneTransition.CUT),
            _create_scene(id=1, transition_out=SceneTransition.CUT),
        ]

        # Create minimal valid video files
        for s in scenes:
            _create_test_video(clips_dir / f"scene_{s.id:03d}.mp4", duration=2)

        composer = VideoComposer()
        clips = [(s, clips_dir / f"scene_{s.id:03d}.mp4") for s in scenes]

        filter_str = composer._build_filter_complex(clips)

        assert "concat" in filter_str

    @SKIP_FFMPEG
    def test_build_filter_with_dissolve(self, tmp_path):
        """Test filter complex with DISSOLVE transition."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        scenes = [
            _create_scene(id=0, transition_out=SceneTransition.DISSOLVE),
            _create_scene(id=1, transition_out=SceneTransition.CUT),
        ]

        for s in scenes:
            _create_test_video(clips_dir / f"scene_{s.id:03d}.mp4", duration=2)

        composer = VideoComposer()
        clips = [(s, clips_dir / f"scene_{s.id:03d}.mp4") for s in scenes]

        filter_str = composer._build_filter_complex(clips)

        assert "xfade" in filter_str
        assert "dissolve" in filter_str


class TestCompositionIntegration:
    """Integration tests requiring FFmpeg."""

    @pytest.mark.integration
    @SKIP_FFMPEG
    def test_compose_with_cuts_only(self, tmp_path):
        """Test composition with all CUT transitions."""
        # Create clips directory with test videos
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        scenes = [
            _create_scene(id=0, start=0, end=2, transition_out=SceneTransition.CUT),
            _create_scene(id=1, start=2, end=4, transition_out=SceneTransition.CUT),
            _create_scene(id=2, start=4, end=6, transition_out=SceneTransition.CUT),
        ]

        for s in scenes:
            _create_test_video(clips_dir / f"scene_{s.id:03d}.mp4", duration=2)

        # Create test audio
        audio_path = tmp_path / "audio.mp3"
        _create_test_audio(audio_path, duration=6)

        scene_plan = _create_scene_plan(scenes)
        output_path = tmp_path / "output.mp4"

        composer = VideoComposer()
        result = composer.compose(
            scene_plan=scene_plan,
            clips_dir=clips_dir,
            audio_path=audio_path,
            output_path=output_path,
        )

        assert output_path.exists()
        assert result.clip_count == 3
        assert result.file_size_bytes > 0
        assert result.transitions_applied.get("cut", 0) == 2

    @pytest.mark.integration
    @SKIP_FFMPEG
    def test_compose_with_fade(self, tmp_path):
        """Test composition with FADE transitions."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        scenes = [
            _create_scene(id=0, start=0, end=2, transition_out=SceneTransition.FADE),
            _create_scene(id=1, start=2, end=4, transition_out=SceneTransition.CUT),
        ]

        for s in scenes:
            _create_test_video(clips_dir / f"scene_{s.id:03d}.mp4", duration=2)

        audio_path = tmp_path / "audio.mp3"
        _create_test_audio(audio_path, duration=4)

        scene_plan = _create_scene_plan(scenes)
        output_path = tmp_path / "output.mp4"

        composer = VideoComposer()
        result = composer.compose(
            scene_plan=scene_plan,
            clips_dir=clips_dir,
            audio_path=audio_path,
            output_path=output_path,
        )

        assert output_path.exists()
        assert result.transitions_applied.get("fade", 0) == 1


# Helper functions for creating test data

def _create_scene(
    id: int = 0,
    start: float = 0.0,
    end: float = 10.0,
    transition_in: SceneTransition = SceneTransition.CUT,
    transition_out: SceneTransition = SceneTransition.CUT,
) -> Scene:
    """Create a test scene."""
    return Scene(
        id=id,
        start=start,
        end=end,
        description=f"Test scene {id}",
        prompt=f"test prompt {id}",
        lyrics_text="",
        section_type="verse",
        mood="neutral",
        energy=0.5,
        transition_in=transition_in,
        transition_out=transition_out,
    )


def _create_scene_plan(scenes: list[Scene]) -> ScenePlan:
    """Create a test scene plan."""
    style_guide = StyleGuide(
        style=VisualStyle.CINEMATIC,
        aesthetic="test aesthetic",
        color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
        lighting="natural",
        camera_style="wide",
        environment_theme="test",
        character_style="",
    )

    narrative = NarrativeAnalysis(
        overall_theme="Test theme",
        story_summary="Test story",
        emotional_arc="rising",
        key_imagery=["test"],
        metaphors=[],
        characters=[],
        settings=["test location"],
        tone="neutral",
        genre_influence="pop",
    )

    duration = scenes[-1].end if scenes else 0.0

    return ScenePlan(
        song_title="Test Song",
        duration=duration,
        master_seed=12345,
        style_guide=style_guide,
        narrative=narrative,
        scenes=scenes,
        target_fps=24.0,
        is_instrumental=False,
    )


def _create_test_video(path: Path, duration: float = 2.0, fps: int = 24):
    """Create a minimal test video file using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:s=320x240:d={duration}:r={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _create_test_audio(path: Path, duration: float = 10.0):
    """Create a minimal test audio file using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
