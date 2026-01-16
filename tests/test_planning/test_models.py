"""Tests for planning models."""

import pytest

from song_to_video.planning.models import (
    Scene,
    ScenePlan,
    StyleGuide,
    NarrativeAnalysis,
    ColorPalette,
    VisualStyle,
    SceneTransition,
)


class TestScene:
    """Tests for Scene data class."""

    def test_scene_duration(self):
        """Test scene duration calculation."""
        scene = Scene(
            id=0,
            start=10.0,
            end=25.0,
            description="Test scene",
            prompt="test prompt",
            lyrics_text="test lyrics",
            section_type="verse",
            mood="happy",
            energy=0.7,
        )
        assert scene.duration == 15.0

    def test_scene_to_dict(self):
        """Test scene serialization."""
        scene = Scene(
            id=1,
            start=0.0,
            end=10.0,
            description="Opening scene",
            prompt="wide shot, sunset",
            lyrics_text="Hello world",
            section_type="intro",
            mood="uplifting",
            energy=0.5,
            transition_in=SceneTransition.FADE,
            transition_out=SceneTransition.CUT,
            camera_movement="slow zoom",
            key_elements=["sunset", "landscape"],
            seed_offset=0,
        )

        data = scene.to_dict()

        assert data["id"] == 1
        assert data["duration"] == 10.0
        assert data["section_type"] == "intro"
        assert data["transition_in"] == "fade"
        assert "sunset" in data["key_elements"]

    def test_scene_from_dict(self):
        """Test scene deserialization."""
        data = {
            "id": 2,
            "start": 30.0,
            "end": 45.0,
            "description": "Chorus scene",
            "prompt": "energetic visuals",
            "lyrics_text": "Sing along",
            "section_type": "chorus",
            "mood": "energetic",
            "energy": 0.9,
            "transition_in": "dissolve",
            "transition_out": "cut",
            "key_elements": ["dancing", "lights"],
        }

        scene = Scene.from_dict(data)

        assert scene.id == 2
        assert scene.duration == 15.0
        assert scene.transition_in == SceneTransition.DISSOLVE


class TestStyleGuide:
    """Tests for StyleGuide data class."""

    def test_style_guide_to_prompt_suffix(self):
        """Test prompt suffix generation."""
        style = StyleGuide(
            style=VisualStyle.CINEMATIC,
            aesthetic="moody noir",
            color_palette=ColorPalette(
                primary="#000000",
                secondary="#333333",
                accent="#FF0000",
                background="#1a1a1a",
            ),
            lighting="dramatic shadows",
            camera_style="dutch angles",
            environment_theme="urban decay",
            character_style="trenchcoat",
            mood_keywords=["noir", "mysterious", "tense"],
        )

        suffix = style.to_prompt_suffix()

        assert "moody noir" in suffix
        assert "dramatic shadows" in suffix
        assert "noir" in suffix

    def test_style_guide_to_dict(self):
        """Test style guide serialization."""
        style = StyleGuide(
            style=VisualStyle.BRIGHT,
            aesthetic="cheerful pop",
            color_palette=ColorPalette(
                primary="#FFD700",
                secondary="#FF69B4",
                accent="#00FF00",
                background="#FFFFFF",
                mood_colors=["#FFC0CB"],
            ),
            lighting="bright daylight",
            camera_style="smooth tracking",
            environment_theme="sunny beach",
            character_style="casual summer",
            mood_keywords=["happy", "fun"],
            negative_prompts=["dark", "gloomy"],
        )

        data = style.to_dict()

        assert data["style"] == "bright"
        assert data["color_palette"]["primary"] == "#FFD700"
        assert "happy" in data["mood_keywords"]
        assert "dark" in data["negative_prompts"]


class TestColorPalette:
    """Tests for ColorPalette data class."""

    def test_color_palette_to_dict(self):
        """Test color palette serialization."""
        palette = ColorPalette(
            primary="#FF0000",
            secondary="#00FF00",
            accent="#0000FF",
            background="#FFFFFF",
            mood_colors=["#FFA500", "#800080"],
        )

        data = palette.to_dict()

        assert data["primary"] == "#FF0000"
        assert len(data["mood_colors"]) == 2

    def test_color_palette_from_dict(self):
        """Test color palette deserialization."""
        data = {
            "primary": "#123456",
            "secondary": "#654321",
            "accent": "#ABCDEF",
            "background": "#000000",
        }

        palette = ColorPalette.from_dict(data)

        assert palette.primary == "#123456"
        assert palette.mood_colors == []


class TestNarrativeAnalysis:
    """Tests for NarrativeAnalysis data class."""

    def test_narrative_to_dict(self):
        """Test narrative serialization."""
        narrative = NarrativeAnalysis(
            overall_theme="Love and loss",
            story_summary="A journey through heartbreak",
            emotional_arc="Starts hopeful, becomes sad, ends accepting",
            key_imagery=["rain", "empty streets", "photographs"],
            metaphors=["rain as tears"],
            characters=["narrator", "lost love"],
            settings=["city apartment", "old cafe"],
            tone="melancholic",
            genre_influence="indie drama",
        )

        data = narrative.to_dict()

        assert data["overall_theme"] == "Love and loss"
        assert "rain" in data["key_imagery"]
        assert data["tone"] == "melancholic"


class TestScenePlan:
    """Tests for ScenePlan data class."""

    def test_scene_plan_scene_count(self):
        """Test scene count property."""
        plan = ScenePlan(
            song_title="Test Song",
            duration=180.0,
            master_seed=12345,
            narrative=NarrativeAnalysis(
                overall_theme="Test",
                story_summary="Test",
                emotional_arc="Test",
                key_imagery=[],
                metaphors=[],
                characters=[],
                settings=[],
                tone="neutral",
                genre_influence="pop",
            ),
            style_guide=StyleGuide(
                style=VisualStyle.CINEMATIC,
                aesthetic="test",
                color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
                lighting="natural",
                camera_style="wide",
                environment_theme="urban",
                character_style="casual",
            ),
            scenes=[
                Scene(
                    id=0, start=0.0, end=30.0, description="S1",
                    prompt="p1", lyrics_text="", section_type="intro",
                    mood="neutral", energy=0.5
                ),
                Scene(
                    id=1, start=30.0, end=60.0, description="S2",
                    prompt="p2", lyrics_text="", section_type="verse",
                    mood="neutral", energy=0.5
                ),
            ],
        )

        assert plan.scene_count == 2

    def test_scene_plan_get_scene_at(self):
        """Test getting scene at specific time."""
        plan = ScenePlan(
            song_title="Test",
            duration=60.0,
            master_seed=0,
            narrative=NarrativeAnalysis(
                overall_theme="", story_summary="", emotional_arc="",
                key_imagery=[], metaphors=[], characters=[], settings=[],
                tone="", genre_influence=""
            ),
            style_guide=StyleGuide(
                style=VisualStyle.CINEMATIC, aesthetic="",
                color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
                lighting="", camera_style="", environment_theme="",
                character_style=""
            ),
            scenes=[
                Scene(
                    id=0, start=0.0, end=20.0, description="First",
                    prompt="", lyrics_text="", section_type="intro",
                    mood="", energy=0.5
                ),
                Scene(
                    id=1, start=20.0, end=40.0, description="Second",
                    prompt="", lyrics_text="", section_type="verse",
                    mood="", energy=0.5
                ),
            ],
        )

        scene = plan.get_scene_at(10.0)
        assert scene.description == "First"

        scene = plan.get_scene_at(30.0)
        assert scene.description == "Second"

        scene = plan.get_scene_at(50.0)
        assert scene is None


class TestVisualStyle:
    """Tests for VisualStyle enum."""

    def test_style_values(self):
        """Test style enum has expected values."""
        assert VisualStyle.CINEMATIC.value == "cinematic"
        assert VisualStyle.ANIMATED.value == "animated"
        assert VisualStyle.DARK.value == "dark"
        assert VisualStyle.DREAMY.value == "dreamy"


class TestSceneTransition:
    """Tests for SceneTransition enum."""

    def test_transition_values(self):
        """Test transition enum has expected values."""
        assert SceneTransition.CUT.value == "cut"
        assert SceneTransition.FADE.value == "fade"
        assert SceneTransition.DISSOLVE.value == "dissolve"
        assert SceneTransition.MORPH.value == "morph"
