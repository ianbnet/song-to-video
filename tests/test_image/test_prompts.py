"""Tests for prompt templating."""

import pytest

from song_to_video.image.prompts import (
    PromptTemplate,
    build_scene_prompt,
    get_template_for_style,
    CINEMATIC_TEMPLATE,
    ANIMATED_TEMPLATE,
)
from song_to_video.planning.models import (
    Scene,
    StyleGuide,
    ColorPalette,
    VisualStyle,
    SceneTransition,
)


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_default_template(self):
        """Test default template values."""
        template = PromptTemplate()

        assert "high quality" in template.quality_suffix
        assert "blurry" in template.negative_prompt

    def test_format_prompt(self):
        """Test prompt formatting - simple prompts for better generation."""
        template = PromptTemplate(
            quality_suffix="masterpiece, best quality",
        )

        scene = Scene(
            id=0,
            start=0.0,
            end=10.0,
            description="A sunset over the ocean",
            prompt="golden sunset, ocean waves",
            lyrics_text="",
            section_type="intro",
            mood="peaceful",
            energy=0.3,
        )

        style_guide = StyleGuide(
            style=VisualStyle.CINEMATIC,
            aesthetic="dramatic cinematography",
            color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
            lighting="golden hour",
            camera_style="wide angle",
            environment_theme="coastal",
            character_style="",
            mood_keywords=["serene", "beautiful"],
        )

        prompt = template.format_prompt(scene, style_guide)

        # Core elements should be present
        assert "golden sunset" in prompt
        assert "golden hour lighting" in prompt
        assert "masterpiece" in prompt
        # Aesthetic is NOT added to keep prompts simple (prevents prompt overload)

    def test_format_negative_prompt(self):
        """Test negative prompt formatting."""
        template = PromptTemplate(
            negative_prompt="blurry, ugly",
        )

        style_guide = StyleGuide(
            style=VisualStyle.CINEMATIC,
            aesthetic="test",
            color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
            lighting="natural",
            camera_style="wide",
            environment_theme="",
            character_style="",
            negative_prompts=["cartoon", "anime"],
        )

        negative = template.format_negative_prompt(style_guide)

        assert "blurry" in negative
        assert "ugly" in negative
        assert "cartoon" in negative
        assert "anime" in negative


class TestGetTemplateForStyle:
    """Tests for get_template_for_style function."""

    def test_cinematic_style(self):
        """Test getting cinematic template."""
        template = get_template_for_style("cinematic")

        assert "cinematic" in template.style_prefix.lower()

    def test_animated_style(self):
        """Test getting animated template."""
        template = get_template_for_style("animated")

        assert "animated" in template.style_prefix.lower()

    def test_unknown_style_returns_default(self):
        """Test unknown style returns default template."""
        template = get_template_for_style("nonexistent_style")

        assert isinstance(template, PromptTemplate)


class TestBuildScenePrompt:
    """Tests for build_scene_prompt function."""

    def test_build_basic_prompt(self):
        """Test building a basic prompt."""
        scene = Scene(
            id=0,
            start=0.0,
            end=10.0,
            description="A beautiful garden",
            prompt="lush garden, flowers blooming",
            lyrics_text="Walking through the garden",
            section_type="verse",
            mood="happy",
            energy=0.5,
        )

        style_guide = StyleGuide(
            style=VisualStyle.BRIGHT,
            aesthetic="vibrant and colorful",
            color_palette=ColorPalette("#FFD700", "#FF69B4", "#00FF00", "#FFFFFF"),
            lighting="bright daylight",
            camera_style="medium shot",
            environment_theme="natural garden",
            character_style="casual",
            mood_keywords=["cheerful", "sunny"],
            negative_prompts=["dark", "gloomy"],
        )

        prompt, negative = build_scene_prompt(scene, style_guide)

        assert "lush garden" in prompt
        assert "flowers blooming" in prompt
        assert "dark" in negative
        assert "gloomy" in negative

    def test_build_prompt_with_lyrics(self):
        """Test building prompt with lyrics - keeps prompts simple."""
        scene = Scene(
            id=0,
            start=0.0,
            end=10.0,
            description="Abstract scene",
            prompt="abstract visuals",
            lyrics_text="Dancing in the moonlight tonight",
            section_type="chorus",
            mood="energetic",
            energy=0.8,
        )

        style_guide = StyleGuide(
            style=VisualStyle.ABSTRACT,
            aesthetic="artistic and expressive",
            color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
            lighting="dramatic",
            camera_style="dynamic",
            environment_theme="abstract space",
            character_style="",
        )

        prompt, _ = build_scene_prompt(scene, style_guide, include_lyrics=True)

        # Core scene prompt should be present
        assert "abstract" in prompt.lower()
        # Style prefix for abstract template should be applied
        assert "artistic" in prompt.lower()
        # Note: Lyrics are NOT directly included in prompts to keep them simple
        # The scene prompt should already capture the essence from lyrics

    def test_build_prompt_without_lyrics(self):
        """Test building prompt without lyrics."""
        scene = Scene(
            id=0,
            start=0.0,
            end=10.0,
            description="Test scene",
            prompt="test visual",
            lyrics_text="Some lyrics here",
            section_type="verse",
            mood="neutral",
            energy=0.5,
        )

        style_guide = StyleGuide(
            style=VisualStyle.CINEMATIC,
            aesthetic="cinematic",
            color_palette=ColorPalette("#000", "#333", "#666", "#FFF"),
            lighting="natural",
            camera_style="wide",
            environment_theme="",
            character_style="",
        )

        prompt, _ = build_scene_prompt(scene, style_guide, include_lyrics=False)

        # Lyrics text should not be directly in prompt
        assert "Some lyrics here" not in prompt
