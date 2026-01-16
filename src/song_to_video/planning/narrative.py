"""Narrative analysis and scene planning using LLM."""

import logging
from pathlib import Path
from typing import Optional

from ..audio.models import Lyrics, AudioFeatures, Mood, SectionType
from .models import (
    Scene,
    ScenePlan,
    StyleGuide,
    NarrativeAnalysis,
    ColorPalette,
    VisualStyle,
    SceneTransition,
    PlanningError,
)
from .ollama import OllamaClient, get_ollama_client
from .seed import generate_master_seed, get_scene_seed

logger = logging.getLogger(__name__)

# System prompt for narrative analysis
NARRATIVE_SYSTEM_PROMPT = """You are a creative director analyzing song lyrics to plan a music video.
Your task is to identify themes, imagery, emotional arcs, and visual concepts from lyrics.
Be specific and visual in your descriptions. Think cinematically."""

# System prompt for scene generation
SCENE_SYSTEM_PROMPT = """You are a music video director creating scene descriptions.
Generate vivid, specific visual descriptions that can be used for AI video generation.
Focus on concrete imagery, colors, lighting, and atmosphere.
Each scene should be visually distinct but maintain overall style coherence."""


class NarrativePlanner:
    """
    Plans music video scenes based on lyrics and audio analysis.

    Uses local LLM (Ollama) to interpret lyrics and generate visual concepts.
    """

    def __init__(self, client: Optional[OllamaClient] = None):
        """
        Initialize the narrative planner.

        Args:
            client: Ollama client (uses global instance if None)
        """
        self.client = client or get_ollama_client()

    def analyze_narrative(
        self,
        lyrics: Lyrics,
        audio_features: AudioFeatures,
    ) -> NarrativeAnalysis:
        """
        Analyze the narrative and themes in the lyrics.

        Args:
            lyrics: Transcribed or provided lyrics
            audio_features: Analyzed audio features

        Returns:
            NarrativeAnalysis with extracted themes and imagery
        """
        logger.info("Analyzing narrative from lyrics")

        # Build context about the song
        lyrics_text = lyrics.text
        mood_str = audio_features.mood.value
        tempo_str = f"{audio_features.tempo:.0f} BPM"
        energy_str = f"{audio_features.average_energy:.0%} average energy"

        prompt = f"""Analyze these song lyrics for a music video:

LYRICS:
{lyrics_text}

AUDIO CHARACTERISTICS:
- Mood: {mood_str}
- Tempo: {tempo_str}
- Energy: {energy_str}
- Key: {audio_features.key}
- Has distinct sections: {len(audio_features.sections)} sections detected

Provide a JSON analysis with these fields:
{{
    "overall_theme": "The main theme or message of the song (1-2 sentences)",
    "story_summary": "Brief narrative summary if there's a story (2-3 sentences)",
    "emotional_arc": "How emotions progress through the song (1-2 sentences)",
    "key_imagery": ["list", "of", "visual", "images", "mentioned or implied"],
    "metaphors": ["key", "metaphors", "or", "symbols"],
    "characters": ["any", "characters", "or", "personas", "mentioned"],
    "settings": ["locations", "or", "environments", "implied"],
    "tone": "overall tone (e.g., playful, serious, melancholic, uplifting)",
    "genre_influence": "visual genre this suggests (e.g., pop, indie, cinematic, abstract)"
}}"""

        try:
            data = self.client.generate_json(
                prompt=prompt,
                system=NARRATIVE_SYSTEM_PROMPT,
                temperature=0.5,
            )

            return NarrativeAnalysis(
                overall_theme=data.get("overall_theme", ""),
                story_summary=data.get("story_summary", ""),
                emotional_arc=data.get("emotional_arc", ""),
                key_imagery=data.get("key_imagery", []),
                metaphors=data.get("metaphors", []),
                characters=data.get("characters", []),
                settings=data.get("settings", []),
                tone=data.get("tone", ""),
                genre_influence=data.get("genre_influence", ""),
            )

        except PlanningError:
            logger.warning("LLM analysis failed, using fallback")
            return self._fallback_narrative(lyrics, audio_features)

    def _fallback_narrative(
        self,
        lyrics: Lyrics,
        audio_features: AudioFeatures,
    ) -> NarrativeAnalysis:
        """Fallback narrative when LLM fails."""
        mood_map = {
            Mood.HAPPY: ("uplifting", "joyful celebration"),
            Mood.SAD: ("melancholic", "emotional reflection"),
            Mood.ENERGETIC: ("dynamic", "high-energy movement"),
            Mood.CALM: ("peaceful", "serene contemplation"),
            Mood.AGGRESSIVE: ("intense", "powerful confrontation"),
            Mood.DARK: ("mysterious", "shadowy exploration"),
            Mood.ROMANTIC: ("intimate", "love and connection"),
        }

        tone, theme = mood_map.get(
            audio_features.mood, ("neutral", "abstract visual journey")
        )

        return NarrativeAnalysis(
            overall_theme=theme,
            story_summary="A visual interpretation of the song's mood and energy.",
            emotional_arc=f"Follows the song's {audio_features.mood.value} mood throughout.",
            key_imagery=["abstract shapes", "flowing colors", "dynamic movement"],
            metaphors=[],
            characters=[],
            settings=["abstract environment"],
            tone=tone,
            genre_influence="abstract",
        )

    def determine_style(
        self,
        narrative: NarrativeAnalysis,
        audio_features: AudioFeatures,
    ) -> StyleGuide:
        """
        Determine the visual style for the video.

        Args:
            narrative: Analyzed narrative
            audio_features: Audio features

        Returns:
            StyleGuide for consistent generation
        """
        logger.info("Determining visual style")

        prompt = f"""Based on this song analysis, create a visual style guide for a music video:

NARRATIVE:
- Theme: {narrative.overall_theme}
- Tone: {narrative.tone}
- Genre influence: {narrative.genre_influence}
- Key imagery: {', '.join(narrative.key_imagery[:5])}
- Settings: {', '.join(narrative.settings[:3])}

AUDIO:
- Mood: {audio_features.mood.value}
- Energy: {audio_features.average_energy:.0%}
- Tempo: {audio_features.tempo:.0f} BPM

Create a JSON style guide:
{{
    "style": "one of: cinematic, animated, abstract, realistic, stylized, vintage, futuristic, dark, bright, dreamy, gritty, minimalist",
    "aesthetic": "detailed description of the visual aesthetic (2-3 sentences)",
    "color_palette": {{
        "primary": "#hexcolor",
        "secondary": "#hexcolor",
        "accent": "#hexcolor",
        "background": "#hexcolor",
        "mood_colors": ["#hex1", "#hex2", "#hex3"]
    }},
    "lighting": "lighting style description",
    "camera_style": "camera angles and movement style",
    "environment_theme": "overall environment description",
    "character_style": "character appearance if applicable (or 'no characters')",
    "mood_keywords": ["keyword1", "keyword2", "keyword3"],
    "negative_prompts": ["things", "to", "avoid", "in", "generation"]
}}"""

        try:
            data = self.client.generate_json(
                prompt=prompt,
                system=SCENE_SYSTEM_PROMPT,
                temperature=0.4,
            )

            # Parse style enum
            style_str = data.get("style", "cinematic").lower()
            try:
                style = VisualStyle(style_str)
            except ValueError:
                style = VisualStyle.CINEMATIC

            # Parse color palette
            palette_data = data.get("color_palette", {})
            palette = ColorPalette(
                primary=palette_data.get("primary", "#1a1a2e"),
                secondary=palette_data.get("secondary", "#16213e"),
                accent=palette_data.get("accent", "#0f3460"),
                background=palette_data.get("background", "#0a0a0a"),
                mood_colors=palette_data.get("mood_colors", []),
            )

            return StyleGuide(
                style=style,
                aesthetic=data.get("aesthetic", ""),
                color_palette=palette,
                lighting=data.get("lighting", "natural"),
                camera_style=data.get("camera_style", "dynamic"),
                environment_theme=data.get("environment_theme", ""),
                character_style=data.get("character_style", ""),
                mood_keywords=data.get("mood_keywords", []),
                negative_prompts=data.get("negative_prompts", []),
            )

        except PlanningError:
            logger.warning("LLM style generation failed, using fallback")
            return self._fallback_style(narrative, audio_features)

    def _fallback_style(
        self,
        narrative: NarrativeAnalysis,
        audio_features: AudioFeatures,
    ) -> StyleGuide:
        """Fallback style when LLM fails."""
        # Map mood to style
        style_map = {
            Mood.HAPPY: (VisualStyle.BRIGHT, "#FFD700", "#FF6B6B"),
            Mood.SAD: (VisualStyle.CINEMATIC, "#4A90A4", "#2C3E50"),
            Mood.ENERGETIC: (VisualStyle.STYLIZED, "#FF4500", "#FF8C00"),
            Mood.CALM: (VisualStyle.DREAMY, "#87CEEB", "#E6E6FA"),
            Mood.AGGRESSIVE: (VisualStyle.GRITTY, "#8B0000", "#2F4F4F"),
            Mood.DARK: (VisualStyle.DARK, "#1a1a2e", "#0f3460"),
        }

        style, primary, secondary = style_map.get(
            audio_features.mood, (VisualStyle.CINEMATIC, "#333333", "#666666")
        )

        return StyleGuide(
            style=style,
            aesthetic=f"A {style.value} visual style matching the {audio_features.mood.value} mood.",
            color_palette=ColorPalette(
                primary=primary,
                secondary=secondary,
                accent="#FFFFFF",
                background="#000000",
            ),
            lighting="dramatic" if audio_features.average_energy > 0.5 else "soft",
            camera_style="dynamic" if audio_features.tempo > 100 else "smooth",
            environment_theme="abstract space",
            character_style="no characters",
            mood_keywords=[audio_features.mood.value, style.value],
            negative_prompts=["blurry", "distorted", "low quality"],
        )

    def generate_scenes(
        self,
        lyrics: Lyrics,
        audio_features: AudioFeatures,
        narrative: NarrativeAnalysis,
        style_guide: StyleGuide,
        master_seed: int,
    ) -> list[Scene]:
        """
        Generate scene descriptions for each section.

        Args:
            lyrics: Song lyrics
            audio_features: Audio analysis
            narrative: Narrative analysis
            style_guide: Visual style guide
            master_seed: Master seed for consistency

        Returns:
            List of Scene objects
        """
        logger.info("Generating scene descriptions")

        scenes = []
        sections = audio_features.sections

        if not sections:
            # Create a single scene for the whole song
            sections = [type('Section', (), {
                'start': 0.0,
                'end': audio_features.duration,
                'type': SectionType.VERSE,
                'label': 'Full Song',
            })()]

        for i, section in enumerate(sections):
            # Get lyrics for this section
            section_lyrics = self._get_lyrics_for_timerange(
                lyrics, section.start, section.end
            )

            # Get energy for this section
            section_energy = audio_features.get_energy_at(
                (section.start + section.end) / 2
            )

            # Generate scene description
            scene = self._generate_single_scene(
                scene_index=i,
                section=section,
                section_lyrics=section_lyrics,
                energy=section_energy,
                narrative=narrative,
                style_guide=style_guide,
                master_seed=master_seed,
                total_sections=len(sections),
            )

            scenes.append(scene)

        # Set transitions
        self._set_transitions(scenes)

        return scenes

    def _get_lyrics_for_timerange(
        self,
        lyrics: Lyrics,
        start: float,
        end: float,
    ) -> str:
        """Get lyrics text within a time range."""
        lines = []
        for line in lyrics.lines:
            # Check if line overlaps with time range
            if line.start < end and line.end > start:
                lines.append(line.text)
        return "\n".join(lines) if lines else ""

    def _generate_single_scene(
        self,
        scene_index: int,
        section,
        section_lyrics: str,
        energy: float,
        narrative: NarrativeAnalysis,
        style_guide: StyleGuide,
        master_seed: int,
        total_sections: int,
    ) -> Scene:
        """Generate a single scene description."""
        section_type = section.type.value if hasattr(section.type, 'value') else str(section.type)

        prompt = f"""Create a visual scene for this part of a music video:

SECTION: {section.label if hasattr(section, 'label') else section_type} ({section_type})
TIME: {section.start:.1f}s - {section.end:.1f}s
ENERGY LEVEL: {energy:.0%}

LYRICS FOR THIS SECTION:
{section_lyrics if section_lyrics else "(instrumental section)"}

OVERALL NARRATIVE:
- Theme: {narrative.overall_theme}
- Story: {narrative.story_summary}
- Tone: {narrative.tone}

VISUAL STYLE:
- Aesthetic: {style_guide.aesthetic}
- Lighting: {style_guide.lighting}
- Environment: {style_guide.environment_theme}

Generate a JSON scene description:
{{
    "description": "Detailed visual description of what happens in this scene (2-3 sentences)",
    "prompt": "SIMPLE image generation prompt (MAX 15 words). Focus on: subject, action, setting. Example: 'woman walking through grocery store aisle, soft lighting'. NO style words, NO quality words.",
    "mood": "emotional tone of this specific scene (one word)",
    "key_elements": ["max", "4", "visual", "elements"],
    "camera_movement": "camera movement (e.g., 'slow zoom', 'static', 'pan left')"
}}"""

        try:
            data = self.client.generate_json(
                prompt=prompt,
                system=SCENE_SYSTEM_PROMPT,
                temperature=0.6,
            )

            # Store ONLY the simple base prompt - style is added during image generation
            base_prompt = data.get("prompt", "")
            # Ensure prompt isn't too long (truncate if needed)
            if len(base_prompt.split()) > 20:
                words = base_prompt.split()[:15]
                base_prompt = " ".join(words)

            return Scene(
                id=scene_index,
                start=section.start,
                end=section.end,
                description=data.get("description", ""),
                prompt=base_prompt,  # Simple prompt only, style added at generation time
                lyrics_text=section_lyrics,
                section_type=section_type,
                mood=data.get("mood", narrative.tone),
                energy=energy,
                camera_movement=data.get("camera_movement", ""),
                key_elements=data.get("key_elements", [])[:4],  # Limit to 4 elements
                seed_offset=scene_index,  # Used with master seed
            )

        except PlanningError:
            logger.warning(f"LLM scene generation failed for scene {scene_index}, using fallback")
            return self._fallback_scene(
                scene_index, section, section_lyrics, energy, style_guide
            )

    def _fallback_scene(
        self,
        scene_index: int,
        section,
        section_lyrics: str,
        energy: float,
        style_guide: StyleGuide,
    ) -> Scene:
        """Fallback scene when LLM fails."""
        section_type = section.type.value if hasattr(section.type, 'value') else str(section.type)

        # Generate basic prompt from style
        prompt = f"Abstract {style_guide.style.value} visuals, {style_guide.lighting} lighting"

        return Scene(
            id=scene_index,
            start=section.start,
            end=section.end,
            description=f"Abstract visualization for {section_type} section.",
            prompt=prompt,
            lyrics_text=section_lyrics,
            section_type=section_type,
            mood=style_guide.style.value,
            energy=energy,
            camera_movement="slow pan",
            key_elements=["abstract shapes", "flowing colors"],
            seed_offset=scene_index,
        )

    def _set_transitions(self, scenes: list[Scene]) -> None:
        """Set transitions between scenes based on content."""
        for i, scene in enumerate(scenes):
            # Intro gets fade in
            if i == 0:
                scene.transition_in = SceneTransition.FADE

            # Outro gets fade out
            if i == len(scenes) - 1:
                scene.transition_out = SceneTransition.FADE
                continue

            # Chorus transitions
            if scene.section_type == "chorus":
                scene.transition_in = SceneTransition.CUT
                scene.transition_out = SceneTransition.CUT
            # Verse to chorus gets dissolve
            elif i + 1 < len(scenes) and scenes[i + 1].section_type == "chorus":
                scene.transition_out = SceneTransition.DISSOLVE
            # Default cut
            else:
                scene.transition_out = SceneTransition.CUT


def create_scene_plan(
    audio_path: Path,
    lyrics: Lyrics,
    audio_features: AudioFeatures,
    song_title: str = "",
    user_seed: Optional[int] = None,
) -> ScenePlan:
    """
    Create a complete scene plan for a song.

    Args:
        audio_path: Path to audio file (for seed generation)
        lyrics: Song lyrics
        audio_features: Audio analysis results
        song_title: Title of the song
        user_seed: Optional user-provided seed

    Returns:
        ScenePlan with all scenes and style guide
    """
    planner = NarrativePlanner()

    # Determine if instrumental
    is_instrumental = not audio_features.has_vocals or len(lyrics.lines) == 0

    if is_instrumental:
        logger.info("Song appears to be instrumental, using mood-based planning")
        narrative = planner._fallback_narrative(lyrics, audio_features)
    else:
        narrative = planner.analyze_narrative(lyrics, audio_features)

    # Determine style
    style_guide = planner.determine_style(narrative, audio_features)

    # Generate master seed
    master_seed = generate_master_seed(
        audio_path=audio_path,
        style_description=style_guide.aesthetic,
        user_seed=user_seed,
    )

    # Generate scenes
    scenes = planner.generate_scenes(
        lyrics=lyrics,
        audio_features=audio_features,
        narrative=narrative,
        style_guide=style_guide,
        master_seed=master_seed,
    )

    return ScenePlan(
        song_title=song_title or audio_path.stem,
        duration=audio_features.duration,
        master_seed=master_seed,
        narrative=narrative,
        style_guide=style_guide,
        scenes=scenes,
        is_instrumental=is_instrumental,
    )
