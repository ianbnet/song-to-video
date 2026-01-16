"""Data models for scene planning and visual style."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VisualStyle(Enum):
    """Visual style categories for video generation."""

    CINEMATIC = "cinematic"
    ANIMATED = "animated"
    ABSTRACT = "abstract"
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    VINTAGE = "vintage"
    FUTURISTIC = "futuristic"
    DARK = "dark"
    BRIGHT = "bright"
    DREAMY = "dreamy"
    GRITTY = "gritty"
    MINIMALIST = "minimalist"


class SceneTransition(Enum):
    """Transition types between scenes."""

    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    MORPH = "morph"


@dataclass
class ColorPalette:
    """Color palette for visual consistency."""

    primary: str  # Hex color code
    secondary: str
    accent: str
    background: str
    mood_colors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "background": self.background,
            "mood_colors": self.mood_colors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColorPalette":
        return cls(
            primary=data.get("primary", "#000000"),
            secondary=data.get("secondary", "#333333"),
            accent=data.get("accent", "#666666"),
            background=data.get("background", "#FFFFFF"),
            mood_colors=data.get("mood_colors", []),
        )


@dataclass
class StyleGuide:
    """Visual style guide for consistent generation."""

    style: VisualStyle
    aesthetic: str  # Detailed style description
    color_palette: ColorPalette
    lighting: str  # Lighting description
    camera_style: str  # Camera angles/movement description
    environment_theme: str  # Overall environment description
    character_style: str  # Character appearance if applicable
    mood_keywords: list[str] = field(default_factory=list)
    negative_prompts: list[str] = field(default_factory=list)  # Things to avoid

    def to_prompt_suffix(self) -> str:
        """Generate a prompt suffix for consistent style."""
        parts = [
            self.aesthetic,
            f"{self.lighting} lighting",
            self.camera_style,
        ]
        if self.mood_keywords:
            # Filter out any non-string items (e.g., dicts from LLM errors)
            keywords = [k for k in self.mood_keywords[:3] if isinstance(k, str)]
            parts.extend(keywords)
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "style": self.style.value,
            "aesthetic": self.aesthetic,
            "color_palette": self.color_palette.to_dict(),
            "lighting": self.lighting,
            "camera_style": self.camera_style,
            "environment_theme": self.environment_theme,
            "character_style": self.character_style,
            "mood_keywords": self.mood_keywords,
            "negative_prompts": self.negative_prompts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StyleGuide":
        return cls(
            style=VisualStyle(data.get("style", "cinematic")),
            aesthetic=data.get("aesthetic", ""),
            color_palette=ColorPalette.from_dict(data.get("color_palette", {})),
            lighting=data.get("lighting", "natural"),
            camera_style=data.get("camera_style", "wide shots"),
            environment_theme=data.get("environment_theme", ""),
            character_style=data.get("character_style", ""),
            mood_keywords=data.get("mood_keywords", []),
            negative_prompts=data.get("negative_prompts", []),
        )


@dataclass
class Scene:
    """A single scene in the video plan."""

    id: int
    start: float  # Start time in seconds
    end: float  # End time in seconds
    description: str  # Visual description for generation
    prompt: str  # Full generation prompt
    lyrics_text: str  # Associated lyrics (if any)
    section_type: str  # verse, chorus, intro, etc.
    mood: str  # Emotional tone
    energy: float  # Energy level 0-1
    transition_in: SceneTransition = SceneTransition.CUT
    transition_out: SceneTransition = SceneTransition.CUT
    camera_movement: str = ""
    key_elements: list[str] = field(default_factory=list)
    seed_offset: int = 0  # Offset from master seed for this scene

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "description": self.description,
            "prompt": self.prompt,
            "lyrics_text": self.lyrics_text,
            "section_type": self.section_type,
            "mood": self.mood,
            "energy": self.energy,
            "transition_in": self.transition_in.value,
            "transition_out": self.transition_out.value,
            "camera_movement": self.camera_movement,
            "key_elements": self.key_elements,
            "seed_offset": self.seed_offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        return cls(
            id=data.get("id", 0),
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            lyrics_text=data.get("lyrics_text", ""),
            section_type=data.get("section_type", "unknown"),
            mood=data.get("mood", "neutral"),
            energy=data.get("energy", 0.5),
            transition_in=SceneTransition(data.get("transition_in", "cut")),
            transition_out=SceneTransition(data.get("transition_out", "cut")),
            camera_movement=data.get("camera_movement", ""),
            key_elements=data.get("key_elements", []),
            seed_offset=data.get("seed_offset", 0),
        )


@dataclass
class NarrativeAnalysis:
    """Analysis of the song's narrative and themes."""

    overall_theme: str
    story_summary: str
    emotional_arc: str
    key_imagery: list[str]
    metaphors: list[str]
    characters: list[str]
    settings: list[str]
    tone: str
    genre_influence: str

    def to_dict(self) -> dict:
        return {
            "overall_theme": self.overall_theme,
            "story_summary": self.story_summary,
            "emotional_arc": self.emotional_arc,
            "key_imagery": self.key_imagery,
            "metaphors": self.metaphors,
            "characters": self.characters,
            "settings": self.settings,
            "tone": self.tone,
            "genre_influence": self.genre_influence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeAnalysis":
        return cls(
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


@dataclass
class ScenePlan:
    """Complete scene plan for video generation."""

    # Metadata
    song_title: str
    duration: float
    master_seed: int

    # Analysis
    narrative: NarrativeAnalysis
    style_guide: StyleGuide

    # Scenes
    scenes: list[Scene]

    # Generation settings
    is_instrumental: bool = False
    target_resolution: str = "1080p"
    target_fps: int = 24

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def get_scene_at(self, time: float) -> Optional[Scene]:
        """Get the scene at the given time."""
        for scene in self.scenes:
            if scene.start <= time <= scene.end:
                return scene
        return None

    def to_dict(self) -> dict:
        return {
            "song_title": self.song_title,
            "duration": self.duration,
            "master_seed": self.master_seed,
            "is_instrumental": self.is_instrumental,
            "target_resolution": self.target_resolution,
            "target_fps": self.target_fps,
            "scene_count": self.scene_count,
            "narrative": self.narrative.to_dict(),
            "style_guide": self.style_guide.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenePlan":
        return cls(
            song_title=data.get("song_title", ""),
            duration=data.get("duration", 0.0),
            master_seed=data.get("master_seed", 0),
            narrative=NarrativeAnalysis.from_dict(data.get("narrative", {})),
            style_guide=StyleGuide.from_dict(data.get("style_guide", {})),
            scenes=[Scene.from_dict(s) for s in data.get("scenes", [])],
            is_instrumental=data.get("is_instrumental", False),
            target_resolution=data.get("target_resolution", "1080p"),
            target_fps=data.get("target_fps", 24),
        )


class PlanningError(Exception):
    """Raised when scene planning fails."""

    pass
