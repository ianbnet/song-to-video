"""Prompt templating for consistent image generation."""

from dataclasses import dataclass
from typing import Optional

from ..planning.models import Scene, StyleGuide, VisualStyle

# Maximum tokens for CLIP-based models (SDXL, SD1.5)
# Flux can handle more, but we keep prompts simple for consistency
MAX_PROMPT_WORDS = 40


@dataclass
class PromptTemplate:
    """Template for generating consistent image prompts."""

    # Core template parts
    style_prefix: str = ""
    quality_suffix: str = "high quality, detailed"
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "multiple people, crowd, busy background, cluttered, text, watermark, "
        "logo, signature, frame, border"
    )

    def format_prompt(
        self,
        scene: Scene,
        style_guide: StyleGuide,
        extra_context: str = "",
    ) -> str:
        """
        Format a complete prompt for image generation.

        Keeps prompts SIMPLE and focused for better results.

        Args:
            scene: Scene with description and prompt
            style_guide: Visual style guide
            extra_context: Additional context to include

        Returns:
            Formatted prompt string (limited length)
        """
        parts = []

        # Add style prefix if set
        if self.style_prefix:
            parts.append(self.style_prefix)

        # Start with scene's simple prompt
        if scene.prompt:
            parts.append(scene.prompt)

        # Add ONE key style element (not all of them)
        if style_guide.lighting:
            parts.append(f"{style_guide.lighting} lighting")

        # Add quality suffix
        parts.append(self.quality_suffix)

        # Join and limit length
        full_prompt = ", ".join(parts)
        return _truncate_prompt(full_prompt, MAX_PROMPT_WORDS)

    def format_negative_prompt(self, style_guide: StyleGuide) -> str:
        """
        Format the negative prompt.

        Args:
            style_guide: Visual style guide with things to avoid

        Returns:
            Formatted negative prompt
        """
        parts = [self.negative_prompt]

        # Add style-specific negative prompts (limit to 3)
        if style_guide.negative_prompts:
            # Filter out non-string items
            valid_negatives = [
                n for n in style_guide.negative_prompts[:3]
                if isinstance(n, str)
            ]
            parts.extend(valid_negatives)

        return ", ".join(parts)


def _truncate_prompt(prompt: str, max_words: int) -> str:
    """Truncate prompt to maximum words."""
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    return " ".join(words[:max_words])


# Pre-defined prompt templates for different styles
# These are intentionally SIMPLE to avoid prompt overload

CINEMATIC_TEMPLATE = PromptTemplate(
    style_prefix="cinematic film still",
    quality_suffix="high quality, 8k, sharp focus",
    negative_prompt=(
        "blurry, low quality, amateur, cartoon, anime, illustration, "
        "multiple people, crowd, busy, cluttered, text, watermark"
    ),
)

ANIMATED_TEMPLATE = PromptTemplate(
    style_prefix="animated style, stylized illustration",
    quality_suffix="high quality, vibrant colors",
    negative_prompt=(
        "blurry, low quality, photorealistic, live action, "
        "multiple characters, busy, cluttered, text, watermark"
    ),
)

ABSTRACT_TEMPLATE = PromptTemplate(
    style_prefix="abstract artistic",
    quality_suffix="high quality, artistic, expressive",
    negative_prompt=(
        "blurry, low quality, photorealistic, mundane, boring, "
        "literal, text, watermark"
    ),
)

DREAMY_TEMPLATE = PromptTemplate(
    style_prefix="dreamy ethereal, soft",
    quality_suffix="high quality, atmospheric",
    negative_prompt=(
        "harsh, sharp edges, realistic, mundane, busy, "
        "cluttered, text, watermark"
    ),
)

STYLIZED_TEMPLATE = PromptTemplate(
    style_prefix="stylized artistic",
    quality_suffix="high quality, vibrant",
    negative_prompt=(
        "blurry, low quality, photorealistic, boring, "
        "busy, cluttered, text, watermark"
    ),
)

BRIGHT_TEMPLATE = PromptTemplate(
    style_prefix="bright colorful",
    quality_suffix="high quality, vivid colors",
    negative_prompt=(
        "dark, gloomy, desaturated, monochrome, "
        "busy, cluttered, text, watermark"
    ),
)

DARK_TEMPLATE = PromptTemplate(
    style_prefix="dark moody atmospheric",
    quality_suffix="high quality, dramatic",
    negative_prompt=(
        "bright, cheerful, colorful, cartoon, "
        "busy, cluttered, text, watermark"
    ),
)


def get_template_for_style(style_name: str) -> PromptTemplate:
    """
    Get a prompt template for a given style.

    Args:
        style_name: Style name (cinematic, animated, etc.)

    Returns:
        Appropriate PromptTemplate
    """
    templates = {
        "cinematic": CINEMATIC_TEMPLATE,
        "animated": ANIMATED_TEMPLATE,
        "abstract": ABSTRACT_TEMPLATE,
        "dreamy": DREAMY_TEMPLATE,
        "stylized": STYLIZED_TEMPLATE,
        "bright": BRIGHT_TEMPLATE,
        "dark": DARK_TEMPLATE,
        "realistic": CINEMATIC_TEMPLATE,
        "vintage": PromptTemplate(
            style_prefix="vintage retro",
            quality_suffix="high quality, film grain",
            negative_prompt="modern, digital, busy, cluttered, text, watermark",
        ),
        "futuristic": PromptTemplate(
            style_prefix="futuristic sci-fi",
            quality_suffix="high quality, detailed",
            negative_prompt="vintage, old, rustic, busy, cluttered, text, watermark",
        ),
        "minimalist": PromptTemplate(
            style_prefix="minimalist simple clean",
            quality_suffix="high quality",
            negative_prompt="busy, cluttered, detailed, complex, text, watermark",
        ),
    }

    return templates.get(style_name.lower(), PromptTemplate())


def build_scene_prompt(
    scene: Scene,
    style_guide: StyleGuide,
    include_lyrics: bool = False,
) -> tuple[str, str]:
    """
    Build complete prompt and negative prompt for a scene.

    Creates SIMPLE, focused prompts that work well with image models.

    Args:
        scene: Scene to generate prompt for
        style_guide: Visual style guide
        include_lyrics: Whether to include lyrics context (usually False for cleaner images)

    Returns:
        Tuple of (prompt, negative_prompt)
    """
    # Get appropriate template
    template = get_template_for_style(style_guide.style.value)

    # Build extra context from lyrics if requested (but keep it minimal)
    extra_context = ""
    if include_lyrics and scene.lyrics_text:
        # Extract just 3-4 key words from lyrics
        lyrics_words = scene.lyrics_text.split()[:4]
        if lyrics_words:
            extra_context = " ".join(lyrics_words)

    # Format prompts
    prompt = template.format_prompt(scene, style_guide, extra_context)
    negative_prompt = template.format_negative_prompt(style_guide)

    return prompt, negative_prompt


def simplify_prompt(prompt: str, max_words: int = 30) -> str:
    """
    Simplify a prompt by keeping only the most important words.

    Useful for cleaning up LLM-generated prompts.

    Args:
        prompt: Original prompt
        max_words: Maximum words to keep

    Returns:
        Simplified prompt
    """
    # Remove common filler words and duplicates
    filler_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "that", "which", "who", "whom", "this", "these", "those", "it", "its",
    }

    words = prompt.lower().replace(",", " ").replace(".", " ").split()
    # Keep unique non-filler words
    seen = set()
    filtered = []
    for word in words:
        if word not in filler_words and word not in seen:
            filtered.append(word)
            seen.add(word)

    return " ".join(filtered[:max_words])
