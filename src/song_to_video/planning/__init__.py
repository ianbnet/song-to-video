"""Scene planning and narrative analysis module."""

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
from .ollama import (
    OllamaClient,
    OllamaResponse,
    get_ollama_client,
    RECOMMENDED_MODELS,
)
from .seed import (
    generate_master_seed,
    get_scene_seed,
    get_frame_seed,
    generate_audio_fingerprint,
)
from .narrative import (
    NarrativePlanner,
    create_scene_plan,
)
from .storyboard import generate_storyboard_html

__all__ = [
    # Models
    "Scene",
    "ScenePlan",
    "StyleGuide",
    "NarrativeAnalysis",
    "ColorPalette",
    "VisualStyle",
    "SceneTransition",
    "PlanningError",
    # Ollama
    "OllamaClient",
    "OllamaResponse",
    "get_ollama_client",
    "RECOMMENDED_MODELS",
    # Seed
    "generate_master_seed",
    "get_scene_seed",
    "get_frame_seed",
    "generate_audio_fingerprint",
    # Narrative
    "NarrativePlanner",
    "create_scene_plan",
    # Storyboard
    "generate_storyboard_html",
]
