"""Master visual seed generation for consistent video generation."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_audio_fingerprint(audio_path: Path) -> str:
    """
    Generate a fingerprint from audio file content.

    Uses a hash of the file content for deterministic fingerprinting.

    Args:
        audio_path: Path to audio file

    Returns:
        Hex string fingerprint
    """
    hasher = hashlib.sha256()

    with open(audio_path, "rb") as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def generate_style_hash(style_description: str) -> str:
    """
    Generate a hash from style description.

    Args:
        style_description: Text description of visual style

    Returns:
        Hex string hash
    """
    return hashlib.md5(style_description.encode()).hexdigest()


def generate_master_seed(
    audio_path: Path,
    style_description: str = "",
    user_seed: Optional[int] = None,
) -> int:
    """
    Generate a deterministic master seed for visual consistency.

    The master seed is derived from:
    1. Audio file fingerprint (ensures same song = same seed)
    2. Style description hash (different styles = different seeds)
    3. Optional user override

    Args:
        audio_path: Path to audio file
        style_description: Visual style description
        user_seed: Optional user-provided seed override

    Returns:
        Integer seed value (0 to 2^31-1 for compatibility)
    """
    # User override takes precedence
    if user_seed is not None:
        logger.info(f"Using user-provided seed: {user_seed}")
        return user_seed % (2**31)

    # Generate from audio + style
    audio_fingerprint = generate_audio_fingerprint(audio_path)
    style_hash = generate_style_hash(style_description)

    # Combine hashes
    combined = audio_fingerprint + style_hash
    combined_hash = hashlib.sha256(combined.encode()).hexdigest()

    # Convert to integer seed (use first 8 hex chars = 32 bits)
    seed = int(combined_hash[:8], 16) % (2**31)

    logger.info(f"Generated master seed: {seed} (from audio fingerprint + style hash)")

    return seed


def get_scene_seed(master_seed: int, scene_index: int) -> int:
    """
    Get a deterministic seed for a specific scene.

    Each scene gets a unique but reproducible seed derived from
    the master seed and scene index.

    Args:
        master_seed: The master seed for the video
        scene_index: Index of the scene (0-based)

    Returns:
        Integer seed for the scene
    """
    # Use a hash to get well-distributed scene seeds
    combined = f"{master_seed}_{scene_index}"
    scene_hash = hashlib.md5(combined.encode()).hexdigest()

    return int(scene_hash[:8], 16) % (2**31)


def get_frame_seed(scene_seed: int, frame_index: int) -> int:
    """
    Get a deterministic seed for a specific frame within a scene.

    Args:
        scene_seed: The seed for the scene
        frame_index: Index of the frame within the scene

    Returns:
        Integer seed for the frame
    """
    combined = f"{scene_seed}_{frame_index}"
    frame_hash = hashlib.md5(combined.encode()).hexdigest()

    return int(frame_hash[:8], 16) % (2**31)
