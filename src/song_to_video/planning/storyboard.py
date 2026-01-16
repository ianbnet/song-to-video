"""Storyboard document generator for human review."""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ScenePlan, Scene


def generate_storyboard_html(
    scene_plan: ScenePlan,
    frames_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate an HTML storyboard document for human review.

    Args:
        scene_plan: The scene plan with narrative, style, and scenes
        frames_dir: Directory containing reference frame images
        output_path: Optional path to save the HTML file

    Returns:
        HTML content as string
    """
    html_parts = [_html_header(scene_plan.song_title)]

    # Summary section
    html_parts.append(_summary_section(scene_plan))

    # Narrative section
    html_parts.append(_narrative_section(scene_plan))

    # Style guide section
    html_parts.append(_style_section(scene_plan))

    # Scenes section with images
    html_parts.append(_scenes_section(scene_plan, frames_dir))

    # Technical details
    html_parts.append(_technical_section(scene_plan))

    html_parts.append(_html_footer())

    html_content = "\n".join(html_parts)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)

    return html_content


def _html_header(title: str) -> str:
    """Generate HTML header with styles."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Storyboard: {title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            line-height: 1.6;
        }}
        h1 {{ color: #e94560; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #0f3460; background: #e94560; padding: 10px 15px; margin-top: 30px; border-radius: 5px; }}
        h3 {{ color: #e94560; margin-top: 20px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e94560;
        }}
        .summary-card label {{ color: #888; font-size: 0.85em; display: block; }}
        .summary-card value {{ font-size: 1.2em; font-weight: bold; }}
        .scene-card {{
            background: #16213e;
            border-radius: 10px;
            margin: 20px 0;
            overflow: hidden;
            display: grid;
            grid-template-columns: 300px 1fr;
        }}
        .scene-image {{
            width: 300px;
            height: 200px;
            object-fit: cover;
            background: #0f3460;
        }}
        .scene-image-placeholder {{
            width: 300px;
            height: 200px;
            background: #0f3460;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }}
        .scene-content {{
            padding: 15px 20px;
        }}
        .scene-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .scene-number {{
            background: #e94560;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .scene-time {{ color: #888; font-size: 0.9em; }}
        .scene-type {{
            background: #0f3460;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
        }}
        .prompt-box {{
            background: #0f3460;
            padding: 12px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
            border-left: 3px solid #e94560;
        }}
        .prompt-label {{
            color: #e94560;
            font-size: 0.8em;
            margin-bottom: 5px;
            font-family: sans-serif;
        }}
        .meta-row {{
            display: flex;
            gap: 20px;
            margin: 8px 0;
            color: #aaa;
            font-size: 0.9em;
        }}
        .meta-row span {{ display: flex; gap: 5px; }}
        .color-palette {{
            display: flex;
            gap: 5px;
            margin: 10px 0;
        }}
        .color-swatch {{
            width: 40px;
            height: 40px;
            border-radius: 5px;
            border: 2px solid #333;
        }}
        .narrative-text {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }}
        .keyword {{
            background: #0f3460;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
        @media (max-width: 700px) {{
            .scene-card {{ grid-template-columns: 1fr; }}
            .scene-image, .scene-image-placeholder {{ width: 100%; height: 180px; }}
        }}
    </style>
</head>
<body>
    <h1>Storyboard: {title}</h1>
"""


def _summary_section(plan: ScenePlan) -> str:
    """Generate summary section."""
    duration_min = plan.duration / 60
    return f"""
    <div class="summary-grid">
        <div class="summary-card">
            <label>Duration</label>
            <value>{duration_min:.1f} minutes</value>
        </div>
        <div class="summary-card">
            <label>Scenes</label>
            <value>{plan.scene_count}</value>
        </div>
        <div class="summary-card">
            <label>Visual Style</label>
            <value>{plan.style_guide.style.value.title()}</value>
        </div>
        <div class="summary-card">
            <label>Master Seed</label>
            <value>{plan.master_seed}</value>
        </div>
        <div class="summary-card">
            <label>Instrumental</label>
            <value>{'Yes' if plan.is_instrumental else 'No'}</value>
        </div>
        <div class="summary-card">
            <label>Target FPS</label>
            <value>{plan.target_fps}</value>
        </div>
    </div>
"""


def _narrative_section(plan: ScenePlan) -> str:
    """Generate narrative analysis section."""
    n = plan.narrative
    return f"""
    <h2>Narrative Analysis</h2>

    <h3>Overall Theme</h3>
    <div class="narrative-text">{n.overall_theme}</div>

    <h3>Story Summary</h3>
    <div class="narrative-text">{n.story_summary}</div>

    <h3>Emotional Arc</h3>
    <div class="narrative-text">{n.emotional_arc}</div>

    <h3>Tone</h3>
    <div class="narrative-text">{n.tone}</div>

    <h3>Key Imagery</h3>
    <div class="keywords">
        {''.join(f'<span class="keyword">{img}</span>' for img in n.key_imagery)}
    </div>

    <h3>Settings</h3>
    <div class="keywords">
        {''.join(f'<span class="keyword">{s}</span>' for s in n.settings)}
    </div>

    <h3>Characters</h3>
    <div class="keywords">
        {''.join(f'<span class="keyword">{c}</span>' for c in n.characters) if n.characters else '<span class="keyword">None specified</span>'}
    </div>
"""


def _style_section(plan: ScenePlan) -> str:
    """Generate style guide section."""
    s = plan.style_guide
    cp = s.color_palette

    return f"""
    <h2>Visual Style Guide</h2>

    <h3>Aesthetic</h3>
    <div class="narrative-text">{s.aesthetic}</div>

    <h3>Lighting</h3>
    <div class="narrative-text">{s.lighting}</div>

    <h3>Camera Style</h3>
    <div class="narrative-text">{s.camera_style}</div>

    <h3>Environment Theme</h3>
    <div class="narrative-text">{s.environment_theme}</div>

    <h3>Color Palette</h3>
    <div class="color-palette">
        <div class="color-swatch" style="background: {cp.primary};" title="Primary: {cp.primary}"></div>
        <div class="color-swatch" style="background: {cp.secondary};" title="Secondary: {cp.secondary}"></div>
        <div class="color-swatch" style="background: {cp.accent};" title="Accent: {cp.accent}"></div>
        <div class="color-swatch" style="background: {cp.background};" title="Background: {cp.background}"></div>
    </div>

    <h3>Mood Keywords</h3>
    <div class="keywords">
        {''.join(f'<span class="keyword">{kw}</span>' for kw in (s.mood_keywords or []))}
    </div>

    <h3>Negative Prompts (Things to Avoid)</h3>
    <div class="prompt-box">
        {', '.join(s.negative_prompts) if s.negative_prompts else 'None specified'}
    </div>
"""


def _scenes_section(plan: ScenePlan, frames_dir: Optional[Path]) -> str:
    """Generate scenes section with images."""
    parts = ["<h2>Scene Breakdown</h2>"]

    for scene in plan.scenes:
        image_html = _get_image_html(scene, frames_dir)
        transition_in = scene.transition_in.value.upper()
        transition_out = scene.transition_out.value.upper()

        parts.append(f"""
    <div class="scene-card">
        {image_html}
        <div class="scene-content">
            <div class="scene-header">
                <span class="scene-number">Scene {scene.id}</span>
                <span class="scene-type">{scene.section_type}</span>
                <span class="scene-time">{_format_time(scene.start)} - {_format_time(scene.end)} ({scene.duration:.1f}s)</span>
            </div>

            <div class="meta-row">
                <span>Mood: <strong>{scene.mood}</strong></span>
                <span>Energy: <strong>{scene.energy:.0%}</strong></span>
                <span>Transitions: {transition_in} in / {transition_out} out</span>
            </div>

            <h4 style="margin: 15px 0 5px 0; color: #e94560;">Description</h4>
            <p style="margin: 0;">{scene.description}</p>

            <div class="prompt-box">
                <div class="prompt-label">IMAGE GENERATION PROMPT</div>
                {scene.prompt}
            </div>

            {f'<p style="color: #888; font-size: 0.9em;"><em>Lyrics: "{scene.lyrics_text}"</em></p>' if scene.lyrics_text else ''}

            {f'<p style="color: #888;">Camera: {scene.camera_movement}</p>' if scene.camera_movement else ''}
        </div>
    </div>
""")

    return "\n".join(parts)


def _get_image_html(scene: Scene, frames_dir: Optional[Path]) -> str:
    """Get image HTML - either embedded base64 or placeholder."""
    if frames_dir:
        image_path = frames_dir / f"scene_{scene.id:03d}.png"
        if image_path.exists():
            # Embed image as base64
            image_data = base64.b64encode(image_path.read_bytes()).decode()
            return f'<img class="scene-image" src="data:image/png;base64,{image_data}" alt="Scene {scene.id}">'

    return f'<div class="scene-image-placeholder">Scene {scene.id}<br>Image not generated</div>'


def _technical_section(plan: ScenePlan) -> str:
    """Generate technical details section."""
    return f"""
    <h2>Technical Details</h2>
    <div class="narrative-text">
        <p><strong>Master Seed:</strong> {plan.master_seed}</p>
        <p><strong>Target FPS:</strong> {plan.target_fps}</p>
        <p><strong>Total Duration:</strong> {plan.duration:.1f} seconds</p>
        <p><strong>Scene Count:</strong> {plan.scene_count}</p>
        <p>Use the same master seed to regenerate identical visuals.</p>
    </div>
"""


def _html_footer() -> str:
    """Generate HTML footer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <div class="footer">
        Generated by song-to-video on {timestamp}
    </div>
</body>
</html>
"""


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"
