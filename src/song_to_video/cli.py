"""Command-line interface for song-to-video."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .memory import (
    detect_system_info,
    get_hardware_tier,
    is_gpu_compatible,
    get_vram_usage,
    flush_vram,
    get_offload_config,
    MemoryMonitor,
    get_current_phase,
    force_unlock,
)

app = typer.Typer(
    name="song-to-video",
    help="Transform songs into AI-generated music videos.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"song-to-video v{__version__}")


@app.command()
def hardware():
    """Display detected hardware and tier classification."""
    info = detect_system_info()
    tier = get_hardware_tier(info.gpu)

    # Create hardware info table
    table = Table(title="Hardware Detection", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # System RAM
    table.add_row("System RAM", f"{info.ram_total_gb:.1f} GB total")
    table.add_row("RAM Available", f"{info.ram_available_gb:.1f} GB")
    table.add_row("", "")  # Spacer

    if info.gpu:
        table.add_row("GPU", info.gpu.name)
        table.add_row("VRAM Total", f"{info.gpu.vram_total_gb:.1f} GB")
        table.add_row("VRAM Available", f"{info.gpu.vram_available_gb:.1f} GB")
        table.add_row(
            "Compute Capability",
            f"{info.gpu.compute_capability[0]}.{info.gpu.compute_capability[1]}",
        )
        table.add_row("CUDA Version", info.gpu.cuda_version)
        table.add_row("Driver Version", info.gpu.driver_version)
        table.add_row("", "")  # Spacer

        compatible = is_gpu_compatible(info.gpu)
        compat_str = "[green]Yes[/green]" if compatible else "[red]No[/red]"
        table.add_row("Compatible", compat_str)
    else:
        table.add_row("GPU", "[yellow]Not detected[/yellow]")

    table.add_row("", "")  # Spacer
    tier_colors = {
        "low": "yellow",
        "mid": "blue",
        "high": "green",
        "cpu": "red",
    }
    tier_color = tier_colors.get(tier.value, "white")
    table.add_row("Hardware Tier", f"[{tier_color}]{tier.value.upper()}[/{tier_color}]")

    console.print(table)

    # Show tier description
    tier_descriptions = {
        "low": "8GB VRAM - Will use aggressive CPU offloading and 4-bit quantization",
        "mid": "12-16GB VRAM - Will use moderate offloading and 8-bit quantization",
        "high": "24GB+ VRAM - Full precision, minimal offloading needed",
        "cpu": "No compatible GPU - Will run on CPU only (very slow)",
    }
    console.print(f"\n[dim]{tier_descriptions.get(tier.value, '')}[/dim]")


@app.command()
def memory(
    diagnostics: bool = typer.Option(False, "--diagnostics", "-d", help="Run memory diagnostics"),
    flush: bool = typer.Option(False, "--flush", "-f", help="Flush VRAM"),
):
    """Display memory status and run diagnostics."""
    monitor = MemoryMonitor()
    status = monitor.get_status()

    if flush:
        console.print("[yellow]Flushing VRAM...[/yellow]")
        result = flush_vram()
        if result.get("status") == "success":
            console.print(
                f"[green]Freed {result.get('freed_gb', 0):.2f} GB VRAM[/green]"
            )
        else:
            console.print(f"[red]Flush failed: {result.get('status')}[/red]")
        console.print("")

    # Memory status table
    table = Table(title="Memory Status", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Used", justify="right")
    table.add_column("Available", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Usage", justify="right")

    # RAM row
    ram_pct = status.ram_percent
    ram_color = "green" if ram_pct < 80 else "yellow" if ram_pct < 95 else "red"
    table.add_row(
        "System RAM",
        f"{status.ram_used_gb:.1f} GB",
        f"{status.ram_available_gb:.1f} GB",
        f"{status.ram_total_gb:.1f} GB",
        f"[{ram_color}]{ram_pct:.1f}%[/{ram_color}]",
    )

    # VRAM row
    if status.vram:
        vram_pct = status.vram.used_percent
        vram_color = "green" if vram_pct < 85 else "yellow" if vram_pct < 95 else "red"
        table.add_row(
            "GPU VRAM",
            f"{status.vram.used_gb:.1f} GB",
            f"{status.vram.free_gb:.1f} GB",
            f"{status.vram.total_gb:.1f} GB",
            f"[{vram_color}]{vram_pct:.1f}%[/{vram_color}]",
        )

        # PyTorch details
        table.add_row(
            "  PyTorch Allocated",
            f"{status.vram.allocated_gb:.2f} GB",
            "",
            "",
            "",
        )
        table.add_row(
            "  PyTorch Reserved",
            f"{status.vram.reserved_gb:.2f} GB",
            "",
            "",
            "",
        )
    else:
        table.add_row("GPU VRAM", "[yellow]N/A[/yellow]", "", "", "")

    console.print(table)

    # Check for warnings
    warnings = monitor.check_health()
    if warnings:
        console.print("")
        for w in warnings:
            color = "yellow" if w.severity.value == "warning" else "red"
            console.print(f"[{color}]{w.severity.value.upper()}: {w.message}[/{color}]")

    # Phase lock status
    current_phase = get_current_phase()
    if current_phase:
        console.print(f"\n[yellow]Active phase: {current_phase}[/yellow]")

    if diagnostics:
        console.print("")
        run_diagnostics()


def run_diagnostics():
    """Run memory diagnostics."""
    console.print(Panel("Running Memory Diagnostics", style="blue"))

    info = detect_system_info()
    tier = get_hardware_tier(info.gpu)

    # Get offload config for this tier
    config = get_offload_config(tier, info.ram_total_gb)

    console.print(f"\n[cyan]Offload Configuration for {tier.value.upper()} tier:[/cyan]")

    config_table = Table(show_header=False)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value")

    config_table.add_row("CPU Offload", "Yes" if config.enable_cpu_offload else "No")
    config_table.add_row("Disk Offload", "Yes" if config.enable_disk_offload else "No")
    config_table.add_row(
        "Max GPU Memory",
        f"{config.max_gpu_memory_gb} GB" if config.max_gpu_memory_gb else "Unlimited",
    )
    config_table.add_row(
        "Max CPU Memory",
        f"{config.max_cpu_memory_gb} GB" if config.max_cpu_memory_gb else "Unlimited",
    )
    config_table.add_row("4-bit Quantization", "Yes" if config.use_4bit else "No")
    config_table.add_row("8-bit Quantization", "Yes" if config.use_8bit else "No")
    config_table.add_row("Attention Slicing", "Yes" if config.enable_attention_slicing else "No")
    config_table.add_row("VAE Slicing", "Yes" if config.enable_vae_slicing else "No")
    config_table.add_row("VAE Tiling", "Yes" if config.enable_vae_tiling else "No")
    config_table.add_row("Sequential Offload", "Yes" if config.enable_sequential_offload else "No")

    console.print(config_table)

    # Test VRAM flush
    console.print("\n[cyan]Testing VRAM flush...[/cyan]")
    result = flush_vram()
    if result.get("status") == "success":
        console.print(f"[green]VRAM flush successful[/green]")
    else:
        console.print(f"[yellow]VRAM flush: {result.get('status')}[/yellow]")

    console.print("\n[green]Diagnostics complete[/green]")


@app.command()
def unlock(
    force: bool = typer.Option(False, "--force", "-f", help="Force unlock without confirmation"),
):
    """Force unlock the phase lock (use after crashes)."""
    current = get_current_phase()

    if not current:
        console.print("[green]No phase lock is currently active[/green]")
        return

    console.print(f"[yellow]Current phase lock: {current}[/yellow]")

    if not force:
        confirm = typer.confirm("Force unlock? This may cause issues if another process is running.")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    if force_unlock():
        console.print("[green]Phase lock released[/green]")
    else:
        console.print("[red]Failed to release lock[/red]")


@app.command()
def transcribe(
    audio_file: str = typer.Argument(..., help="Path to audio file (MP3, WAV, etc.)"),
    lyrics: Optional[str] = typer.Option(
        None, "--lyrics", "-l", help="Manual lyrics file (SRT, LRC, TXT)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON or SRT)"
    ),
    embedded_only: bool = typer.Option(
        False, "--embedded-only", "-e", help="Only show embedded lyrics, don't transcribe"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Whisper model size (tiny, small, medium, large-v3)"
    ),
):
    """Transcribe audio to lyrics with timestamps."""
    from .audio import (
        load_audio,
        extract_embedded_lyrics,
        parse_lyrics_file,
        transcribe_audio,
        get_recommended_model,
        AudioValidationError,
        LyricsParseError,
        TranscriptionError,
    )

    audio_path = Path(audio_file)

    # Validate audio file
    try:
        audio_info = load_audio(audio_path)
        console.print(f"[cyan]Audio:[/cyan] {audio_info}")
    except AudioValidationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    result_lyrics = None

    # Priority 1: Manual lyrics file
    if lyrics:
        lyrics_path = Path(lyrics)
        try:
            result_lyrics = parse_lyrics_file(lyrics_path)
            console.print(f"[green]Loaded manual lyrics:[/green] {lyrics_path.name}")
        except LyricsParseError as e:
            console.print(f"[red]Error parsing lyrics:[/red] {e}")
            raise typer.Exit(1)

    # Priority 2: Embedded lyrics
    if result_lyrics is None:
        embedded = extract_embedded_lyrics(audio_path)
        if embedded:
            result_lyrics = embedded
            console.print("[green]Found embedded lyrics in audio file[/green]")
        elif embedded_only:
            console.print("[yellow]No embedded lyrics found[/yellow]")
            raise typer.Exit(0)

    # Priority 3: Transcription
    if result_lyrics is None:
        if model is None:
            model = get_recommended_model()
            console.print(f"[dim]Using Whisper model: {model}[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Transcribing audio...", total=None)
                result_lyrics = transcribe_audio(audio_path, model_size=model)

            console.print(
                f"[green]Transcription complete[/green] "
                f"(confidence: {result_lyrics.confidence:.0%})"
            )
        except TranscriptionError as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            raise typer.Exit(1)

    # Display lyrics
    console.print(f"\n[cyan]Source:[/cyan] {result_lyrics.source.value}")
    console.print(f"[cyan]Lines:[/cyan] {len(result_lyrics.lines)}")
    console.print(f"[cyan]Word timing:[/cyan] {'Yes' if result_lyrics.has_word_timing else 'No'}")
    console.print("")

    for line in result_lyrics.lines:
        time_str = f"[{line.start:06.2f} - {line.end:06.2f}]"
        console.print(f"[dim]{time_str}[/dim] {line.text}")

    # Save output if requested
    if output:
        output_path = Path(output)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(result_lyrics.to_dict(), indent=2))
        elif output_path.suffix.lower() == ".srt":
            output_path.write_text(result_lyrics.to_srt())
        elif output_path.suffix.lower() == ".lrc":
            output_path.write_text(result_lyrics.to_lrc())
        else:
            output_path.write_text(json.dumps(result_lyrics.to_dict(), indent=2))

        console.print(f"\n[green]Saved to:[/green] {output_path}")


@app.command()
def analyze(
    audio_file: str = typer.Argument(..., help="Path to audio file (MP3, WAV, etc.)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed analysis"
    ),
):
    """Analyze audio to extract musical features (tempo, sections, mood)."""
    from .audio import (
        load_audio,
        analyze_audio,
        AudioValidationError,
    )

    audio_path = Path(audio_file)

    # Validate audio file
    try:
        audio_info = load_audio(audio_path)
        console.print(f"[cyan]Audio:[/cyan] {audio_info}")
    except AudioValidationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Analyze audio
    console.print("")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Analyzing audio...", total=None)
            features = analyze_audio(audio_path)

        console.print("[green]Analysis complete[/green]\n")
    except AudioValidationError as e:
        console.print(f"[red]Analysis failed:[/red] {e}")
        raise typer.Exit(1)

    # Display results
    table = Table(title="Audio Analysis", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Duration", f"{features.duration:.1f}s ({features.duration/60:.1f} min)")
    table.add_row("Sample Rate", f"{features.sample_rate} Hz")
    table.add_row("", "")

    # Tempo and rhythm
    table.add_row("Tempo", f"{features.tempo:.1f} BPM")
    table.add_row("Tempo Confidence", f"{features.tempo_confidence:.0%}")
    table.add_row("Beat Count", str(len(features.beats)))
    table.add_row("", "")

    # Key
    if features.key:
        table.add_row("Key", features.key)
        table.add_row("Key Confidence", f"{features.key_confidence:.0%}")
        table.add_row("", "")

    # Energy
    table.add_row("Average Energy", f"{features.average_energy:.0%}")
    table.add_row("Energy Variance", f"{features.energy_variance:.3f}")
    table.add_row("", "")

    # Vocals
    vocals_str = "[green]Yes[/green]" if features.has_vocals else "[yellow]No[/yellow]"
    table.add_row("Has Vocals", vocals_str)
    table.add_row("Vocal Ratio", f"{features.vocal_ratio:.0%}")
    table.add_row("", "")

    # Mood
    mood_colors = {
        "happy": "green",
        "sad": "blue",
        "energetic": "yellow",
        "calm": "cyan",
        "aggressive": "red",
        "dark": "magenta",
        "uplifting": "green",
        "melancholic": "blue",
    }
    mood_color = mood_colors.get(features.mood.value, "white")
    table.add_row("Mood", f"[{mood_color}]{features.mood.value.title()}[/{mood_color}]")
    table.add_row("Mood Confidence", f"{features.mood_confidence:.0%}")

    # Genre tags
    if features.genre_tags:
        table.add_row("Tags", ", ".join(features.genre_tags))

    console.print(table)

    # Sections
    if features.sections:
        console.print("\n[cyan]Sections:[/cyan]")
        sections_table = Table(show_header=True)
        sections_table.add_column("Time", style="dim")
        sections_table.add_column("Duration", style="dim")
        sections_table.add_column("Type", style="cyan")
        sections_table.add_column("Label")

        for section in features.sections:
            time_str = f"{section.start:.1f}s - {section.end:.1f}s"
            duration_str = f"{section.duration:.1f}s"
            sections_table.add_row(
                time_str,
                duration_str,
                section.type.value,
                section.label,
            )

        console.print(sections_table)

    # Verbose output - show beats
    if verbose and features.beats:
        console.print(f"\n[cyan]Beats:[/cyan] (showing first 20 of {len(features.beats)})")
        beat_times = [f"{b.time:.2f}s" for b in features.beats[:20]]
        console.print(f"[dim]{', '.join(beat_times)}{'...' if len(features.beats) > 20 else ''}[/dim]")

    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(features.to_dict(), indent=2))
        console.print(f"\n[green]Saved to:[/green] {output_path}")


@app.command()
def plan(
    audio_file: str = typer.Argument(..., help="Path to audio file (MP3, WAV, etc.)"),
    lyrics: Optional[str] = typer.Option(
        None, "--lyrics", "-l", help="Manual lyrics file (SRT, LRC, TXT)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Override master visual seed"
    ),
    scenes_only: bool = typer.Option(
        False, "--scenes-only", help="Output only scene plan JSON (no display)"
    ),
):
    """Plan video scenes from audio and lyrics using local LLM."""
    from .audio import (
        load_audio,
        analyze_audio,
        extract_embedded_lyrics,
        parse_lyrics_file,
        transcribe_audio,
        get_recommended_model,
        AudioValidationError,
        LyricsParseError,
        TranscriptionError,
    )
    from .planning import (
        create_scene_plan,
        get_ollama_client,
        PlanningError,
    )

    audio_path = Path(audio_file)

    # Validate audio file
    try:
        audio_info = load_audio(audio_path)
        if not scenes_only:
            console.print(f"[cyan]Audio:[/cyan] {audio_info}")
    except AudioValidationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Check Ollama availability
    client = get_ollama_client()
    if not client.is_available():
        console.print("[red]Error:[/red] Ollama is not running. Start it with: ollama serve")
        raise typer.Exit(1)

    if not scenes_only:
        console.print(f"[dim]Using LLM: {client.model}[/dim]")

    # Step 1: Get lyrics
    result_lyrics = None

    if lyrics:
        lyrics_path = Path(lyrics)
        try:
            result_lyrics = parse_lyrics_file(lyrics_path)
            if not scenes_only:
                console.print(f"[green]Loaded manual lyrics:[/green] {lyrics_path.name}")
        except LyricsParseError as e:
            console.print(f"[red]Error parsing lyrics:[/red] {e}")
            raise typer.Exit(1)

    if result_lyrics is None:
        embedded = extract_embedded_lyrics(audio_path)
        if embedded:
            result_lyrics = embedded
            if not scenes_only:
                console.print("[green]Found embedded lyrics in audio file[/green]")

    if result_lyrics is None:
        if not scenes_only:
            model = get_recommended_model()
            console.print(f"[dim]Transcribing with Whisper ({model})...[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                disable=scenes_only,
            ) as progress:
                progress.add_task("Transcribing audio...", total=None)
                result_lyrics = transcribe_audio(audio_path)
        except TranscriptionError as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            raise typer.Exit(1)

    # Step 2: Analyze audio
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=scenes_only,
        ) as progress:
            progress.add_task("Analyzing audio...", total=None)
            features = analyze_audio(audio_path)
    except AudioValidationError as e:
        console.print(f"[red]Analysis failed:[/red] {e}")
        raise typer.Exit(1)

    # Step 3: Create scene plan
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=scenes_only,
        ) as progress:
            progress.add_task("Planning scenes with LLM...", total=None)
            scene_plan = create_scene_plan(
                audio_path=audio_path,
                lyrics=result_lyrics,
                audio_features=features,
                song_title=audio_path.stem,
                user_seed=seed,
            )
    except PlanningError as e:
        console.print(f"[red]Planning failed:[/red] {e}")
        raise typer.Exit(1)

    # Output
    if scenes_only:
        # Just output JSON
        print(json.dumps(scene_plan.to_dict(), indent=2))
    else:
        # Display summary
        console.print("\n[green]Scene plan created[/green]\n")

        # Summary table
        table = Table(title="Scene Plan Summary", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Song Title", scene_plan.song_title)
        table.add_row("Duration", f"{scene_plan.duration:.1f}s ({scene_plan.duration/60:.1f} min)")
        table.add_row("Master Seed", str(scene_plan.master_seed))
        table.add_row("Scene Count", str(scene_plan.scene_count))
        table.add_row("Is Instrumental", "Yes" if scene_plan.is_instrumental else "No")
        table.add_row("", "")
        table.add_row("Visual Style", scene_plan.style_guide.style.value)
        table.add_row("Theme", scene_plan.narrative.overall_theme[:60] + "..." if len(scene_plan.narrative.overall_theme) > 60 else scene_plan.narrative.overall_theme)
        table.add_row("Tone", scene_plan.narrative.tone)

        console.print(table)

        # Scene list
        console.print("\n[cyan]Scenes:[/cyan]")
        scenes_table = Table(show_header=True)
        scenes_table.add_column("#", style="dim", width=3)
        scenes_table.add_column("Time", style="dim", width=14)
        scenes_table.add_column("Type", style="cyan", width=12)
        scenes_table.add_column("Mood", width=12)
        scenes_table.add_column("Description", width=50)

        for scene in scene_plan.scenes:
            time_str = f"{scene.start:.1f}s-{scene.end:.1f}s"
            desc = scene.description[:47] + "..." if len(scene.description) > 50 else scene.description
            scenes_table.add_row(
                str(scene.id),
                time_str,
                scene.section_type,
                scene.mood,
                desc,
            )

        console.print(scenes_table)

        # Style guide summary
        console.print("\n[cyan]Style Guide:[/cyan]")
        console.print(f"[dim]Aesthetic:[/dim] {scene_plan.style_guide.aesthetic}")
        console.print(f"[dim]Lighting:[/dim] {scene_plan.style_guide.lighting}")
        console.print(f"[dim]Camera:[/dim] {scene_plan.style_guide.camera_style}")

    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(scene_plan.to_dict(), indent=2))
        if not scenes_only:
            console.print(f"\n[green]Saved to:[/green] {output_path}")


@app.command()
def frames(
    plan_file: str = typer.Argument(..., help="Path to scene plan JSON file"),
    output_dir: str = typer.Option(
        "./frames", "--output", "-o", help="Output directory for frames"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use (flux-schnell, flux-dev, sdxl, sd-1.5)"
    ),
    quality: str = typer.Option(
        "standard", "--quality", "-q", help="Quality preset (draft, standard, high)"
    ),
    scene: Optional[int] = typer.Option(
        None, "--scene", "-s", help="Generate only this scene (0-indexed)"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--regenerate", help="Skip scenes with existing frames"
    ),
):
    """Generate reference frames for a scene plan."""
    from .planning.models import ScenePlan
    from .image import (
        generate_reference_frames,
        FluxGenerator,
        ImageModel,
        ImageQuality,
        ImageGenerationError,
    )

    plan_path = Path(plan_file)
    output_path = Path(output_dir)

    # Load scene plan
    if not plan_path.exists():
        console.print(f"[red]Error:[/red] Scene plan not found: {plan_path}")
        raise typer.Exit(1)

    try:
        plan_data = json.loads(plan_path.read_text())
        scene_plan = ScenePlan.from_dict(plan_data)
        console.print(f"[cyan]Loaded scene plan:[/cyan] {scene_plan.song_title}")
        console.print(f"[dim]Scenes: {scene_plan.scene_count}, Master seed: {scene_plan.master_seed}[/dim]")
    except Exception as e:
        console.print(f"[red]Error loading scene plan:[/red] {e}")
        raise typer.Exit(1)

    # Parse model
    image_model = None
    if model:
        model_map = {
            "flux-schnell": ImageModel.FLUX_SCHNELL,
            "flux-dev": ImageModel.FLUX_DEV,
            "sdxl": ImageModel.SDXL,
            "sd-1.5": ImageModel.SD15,
        }
        image_model = model_map.get(model.lower())
        if not image_model:
            console.print(f"[red]Unknown model:[/red] {model}")
            console.print("[dim]Available: flux-schnell, flux-dev, sdxl, sd-1.5[/dim]")
            raise typer.Exit(1)

    # Parse quality
    quality_map = {
        "draft": ImageQuality.DRAFT,
        "standard": ImageQuality.STANDARD,
        "high": ImageQuality.HIGH,
    }
    image_quality = quality_map.get(quality.lower(), ImageQuality.STANDARD)

    # Filter to single scene if specified
    if scene is not None:
        if scene < 0 or scene >= len(scene_plan.scenes):
            console.print(f"[red]Invalid scene:[/red] {scene} (valid: 0-{len(scene_plan.scenes)-1})")
            raise typer.Exit(1)
        scene_plan.scenes = [scene_plan.scenes[scene]]
        console.print(f"[dim]Generating only scene {scene}[/dim]")

    # Show what we're doing
    generator = FluxGenerator(model=image_model)
    console.print(f"[cyan]Model:[/cyan] {generator.model.value}")
    console.print(f"[cyan]Quality:[/cyan] {image_quality.value}")
    console.print(f"[cyan]Resolution:[/cyan] {generator.config.width}x{generator.config.height}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print("")

    # Generate frames
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating frames...",
                total=len(scene_plan.scenes),
            )

            # Generate each frame with progress updates
            output_path.mkdir(parents=True, exist_ok=True)
            frames_result = []
            total_time = 0

            for i, scene_obj in enumerate(scene_plan.scenes):
                frame_path = output_path / f"scene_{scene_obj.id:03d}.png"

                if skip_existing and frame_path.exists():
                    progress.update(task, advance=1, description=f"Skipped scene {scene_obj.id}")
                    continue

                progress.update(task, description=f"Scene {scene_obj.id}: {scene_obj.section_type}")

                frame = generator.generate_for_scene(
                    scene=scene_obj,
                    style_guide=scene_plan.style_guide,
                    master_seed=scene_plan.master_seed,
                )

                frame.save(frame_path)
                frames_result.append(frame)
                total_time += frame.generation_time_ms

                progress.update(task, advance=1)

        # Cleanup
        generator.unload()

        console.print(f"\n[green]Generated {len(frames_result)} frames[/green]")
        console.print(f"[dim]Total time: {total_time/1000:.1f}s[/dim]")
        console.print(f"[dim]Output: {output_path}[/dim]")

    except ImageGenerationError as e:
        console.print(f"[red]Generation failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def clips(
    plan_file: str = typer.Argument(..., help="Path to scene plan JSON file"),
    frames_dir: str = typer.Option(None, "--frames", "-f", help="Directory with reference frames"),
    output_dir: str = typer.Option("./clips", "--output", "-o", help="Output directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Video model (ltx-video, wan-i2v)"),
    quality: str = typer.Option("standard", "--quality", "-q", help="Quality preset (draft, standard, high)"),
    workflow: str = typer.Option("img2vid", "--workflow", "-w", help="Workflow (img2vid, txt2vid)"),
    scene: Optional[int] = typer.Option(None, "--scene", "-s", help="Generate only specific scene"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--regenerate", help="Skip existing clips"),
):
    """Generate video clips for a scene plan."""
    from .video import (
        VideoGenerator,
        VideoConfig,
        VideoModel,
        VideoQuality,
        VideoWorkflow,
        VideoGenerationError,
        generate_scene_clips,
    )
    from .planning.models import ScenePlan

    plan_path = Path(plan_file)
    if not plan_path.exists():
        console.print(f"[red]Plan file not found:[/red] {plan_file}")
        raise typer.Exit(1)

    # Load scene plan
    console.print(f"[cyan]Loading scene plan:[/cyan] {plan_file}")
    with open(plan_path) as f:
        plan_data = json.load(f)
    scene_plan = ScenePlan.from_dict(plan_data)

    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse quality
    try:
        video_quality = VideoQuality(quality.lower())
    except ValueError:
        console.print(f"[yellow]Unknown quality '{quality}', using standard[/yellow]")
        video_quality = VideoQuality.STANDARD

    # Parse workflow
    try:
        video_workflow = VideoWorkflow(workflow.lower())
    except ValueError:
        console.print(f"[yellow]Unknown workflow '{workflow}', using img2vid[/yellow]")
        video_workflow = VideoWorkflow.IMG2VID

    # Create config
    config = VideoConfig.for_quality(video_quality)
    config.workflow = video_workflow

    # Parse model if specified
    if model:
        try:
            config.model = VideoModel(model.lower())
        except ValueError:
            console.print(f"[yellow]Unknown model '{model}', using auto-selection[/yellow]")

    # Load reference frames if provided
    reference_frames = {}
    if frames_dir:
        frames_path = Path(frames_dir)
        if frames_path.exists():
            for frame_file in frames_path.glob("scene_*.png"):
                try:
                    scene_id = int(frame_file.stem.split("_")[1])
                    reference_frames[scene_id] = frame_file
                except (ValueError, IndexError):
                    pass
            console.print(f"[cyan]Loaded {len(reference_frames)} reference frames[/cyan]")

    # Filter scenes if specific scene requested
    scenes = scene_plan.scenes
    if scene is not None:
        scenes = [s for s in scenes if s.id == scene]
        if not scenes:
            console.print(f"[red]Scene {scene} not found in plan[/red]")
            raise typer.Exit(1)

    console.print(f"[cyan]Model:[/cyan] {config.model.value}")
    console.print(f"[cyan]Quality:[/cyan] {config.quality.value} ({config.width}x{config.height})")
    console.print(f"[cyan]Workflow:[/cyan] {config.workflow.value}")
    console.print(f"[cyan]Scenes to generate:[/cyan] {len(scenes)}")
    console.print()

    try:
        generator = VideoGenerator(config=config)

        for s in scenes:
            console.print(f"[yellow]Scene {s.id}:[/yellow] {s.description[:50]}...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Generating clip for scene {s.id}...", total=None)

                ref_frame = reference_frames.get(s.id)
                clip = generator.generate_for_scene(
                    scene=s,
                    style_guide=scene_plan.style_guide,
                    master_seed=scene_plan.master_seed,
                    reference_image=ref_frame,
                )

                progress.remove_task(task)

            # Save clip
            clip_path = output_path / f"scene_{s.id:03d}.mp4"
            clip.save(clip_path)
            console.print(f"[green]Saved:[/green] {clip_path} ({clip.duration_seconds:.1f}s, {clip.generation_time_ms/1000:.1f}s gen)")
            console.print()

        console.print(f"[bold green]Generated {len(scenes)} video clips![/bold green]")
        console.print(f"[dim]Output directory: {output_path}[/dim]")

    except VideoGenerationError as e:
        console.print(f"[red]Video generation failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def compose(
    plan_file: str = typer.Argument(..., help="Path to scene plan JSON file"),
    clips_dir: str = typer.Argument(..., help="Directory containing video clips"),
    audio_file: str = typer.Argument(..., help="Path to original audio file"),
    output: str = typer.Option("output.mp4", "--output", "-o", help="Output video path"),
    quality: str = typer.Option("standard", "--quality", "-q", help="Quality preset (draft, standard, high)"),
    transition_duration: int = typer.Option(500, "--transition-duration", "-t", help="Transition duration in ms"),
):
    """Compose video clips into final music video with audio."""
    from .video.compose import (
        VideoComposer,
        CompositionConfig,
        CompositionError,
        COMPOSITION_QUALITY,
    )
    from .planning.models import ScenePlan

    plan_path = Path(plan_file)
    clips_path = Path(clips_dir)
    audio_path = Path(audio_file)
    output_path = Path(output)

    # Validate inputs
    if not plan_path.exists():
        console.print(f"[red]Error:[/red] Plan file not found: {plan_file}")
        raise typer.Exit(1)

    if not clips_path.exists():
        console.print(f"[red]Error:[/red] Clips directory not found: {clips_dir}")
        raise typer.Exit(1)

    if not audio_path.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio_file}")
        raise typer.Exit(1)

    # Load scene plan
    try:
        plan_data = json.loads(plan_path.read_text())
        scene_plan = ScenePlan.from_dict(plan_data)
        console.print(f"[cyan]Scene Plan:[/cyan] {scene_plan.song_title}")
        console.print(f"[dim]Scenes: {scene_plan.scene_count}[/dim]")
    except Exception as e:
        console.print(f"[red]Error loading scene plan:[/red] {e}")
        raise typer.Exit(1)

    # Get quality preset
    config = COMPOSITION_QUALITY.get(quality.lower(), CompositionConfig())
    config.transition_duration_ms = transition_duration

    console.print(f"[cyan]Quality:[/cyan] {quality}")
    console.print(f"[cyan]CRF:[/cyan] {config.crf}")
    console.print(f"[cyan]Transition Duration:[/cyan] {config.transition_duration_ms}ms")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print()

    # Compose video
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Composing video...", total=None)

            composer = VideoComposer(config=config)
            result = composer.compose(
                scene_plan=scene_plan,
                clips_dir=clips_path,
                audio_path=audio_path,
                output_path=output_path,
            )

        console.print(f"\n[bold green]Video composed successfully![/bold green]")
        console.print(f"[cyan]Output:[/cyan] {result.output_path}")
        console.print(f"[cyan]Duration:[/cyan] {result.duration_seconds:.1f}s")
        console.print(f"[cyan]Clips:[/cyan] {result.clip_count}")
        console.print(f"[cyan]File Size:[/cyan] {result.file_size_bytes / 1024 / 1024:.1f} MB")
        console.print(f"[cyan]Composition Time:[/cyan] {result.composition_time_ms / 1000:.1f}s")

        if result.transitions_applied:
            transitions_str = ", ".join(f"{k}: {v}" for k, v in result.transitions_applied.items())
            console.print(f"[cyan]Transitions:[/cyan] {transitions_str}")

    except CompositionError as e:
        console.print(f"[red]Composition failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def preview(
    plan_file: str = typer.Argument(..., help="Path to scene plan JSON file"),
    frames_dir: Optional[str] = typer.Option(None, "--frames", "-f", help="Directory with reference frames"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output HTML file (default: storyboard.html)"),
):
    """Generate a storyboard preview document for review before video generation."""
    from .planning.models import ScenePlan
    from .planning.storyboard import generate_storyboard_html

    plan_path = Path(plan_file)

    # Validate plan file
    if not plan_path.exists():
        console.print(f"[red]Error:[/red] Plan file not found: {plan_file}")
        raise typer.Exit(1)

    # Load scene plan
    try:
        plan_data = json.loads(plan_path.read_text())
        scene_plan = ScenePlan.from_dict(plan_data)
        console.print(f"[cyan]Scene Plan:[/cyan] {scene_plan.song_title}")
        console.print(f"[dim]Scenes: {scene_plan.scene_count}, Seed: {scene_plan.master_seed}[/dim]")
    except Exception as e:
        console.print(f"[red]Error loading scene plan:[/red] {e}")
        raise typer.Exit(1)

    # Set up frames directory
    frames_path = Path(frames_dir) if frames_dir else None
    if frames_path and not frames_path.exists():
        console.print(f"[yellow]Warning:[/yellow] Frames directory not found: {frames_dir}")
        frames_path = None

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = plan_path.parent / "storyboard.html"

    # Generate storyboard
    console.print(f"\n[cyan]Generating storyboard...[/cyan]")
    html_content = generate_storyboard_html(
        scene_plan=scene_plan,
        frames_dir=frames_path,
        output_path=output_path,
    )

    console.print(f"\n[bold green]Storyboard generated![/bold green]")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Scenes:[/cyan] {scene_plan.scene_count}")

    if frames_path:
        frame_count = len(list(frames_path.glob("scene_*.png")))
        console.print(f"[cyan]Frames embedded:[/cyan] {frame_count}")

    console.print(f"\n[dim]Open in browser to review before video generation[/dim]")


@app.command()
def generate(
    input_file: str = typer.Argument(..., help="Path to MP3 file"),
    output: str = typer.Option("output.mp4", "--output", "-o", help="Output video path"),
    lyrics: Optional[str] = typer.Option(None, "--lyrics", "-l", help="Manual lyrics file"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Master visual seed"),
    quality: str = typer.Option("standard", "--quality", "-q", help="Quality preset"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--regenerate", help="Skip existing artifacts"),
    preview_only: bool = typer.Option(False, "--preview", "-p", help="Stop after frames and generate preview storyboard"),
):
    """Generate a music video from a song (end-to-end pipeline)."""
    from .audio import (
        load_audio,
        analyze_audio,
        extract_embedded_lyrics,
        parse_lyrics_file,
        transcribe_audio,
        get_recommended_model,
        AudioValidationError,
        LyricsParseError,
        TranscriptionError,
    )
    from .planning import create_scene_plan, get_ollama_client, PlanningError, generate_storyboard_html
    from .planning.models import ScenePlan
    from .image import FluxGenerator, ImageQuality, ImageGenerationError
    from .video import VideoGenerator, VideoConfig, VideoQuality, VideoGenerationError
    from .video.compose import VideoComposer, CompositionConfig, CompositionError, COMPOSITION_QUALITY

    audio_path = Path(input_file)
    output_path = Path(output)

    # Create work directory
    work_dir = output_path.parent / f".{output_path.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    plan_path = work_dir / "scene_plan.json"
    frames_dir = work_dir / "frames"
    clips_dir = work_dir / "clips"

    console.print(Panel(f"[bold]Generating Music Video[/bold]\n{audio_path.name} -> {output_path.name}", style="blue"))

    # Validate audio file
    try:
        audio_info = load_audio(audio_path)
        console.print(f"[cyan]Audio:[/cyan] {audio_info}")
    except AudioValidationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Check Ollama
    client = get_ollama_client()
    if not client.is_available():
        console.print("[red]Error:[/red] Ollama is not running. Start it with: ollama serve")
        raise typer.Exit(1)

    # STEP 1: Get lyrics
    console.print("\n[bold cyan]Step 1/6:[/bold cyan] Extracting lyrics")
    result_lyrics = None

    if lyrics:
        try:
            result_lyrics = parse_lyrics_file(Path(lyrics))
            console.print(f"[green]Loaded manual lyrics[/green]")
        except LyricsParseError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if result_lyrics is None:
        embedded = extract_embedded_lyrics(audio_path)
        if embedded:
            result_lyrics = embedded
            console.print("[green]Found embedded lyrics[/green]")

    if result_lyrics is None:
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                progress.add_task("Transcribing audio...", total=None)
                result_lyrics = transcribe_audio(audio_path)
            console.print(f"[green]Transcription complete[/green] ({result_lyrics.confidence:.0%} confidence)")
        except TranscriptionError as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            raise typer.Exit(1)

    # STEP 2: Analyze audio
    console.print("\n[bold cyan]Step 2/6:[/bold cyan] Analyzing audio")
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Analyzing audio...", total=None)
            features = analyze_audio(audio_path)
        console.print(f"[green]Analysis complete[/green] ({features.tempo:.0f} BPM, {len(features.sections)} sections)")
    except AudioValidationError as e:
        console.print(f"[red]Analysis failed:[/red] {e}")
        raise typer.Exit(1)

    # STEP 3: Create scene plan
    console.print("\n[bold cyan]Step 3/6:[/bold cyan] Planning scenes")
    if skip_existing and plan_path.exists():
        plan_data = json.loads(plan_path.read_text())
        scene_plan = ScenePlan.from_dict(plan_data)
        console.print(f"[yellow]Using existing plan[/yellow] ({scene_plan.scene_count} scenes)")
    else:
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                progress.add_task("Planning with LLM...", total=None)
                scene_plan = create_scene_plan(
                    audio_path=audio_path,
                    lyrics=result_lyrics,
                    audio_features=features,
                    song_title=audio_path.stem,
                    user_seed=seed,
                )
            plan_path.write_text(json.dumps(scene_plan.to_dict(), indent=2))
            console.print(f"[green]Plan created[/green] ({scene_plan.scene_count} scenes)")
        except PlanningError as e:
            console.print(f"[red]Planning failed:[/red] {e}")
            raise typer.Exit(1)

    # STEP 4: Generate reference frames
    console.print("\n[bold cyan]Step 4/6:[/bold cyan] Generating reference frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(frames_dir.glob("scene_*.png"))
    if skip_existing and len(existing_frames) >= len(scene_plan.scenes):
        console.print(f"[yellow]Using existing frames[/yellow] ({len(existing_frames)} frames)")
    else:
        try:
            quality_map = {"draft": ImageQuality.DRAFT, "standard": ImageQuality.STANDARD, "high": ImageQuality.HIGH}
            image_quality = quality_map.get(quality.lower(), ImageQuality.STANDARD)
            generator = FluxGenerator()

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Generating frames...", total=len(scene_plan.scenes))

                for scene_obj in scene_plan.scenes:
                    frame_path = frames_dir / f"scene_{scene_obj.id:03d}.png"
                    if skip_existing and frame_path.exists():
                        progress.update(task, advance=1)
                        continue

                    frame = generator.generate_for_scene(
                        scene=scene_obj,
                        style_guide=scene_plan.style_guide,
                        master_seed=scene_plan.master_seed,
                    )
                    frame.save(frame_path)
                    progress.update(task, advance=1)

            generator.unload()
            console.print(f"[green]Frames generated[/green]")
        except ImageGenerationError as e:
            console.print(f"[red]Frame generation failed:[/red] {e}")
            raise typer.Exit(1)

    # Generate storyboard preview
    storyboard_path = work_dir / "storyboard.html"
    console.print(f"\n[cyan]Generating storyboard preview...[/cyan]")
    generate_storyboard_html(
        scene_plan=scene_plan,
        frames_dir=frames_dir,
        output_path=storyboard_path,
    )
    console.print(f"[green]Storyboard saved:[/green] {storyboard_path}")

    if preview_only:
        console.print(f"\n[bold yellow]Preview mode - stopping before video generation[/bold yellow]")
        console.print(f"[dim]Review the storyboard, then run again without --preview to complete[/dim]")
        console.print(f"\n[cyan]Work directory:[/cyan] {work_dir}")
        console.print(f"[cyan]Scene plan:[/cyan] {plan_path}")
        console.print(f"[cyan]Frames:[/cyan] {frames_dir}")
        console.print(f"[cyan]Storyboard:[/cyan] {storyboard_path}")
        raise typer.Exit(0)

    # STEP 5: Generate video clips
    console.print("\n[bold cyan]Step 5/6:[/bold cyan] Generating video clips")
    clips_dir.mkdir(parents=True, exist_ok=True)

    existing_clips = list(clips_dir.glob("scene_*.mp4"))
    if skip_existing and len(existing_clips) >= len(scene_plan.scenes):
        console.print(f"[yellow]Using existing clips[/yellow] ({len(existing_clips)} clips)")
    else:
        try:
            quality_map = {"draft": VideoQuality.DRAFT, "standard": VideoQuality.STANDARD, "high": VideoQuality.HIGH}
            video_quality = quality_map.get(quality.lower(), VideoQuality.STANDARD)
            config = VideoConfig.for_quality(video_quality)
            video_gen = VideoGenerator(config=config)

            # Load reference frames
            reference_frames = {}
            for frame_file in frames_dir.glob("scene_*.png"):
                try:
                    scene_id = int(frame_file.stem.split("_")[1])
                    reference_frames[scene_id] = frame_file
                except (ValueError, IndexError):
                    pass

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Generating clips...", total=len(scene_plan.scenes))

                for scene_obj in scene_plan.scenes:
                    clip_path = clips_dir / f"scene_{scene_obj.id:03d}.mp4"
                    if skip_existing and clip_path.exists():
                        progress.update(task, advance=1)
                        continue

                    ref_frame = reference_frames.get(scene_obj.id)
                    clip = video_gen.generate_for_scene(
                        scene=scene_obj,
                        style_guide=scene_plan.style_guide,
                        master_seed=scene_plan.master_seed,
                        reference_image=ref_frame,
                    )
                    clip.save(clip_path)
                    progress.update(task, advance=1)

            console.print(f"[green]Clips generated[/green]")
        except VideoGenerationError as e:
            console.print(f"[red]Clip generation failed:[/red] {e}")
            raise typer.Exit(1)

    # STEP 6: Compose final video
    console.print("\n[bold cyan]Step 6/6:[/bold cyan] Composing final video")
    try:
        comp_config = COMPOSITION_QUALITY.get(quality.lower(), CompositionConfig())
        composer = VideoComposer(config=comp_config)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            progress.add_task("Composing video...", total=None)
            result = composer.compose(
                scene_plan=scene_plan,
                clips_dir=clips_dir,
                audio_path=audio_path,
                output_path=output_path,
            )

        console.print(f"\n[bold green]Music video generated successfully![/bold green]")
        console.print(f"[cyan]Output:[/cyan] {result.output_path}")
        console.print(f"[cyan]Duration:[/cyan] {result.duration_seconds:.1f}s")
        console.print(f"[cyan]File Size:[/cyan] {result.file_size_bytes / 1024 / 1024:.1f} MB")

    except CompositionError as e:
        console.print(f"[red]Composition failed:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
