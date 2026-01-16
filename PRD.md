# Song-to-Video: Product Requirements Document

**Version:** 1.1
**Last Updated:** 2026-01-15
**Author:** Ian Bergman

---

## 1. Vision

Create a fully local, AI-powered application that transforms any MP3 song into a cohesive, narrative-driven animated music video. The system analyzes lyrics, musical mood, and song structure to generate visually consistent video clips that tell the song's story.

---

## 2. Problem Statement

Creating music videos is expensive, time-consuming, and requires specialized skills. Independent artists, hobbyists, and content creators lack accessible tools to generate compelling visual content for their music. Existing solutions either:
- Require cloud processing (privacy concerns, ongoing costs)
- Produce only abstract visualizers without narrative coherence
- Lack visual consistency across scenes
- Require significant manual intervention

---

## 3. Target Users

### Primary (v1)
- **Independent musicians** wanting promotional content
- **Content creators** needing lyric visualizations
- **Hobbyists** exploring AI-generated media

### Secondary (Future)
- Music labels seeking rapid prototyping
- Podcast producers wanting visual content
- Educators creating multimedia materials

---

## 4. Core Requirements

### 4.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Accept MP3 audio file as input | Must Have |
| FR-2 | Transcribe lyrics using local AI (Whisper) | Must Have |
| FR-2a | Accept manual lyrics file override (SRT/LRC/TXT) | Must Have |
| FR-3 | Extract embedded lyrics metadata if available | Should Have |
| FR-4 | Analyze song structure (verses, chorus, bridge, etc.) | Must Have |
| FR-5 | Detect musical mood, tempo, and energy levels | Must Have |
| FR-6 | Generate scene descriptions based on lyric narrative | Must Have |
| FR-7 | Create visual style guide for consistency | Must Have |
| FR-8 | Generate 5-20 second video clips per scene | Must Have |
| FR-9 | Maintain visual consistency across all clips | Must Have |
| FR-10 | Stitch clips into final video synced to audio | Must Have |
| FR-11 | Output final video in standard formats (MP4/WebM) | Must Have |
| FR-12 | Generate mood-based visuals for instrumental sections | Must Have |

### 4.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Output resolution | 1080p (1920x1080) |
| NFR-2 | Output framerate | 24-30 fps |
| NFR-3 | Entirely local processing | No internet required after setup |
| NFR-4 | GPU memory usage | Optimize for 8-24GB VRAM (see Hardware Tiers) |
| NFR-5 | Hardware compatibility | NVIDIA RTX 4000/5000 series |
| NFR-9 | Memory management | Mandatory VRAM flush between pipeline phases |
| NFR-6 | Operating environment | WSL2 Ubuntu on Windows |
| NFR-7 | Maximum song length | 6 minutes |
| NFR-8 | Language support | English lyrics only (v1) |

---

## 5. Technical Architecture (Proposed)

### 5.1 High-Level Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   INPUT     │    │   ANALYSIS       │    │   PLANNING      │
│   MP3 File  │───▶│  - Transcription │───▶│  - Story arcs   │
│             │    │  - Mood analysis │    │  - Scene breaks │
│             │    │  - Structure     │    │  - Style guide  │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OUTPUT    │    │   ASSEMBLY       │    │   GENERATION    │
│   MP4 Video │◀───│  - Clip stitching│◀───│  - Video clips  │
│             │    │  - Audio sync    │    │  - Transitions  │
│             │    │  - Encoding      │    │  - Consistency  │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

### 5.2 Component Stack (Preliminary)

| Component | Candidate Technologies | Notes |
|-----------|----------------------|-------|
| **Audio Transcription** | OpenAI Whisper, WhisperX | WhisperX provides word-level timestamps |
| **Lyrics Extraction** | mutagen, tinytag, eyeD3 | For embedded ID3 lyrics |
| **Audio Analysis** | librosa, essentia | Tempo, energy, structure detection |
| **LLM Orchestration** | Ollama + Llama 3.1 14B / Mistral-Nemo 12B | Local narrative understanding (13-14B class) |
| **Image Generation** | Flux (default), SDXL | Reference frame generation for consistency |
| **Video Generation** | LTX-2, Wan 2.2, AnimateDiff | See Section 6 for comparison |
| **Video Processing** | FFmpeg | Stitching, encoding, sync |
| **Workflow Engine** | ComfyUI or custom Python | Node-based or scripted pipeline |

### 5.3 Hardware Tiers

The application auto-detects available VRAM and selects appropriate model configurations:

| Tier | Target Hardware | VRAM | Default Configuration |
|------|-----------------|------|----------------------|
| **Low-End** | RTX 4060 Mobile, RTX 4060 | 8GB | Wan 2.2 (5B) or LTX-2 (Distilled/GGUF) @ 480p, aggressive RAM offloading |
| **Mid-Range** | RTX 4070, RTX 4080 | 12-16GB | LTX-2 (quantized) @ 720p, moderate offloading |
| **High-End** | RTX 4090, RTX 5090 | 24GB+ | LTX-2 (19B) or HunyuanVideo 1.5 @ 1080p/4K, native processing |

**Memory Management Strategy:**
- **Mandatory VRAM Flush:** Each pipeline component (Whisper → LLM → Flux → Video Model) must fully unload before the next starts
- **System RAM Spillover:** On 8GB cards, use 64GB+ system RAM for model layer offloading
- **Phase Isolation:** No two large models may be loaded simultaneously

### 5.4 LLM Configuration by Hardware

| Hardware Tier | LLM Model | Quantization | Memory Strategy |
|---------------|-----------|--------------|-----------------|
| **Low-End (8GB)** | Llama 3.1 14B / Gemma 3 12B | Q4_K_M | Layer spilling to 64GB system RAM |
| **Mid-Range (12-16GB)** | Llama 3.1 14B | Q8_0 | Partial VRAM, partial RAM |
| **High-End (24GB+)** | Llama 3.1 14B | FP16 | Full VRAM |

---

## 6. Video Generation Technology Assessment

Based on current (2026) state of the art for local video generation:

### 6.1 Model Comparison

| Model | Max Length | Resolution | VRAM Required | Strengths |
|-------|-----------|------------|---------------|-----------|
| **LTX-2** | 20 sec | Up to 4K | 12-16GB (quantized) | Longest clips, built-in audio support |
| **Wan 2.2** | ~10 sec | 720p @ 24fps | 8-16GB | Excellent stylization control, efficient |
| **AnimateDiff-Lightning** | ~4 sec | 512-768px | 8GB | Fast iteration, LoRA compatible |
| **CogVideoX-5B** | ~6 sec | 720p | 12GB | Good text understanding |
| **HunyuanVideo 1.5** | ~8 sec | 720p | 14GB (with offload) | Strong motion fidelity |

### 6.2 Recommended Approach

**Primary:** LTX-2 for main clip generation (best length/quality balance)
**Fallback:** Wan 2.2 for lower VRAM systems or style-specific needs
**Rapid Prototyping:** AnimateDiff-Lightning for quick iterations

### 6.3 Generation Workflow Options

| Workflow | Description | Use Case |
|----------|-------------|----------|
| **Image-to-Video (Default)** | Generate Flux reference images → animate with LTX-2/Wan | Best consistency |
| **Text-to-Video Direct** | Generate video directly from prompts | Faster, less consistent |

Users can toggle between workflows via CLI flag `--workflow <img2vid|txt2vid>`

### 6.4 Consistency Strategy

Maintaining visual consistency across clips is critical. Approaches:
1. **Master Visual Seed** - Generate a global seed value in Phase 3 used across all image and video generation
2. **Style Guide LoRA** - Create a lightweight style adapter during visual planning for consistent aesthetics
3. **Reference image conditioning** - Generate a "style frame" and use image-to-video
4. **Shared embeddings** - Use consistent character/style embeddings across generations
5. **Prompt engineering** - Detailed, consistent style descriptions with shared style tokens

**Global Seeding Implementation:**
- A deterministic "Master Seed" is generated from song audio fingerprint + style hash
- All Flux image generations use this seed with scene-specific offsets
- All video generations inherit the seed for motion consistency
- Users can override with `--seed <value>` for reproducibility

---

## 7. Processing Pipeline (Detailed)

### Phase 1: Audio Ingestion & Analysis
1. Load MP3 file
2. Check for embedded lyrics (ID3 tags)
3. If no embedded lyrics, run Whisper transcription
4. Detect if song is instrumental (no vocals detected)
   - If instrumental: Skip to Phase 1.5 (audio-only analysis)
   - If vocals present: Extract word-level timestamps (WhisperX)
5. Analyze audio features:
   - Tempo (BPM)
   - Energy curve over time
   - Beat detection
   - Section boundaries (verse/chorus/bridge)

### Phase 1.5: Instrumental Mode (when no lyrics detected)
1. Analyze musical characteristics:
   - Genre classification
   - Mood/energy progression
   - Key musical moments (drops, builds, breakdowns)
2. Generate abstract/mood-based scene concepts
3. Skip to Phase 3 (Visual Planning)

### Phase 2: Narrative Understanding (lyrics present)
1. Send lyrics + timestamps to local LLM
2. Identify:
   - Overall theme/mood
   - Narrative arc
   - Key imagery and metaphors
   - Emotional progression
3. Map sections to visual concepts
4. Generate scene breakdown with timing

### Phase 3: Visual Planning
1. Determine overall visual style based on:
   - Lyrical content
   - Musical genre
   - Mood analysis
2. Create style guide:
   - Color palette
   - Visual aesthetic (e.g., "cinematic noir", "watercolor dreamscape")
   - Character descriptions (if applicable)
   - Environment themes
3. Generate reference images for consistency
4. Create detailed prompts for each scene

### Phase 4: Video Generation
1. For each scene:
   - Generate video clip using text-to-video or image-to-video
   - Apply consistency mechanisms
   - Validate output quality
2. Handle transitions between scenes:
   - **Default:** Simple cuts and cross-dissolves (FFmpeg)
   - **Optional:** AI-generated morphing transitions (slower, higher quality)
3. Generate instrumental section visuals based on:
   - Musical energy
   - Surrounding scene context
   - Mood interpolation

### Phase 5: Assembly & Output
1. Sequence all clips
2. Add transition effects
3. Sync to original audio
4. Encode to final format (MP4 H.264/H.265)
5. Optional: Generate thumbnail

---

## 8. User Experience (v1 - CLI)

### 8.1 Basic Usage
```bash
song-to-video generate song.mp3 --output video.mp4
```

### 8.2 Optional Flags
```bash
--style <style>         # Override AI-determined style
--resolution <res>      # 720p, 1080p (default), 4k
--quality <level>       # draft, standard, high
--preview               # Generate low-res preview first
--scenes-only           # Output scene plan without video
--verbose               # Detailed progress logging
--workflow <mode>       # img2vid (default) or txt2vid
--transitions <type>    # simple (default) or morph
--lyrics <file>         # Manual lyrics override (SRT/LRC/TXT)
--seed <value>          # Override master visual seed for reproducibility
--hardware <tier>       # Force hardware tier (low/mid/high)
```

### 8.3 Progress Feedback
- Phase indicators (Analyzing → Planning → Generating → Assembling)
- Per-scene progress for generation phase
- Estimated progress percentage
- GPU memory usage monitoring

---

## 9. Future Enhancements (v2+)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Web UI** | Browser-based interface for ease of use | High |
| **Style Selection** | User chooses from preset visual styles | High |
| **Scene Editor** | Review and edit scene plan before generation | High |
| **Character Consistency** | Named characters that persist across scenes | Medium |
| **Custom Training** | Fine-tune on user's preferred aesthetic | Medium |
| **Batch Processing** | Process multiple songs | Medium |
| **Album Mode** | Consistent style across album tracks | Low |
| **Live Preview** | Real-time preview during generation | Low |

---

## 10. Technical Constraints & Risks

### 10.1 Known Constraints
- **Generation time:** Full video may require significant processing time
- **VRAM limits:** Larger models need 16-24GB VRAM
- **Visual consistency:** AI models struggle with perfect consistency
- **Lyric accuracy:** Transcription may have errors on unclear vocals

### 10.2 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Out of VRAM | Mandatory phase isolation, VRAM flush, hardware tier auto-detection |
| Poor transcription | Manual lyrics override (SRT/LRC/TXT) - now a Must Have |
| Inconsistent visuals | Master seed, style LoRA, reference image conditioning |
| Long generation times | Preview mode, hardware-appropriate quality defaults |
| Multi-model OOM | Phase 0 memory management enforces single-model loading |

---

## 11. Decisions & Constraints (v1)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language Support** | English only | Simplifies v1; Whisper supports 99 languages for future expansion |
| **Max Song Length** | 6 minutes | Balances capability with reasonable generation times |
| **Model Distribution** | Download on first run | Smaller initial install (~50MB); models (~10-50GB) fetched when needed |
| **Explicit Content** | Generate matching visuals | AI creates visuals matching song tone, including dark/edgy themes |
| **Failure Recovery** | Restart required (v1) | No checkpoint/resume in v1; simplifies implementation |
| **Lyrics Input** | Transcription + manual override | Accept SRT/LRC/TXT files; critical fallback when Whisper fails on processed vocals |
| **Generation Workflow** | Image-to-video (default) | Flux generates reference frames; user can switch to text-to-video |
| **Transitions** | Simple cuts (default) | Cross-dissolves via FFmpeg; AI morphs available as option |
| **Instrumental Songs** | Fully supported | Mood-based visuals from audio analysis when no lyrics detected |
| **LLM Size** | 13-14B class | Better narrative understanding; Llama 3.1 14B or Mistral-Nemo 12B |

---

## 12. Implementation Phases

### Phase 0: Memory Management Infrastructure
**Goal:** Establish VRAM/RAM management to prevent OOM errors across all hardware tiers

| Task | Description | Dependencies |
|------|-------------|--------------|
| 0.1 | Hardware detection (VRAM, system RAM, GPU model) | None |
| 0.2 | Hardware tier classification logic | 0.1 |
| 0.3 | VRAM flush utility (complete model unloading) | 0.1 |
| 0.4 | System RAM offloading infrastructure | 0.1 |
| 0.5 | Memory monitoring and alerts | 0.3 |
| 0.6 | Phase isolation enforcement (prevent concurrent model loading) | 0.3 |

**Deliverable:** Memory management module that enforces clean handoffs between pipeline phases

---

### Phase 1: Foundation & Audio Pipeline
**Goal:** Establish project structure and complete audio ingestion

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.1 | Project scaffolding (Python, pyproject.toml, folder structure) | None |
| 1.2 | CLI framework setup (Click or Typer) | 1.1 |
| 1.3 | Audio file loading and validation | 1.1 |
| 1.4 | Whisper integration for transcription | 1.3 |
| 1.5 | WhisperX integration for word-level timestamps | 1.4 |
| 1.6 | Embedded lyrics extraction (ID3 tags via mutagen) | 1.3 |
| 1.7 | Manual lyrics file parser (SRT/LRC/TXT formats) | 1.3 |
| 1.8 | Lyrics source priority logic (manual > embedded > transcription) | 1.4, 1.6, 1.7 |
| 1.9 | Model download manager (first-run downloads) | 1.1 |

**Deliverable:** CLI that accepts MP3 (+ optional lyrics file) and outputs timestamped lyrics

---

### Phase 2: Audio Analysis
**Goal:** Extract musical features for scene planning

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.1 | librosa integration | Phase 1 |
| 2.2 | Tempo (BPM) detection | 2.1 |
| 2.3 | Energy curve analysis over time | 2.1 |
| 2.4 | Beat detection and grid | 2.1 |
| 2.5 | Section boundary detection (verse/chorus/bridge) | 2.1, 2.3 |
| 2.6 | Instrumental vs. vocal detection | 2.1 |
| 2.7 | Genre/mood classification | 2.1 |

**Deliverable:** Audio analysis module that outputs structured song metadata (JSON)

---

### Phase 3: LLM Integration & Narrative Planning
**Goal:** Use local LLM to interpret lyrics and plan scenes

| Task | Description | Dependencies |
|------|-------------|--------------|
| 3.1 | Ollama integration and model management | Phase 1 |
| 3.2 | Narrative analysis prompt engineering | 3.1 |
| 3.3 | Theme/mood extraction from lyrics | 3.2, Phase 2 |
| 3.4 | Scene breakdown generation with timing | 3.3 |
| 3.5 | Visual style determination logic | 3.3 |
| 3.6 | Master Visual Seed generation (audio fingerprint + style hash) | 3.5 |
| 3.7 | Style Guide LoRA creation for consistency | 3.5 |
| 3.8 | Instrumental mode scene planning | 3.4, 2.6 |
| 3.9 | Scene plan output format (JSON schema) | 3.4, 3.6 |

**Deliverable:** `--scenes-only` flag works, outputting complete scene plan JSON with master seed

---

### Phase 4: Image Generation
**Goal:** Generate reference frames for visual consistency

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.1 | Flux model integration (with hardware tier configs) | Phase 0, Phase 1 |
| 4.2 | Style guide to image prompt translation | Phase 3 |
| 4.3 | Reference frame generation per scene | 4.1, 4.2 |
| 4.4 | Prompt templating system for consistency | 4.2 |
| 4.5 | Image quality validation | 4.3 |
| 4.6 | Master seed propagation (seed + scene offset) | 4.3, 3.6 |
| 4.7 | Style LoRA application | 4.3, 3.7 |
| 4.8 | VRAM flush before/after image generation | Phase 0 |

**Deliverable:** Generate consistent reference images for all scenes using master seed

---

### Phase 5: Video Generation
**Goal:** Animate reference frames into video clips

| Task | Description | Dependencies |
|------|-------------|--------------|
| 5.1 | LTX-2 model integration (with hardware tier configs) | Phase 0, Phase 1 |
| 5.2 | Wan 2.2 integration (low-end fallback) | Phase 0, Phase 1 |
| 5.3 | Image-to-video workflow implementation | 5.1, Phase 4 |
| 5.4 | Text-to-video workflow implementation | 5.1 |
| 5.5 | Workflow selection logic (`--workflow` flag) | 5.3, 5.4 |
| 5.6 | Clip duration management (match scene timing) | 5.3 |
| 5.7 | VRAM management and model offloading | 5.1, 5.2, Phase 0 |
| 5.8 | Quality tier implementation (draft/standard/high) | 5.1 |
| 5.9 | Master seed propagation for motion consistency | 5.3, 3.6 |
| 5.10 | VRAM flush before/after video generation | Phase 0 |

**Deliverable:** Generate individual video clips for each scene with consistent motion

---

### Phase 6: Assembly & Output
**Goal:** Stitch clips into final synchronized video

| Task | Description | Dependencies |
|------|-------------|--------------|
| 6.1 | FFmpeg integration | Phase 1 |
| 6.2 | Clip sequencing by timestamp | 6.1, Phase 5 |
| 6.3 | Simple transitions (cuts, cross-dissolves) | 6.2 |
| 6.4 | AI morph transitions (optional) | 6.2, 5.1 |
| 6.5 | Transition selection logic (`--transitions` flag) | 6.3, 6.4 |
| 6.6 | Audio track synchronization | 6.2 |
| 6.7 | Final encoding (H.264/H.265) | 6.6 |
| 6.8 | Resolution scaling (`--resolution` flag) | 6.7 |
| 6.9 | Thumbnail generation | 6.7 |

**Deliverable:** Complete video output with synced audio

---

### Phase 7: CLI Polish & Error Handling
**Goal:** Production-ready CLI experience

| Task | Description | Dependencies |
|------|-------------|--------------|
| 7.1 | Full CLI flag implementation | All phases |
| 7.2 | Progress reporting (phase indicators, percentages) | 7.1 |
| 7.3 | GPU memory monitoring and display | 7.1 |
| 7.4 | Error handling and user-friendly messages | 7.1 |
| 7.5 | Input validation (file format, duration limits) | 7.1 |
| 7.6 | Preview mode implementation (`--preview`) | Phase 5, 6 |
| 7.7 | Verbose logging (`--verbose`) | 7.1 |
| 7.8 | Graceful cancellation (Ctrl+C handling) | 7.1 |

**Deliverable:** Polished CLI with all documented flags working

---

### Phase 8: Testing & Documentation
**Goal:** Ensure reliability and ease of setup

| Task | Description | Dependencies |
|------|-------------|--------------|
| 8.1 | Unit tests for each module | All phases |
| 8.2 | Integration tests (end-to-end pipeline) | 8.1 |
| 8.3 | Hardware compatibility testing (various VRAM configs) | 8.2 |
| 8.4 | Installation documentation | 8.2 |
| 8.5 | Usage guide with examples | 8.4 |
| 8.6 | Troubleshooting guide | 8.3 |
| 8.7 | Performance benchmarks on reference hardware | 8.3 |

**Deliverable:** v1.0 release-ready with documentation

---

### Implementation Order Summary

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8
  │           │           │           │           │           │           │
  │           │           │           │           │           │           └─► Final video output
  │           │           │           │           │           └─► Video clips generated
  │           │           │           │           └─► Reference images ready
  │           │           │           └─► Scene plan + master seed
  │           │           └─► Audio features extracted
  │           └─► Lyrics transcribed (or loaded from file)
  └─► Memory management infrastructure
```

### Milestone Checkpoints

| Milestone | After Phase | Verification |
|-----------|-------------|--------------|
| **M0: Memory Safe** | Phase 0 | VRAM flush works, hardware tiers detected |
| **M1: Audio Works** | Phase 2 | Can transcribe/load lyrics and analyze any MP3 |
| **M2: Planning Works** | Phase 3 | `--scenes-only` produces scene plan with master seed |
| **M3: Images Work** | Phase 4 | Reference frames generated with seed consistency |
| **M4: Video Works** | Phase 5 | Individual clips generated across hardware tiers |
| **M5: End-to-End** | Phase 6 | Complete video output for test song |
| **M6: Release Ready** | Phase 8 | All tests pass on low/mid/high hardware |

---

## 13. Success Metrics (v1)

| Metric | Target |
|--------|--------|
| Successful video generation rate | >90% of attempts |
| Visual-lyric alignment | Subjective user satisfaction |
| Processing time (3-min song) | TBD based on hardware |
| VRAM peak usage | <20GB for standard quality |

---

## 14. References & Resources

### Video Generation
- [NVIDIA RTX AI Video Generation Announcements (CES 2026)](https://blogs.nvidia.com/blog/rtx-ai-garage-ces-2026-open-models-video-generation/)
- [LTX-2 Technical Overview](https://www.ltx2.tech/)
- [Best Open Source Video Generation Models 2026](https://www.hyperstack.cloud/blog/case-study/best-open-source-video-generation-models)
- [Hugging Face Video Generation State](https://huggingface.co/blog/video_gen)

### Audio Processing
- [Whisper GPU Benchmarks](https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked)
- [Faster Whisper Transcription Techniques](https://modal.com/blog/faster-transcription)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **LoRA** | Low-Rank Adaptation - lightweight model fine-tuning technique |
| **VRAM** | Video RAM - GPU memory |
| **DiT** | Diffusion Transformer - architecture used by newer video models |
| **ComfyUI** | Node-based workflow tool for AI image/video generation |
| **Whisper** | OpenAI's open-source speech recognition model |
| **Flux** | Black Forest Labs' open-source image generation model |
| **Image-to-Video (I2V)** | Generating video by animating a static reference image |
| **Text-to-Video (T2V)** | Generating video directly from text prompts |
| **Master Visual Seed** | A deterministic seed value derived from audio fingerprint + style, ensuring reproducible and consistent visual generation |
| **Q4_K_M** | A 4-bit quantization format for LLMs that balances quality and memory usage |
| **GGUF** | A file format for quantized models optimized for CPU/GPU inference |

---

*This is a living document. Updates will be tracked with version numbers.*
