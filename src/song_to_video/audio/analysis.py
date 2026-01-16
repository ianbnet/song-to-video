"""Audio analysis using librosa for feature extraction."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .models import (
    AudioFeatures,
    Beat,
    Section,
    SectionType,
    EnergyPoint,
    Mood,
    AudioValidationError,
)

logger = logging.getLogger(__name__)

# Default analysis parameters
DEFAULT_HOP_LENGTH = 512
DEFAULT_ENERGY_RESOLUTION = 0.1  # seconds between energy samples


class AudioAnalyzer:
    """
    Analyzes audio files to extract musical features.

    Uses librosa for tempo, beat, energy, and structural analysis.
    """

    def __init__(
        self,
        hop_length: int = DEFAULT_HOP_LENGTH,
        energy_resolution: float = DEFAULT_ENERGY_RESOLUTION,
    ):
        """
        Initialize the analyzer.

        Args:
            hop_length: Number of samples between analysis frames
            energy_resolution: Time resolution for energy curve (seconds)
        """
        self.hop_length = hop_length
        self.energy_resolution = energy_resolution

    def analyze(self, audio_path: str | Path) -> AudioFeatures:
        """
        Perform complete audio analysis.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioFeatures with extracted features

        Raises:
            AudioValidationError: If analysis fails
        """
        audio_path = Path(audio_path)

        try:
            import librosa
        except ImportError:
            raise AudioValidationError("librosa not installed. Run: pip install librosa")

        logger.info(f"Analyzing audio: {audio_path.name}")

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(y) / sr

            logger.debug(f"Loaded audio: {duration:.1f}s, {sr}Hz")

            # Extract features
            tempo, tempo_confidence, beats, downbeats = self._analyze_tempo_beats(y, sr)
            energy_curve, avg_energy, energy_var = self._analyze_energy(y, sr)
            sections = self._detect_sections(y, sr, beats, energy_curve)
            has_vocals, vocal_ratio = self._detect_vocals(y, sr)
            mood, mood_confidence = self._classify_mood(y, sr, tempo, avg_energy)
            key, key_confidence = self._detect_key(y, sr)
            genre_tags = self._classify_genre(y, sr, tempo)

            features = AudioFeatures(
                duration=duration,
                sample_rate=sr,
                tempo=tempo,
                tempo_confidence=tempo_confidence,
                beats=beats,
                downbeats=downbeats,
                energy_curve=energy_curve,
                average_energy=avg_energy,
                energy_variance=energy_var,
                sections=sections,
                has_vocals=has_vocals,
                vocal_ratio=vocal_ratio,
                mood=mood,
                mood_confidence=mood_confidence,
                genre_tags=genre_tags,
                key=key,
                key_confidence=key_confidence,
            )

            logger.info(
                f"Analysis complete: {tempo:.1f} BPM, {len(sections)} sections, "
                f"mood={mood.value}, vocals={'yes' if has_vocals else 'no'}"
            )

            return features

        except Exception as e:
            raise AudioValidationError(f"Audio analysis failed: {e}")

    def _analyze_tempo_beats(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, float, list[Beat], list[float]]:
        """Detect tempo and beat positions."""
        import librosa

        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Detect tempo
        tempo_estimates = librosa.feature.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            aggregate=None,
        )

        # Get primary tempo
        if len(tempo_estimates) > 0:
            tempo = float(np.median(tempo_estimates))
            # Confidence based on how consistent the estimates are
            tempo_std = float(np.std(tempo_estimates))
            tempo_confidence = max(0.0, min(1.0, 1.0 - tempo_std / 50.0))
        else:
            tempo = 120.0
            tempo_confidence = 0.5

        # Detect beats
        beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            bpm=tempo,
        )[1]

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

        # Create Beat objects with strength based on onset envelope
        beats = []
        for frame, time in zip(beat_frames, beat_times):
            if frame < len(onset_env):
                strength = float(onset_env[frame]) / (np.max(onset_env) + 1e-6)
            else:
                strength = 0.5
            beats.append(Beat(time=float(time), strength=min(1.0, strength)))

        # Estimate downbeats (first beat of each measure, assuming 4/4)
        downbeats = [beat_times[i] for i in range(0, len(beat_times), 4)]

        logger.debug(f"Detected {len(beats)} beats at {tempo:.1f} BPM")

        return tempo, tempo_confidence, beats, [float(d) for d in downbeats]

    def _analyze_energy(
        self, y: np.ndarray, sr: int
    ) -> tuple[list[EnergyPoint], float, float]:
        """Analyze energy curve over time."""
        import librosa

        # Compute RMS energy
        frame_length = int(self.energy_resolution * sr)
        hop_length = frame_length // 2

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        # Normalize to 0-1 range
        rms_max = np.max(rms) + 1e-6
        normalized = rms / rms_max

        # Create energy points
        energy_curve = [
            EnergyPoint(time=float(t), energy=float(e), rms=float(r))
            for t, e, r in zip(times, normalized, rms)
        ]

        avg_energy = float(np.mean(normalized))
        energy_var = float(np.var(normalized))

        logger.debug(f"Energy: avg={avg_energy:.2f}, variance={energy_var:.3f}")

        return energy_curve, avg_energy, energy_var

    def _detect_sections(
        self,
        y: np.ndarray,
        sr: int,
        beats: list[Beat],
        energy_curve: list[EnergyPoint],
    ) -> list[Section]:
        """Detect song sections (verse, chorus, etc.)."""
        import librosa

        duration = len(y) / sr

        # Use structural segmentation based on self-similarity matrix
        try:
            # Compute chromagram for harmonic content
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)

            # Compute self-similarity matrix
            rec = librosa.segment.recurrence_matrix(
                chroma,
                mode="affinity",
                metric="cosine",
                sparse=True,
            )

            # Find segment boundaries using agglomerative clustering
            # Use a reasonable number of clusters based on song duration
            num_sections = max(4, min(12, int(duration / 15)))  # ~15 sec per section
            bounds = librosa.segment.agglomerative(chroma, k=num_sections)
            bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=self.hop_length)

            # Ensure we have start and end
            if len(bound_times) == 0 or bound_times[0] > 0.5:
                bound_times = np.concatenate([[0.0], bound_times])
            if bound_times[-1] < duration - 0.5:
                bound_times = np.concatenate([bound_times, [duration]])

        except Exception as e:
            logger.warning(f"Segmentation failed, using fallback: {e}")
            # Fallback: simple energy-based segmentation
            bound_times = self._fallback_segmentation(energy_curve, duration)

        # Classify each section
        sections = []
        for i in range(len(bound_times) - 1):
            start = float(bound_times[i])
            end = float(bound_times[i + 1])

            # Skip very short sections
            if end - start < 2.0:
                continue

            # Classify section type based on position and energy
            section_type = self._classify_section(
                start, end, duration, energy_curve, i, len(bound_times) - 1
            )

            sections.append(
                Section(
                    start=start,
                    end=end,
                    type=section_type,
                    confidence=0.7,
                    label=self._generate_section_label(section_type, sections),
                )
            )

        logger.debug(f"Detected {len(sections)} sections")

        return sections

    def _fallback_segmentation(
        self, energy_curve: list[EnergyPoint], duration: float
    ) -> np.ndarray:
        """Fallback segmentation based on energy changes."""
        if not energy_curve:
            return np.array([0.0, duration])

        # Find significant energy changes
        energies = np.array([p.energy for p in energy_curve])
        times = np.array([p.time for p in energy_curve])

        # Smooth energy
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(energies, sigma=5)

        # Find peaks in the derivative (significant changes)
        diff = np.abs(np.diff(smoothed))
        threshold = np.percentile(diff, 90)

        change_indices = np.where(diff > threshold)[0]
        change_times = times[change_indices]

        # Add boundaries
        boundaries = np.concatenate([[0.0], change_times, [duration]])

        # Merge close boundaries
        merged = [boundaries[0]]
        for t in boundaries[1:]:
            if t - merged[-1] > 8.0:  # Minimum 8 seconds between sections
                merged.append(t)
        if merged[-1] < duration:
            merged.append(duration)

        return np.array(merged)

    def _classify_section(
        self,
        start: float,
        end: float,
        duration: float,
        energy_curve: list[EnergyPoint],
        section_idx: int,
        total_sections: int,
    ) -> SectionType:
        """Classify a section based on position and energy."""
        # Get average energy for this section
        section_energy = np.mean([
            p.energy for p in energy_curve
            if start <= p.time <= end
        ]) if energy_curve else 0.5

        # Position-based heuristics
        relative_start = start / duration
        relative_end = end / duration

        # First section is likely intro
        if section_idx == 0 and relative_end < 0.15:
            return SectionType.INTRO

        # Last section is likely outro
        if section_idx == total_sections - 1 and relative_start > 0.85:
            return SectionType.OUTRO

        # High energy sections near middle are likely chorus
        if section_energy > 0.6 and 0.2 < relative_start < 0.8:
            return SectionType.CHORUS

        # Low energy followed by high energy suggests buildup/drop
        if section_energy < 0.3:
            return SectionType.BREAKDOWN

        # Medium energy sections are likely verses
        if section_energy < 0.6:
            return SectionType.VERSE

        # Default to verse
        return SectionType.VERSE

    def _generate_section_label(
        self, section_type: SectionType, existing_sections: list[Section]
    ) -> str:
        """Generate a label like 'Verse 1', 'Chorus 2', etc."""
        # Count existing sections of this type
        count = sum(1 for s in existing_sections if s.type == section_type)
        count += 1

        type_name = section_type.value.title()

        # Only add number if there might be multiples
        if section_type in (SectionType.VERSE, SectionType.CHORUS, SectionType.INSTRUMENTAL):
            return f"{type_name} {count}"
        return type_name

    def _detect_vocals(self, y: np.ndarray, sr: int) -> tuple[bool, float]:
        """Detect presence of vocals in the audio."""
        import librosa

        # Use harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Vocals typically have strong harmonic content in the 300-3000Hz range
        # Compute spectral centroid
        cent = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=self.hop_length)[0]

        # Vocals tend to have centroid in 1000-2500Hz range
        vocal_range = (cent > 1000) & (cent < 2500)
        vocal_ratio = float(np.mean(vocal_range))

        # Also check for high harmonic-to-percussive ratio in vocal frequency range
        harmonic_energy = np.sum(y_harmonic ** 2)
        total_energy = np.sum(y ** 2) + 1e-6
        harmonic_ratio = harmonic_energy / total_energy

        # Combined heuristic
        has_vocals = vocal_ratio > 0.3 and harmonic_ratio > 0.4

        logger.debug(f"Vocal detection: ratio={vocal_ratio:.2f}, harmonic={harmonic_ratio:.2f}")

        return has_vocals, vocal_ratio

    def _classify_mood(
        self, y: np.ndarray, sr: int, tempo: float, avg_energy: float
    ) -> tuple[Mood, float]:
        """Classify the overall mood of the song."""
        import librosa

        # Use multiple features for mood classification

        # 1. Tempo-based mood
        if tempo > 140:
            tempo_mood = Mood.ENERGETIC
        elif tempo > 100:
            tempo_mood = Mood.UPLIFTING if avg_energy > 0.5 else Mood.NEUTRAL
        elif tempo > 70:
            tempo_mood = Mood.CALM if avg_energy < 0.4 else Mood.NEUTRAL
        else:
            tempo_mood = Mood.MELANCHOLIC if avg_energy < 0.3 else Mood.SAD

        # 2. Spectral characteristics
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=self.hop_length
        )
        mean_contrast = float(np.mean(spectral_contrast))

        # High contrast suggests more energetic/aggressive
        if mean_contrast > 30:
            contrast_mood = Mood.AGGRESSIVE
        elif mean_contrast > 20:
            contrast_mood = Mood.ENERGETIC
        else:
            contrast_mood = Mood.CALM

        # 3. Mode detection (major = happy, minor = sad)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        # Simple major/minor detection based on 3rd degree
        major_score = float(np.mean(chroma[4]))  # Major 3rd
        minor_score = float(np.mean(chroma[3]))  # Minor 3rd

        if major_score > minor_score * 1.2:
            key_mood = Mood.HAPPY if tempo > 100 else Mood.ROMANTIC
        elif minor_score > major_score * 1.2:
            key_mood = Mood.SAD if avg_energy < 0.4 else Mood.DARK
        else:
            key_mood = Mood.NEUTRAL

        # Combine moods with weighting
        mood_scores = {
            tempo_mood: 0.3,
            contrast_mood: 0.3,
            key_mood: 0.4,
        }

        # If multiple moods agree, increase confidence
        if tempo_mood == contrast_mood == key_mood:
            return tempo_mood, 0.9
        elif tempo_mood == key_mood or contrast_mood == key_mood:
            return key_mood, 0.7
        else:
            # Default to energy-based mood
            if avg_energy > 0.6:
                return Mood.ENERGETIC, 0.5
            elif avg_energy < 0.3:
                return Mood.CALM, 0.5
            else:
                return Mood.NEUTRAL, 0.4

    def _detect_key(self, y: np.ndarray, sr: int) -> tuple[str, float]:
        """Detect the musical key of the song."""
        import librosa

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        chroma_mean = np.mean(chroma, axis=1)

        # Key names
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Find the strongest pitch class
        key_idx = int(np.argmax(chroma_mean))
        key_strength = float(chroma_mean[key_idx])

        # Determine major or minor
        # Major: root, major 3rd (4 semitones), perfect 5th (7 semitones)
        # Minor: root, minor 3rd (3 semitones), perfect 5th (7 semitones)
        major_third = chroma_mean[(key_idx + 4) % 12]
        minor_third = chroma_mean[(key_idx + 3) % 12]
        fifth = chroma_mean[(key_idx + 7) % 12]

        if major_third > minor_third:
            mode = "major"
            mode_confidence = float(major_third / (minor_third + 1e-6))
        else:
            mode = "minor"
            mode_confidence = float(minor_third / (major_third + 1e-6))

        key = f"{key_names[key_idx]} {mode}"
        confidence = min(1.0, key_strength * min(2.0, mode_confidence) / 2)

        logger.debug(f"Key detection: {key} (confidence={confidence:.2f})")

        return key, confidence

    def _classify_genre(
        self, y: np.ndarray, sr: int, tempo: float
    ) -> list[str]:
        """Classify genre tags for the song."""
        import librosa

        tags = []

        # Tempo-based tags
        if tempo > 160:
            tags.append("fast")
        elif tempo > 120:
            tags.append("upbeat")
        elif tempo < 80:
            tags.append("slow")

        # Compute spectral features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
        mean_rolloff = float(np.mean(spectral_rolloff))

        # High frequency content suggests electronic/rock
        if mean_rolloff > 5000:
            tags.append("bright")
        elif mean_rolloff < 2000:
            tags.append("warm")

        # Percussive vs harmonic
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)

        if percussive_energy > harmonic_energy:
            tags.append("rhythmic")
        else:
            tags.append("melodic")

        # Zero crossing rate can indicate speech/vocals vs instruments
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))

        if mean_zcr > 0.1:
            tags.append("noisy")
        elif mean_zcr < 0.03:
            tags.append("clean")

        return tags


def analyze_audio(audio_path: str | Path) -> AudioFeatures:
    """
    Convenience function to analyze an audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        AudioFeatures with extracted features
    """
    analyzer = AudioAnalyzer()
    return analyzer.analyze(audio_path)
