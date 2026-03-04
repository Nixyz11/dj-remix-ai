"""Hybrid Remix Engine — combines stem manipulation with AI-generated elements."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional

from effects import AudioEffects
from ai_generator import AIGenerator


class RemixEngine:
    """Core remix engine that creates DJ-style remixes.

    Hybrid approach:
    1. Stems are separated and individually processed with FX
    2. AI generates new elements (drops, transitions, beats)
    3. Everything is mixed together into a final remix
    """

    STYLE_PRESETS = {
        "party": {
            "bpm_range": (124, 132),
            "vocal_fx": ["reverb", "echo"],
            "drum_boost": 1.3,
            "bass_boost": 1.2,
            "ai_elements": ["drop", "buildup", "beat"],
            "energy_curve": "build_and_drop",
        },
        "edm": {
            "bpm_range": (126, 140),
            "vocal_fx": ["reverb", "high_pass"],
            "drum_boost": 1.5,
            "bass_boost": 1.4,
            "ai_elements": ["drop", "buildup", "bass_line"],
            "energy_curve": "constant_high",
        },
        "chill": {
            "bpm_range": (90, 110),
            "vocal_fx": ["reverb", "echo"],
            "drum_boost": 0.8,
            "bass_boost": 1.0,
            "ai_elements": ["transition"],
            "energy_curve": "smooth",
        },
        "bass_heavy": {
            "bpm_range": (130, 150),
            "vocal_fx": ["high_pass"],
            "drum_boost": 1.4,
            "bass_boost": 1.8,
            "ai_elements": ["drop", "bass_line", "beat"],
            "energy_curve": "build_and_drop",
        },
    }

    def __init__(self):
        self.effects = AudioEffects()
        self.ai_gen = AIGenerator()
        self.sr = 22050  # Working sample rate

    def create_remix(
        self,
        stems: Dict[str, str],
        analysis: Dict,
        style: str = "party",
        bpm_target: Optional[int] = None,
        intensity: float = 0.7,
        output_path: str = "remix.wav",
    ) -> str:
        """Create a full remix from separated stems + AI generation.

        Args:
            stems: Dict of stem paths {"vocals": "...", "drums": "...", ...}
            analysis: Audio analysis dict from AudioAnalyzer
            style: Remix style preset name
            bpm_target: Target BPM (None = auto from preset)
            intensity: How heavy the remix is (0.0 - 1.0)
            output_path: Where to save the final remix

        Returns:
            Path to the output remix file
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["party"])

        # Load stems
        loaded_stems = {}
        for name, path in stems.items():
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            loaded_stems[name] = y

        original_bpm = analysis["bpm"]
        if bpm_target is None:
            bpm_low, bpm_high = preset["bpm_range"]
            bpm_target = max(bpm_low, min(original_bpm, bpm_high))

        # Step 1: Time-stretch stems to target BPM
        stretch_rate = bpm_target / original_bpm
        if abs(stretch_rate - 1.0) > 0.02:
            for name in loaded_stems:
                loaded_stems[name] = self.effects.time_stretch(loaded_stems[name], stretch_rate)

        # Ensure all stems have the same length
        min_len = min(len(s) for s in loaded_stems.values())
        for name in loaded_stems:
            loaded_stems[name] = loaded_stems[name][:min_len]

        # Step 2: Apply FX to individual stems
        processed_stems = self._process_stems(loaded_stems, preset, intensity)

        # Step 3: Generate AI elements
        ai_layers = self._generate_ai_layers(
            preset=preset,
            bpm=bpm_target,
            key=analysis.get("key_full", "C minor"),
            track_length=min_len,
            intensity=intensity,
        )

        # Step 4: Mix everything together
        remix = self._final_mix(processed_stems, ai_layers, preset, intensity)

        # Step 5: Apply master FX
        remix = self._master_chain(remix, preset, bpm_target)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, remix, self.sr)
        return output_path

    def _process_stems(self, stems: Dict[str, np.ndarray], preset: dict, intensity: float) -> Dict[str, np.ndarray]:
        """Apply FX to individual stems based on style preset."""
        processed = {}

        # Vocals
        if "vocals" in stems:
            vocals = stems["vocals"]
            for fx in preset.get("vocal_fx", []):
                if fx == "reverb":
                    vocals = self.effects.apply_reverb(vocals, decay=0.3 * intensity)
                elif fx == "echo":
                    vocals = self.effects.apply_echo(vocals, delay_ms=250, decay=0.3 * intensity)
                elif fx == "high_pass":
                    vocals = self.effects.high_pass_filter(vocals, cutoff_hz=300)
            processed["vocals"] = vocals

        # Drums — boost and add sidechain
        if "drums" in stems:
            drums = stems["drums"] * preset.get("drum_boost", 1.0)
            processed["drums"] = drums

        # Bass — boost
        if "bass" in stems:
            bass = stems["bass"] * preset.get("bass_boost", 1.0)
            bass = self.effects.low_pass_filter(bass, cutoff_hz=250)
            processed["bass"] = bass

        # Other (melody/instruments)
        if "other" in stems:
            other = stems["other"]
            if intensity > 0.5:
                other = self.effects.apply_reverb(other, decay=0.2)
            processed["other"] = other

        return processed

    def _generate_ai_layers(
        self, preset: dict, bpm: float, key: str,
        track_length: int, intensity: float
    ) -> Dict[str, np.ndarray]:
        """Generate AI elements and fit them to the track length."""
        ai_layers = {}

        if intensity < 0.3:
            return ai_layers  # Low intensity = no AI elements

        for element_type in preset.get("ai_elements", []):
            try:
                # Generate 8-second AI clip
                ai_audio = self.ai_gen.generate_remix_element(
                    element_type=element_type,
                    style=list(self.STYLE_PRESETS.keys())[
                        list(self.STYLE_PRESETS.values()).index(preset)
                    ],
                    bpm=bpm,
                    key=key,
                    duration_seconds=8,
                )

                # MusicGen outputs at 32kHz, resample to our SR
                ai_audio = librosa.resample(ai_audio, orig_sr=32000, target_sr=self.sr)

                ai_layers[element_type] = ai_audio
            except Exception as e:
                print(f"Warning: Failed to generate {element_type}: {e}")
                continue

        return ai_layers

    def _final_mix(
        self, stems: Dict[str, np.ndarray],
        ai_layers: Dict[str, np.ndarray],
        preset: dict, intensity: float
    ) -> np.ndarray:
        """Mix processed stems and AI layers into final audio."""
        # Get track length from stems
        track_length = max(len(s) for s in stems.values())

        # Start with stem mix
        mix = np.zeros(track_length, dtype=np.float64)

        # Stem mixing levels
        stem_levels = {
            "vocals": 0.9,
            "drums": 0.85,
            "bass": 0.75,
            "other": 0.6,
        }

        for name, audio in stems.items():
            level = stem_levels.get(name, 0.7)
            padded = np.zeros(track_length)
            padded[:len(audio)] = audio
            mix += padded * level

        # Layer AI elements at strategic positions
        if ai_layers:
            energy_curve = preset.get("energy_curve", "build_and_drop")
            positions = self._get_ai_placement_positions(track_length, energy_curve, len(ai_layers))

            ai_level = 0.3 * intensity
            for (element_type, audio), position in zip(ai_layers.items(), positions):
                start = int(position * track_length)
                end = min(start + len(audio), track_length)
                actual_len = end - start

                # Fade in/out the AI layer
                fade_len = min(int(0.5 * self.sr), actual_len // 4)
                ai_segment = audio[:actual_len] * ai_level

                if fade_len > 0 and actual_len > fade_len * 2:
                    ai_segment[:fade_len] *= np.linspace(0, 1, fade_len)
                    ai_segment[-fade_len:] *= np.linspace(1, 0, fade_len)

                mix[start:end] += ai_segment

        # Normalize
        max_val = np.max(np.abs(mix))
        if max_val > 0:
            mix = mix / max_val * 0.95

        return mix.astype(np.float32)

    def _get_ai_placement_positions(self, track_length: int, energy_curve: str, n_elements: int) -> list:
        """Determine where to place AI elements in the track."""
        if energy_curve == "build_and_drop":
            # Place elements at buildups and drops (40%, 60%, 80% of track)
            positions = [0.4, 0.6, 0.8]
        elif energy_curve == "constant_high":
            # Evenly spread throughout
            positions = [i / (n_elements + 1) for i in range(1, n_elements + 1)]
        elif energy_curve == "smooth":
            # Middle of the track
            positions = [0.3, 0.5, 0.7]
        else:
            positions = [0.5]

        return positions[:n_elements]

    def _master_chain(self, audio: np.ndarray, preset: dict, bpm: float) -> np.ndarray:
        """Apply master chain FX to the final mix."""
        energy_curve = preset.get("energy_curve", "build_and_drop")

        if energy_curve == "build_and_drop":
            # Add a buildup → drop at the middle
            audio = self.effects.build_drop(audio, drop_position=0.5)

        # Light sidechain for EDM/party styles
        if energy_curve in ("build_and_drop", "constant_high"):
            audio = self.effects.sidechain_compression(audio, kick_pattern_bpm=bpm, depth=0.15)

        return audio
