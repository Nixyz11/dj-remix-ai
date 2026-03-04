"""Audio analysis — BPM detection, key estimation, energy profiling."""

import logging
import librosa
import numpy as np
from typing import Dict, Any

log = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyzes audio files for BPM, musical key, energy, and structure."""

    KEY_NAMES = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Full analysis of an audio track."""
        log.info("Analyzing: %s", audio_path)
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        bpm = self._detect_bpm(y, sr)
        key, mode = self._detect_key(y, sr)
        energy_profile = self._energy_profile(y, sr)
        onsets = self._detect_onsets(y, sr)
        duration = float(librosa.get_duration(y=y, sr=sr))
        sections = self._detect_sections(y, sr)
        avg_energy = float(np.mean(energy_profile)) if energy_profile else 0.0

        result = {
            "bpm": round(bpm, 1),
            "key": key,
            "mode": mode,
            "key_full": f"{key} {mode}",
            "duration_seconds": round(duration, 2),
            "energy_profile": energy_profile,
            "average_energy": round(avg_energy, 4),
            "onset_count": len(onsets),
            "onset_times": onsets[:50],
            "sections": sections,
            "sample_rate": sr,
        }
        log.info("Analysis complete: %.1f BPM, %s %s, %.1fs duration",
                 bpm, key, mode, duration)
        return result

    def analyze_quick(self, audio_path: str) -> Dict[str, Any]:
        """Quick analysis — BPM and key only, no sections/onsets."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        bpm = self._detect_bpm(y, sr)
        key, mode = self._detect_key(y, sr)
        duration = float(librosa.get_duration(y=y, sr=sr))
        return {
            "bpm": round(bpm, 1),
            "key": key,
            "mode": mode,
            "key_full": f"{key} {mode}",
            "duration_seconds": round(duration, 2),
        }

    def _detect_bpm(self, y: np.ndarray, sr: int) -> float:
        """Detect BPM using librosa's beat tracker."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        return float(tempo)

    def _detect_key(self, y: np.ndarray, sr: int) -> tuple:
        """Estimate musical key using chroma features."""
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_corr = -1
        best_key = 0
        best_mode = "major"

        for i in range(12):
            shifted = np.roll(chroma_mean, -i)
            corr_major = float(np.corrcoef(shifted, self.MAJOR_PROFILE)[0, 1])
            corr_minor = float(np.corrcoef(shifted, self.MINOR_PROFILE)[0, 1])

            if corr_major > best_corr:
                best_corr = corr_major
                best_key = i
                best_mode = "major"
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = i
                best_mode = "minor"

        return self.KEY_NAMES[best_key], best_mode

    def _energy_profile(self, y: np.ndarray, sr: int) -> list:
        """Compute RMS energy over time, sampled at ~1 per second."""
        hop_length = sr
        frame_length = min(sr * 2, len(y))
        if frame_length < 1:
            return []
        rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)[0]
        return [round(float(x), 4) for x in rms]

    def _detect_onsets(self, y: np.ndarray, sr: int) -> list:
        """Detect onset times (transients) in seconds."""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return [round(float(t), 3) for t in onset_times]

    def _detect_sections(self, y: np.ndarray, sr: int) -> list:
        """Section detection based on spectral clustering."""
        duration = len(y) / sr
        # Adjust number of sections based on track length
        n_sections = max(2, min(int(duration / 15), 10))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < n_sections:
            return [{"start": 0.0, "label": "section_1"}]

        bounds = librosa.segment.agglomerative(mfcc, k=n_sections)
        bound_times = librosa.frames_to_time(bounds, sr=sr)

        sections = []
        for i, t in enumerate(bound_times):
            sections.append({
                "start": round(float(t), 2),
                "label": f"section_{i+1}",
            })
        return sections
