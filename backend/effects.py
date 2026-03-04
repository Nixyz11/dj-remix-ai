"""Audio effects chain — reverb, filters, transitions, DJ-style FX."""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal


class AudioEffects:
    """DJ-style audio effects processor."""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def apply_reverb(self, y: np.ndarray, decay: float = 0.5, delay_ms: int = 40) -> np.ndarray:
        """Simple reverb using comb filter."""
        delay_samples = int(self.sr * delay_ms / 1000)
        output = np.copy(y).astype(np.float64)

        for i in range(delay_samples, len(output)):
            output[i] += decay * output[i - delay_samples]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95
        return output.astype(np.float32)

    def apply_echo(self, y: np.ndarray, delay_ms: int = 300, decay: float = 0.4, repeats: int = 3) -> np.ndarray:
        """Echo / delay effect."""
        delay_samples = int(self.sr * delay_ms / 1000)
        output_length = len(y) + delay_samples * repeats
        output = np.zeros(output_length, dtype=np.float64)
        output[:len(y)] = y

        for r in range(1, repeats + 1):
            offset = delay_samples * r
            gain = decay ** r
            end = min(offset + len(y), output_length)
            output[offset:end] += y[:end - offset] * gain

        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95
        return output.astype(np.float32)

    def low_pass_filter(self, y: np.ndarray, cutoff_hz: float = 800) -> np.ndarray:
        """Low-pass filter (for filter sweeps / buildups)."""
        nyquist = self.sr / 2
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, y).astype(np.float32)

    def high_pass_filter(self, y: np.ndarray, cutoff_hz: float = 2000) -> np.ndarray:
        """High-pass filter."""
        nyquist = self.sr / 2
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, y).astype(np.float32)

    def filter_sweep(self, y: np.ndarray, start_hz: float = 200, end_hz: float = 8000) -> np.ndarray:
        """Gradual filter sweep from low to high (buildup effect)."""
        chunk_size = len(y) // 20
        output = np.zeros_like(y)

        for i in range(20):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(y))
            progress = i / 19
            cutoff = start_hz + (end_hz - start_hz) * progress
            chunk = y[start_idx:end_idx]
            output[start_idx:end_idx] = self.low_pass_filter(chunk, cutoff)

        return output

    def sidechain_compression(self, y: np.ndarray, kick_pattern_bpm: float = 128,
                               depth: float = 0.7) -> np.ndarray:
        """Simulate sidechain compression (pumping effect common in EDM)."""
        beat_duration = 60.0 / kick_pattern_bpm
        beat_samples = int(beat_duration * self.sr)
        output = np.copy(y).astype(np.float64)

        for i in range(0, len(output), beat_samples):
            # Create a pump envelope: quick duck then release
            chunk_end = min(i + beat_samples, len(output))
            chunk_len = chunk_end - i
            t = np.linspace(0, 1, chunk_len)
            # Fast attack, slow release envelope
            envelope = 1.0 - depth * np.exp(-t * 8)
            output[i:chunk_end] *= envelope

        return output.astype(np.float32)

    def time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        """Time-stretch audio without changing pitch."""
        return librosa.effects.time_stretch(y, rate=rate)

    def pitch_shift(self, y: np.ndarray, n_steps: float) -> np.ndarray:
        """Pitch shift by n semitones."""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

    def crossfade(self, y1: np.ndarray, y2: np.ndarray, fade_samples: int = None) -> np.ndarray:
        """Crossfade between two audio clips."""
        if fade_samples is None:
            fade_samples = min(len(y1), len(y2), self.sr * 2)  # 2 second default

        fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)

        # Apply fades
        y1_tail = y1[-fade_samples:] * fade_out
        y2_head = y2[:fade_samples] * fade_in
        crossfaded = y1_tail + y2_head

        # Concatenate: y1 (minus tail) + crossfade + y2 (minus head)
        result = np.concatenate([y1[:-fade_samples], crossfaded, y2[fade_samples:]])
        return result

    def build_drop(self, y: np.ndarray, drop_position: float = 0.5, buildup_seconds: float = 4) -> np.ndarray:
        """Create a buildup → drop effect at the specified position."""
        drop_sample = int(len(y) * drop_position)
        buildup_samples = int(buildup_seconds * self.sr)
        buildup_start = max(0, drop_sample - buildup_samples)

        output = np.copy(y)

        # Buildup: filter sweep + volume rise
        buildup = output[buildup_start:drop_sample]
        buildup = self.filter_sweep(buildup, start_hz=300, end_hz=6000)
        output[buildup_start:drop_sample] = buildup

        # Drop: brief silence then full energy
        silence_samples = int(0.1 * self.sr)
        if drop_sample + silence_samples < len(output):
            output[drop_sample:drop_sample + silence_samples] = 0

        return output
