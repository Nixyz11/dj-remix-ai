"""Audio effects chain — reverb, filters, transitions, DJ-style FX."""

import logging
import numpy as np
import librosa
from scipy import signal

log = logging.getLogger(__name__)

# Minimum chunk length for filter operations (avoid scipy butter crashes)
MIN_FILTER_SAMPLES = 64


class AudioEffects:
    """DJ-style audio effects processor."""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def apply_reverb(self, y: np.ndarray, decay: float = 0.5, delay_ms: int = 40) -> np.ndarray:
        """Reverb using vectorized IIR comb filter."""
        delay_samples = int(self.sr * delay_ms / 1000)
        if delay_samples <= 0 or len(y) <= delay_samples:
            return y.copy()

        output = np.copy(y).astype(np.float64)
        # Vectorized IIR: split into passes to avoid sample-by-sample loop
        # Apply multiple taps at increasing delays for a richer reverb
        for tap in range(1, 5):
            d = delay_samples * tap
            gain = decay ** tap
            if d >= len(output):
                break
            output[d:] += output[:-d] * gain if d < len(output) else 0

        return _normalize(output)

    def apply_echo(self, y: np.ndarray, delay_ms: int = 300, decay: float = 0.4,
                   repeats: int = 3, match_length: bool = True) -> np.ndarray:
        """Echo / delay effect. If match_length=True, output matches input length."""
        delay_samples = int(self.sr * delay_ms / 1000)
        if match_length:
            output = np.copy(y).astype(np.float64)
            for r in range(1, repeats + 1):
                offset = delay_samples * r
                gain = decay ** r
                if offset >= len(output):
                    break
                remaining = len(output) - offset
                output[offset:] += y[:remaining] * gain
        else:
            output_length = len(y) + delay_samples * repeats
            output = np.zeros(output_length, dtype=np.float64)
            output[:len(y)] = y
            for r in range(1, repeats + 1):
                offset = delay_samples * r
                gain = decay ** r
                end = min(offset + len(y), output_length)
                output[offset:end] += y[:end - offset] * gain

        return _normalize(output)

    def low_pass_filter(self, y: np.ndarray, cutoff_hz: float = 800) -> np.ndarray:
        """Low-pass filter (for filter sweeps / buildups)."""
        if len(y) < MIN_FILTER_SAMPLES:
            return y.copy()
        nyquist = self.sr / 2
        normalized_cutoff = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, y).astype(np.float32)

    def high_pass_filter(self, y: np.ndarray, cutoff_hz: float = 2000) -> np.ndarray:
        """High-pass filter."""
        if len(y) < MIN_FILTER_SAMPLES:
            return y.copy()
        nyquist = self.sr / 2
        normalized_cutoff = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, y).astype(np.float32)

    def filter_sweep(self, y: np.ndarray, start_hz: float = 200, end_hz: float = 8000,
                     n_steps: int = 20) -> np.ndarray:
        """Gradual filter sweep from low to high (buildup effect)."""
        if len(y) < MIN_FILTER_SAMPLES * n_steps:
            n_steps = max(1, len(y) // MIN_FILTER_SAMPLES)

        chunk_size = len(y) // n_steps
        if chunk_size < MIN_FILTER_SAMPLES:
            return y.copy()

        output = np.zeros_like(y)
        for i in range(n_steps):
            start_idx = i * chunk_size
            end_idx = len(y) if i == n_steps - 1 else (i + 1) * chunk_size
            progress = i / max(n_steps - 1, 1)
            cutoff = start_hz + (end_hz - start_hz) * progress
            chunk = y[start_idx:end_idx]
            if len(chunk) >= MIN_FILTER_SAMPLES:
                output[start_idx:end_idx] = self.low_pass_filter(chunk, cutoff)
            else:
                output[start_idx:end_idx] = chunk

        return output

    def sidechain_compression(self, y: np.ndarray, kick_pattern_bpm: float = 128,
                               depth: float = 0.7) -> np.ndarray:
        """Simulate sidechain compression (pumping effect common in EDM)."""
        beat_duration = 60.0 / kick_pattern_bpm
        beat_samples = int(beat_duration * self.sr)
        if beat_samples <= 0:
            return y.copy()

        output = np.copy(y).astype(np.float64)

        for i in range(0, len(output), beat_samples):
            chunk_end = min(i + beat_samples, len(output))
            chunk_len = chunk_end - i
            t = np.linspace(0, 1, chunk_len)
            envelope = 1.0 - depth * np.exp(-t * 8)
            output[i:chunk_end] *= envelope

        return output.astype(np.float32)

    def time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        """Time-stretch audio without changing pitch."""
        if abs(rate - 1.0) < 0.01:
            return y.copy()
        return librosa.effects.time_stretch(y, rate=rate)

    def pitch_shift(self, y: np.ndarray, n_steps: float) -> np.ndarray:
        """Pitch shift by n semitones."""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

    def crossfade(self, y1: np.ndarray, y2: np.ndarray, fade_samples: int = None) -> np.ndarray:
        """Crossfade between two audio clips."""
        if fade_samples is None:
            fade_samples = min(len(y1), len(y2), self.sr * 2)

        fade_samples = min(fade_samples, len(y1), len(y2))
        if fade_samples <= 0:
            return np.concatenate([y1, y2])

        fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)

        y1_tail = y1[-fade_samples:] * fade_out
        y2_head = y2[:fade_samples] * fade_in
        crossfaded = y1_tail + y2_head

        return np.concatenate([y1[:-fade_samples], crossfaded, y2[fade_samples:]])

    def build_drop(self, y: np.ndarray, drop_position: float = 0.5,
                   buildup_seconds: float = 4) -> np.ndarray:
        """Create a buildup -> drop effect at the specified position."""
        drop_sample = int(len(y) * drop_position)
        buildup_samples = int(buildup_seconds * self.sr)
        buildup_start = max(0, drop_sample - buildup_samples)

        output = np.copy(y)

        # Buildup: filter sweep + volume rise
        buildup_region = output[buildup_start:drop_sample]
        if len(buildup_region) >= MIN_FILTER_SAMPLES:
            buildup_region = self.filter_sweep(buildup_region, start_hz=300, end_hz=6000)
            # Add volume ramp
            ramp = np.linspace(0.6, 1.0, len(buildup_region)).astype(np.float32)
            buildup_region *= ramp
            output[buildup_start:drop_sample] = buildup_region

        # Drop: brief silence then full energy
        silence_samples = int(0.05 * self.sr)
        if drop_sample + silence_samples < len(output):
            output[drop_sample:drop_sample + silence_samples] = 0

        return output

    def stutter(self, y: np.ndarray, position: float = 0.5, repeat_ms: int = 100,
                repeats: int = 8) -> np.ndarray:
        """Stutter/glitch effect — repeats a tiny slice of audio."""
        pos_sample = int(len(y) * position)
        slice_len = int(self.sr * repeat_ms / 1000)
        if pos_sample + slice_len > len(y):
            return y.copy()

        output = np.copy(y)
        snippet = y[pos_sample:pos_sample + slice_len]
        total_stutter = slice_len * repeats

        if pos_sample + total_stutter > len(output):
            return output

        for r in range(repeats):
            start = pos_sample + r * slice_len
            decay = 1.0 - (r / repeats) * 0.3
            output[start:start + slice_len] = snippet * decay

        return output

    def vinyl_brake(self, y: np.ndarray, position: float = 0.5,
                    duration_seconds: float = 1.5) -> np.ndarray:
        """Simulates a vinyl record slowing down and stopping."""
        pos_sample = int(len(y) * position)
        brake_len = int(self.sr * duration_seconds)
        end_sample = min(pos_sample + brake_len, len(y))
        actual_len = end_sample - pos_sample

        if actual_len < 100:
            return y.copy()

        output = np.copy(y)
        segment = output[pos_sample:end_sample].astype(np.float64)

        # Pitch drops to zero over the duration
        t = np.linspace(1.0, 0.0, actual_len)
        # Apply progressively slower playback by phase manipulation
        phase = np.cumsum(t) / np.sum(t) * actual_len
        indices = np.clip(phase.astype(int), 0, actual_len - 1)
        output[pos_sample:end_sample] = segment[indices] * t

        return output.astype(np.float32)


def _normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * target_peak
    return audio.astype(np.float32)
