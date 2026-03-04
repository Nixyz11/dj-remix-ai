"""Stem separation using Demucs v4 (Meta's htdemucs model) — Python API."""

import logging
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)


class StemSeparator:
    """Separates audio into stems using Demucs v4 via Python API.

    Outputs 4 stems: vocals, drums, bass, other (melody/instruments).
    """

    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy-load the demucs model."""
        if self.model is not None:
            return
        from demucs.pretrained import get_model
        log.info("Loading Demucs model '%s' on %s...", self.model_name, self.device)
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        log.info("Demucs loaded (sources: %s, samplerate: %d)",
                 self.model.sources, self.model.samplerate)

    def separate(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio file into stems.

        Returns:
            Dict mapping stem name to file path
        """
        self._load_model()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Separating stems for: %s", audio_path)

        # Load audio with librosa (reliable cross-platform, no ffmpeg DLL issues)
        model_sr = self.model.samplerate
        audio, orig_sr = librosa.load(audio_path, sr=model_sr, mono=False)
        # audio shape: (channels, samples) if stereo, (samples,) if mono

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] > 2:
            audio = audio[:2]

        wav = torch.from_numpy(audio).float().to(self.device)

        # Add batch dimension: (batch=1, channels=2, samples)
        wav = wav.unsqueeze(0)

        # Run separation
        with torch.no_grad():
            from demucs.apply import apply_model
            sources = apply_model(self.model, wav, device=self.device)

        # sources shape: (batch=1, n_sources, channels=2, samples)
        sources = sources[0]  # Remove batch dim -> (n_sources, channels, samples)

        stem_names = self.model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']

        stems = {}
        for i, name in enumerate(stem_names):
            stem_audio = sources[i].cpu().numpy()
            # (channels, samples) -> (samples, channels) for soundfile
            stem_audio = stem_audio.T

            stem_path = output_dir / f"{name}.wav"
            sf.write(str(stem_path), stem_audio, model_sr)
            stems[name] = str(stem_path)
            log.info("  %s: %.1fs", name, stem_audio.shape[0] / model_sr)

        if not stems:
            raise RuntimeError("No stems produced")

        return stems
