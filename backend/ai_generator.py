"""AI music generation using Facebook's MusicGen via HuggingFace Transformers."""

import logging
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

log = logging.getLogger(__name__)


class AIGenerator:
    """Generates musical elements using MusicGen-small.

    Used to create:
    - New beats/drops conditioned on the remix style
    - Transition fills
    - Additional melodic layers
    """

    SAMPLE_RATE = 32000  # MusicGen native output rate

    def __init__(self, model_name: str = "facebook/musicgen-small", enabled: bool = True):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.enabled = enabled
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_failed = False

    def is_available(self) -> bool:
        """Check if AI generation is available and enabled."""
        if not self.enabled or self._load_failed:
            return False
        return True

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is not None:
            return
        if self._load_failed:
            raise RuntimeError("Model previously failed to load")

        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            log.info("Loading MusicGen model (%s) on %s...", self.model_name, self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            log.info("MusicGen loaded successfully.")
        except Exception as e:
            self._load_failed = True
            log.error("Failed to load MusicGen: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        duration_seconds: float = 10,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Generate audio from a text prompt.

        Returns numpy array of audio samples at 32kHz.
        """
        if not self.enabled:
            raise RuntimeError("AI generation is disabled")

        self._load_model()

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # MusicGen generates at 32kHz, ~50 tokens per second
        max_new_tokens = int(duration_seconds * 50)

        log.info("Generating %ds audio: '%s'", duration_seconds, prompt[:80])
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
            )

        audio = audio_values[0, 0].cpu().numpy()
        log.info("Generated %d samples (%.1fs)", len(audio), len(audio) / self.SAMPLE_RATE)
        return audio

    def generate_remix_element(
        self,
        element_type: str,
        style: str = "edm",
        bpm: float = 128,
        key: str = "C minor",
        duration_seconds: float = 8,
    ) -> np.ndarray:
        """Generate a specific remix element conditioned on the track analysis."""
        element_descriptions = {
            "drop": f"powerful {style} drop at {bpm} BPM in {key}, heavy kick drums and bass",
            "buildup": f"rising {style} buildup tension at {bpm} BPM in {key}, snare rolls and risers",
            "transition": f"smooth DJ transition {style} music at {bpm} BPM in {key}",
            "beat": f"crisp drum pattern {style} beat at {bpm} BPM",
            "fill": f"drum fill transition at {bpm} BPM, {style} style",
            "bass_line": f"deep bass line at {bpm} BPM in {key}, {style} style",
        }

        prompt = element_descriptions.get(
            element_type,
            f"energetic {style} music at {bpm} BPM in {key}"
        )

        return self.generate(prompt=prompt, duration_seconds=duration_seconds)

    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = 32000):
        """Save generated audio to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)
        return output_path
