"""AI music generation using Facebook's MusicGen via HuggingFace Transformers."""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


class AIGenerator:
    """Generates musical elements using MusicGen-small.

    Used to create:
    - New beats/drops conditioned on the remix style
    - Transition fills
    - Additional melodic layers
    """

    def __init__(self, model_name: str = "facebook/musicgen-small"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is not None:
            return

        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        print(f"Loading MusicGen model ({self.model_name}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        print("MusicGen loaded successfully.")

    def generate(
        self,
        prompt: str,
        duration_seconds: float = 10,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Generate audio from a text prompt.

        Args:
            prompt: Text description (e.g. "energetic EDM drop with heavy bass")
            duration_seconds: Length of generated audio (max ~30s for small model)
            guidance_scale: How closely to follow the prompt (higher = more faithful)
            temperature: Randomness (higher = more creative)

        Returns:
            numpy array of audio samples at 32kHz
        """
        self._load_model()

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # MusicGen generates at 32kHz, ~50 tokens per second
        max_new_tokens = int(duration_seconds * 50)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
            )

        # Convert to numpy
        audio = audio_values[0, 0].cpu().numpy()
        return audio

    def generate_remix_element(
        self,
        element_type: str,
        style: str = "edm",
        bpm: float = 128,
        key: str = "C minor",
        duration_seconds: float = 8,
    ) -> np.ndarray:
        """Generate a specific remix element conditioned on the track analysis.

        Args:
            element_type: One of "drop", "buildup", "transition", "beat", "fill", "bass_line"
            style: Remix style (party, edm, chill, bass_heavy)
            bpm: Target BPM
            key: Musical key
            duration_seconds: Length
        """
        style_descriptions = {
            "party": "energetic party music with driving beats",
            "edm": "electronic dance music with synthesizers and heavy drops",
            "chill": "chill downtempo electronic with smooth pads",
            "bass_heavy": "heavy bass music with sub-bass drops and wobble",
        }

        element_descriptions = {
            "drop": f"powerful {style} drop at {bpm} BPM in {key}, heavy kick drums and bass",
            "buildup": f"rising {style} buildup tension at {bpm} BPM in {key}, snare rolls and risers",
            "transition": f"smooth DJ transition {style} music at {bpm} BPM in {key}",
            "beat": f"crisp drum pattern {style} beat at {bpm} BPM",
            "fill": f"drum fill transition at {bpm} BPM, {style} style",
            "bass_line": f"deep bass line at {bpm} BPM in {key}, {style} style",
        }

        style_desc = style_descriptions.get(style, style_descriptions["party"])
        prompt = element_descriptions.get(element_type, f"{style_desc} at {bpm} BPM in {key}")

        return self.generate(
            prompt=prompt,
            duration_seconds=duration_seconds,
        )

    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = 32000):
        """Save generated audio to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)
        return output_path
