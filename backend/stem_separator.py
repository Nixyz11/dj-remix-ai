"""Stem separation using Demucs v4 (Meta's htdemucs model)."""

import os
import subprocess
from pathlib import Path
from typing import Dict


class StemSeparator:
    """Separates audio into stems using Demucs v4.

    Outputs 4 stems: vocals, drums, bass, other (melody/instruments).
    """

    def __init__(self, model: str = "htdemucs"):
        self.model = model

    def separate(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio file into stems.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save stems

        Returns:
            Dict mapping stem name to file path:
            {"vocals": "path/vocals.wav", "drums": "path/drums.wav", ...}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run demucs separation
        cmd = [
            "python", "-m", "demucs",
            "--name", self.model,
            "--out", str(output_dir),
            "--two-stems" if False else "",  # Use 4-stem mode
            str(audio_path),
        ]
        # Filter empty strings from command
        cmd = [c for c in cmd if c]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Demucs outputs to: output_dir/htdemucs/track_name/stem.wav
        track_name = Path(audio_path).stem
        stems_dir = output_dir / self.model / track_name

        stems = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            stem_path = stems_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
            else:
                print(f"Warning: stem {stem_name} not found at {stem_path}")

        if not stems:
            raise RuntimeError(f"No stems found in {stems_dir}. Demucs may have failed.")

        return stems
