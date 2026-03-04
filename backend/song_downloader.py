"""Song downloader — download audio from YouTube/SoundCloud URLs via yt-dlp."""

import subprocess
from pathlib import Path


class SongDownloader:
    """Downloads audio from URLs using yt-dlp."""

    def download(self, url: str, output_dir: str, job_id: str) -> str:
        """Download audio from URL and convert to WAV.

        Args:
            url: YouTube, SoundCloud, or other supported URL
            output_dir: Directory to save the downloaded file
            job_id: Unique job identifier for the filename

        Returns:
            Path to the downloaded audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{job_id}_downloaded.%(ext)s")

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", output_path,
            "--no-playlist",
            "--max-filesize", "100M",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        # Find the downloaded file
        wav_path = output_dir / f"{job_id}_downloaded.wav"
        if wav_path.exists():
            return str(wav_path)

        # yt-dlp might output with different extension patterns
        for f in output_dir.glob(f"{job_id}_downloaded*"):
            return str(f)

        raise RuntimeError("Downloaded file not found")
