# DJ Remix AI

AI-powered DJ remixing engine that transforms any song into a party-ready remix.

## How It Works

1. **Stem Separation** — Splits song into vocals, drums, bass, melody (Demucs v4)
2. **Audio Analysis** — Detects BPM, key, energy, and structure (Librosa)
3. **Hybrid Remix** — Applies DJ effects + AI-generated elements (MusicGen)
4. **Export** — Polished remix ready to play

## Quick Start

```bash
# Install
bash scripts/install.sh

# Download AI models (~2GB)
bash scripts/download_models.sh

# Run server
cd backend && uvicorn main:app --reload
```

API available at `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/remix/upload` | Upload audio file to remix |
| POST | `/remix/url` | Remix from YouTube/SoundCloud URL |
| GET | `/status/{job_id}` | Check remix progress |
| GET | `/download/{job_id}` | Download finished remix |

## Remix Styles

- **party** — Driving beats, energetic drops (124-132 BPM)
- **edm** — Heavy synths, big drops (126-140 BPM)
- **chill** — Smooth downtempo (90-110 BPM)
- **bass_heavy** — Sub-bass and wobble (130-150 BPM)

## Tech Stack

- **Backend**: Python + FastAPI
- **Stem Separation**: Demucs v4 (Meta)
- **AI Generation**: MusicGen-small (Meta/HuggingFace)
- **Audio Analysis**: Librosa
- **Audio FX**: Custom DSP (reverb, echo, filters, sidechain)
- **Downloading**: yt-dlp

## Requirements

- Python 3.10+
- ffmpeg
- ~4GB RAM (CPU mode), GPU recommended for faster AI generation
