#!/bin/bash
# DJ Remix AI — Installation Script

set -e

echo "🎧 Installing DJ Remix AI..."

# Check Python version
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install yt-dlp
pip install yt-dlp

# Check ffmpeg
ffmpeg -version > /dev/null 2>&1 || { echo "ffmpeg not found. Please install ffmpeg."; exit 1; }

# Create directories
mkdir -p uploads output models

echo ""
echo "Installation complete!"
echo "Run the server: uvicorn backend.main:app --reload"
