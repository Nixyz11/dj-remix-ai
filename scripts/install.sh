#!/bin/bash
# DJ Remix AI — Installation Script (venv-based)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== DJ Remix AI — Setup ==="

# Check Python
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found. Install Python 3.10+."
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate venv (cross-platform)
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows Git Bash
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate      # Linux/Mac
fi

# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies from frozen requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo ""
    echo "WARNING: ffmpeg not found!"
    echo "  Windows: choco install ffmpeg  OR  scoop install ffmpeg"
    echo "  Mac:     brew install ffmpeg"
    echo "  Linux:   sudo apt install ffmpeg"
fi

# Create directories
mkdir -p uploads output models

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the server:"
echo "  bash run.sh"
echo ""
echo "To download AI models (~2GB):"
echo "  bash scripts/download_models.sh"
