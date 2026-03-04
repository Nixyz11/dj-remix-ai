#!/bin/bash
# DJ Remix AI — Run server using venv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows Git Bash
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate      # Linux/Mac
else
    echo "Error: venv not found. Run setup.sh first."
    exit 1
fi

echo "Starting DJ Remix AI server..."
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
