#!/bin/bash
# Download HuggingFace models for offline use

set -e

echo "Downloading MusicGen Small model..."
python3 -c "
from transformers import AutoProcessor, MusicgenForConditionalGeneration
print('Downloading MusicGen-small...')
processor = AutoProcessor.from_pretrained('facebook/musicgen-small')
model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-small')
print('MusicGen-small downloaded successfully!')
"

echo "Downloading Demucs htdemucs model..."
python3 -c "
import demucs.pretrained
print('Downloading Demucs htdemucs...')
model = demucs.pretrained.get_model('htdemucs')
print('Demucs htdemucs downloaded successfully!')
"

echo ""
echo "All models downloaded!"
