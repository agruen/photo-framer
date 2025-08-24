#!/bin/bash
# Face-Aware Photo Cropper Runner Script
# This script activates the virtual environment and runs the cropping tool

set -e

echo "🖼️  Face-Aware Photo Cropper"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install opencv-python Pillow numpy"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the cropping tool with all arguments passed through
echo "🚀 Starting photo processing..."
python face_crop_tool.py "$@"

echo "✅ Done!"