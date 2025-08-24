#!/bin/bash
# Complete Photo Frame Slideshow Generator
# This script crops photos and creates a slideshow with weather integration

set -e

echo "🖼️  Photo Frame Slideshow Generator"
echo "====================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Setting up now..."
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "✅ Virtual environment setup complete!"
    echo
else
    # Activate virtual environment
    source venv/bin/activate
fi

# Collect user inputs
echo "📝 Please provide the following information:"
echo

# Input folder
read -p "📁 Input folder containing photos (default: photos): " INPUT_FOLDER
INPUT_FOLDER=${INPUT_FOLDER:-photos}

# Output folder
read -p "💾 Output folder for processed photos (default: output): " OUTPUT_FOLDER
OUTPUT_FOLDER=${OUTPUT_FOLDER:-output}

# Screen dimensions
read -p "📐 Screen width in pixels (default: 1280): " SCREEN_WIDTH
SCREEN_WIDTH=${SCREEN_WIDTH:-1280}

read -p "📐 Screen height in pixels (default: 800): " SCREEN_HEIGHT  
SCREEN_HEIGHT=${SCREEN_HEIGHT:-800}

# Weather API setup
echo
echo "🌤️  Weather Integration Setup"
echo "For weather functionality, you'll need a free API key from OpenWeatherMap:"
echo "1. Go to: https://openweathermap.org/api"
echo "2. Sign up for a free account"
echo "3. Get your API key from the dashboard"
echo

read -p "🔑 OpenWeatherMap API key (or press Enter to skip weather): " API_KEY

if [ ! -z "$API_KEY" ]; then
    read -p "📍 ZIP code for weather data (e.g., 10001): " ZIP_CODE
    ZIP_CODE=${ZIP_CODE:-10001}
else
    API_KEY="YOUR_API_KEY_HERE"
    ZIP_CODE="10001"
    echo "⚠️  Weather functionality will be disabled"
fi

# Confirm settings
echo
echo "📋 Configuration Summary:"
echo "=========================="
echo "Input folder:     $INPUT_FOLDER"
echo "Output folder:    $OUTPUT_FOLDER" 
echo "Screen size:      ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
if [ "$API_KEY" != "YOUR_API_KEY_HERE" ]; then
    echo "Weather ZIP:      $ZIP_CODE"
    echo "Weather API:      Enabled"
else
    echo "Weather API:      Disabled"
fi
echo

read -p "🚀 Proceed with these settings? (y/N): " PROCEED
if [[ ! $PROCEED =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 0
fi

echo
echo "🔄 Starting photo processing pipeline..."

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "❌ Input folder '$INPUT_FOLDER' does not exist"
    echo "Please create the folder and add your photos, then run this script again"
    exit 1
fi

# Count images in input folder
IMAGE_COUNT=$(find "$INPUT_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ No image files found in '$INPUT_FOLDER'"
    echo "Please add some photos to the folder and run this script again"
    exit 1
fi

echo "📸 Found $IMAGE_COUNT images to process"

# Step 1: Crop images with face detection
echo
echo "🎯 Step 1: Processing images with face-aware cropping..."
python face_crop_tool.py \
    --input "$INPUT_FOLDER" \
    --output "$OUTPUT_FOLDER" \
    --width "$SCREEN_WIDTH" \
    --height "$SCREEN_HEIGHT"

# Check if cropping was successful
PROCESSED_COUNT=$(find "$OUTPUT_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" \) 2>/dev/null | wc -l)
if [ "$PROCESSED_COUNT" -eq 0 ]; then
    echo "❌ No images were successfully processed"
    exit 1
fi

echo "✅ Successfully processed $PROCESSED_COUNT images"

# Step 2: Generate slideshow HTML
echo
echo "🎭 Step 2: Generating slideshow with weather integration..."
python slideshow_generator.py \
    --folder "$OUTPUT_FOLDER" \
    --zip "$ZIP_CODE" \
    --api-key "$API_KEY" \
    --width "$SCREEN_WIDTH" \
    --height "$SCREEN_HEIGHT"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Complete! Your photo slideshow is ready!"
    echo "==============================================="
    echo
    echo "📁 Processed images: $OUTPUT_FOLDER/"
    echo "🌐 Slideshow HTML:   $OUTPUT_FOLDER/slideshow.html"
    echo
    echo "📖 Instructions:"
    echo "1. Open slideshow.html in a web browser"
    echo "2. Press F11 for full-screen mode"
    echo "3. Images will rotate every 60 seconds"
    echo "4. Clock and weather update automatically"
    echo
    if [ "$API_KEY" = "YOUR_API_KEY_HERE" ]; then
        echo "💡 To enable weather:"
        echo "   - Get a free API key from openweathermap.org/api"
        echo "   - Run this script again with your API key"
    fi
    echo
    echo "✨ Enjoy your digital photo frame!"
else
    echo "❌ Error generating slideshow"
    exit 1
fi