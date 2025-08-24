# Photo Frame Slideshow Generator

A complete Python toolkit that intelligently crops photos with face detection and creates beautiful HTML slideshows for digital photo frames. Features weather integration, automatic image rotation, and real-time clock display.

## Features

### Photo Processing
- **Advanced Face Detection**: Multi-method approach using DNN models (when available) and Haar cascade classifiers
- **Intelligent Cropping**: Rule-of-thirds composition with sophisticated face-aware positioning
- **Enhanced Detection**: Eye-based face estimation as fallback, with duplicate removal
- **Smart Composition**: Positions faces optimally using portrait photography principles
- **Adaptive Image Enhancement**: Dynamic sharpening and quality optimization based on scaling
- **Batch Processing**: Processes entire folders of images efficiently with multiprocessing
- **Multiple Face Support**: Handles complex group photos with many faces
- **Quality Preservation**: LANCZOS resampling with progressive JPEG optimization

### Slideshow Generation  
- **HTML Slideshow**: Full-screen responsive slideshow optimized for digital photo frames
- **Weather Integration**: Real-time weather display using OpenWeatherMap API
- **Digital Clock**: Live 12-hour format clock display with automatic updates
- **Auto Image Rotation**: Random image changes every 60 seconds
- **Responsive Design**: Configurable screen dimensions for any display device
- **Elegant Overlays**: Weather and clock with attractive text shadows for readability

## Setup

1. **Quick Setup (Recommended):**
   ```bash
   git clone <your-repo>
   cd Photo-Framer
   ./run_complete.sh
   ```
   The script will automatically set up the virtual environment and install dependencies.

2. **Manual Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install opencv-python Pillow numpy
   ```

3. **Get Weather API Key (Optional):**
   - Go to [OpenWeatherMap API](https://openweathermap.org/api)
   - Sign up for free account
   - Get your API key from the dashboard

## Usage

### Complete Slideshow Generation (Recommended)
The interactive script will prompt you for all settings:
```bash
./run_complete.sh
```

### Advanced Usage
Use the integrated Python script directly:
```bash
python photo_frame_complete.py \
  --input photos \
  --output slideshow \
  --width 1280 \
  --height 800 \
  --zip 10001 \
  --api-key YOUR_OPENWEATHER_API_KEY
```

### Legacy Cropping Only
Process images without slideshow generation:
```bash
./run.sh --input photos --output cropped_photos
```

### All Available Options
```bash
python photo_frame_complete.py \
  --input photos \           # Input folder containing photos
  --output slideshow \       # Output folder for processed photos and slideshow  
  --width 1280 \            # Target screen width (default: 1280)
  --height 800 \            # Target screen height (default: 800)
  --zip 10001 \             # ZIP code for weather data (default: 10001)
  --api-key YOUR_KEY \      # OpenWeatherMap API key
  --processes 4             # Number of processes for parallel processing
```

## How It Works

1. **Multi-Method Face Detection**: 
   - Attempts DNN-based face detection for highest accuracy
   - Falls back to multiple Haar cascade configurations
   - Uses eye detection to estimate face regions when needed
   - Removes duplicate detections automatically

2. **Advanced Composition Algorithm**:
   - Calculates optimal face positioning using rule of thirds
   - Evaluates multiple crop candidates for best composition
   - Ensures all faces are preserved with dynamic padding
   - Applies portrait photography principles for pleasing results

3. **Smart Image Processing**:
   - Adaptive sharpening based on scaling factors
   - Subtle contrast and saturation enhancement
   - Progressive JPEG optimization for smaller files
   - High-quality LANCZOS resampling for final resize

## Fallback Behavior

If no faces are detected in an image, the tool will:
- Crop from the center of the image
- Maintain the target aspect ratio
- Provide the best possible crop of the original image

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png) 
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Output

- All cropped images are saved with `_cropped` suffix
- Output format matches input format
- Images are saved with high quality (95% JPEG quality)
- Exact dimensions: 1280x800 pixels (or custom dimensions specified)

## Example Output

```
🖼️  Photo Frame Slideshow Generator
=====================================

📸 Found 15 images to process

🎯 Step 1: Processing images with face-aware cropping...
Found 15 image files to process
Using 4 processes for parallel processing

Progress: 15/15 (100.0%) - Success: 15, Failed: 0

Processing complete!
Total time: 12.3 seconds
Successfully processed: 15/15 images

🎭 Step 2: Generating slideshow with weather integration...
Found 15 images for slideshow
✅ Slideshow generated: /path/to/slideshow/slideshow.html

🎉 Complete! Your photo slideshow is ready!
==================================================

📁 Processed images: slideshow/
🌐 Slideshow HTML:   slideshow/slideshow.html

📖 Instructions:
1. Open slideshow.html in a web browser
2. Press F11 for full-screen mode  
3. Images will rotate every 60 seconds
4. Clock and weather update automatically

✨ Enjoy your digital photo frame!
```

## Slideshow Features

- **Full-Screen Display**: Optimized for any screen resolution
- **Automatic Image Rotation**: Random image every 60 seconds
- **Live Clock**: Updates every second with current time
- **Weather Display**: Shows current temperature and weather icon
- **Responsive Design**: Works on any device or screen size
- **Error Handling**: Graceful fallback if weather API is unavailable