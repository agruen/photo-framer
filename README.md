# Face-Aware Photo Cropper

A Python tool that intelligently crops photos to exactly 1280x800 pixels while using advanced face detection to ensure that people's faces remain visible in the cropped images.

## Features

- **Advanced Face Detection**: Multi-method approach using DNN models (when available) and Haar cascade classifiers
- **Intelligent Cropping**: Rule-of-thirds composition with sophisticated face-aware positioning
- **Enhanced Detection**: Eye-based face estimation as fallback, with duplicate removal
- **Smart Composition**: Positions faces optimally using portrait photography principles
- **Adaptive Image Enhancement**: Dynamic sharpening and quality optimization based on scaling
- **Batch Processing**: Processes entire folders of images efficiently
- **Multiple Face Support**: Handles complex group photos with many faces
- **Quality Preservation**: LANCZOS resampling with progressive JPEG optimization

## Setup

1. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install opencv-python Pillow numpy
   ```

## Usage

### Basic Usage
Process all images in the `photos` folder and save to `output`:
```bash
./run.sh
```

### Custom Input/Output Folders
```bash
./run.sh --input my_photos --output cropped_photos
```

### Custom Dimensions
```bash
./run.sh --width 1920 --height 1080
```

### Multiprocessing Options
```bash
# Use specific number of processes
./run.sh --processes 4

# Force single-threaded (for comparison)
./run.sh --single-threaded

# Benchmark both modes
./run.sh --benchmark --input test_photos
```

### All Options
```bash
./run.sh --input photos --output output --width 1280 --height 800 --processes 4
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
🖼️  Face-Aware Photo Cropper
================================
🚀 Starting photo processing...
Found 3 image files to process

[1/3] Processing photos/family.jpeg: 3616x2411
Detected 3 face(s)
Saved cropped image to output/family_cropped.jpeg

[2/3] Processing photos/portrait.jpeg: 1935x2592
Detected 1 face(s)
Saved cropped image to output/portrait_cropped.jpeg

[3/3] Processing photos/group.jpeg: 4032x3024
Detected 5 face(s)
Saved cropped image to output/group_cropped.jpeg

Processing complete: 3/3 images processed successfully
✅ Done!
```