# Photo Frame Slideshow Generator

A complete Python toolkit and web server for creating beautiful digital photo frame slideshows. Features intelligent face-aware cropping, weather integration, and a modern web admin interface for managing multiple slideshows.

## ✨ New: Bulk Folder Upload Support

🚀 **Major Update - Now supports uploading thousands of images:**
- **Folder Upload**: Select entire photo folders directly (no ZIP required!)
- **Batch Processing**: Handles hundreds or thousands of images efficiently
- **Real-time Progress**: Watch individual file uploads with live progress bars
- **Memory Optimized**: Processes files in small batches to avoid memory issues
- **Maintains Structure**: Preserves your original folder organization

## Complete Dockerized Solution

🖥️ **Web Server with Admin Interface:**
- **Dual Upload Options**: Upload entire folders OR ZIP files via drag-and-drop
- **Background Processing**: Real-time progress updates with WebSocket notifications
- **Secure Sharing**: Generate 256-character random URLs for sharing slideshows
- **Multi-user Hosting**: Host multiple slideshows simultaneously 
- **Cross-platform**: Works on AMD64 and ARM64 (Raspberry Pi compatible)

💻 **Legacy Command Line Tools:**
- Original face cropping script: `face_crop_tool.py`
- Slideshow generator: `slideshow_generator.py`
- Simple runner: `run.sh`

## Features

### Photo Upload & Processing
- **Multiple Upload Methods**: Choose between folder upload or ZIP file upload
- **Scalable Processing**: Handles 10s to 1000s of images with batch processing
- **Advanced Face Detection**: Multi-method approach using DNN models and Haar cascade classifiers
- **Intelligent Cropping**: Rule-of-thirds composition with sophisticated face-aware positioning
- **Enhanced Detection**: Eye-based face estimation as fallback, with duplicate removal
- **Smart Composition**: Positions faces optimally using portrait photography principles
- **Adaptive Image Enhancement**: Dynamic sharpening and quality optimization based on scaling
- **Multiple Face Support**: Handles complex group photos with many faces
- **Quality Preservation**: LANCZOS resampling with progressive JPEG optimization

### Slideshow Generation  
- **HTML Slideshow**: Full-screen responsive slideshow optimized for digital photo frames
- **Weather Integration**: Real-time weather display using OpenWeatherMap API
- **Digital Clock**: Live 12-hour format clock display with automatic updates
- **Auto Image Rotation**: Random image changes every 60 seconds
- **Responsive Design**: Configurable screen dimensions for any display device
- **Elegant Overlays**: Weather and clock with attractive text shadows for readability

## Quick Start

### 🖥️ Web Server Setup (Recommended)

1. **Start the server:**
   ```bash
   git clone <your-repo>
   cd Photo-Framer
   ./start-server.sh
   ```

2. **Access admin interface:**
   - Open http://localhost:5000 in your browser
   - Login with password: `changeme123` (change this!)
   - Click "Create New Slideshow"

3. **Create slideshow:**
   - **Option A**: Click "Select Folder" to upload entire photo folders (thousands of images!)
   - **Option B**: Drag and drop ZIP file with photos (legacy method)
   - Configure screen size and weather settings
   - Watch real-time batch processing with live progress bars
   - Generate secure URLs to share

📖 **Full server documentation:** [README-SERVER.md](README-SERVER.md)

---

### 💻 Command Line Setup (Original)

1. **Quick Setup:**
   ```bash
   git clone <your-repo>
   cd Photo-Framer
   ./run_complete.sh
   ```

2. **Manual Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install opencv-python Pillow numpy
   ```

## Usage

### Web Interface (Recommended)
```bash
./start-server.sh                    # Start web server
# Open http://localhost:5000 in browser
```

### Command Line Tools (Legacy)
```bash
./run.sh --input photos --output cropped_photos
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
- WEBP (.webp)

## Output

- All cropped images are saved with `_cropped` suffix
- Output format matches input format
- Images are saved with high quality (95% JPEG quality)
- Exact dimensions: 1280x800 pixels (or custom dimensions specified)

## Web Interface Example

When uploading a folder with hundreds of photos, you'll see:

```
🖼️ Photo Frame Slideshow Creator
=================================

📁 Selected folder: "Family Photos 2024"
📸 Images found: 847
💾 Total size: 2.3 GB

🚀 Upload Method: Folder Upload

📤 Uploading batch 1...
Files 1-20 of 847
Progress: 847/847 files (100%)

✓ Processed 1/847: IMG_001.jpg
✓ Processed 2/847: IMG_002.jpg
✓ Processed 3/847: IMG_003.jpg
...

🎭 Processing images with face detection...
⚡ Found faces in 623 photos
🎯 Applied intelligent cropping to all images

✅ Slideshow generated successfully!
🔗 Secure URL: https://yourserver.com/s/Abc123...
```

## Slideshow Features

- **Full-Screen Display**: Optimized for any screen resolution
- **Automatic Image Rotation**: Random image every 60 seconds
- **Live Clock**: Updates every second with current time
- **Weather Display**: Shows current temperature and weather icon
- **Responsive Design**: Works on any device or screen size
- **Error Handling**: Graceful fallback if weather API is unavailable
- **Secure URLs**: Each slideshow gets a unique 256-character access URL
- **Multi-device Access**: Share slideshows across multiple devices simultaneously

## Browser Support

The folder upload feature requires a modern browser that supports:
- **HTML5 File API**: For folder selection (`webkitdirectory`)
- **Drag & Drop API**: For drag-and-drop functionality
- **Fetch API**: For batch file uploads

**Supported Browsers:**
- ✅ Chrome/Chromium (all versions)
- ✅ Firefox (50+)
- ✅ Safari (11.1+)
- ✅ Edge (79+)

*ZIP upload works in all browsers as fallback option*