# Photo Frame Slideshow Server

A complete, dockerized solution for creating beautiful digital photo frame slideshows. Features **Smart Body-Aware Cropping (V2)**, weather integration, and a modern web interface.

## ✨ Key Features

*   **Smart Body-Aware Cropping (V2)**:
    *   **Face Safety**: Uses MediaPipe to strictly ensure faces are *never* cut off.
    *   **Max Context**: Preserves the maximum possible background context.
    *   **Pose Optimization**: Intelligently shifts the crop to include as much of the person's body/pose as possible (e.g., arms, torso) without sacrificing face visibility.
*   **Dockerized**: Runs anywhere (Mac, Linux, Raspberry Pi/ARM64).
*   **Web Interface**:
    *   Upload entire folders or ZIP files.
    *   Real-time progress tracking.
    *   Manage multiple slideshows.
*   **Digital Frame Mode**:
    *   Full-screen responsive slideshow.
    *   Live clock and weather (OpenWeatherMap).
    *   Auto-rotation (60s).

---

## 🚀 Quick Start (Docker)

The easiest way to run the server is with Docker.

### 1. Start the Server
```bash
git clone https://github.com/agruen/photo-framer.git
cd photo-framer
docker compose up -d
```

### 2. Setup
1.  Open **http://localhost:8012** in your browser.
2.  Follow the setup wizard (create admin password, set resolution).
3.  Upload photos via the web interface.

---

## 🛠️ Standalone Tools

You can use the intelligent cropping logic without running the full web server.

### Verification Tool (`verify_crop.py`)
Visualize how the cropper makes decisions. This draws the Face (Red), Body (Blue), and Final Crop (Yellow) on an image.

**Run via Docker (Recommended):**
```bash
# Build the image first
docker build -t photo-framer .

# Run verification on a local image
docker run --rm \
  -v "$(pwd):/app" \
  -v "/path/to/your/photos:/photos" \
  photo-framer \
  python3 verify_crop.py /photos/input.jpg /photos/debug_output.jpg
```

### Batch Cropper (`face_crop_tool.py`)
Process a folder of images via command line.

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -v "/path/to/input:/input" \
  -v "/path/to/output:/output" \
  photo-framer \
  python3 face_crop_tool.py --input /input --output /output --width 1280 --height 800
```

---

## 🧠 How It Works (Smart Crop V2)

The cropping engine uses a hierarchy of needs to determine the perfect crop:

1.  **Maximize Context**: It starts with the largest possible crop that fits the target aspect ratio (e.g., 16:10).
2.  **Face Safety Constraint**: It calculates a "Valid Range" where the crop *must* exist to keep all faces fully visible.
3.  **Pose Optimization**: Within that valid range, it slides the crop window to maximize overlap with the detected body (Pose).
    *   *Horizontal*: Centers on the body to capture width/arms.
    *   *Vertical*: Aligns with the top of the body to capture head/torso/legs.

If a face cannot fit (e.g., extreme close-up vs landscape target), it falls back to a high-quality blurred background composite.

---

## 💻 Development

### Requirements
*   Docker & Docker Compose
*   Python 3.11+ (if running locally without Docker)

### Local Setup (No Docker)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-server.txt
python -m app.app
```