#!/usr/bin/env python3
"""
Setup script to install required dependencies for the face-aware photo cropper.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages."""
    print("Installing required dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    print("Face-Aware Photo Cropper Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nUsage:")
    print("  python face_crop_tool.py                    # Process 'photos' folder")
    print("  python face_crop_tool.py --input my_photos  # Process custom folder")
    print("  python face_crop_tool.py --width 1920 --height 1080  # Custom dimensions")

if __name__ == '__main__':
    main()