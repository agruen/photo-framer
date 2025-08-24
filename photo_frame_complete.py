#!/usr/bin/env python3
"""
Complete Photo Frame Slideshow Generator

This script combines face-aware photo cropping with HTML slideshow generation.
It processes images with face detection and creates a digital photo frame slideshow
with weather integration and clock display.

Features:
- Face-aware intelligent cropping to preserve faces
- Multiprocessing for fast batch operations  
- HTML slideshow with weather and time display
- Configurable screen resolution targeting
- OpenWeatherMap API integration for weather data

Usage:
    python photo_frame_complete.py --input photos --output slideshow --zip 10001 --api-key YOUR_KEY
"""

import os
import sys
import argparse
import json
from pathlib import Path
from face_crop_tool import FaceAwareCropper
from slideshow_generator import generate_slideshow_html


def process_photos_and_create_slideshow(input_folder, output_folder, screen_width=1280, screen_height=800, 
                                       zip_code="10001", api_key="YOUR_API_KEY_HERE", max_workers=None):
    """
    Complete pipeline: crop photos with face detection and generate slideshow.
    
    Args:
        input_folder (str): Path to folder containing source images
        output_folder (str): Path to folder where processed images and slideshow will be saved
        screen_width (int): Target screen width in pixels
        screen_height (int): Target screen height in pixels
        zip_code (str): ZIP code for weather data
        api_key (str): OpenWeatherMap API key
        max_workers (int): Number of processes for parallel processing
        
    Returns:
        tuple: (success_count, html_path) - number of successfully processed images and path to slideshow
    """
    
    print(f"🖼️  Complete Photo Frame Pipeline")
    print(f"=================================")
    print(f"Input folder:    {input_folder}")
    print(f"Output folder:   {output_folder}")
    print(f"Screen size:     {screen_width}x{screen_height}")
    print(f"Weather ZIP:     {zip_code}")
    print(f"Weather API:     {'Enabled' if api_key != 'YOUR_API_KEY_HERE' else 'Disabled'}")
    print()
    
    # Validate input folder
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist")
        return 0, None
    
    # Count input images
    input_path = Path(input_folder)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    input_images = []
    for ext in supported_extensions:
        input_images.extend(input_path.glob(f'*{ext}'))
        input_images.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not input_images:
        print(f"❌ No image files found in '{input_folder}'")
        return 0, None
    
    print(f"📸 Found {len(input_images)} images to process")
    print()
    
    # Step 1: Process images with face-aware cropping
    print("🎯 Step 1: Processing images with face-aware cropping...")
    cropper = FaceAwareCropper()
    
    # Use the existing process_folder method for consistent results
    cropper.process_folder(input_folder, output_folder, screen_width, screen_height, max_workers)
    
    # Count processed images
    output_path = Path(output_folder)
    processed_images = []
    for ext in supported_extensions:
        processed_images.extend(output_path.glob(f'*{ext}'))
        processed_images.extend(output_path.glob(f'*{ext.upper()}'))
    
    if not processed_images:
        print(f"❌ No images were successfully processed")
        return 0, None
    
    success_count = len(processed_images)
    print(f"✅ Successfully processed {success_count} images")
    print()
    
    # Step 2: Generate slideshow HTML
    print("🎭 Step 2: Generating slideshow with weather integration...")
    
    # Get just the filenames for the slideshow
    processed_filenames = [img.name for img in processed_images]
    
    html_path = generate_slideshow_html(
        processed_filenames, output_folder, zip_code, api_key, screen_width, screen_height
    )
    
    print(f"✅ Slideshow generated: {html_path}")
    print()
    
    return success_count, html_path


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Complete photo frame slideshow generator with face-aware cropping')
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input folder containing photos')
    parser.add_argument('--output', '-o', required=True,
                       help='Output folder for processed photos and slideshow')
    
    # Screen dimensions
    parser.add_argument('--width', '-w', type=int, default=1280,
                       help='Target screen width (default: 1280)')
    parser.add_argument('--height', '-ht', type=int, default=800,
                       help='Target screen height (default: 800)')
    
    # Weather integration
    parser.add_argument('--zip', '-z', default="10001",
                       help='ZIP code for weather data (default: 10001)')
    parser.add_argument('--api-key', '-k', default="YOUR_API_KEY_HERE",
                       help='OpenWeatherMap API key (get free at openweathermap.org/api)')
    
    # Processing options
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='Number of processes to use (default: auto-detect based on CPU cores)')
    
    args = parser.parse_args()
    
    # Validate required inputs
    if not os.path.exists(args.input):
        print(f"❌ Error: Input folder '{args.input}' does not exist")
        print(f"Please create the folder and add your photos, then run this script again")
        sys.exit(1)
    
    # Process photos and create slideshow
    success_count, html_path = process_photos_and_create_slideshow(
        args.input, args.output, args.width, args.height, 
        args.zip, args.api_key, args.processes
    )
    
    if success_count > 0 and html_path:
        print("🎉 Complete! Your photo slideshow is ready!")
        print("=" * 50)
        print()
        print(f"📁 Processed images: {args.output}/")
        print(f"🌐 Slideshow HTML:   {html_path}")
        print()
        print("📖 Instructions:")
        print("1. Open slideshow.html in a web browser")
        print("2. Press F11 for full-screen mode")
        print("3. Images will rotate every 60 seconds")
        print("4. Clock and weather update automatically")
        print()
        if args.api_key == "YOUR_API_KEY_HERE":
            print("💡 To enable weather:")
            print("   - Get a free API key from openweathermap.org/api")
            print("   - Run this script again with --api-key YOUR_KEY")
        print()
        print("✨ Enjoy your digital photo frame!")
    else:
        print("❌ Failed to create slideshow")
        sys.exit(1)


if __name__ == '__main__':
    import multiprocessing as mp
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()