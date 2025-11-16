#!/usr/bin/env python3
"""
Face-aware photo cropper that crops images to 1280x800 while ensuring faces are preserved.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import argparse
import sys
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import sys
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import dlib
import requests
import gzip
import shutil

def process_single_image_worker(args):
    """Standalone function for multiprocessing - processes a single image."""
    image_file, output_file, target_width, target_height = args
    
    # Create a fresh FaceAwareCropper instance for this process
    cropper = FaceAwareCropper()
    success = cropper.crop_image(str(image_file), str(output_file), target_width, target_height, verbose=False)
    return success, str(image_file.name)


class FaceAwareCropper:
    def __init__(self):
        """
        Initializes the FaceAwareCropper, loading dlib models and downloading them if necessary.
        """
        self.model_filename = "shape_predictor_68_face_landmarks.dat"
        
        # Download the dlib model if it doesn't exist
        if not os.path.exists(self.model_filename):
            print(f"Dlib landmark model '{self.model_filename}' not found.")
            self._download_dlib_model()

        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.model_filename)
            print("✓ Dlib face and landmark detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing dlib detectors: {e}")
            print("Please ensure 'shape_predictor_68_face_landmarks.dat' is in the correct directory.")
            sys.exit(1)

    def _download_dlib_model(self):
        """Downloads and extracts the dlib facial landmark model."""
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_filename = self.model_filename + ".bz2"
        
        print(f"Downloading {model_url}...")
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(bz2_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print("Download complete. Decompressing...")
            # dlib model is compressed with bz2, not gzip
            import bz2
            with bz2.BZ2File(bz2_filename, 'rb') as f_in:
                with open(self.model_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Clean up the compressed file
            os.remove(bz2_filename)
            print("✓ Dlib model decompressed and ready.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading dlib model: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error decompressing dlib model: {e}")
            sys.exit(1)

    def detect_faces(self, image):
        """
        Detects faces and their 68-point landmarks using dlib.
        Returns a list of dlib shape objects.
        """
        # Dlib works with grayscale or RGB images.
        # Using grayscale is slightly faster for detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face rectangles
        face_rects = self.detector(gray, 1)
        
        if not face_rects:
            return []

        # For each detected face, find the landmarks.
        shapes = []
        for rect in face_rects:
            shape = self.predictor(gray, rect)
            shapes.append(shape)
            
        return shapes

    def _get_roi_from_landmarks(self, shapes):
        """
        Calculates a single Region of Interest bounding box that encompasses all
        facial landmarks from a list of shapes.
        """
        if not shapes:
            return None
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for shape in shapes:
            for i in range(shape.num_parts):
                p = shape.part(i)
                min_x = min(min_x, p.x)
                min_y = min(min_y, p.y)
                max_x = max(max_x, p.x)
                max_y = max(max_y, p.y)
        
        return (min_x, min_y, max_x, max_y)

    def calculate_smart_crop(self, image_width, image_height, shapes, target_width=1280, target_height=800):
        """
        Calculates the optimal crop box based on facial landmarks.
        """
        target_ratio = target_width / target_height

        if not shapes:
            return self._calculate_content_aware_crop(image_width, image_height, target_width, target_height)

        # --- Stage 1 & 2: Determine Padded ROI from Landmarks ---
        roi = self._get_roi_from_landmarks(shapes)
        roi_width = roi[2] - roi[0]
        roi_height = roi[3] - roi[1]

        # Get specific landmarks for more precise calculations
        # For simplicity, we'll use the overall landmark ROI, but this could be refined
        # e.g., using chin (point 8) and eyebrows (points 19, 24)
        
        # More robust padding based on landmark ROI
        padding_top = roi_height * 0.7
        padding_bottom = roi_height * 0.4
        padding_horizontal = roi_width * 0.5

        padded_roi = {
            'x': roi[0] - padding_horizontal,
            'y': roi[1] - padding_top,
            'width': roi_width + (2 * padding_horizontal),
            'height': roi_height + padding_top + padding_bottom
        }

        # --- Stage 3: Calculate Ideal Final Crop Dimensions ---
        padded_roi_ratio = padded_roi['width'] / padded_roi['height']

        if padded_roi_ratio > target_ratio:
            ideal_crop_width = padded_roi['width']
            ideal_crop_height = ideal_crop_width / target_ratio
        else:
            ideal_crop_height = padded_roi['height']
            ideal_crop_width = ideal_crop_height * target_ratio

        # --- Stage 3.5: The "Maximal Crop" Strategy ---
        if ideal_crop_width > image_width or ideal_crop_height > image_height:
            if image_width / image_height > target_ratio:
                final_crop_height = image_height
                final_crop_width = final_crop_height * target_ratio
            else:
                final_crop_width = image_width
                final_crop_height = final_crop_width / target_ratio
        else:
            final_crop_width = ideal_crop_width
            final_crop_height = ideal_crop_height

        # --- Stage 4: Position the Crop Box ---
        # Get eye landmarks for Rule of Thirds positioning
        # Left eye: 36-41, Right eye: 42-47
        eye_landmarks = []
        for shape in shapes:
            for i in range(36, 48):
                eye_landmarks.append(shape.part(i))
        
        if eye_landmarks:
            eye_center_y = sum([p.y for p in eye_landmarks]) / len(eye_landmarks)
        else: # Fallback to ROI center if eyes not in landmarks (should not happen with dlib)
            eye_center_y = roi[1] + (roi_height / 2)

        ideal_crop_y = eye_center_y - (final_crop_height / 3)
        
        roi_center_x = roi[0] + (roi_width / 2)
        ideal_crop_x = roi_center_x - (final_crop_width / 2)

        # Boundary correction
        crop_x = max(0, ideal_crop_x)
        crop_y = max(0, ideal_crop_y)

        if crop_x + final_crop_width > image_width:
            crop_x = image_width - final_crop_width
        if crop_y + final_crop_height > image_height:
            crop_y = image_height - final_crop_height

        return (int(crop_x), int(crop_y), int(crop_x + final_crop_width), int(crop_y + final_crop_height))

    def _calculate_content_aware_crop(self, image_width, image_height, crop_width, crop_height):
        """Calculate crop for images without faces using content analysis."""
        target_ratio = crop_width / crop_height
        
        if image_width / image_height > target_ratio:
            crop_height = image_height
            crop_width = int(crop_height * target_ratio)
        else:
            crop_width = image_width
            crop_height = int(crop_width / target_ratio)

        crop_x = (image_width - crop_width) / 2
        crop_y = (image_height - crop_height) * 0.45
        
        return (int(crop_x), int(crop_y), int(crop_x + crop_width), int(crop_y + crop_height))
    
    def _enhance_image_quality(self, image, original_size, final_size):
        """Apply adaptive image enhancement based on scaling and quality."""
        from PIL import ImageEnhance, ImageFilter
        
        original_width, original_height = original_size
        final_width, final_height = final_size
        
        width_scale = final_width / original_width if original_width > 0 else 0
        height_scale = final_height / original_height if original_height > 0 else 0
        min_scale = min(width_scale, height_scale)
        
        enhanced = image
        
        if min_scale > 1.2:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.15)
        elif min_scale < 0.5:
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.05)
        
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.03)
        
        return enhanced
    
    def _create_blurred_background_composite(self, pil_image, target_width, target_height, verbose=False):
        """Creates a composite with a blurred background."""
        if verbose: print("Creating blurred background composite.")
        
        original_width, original_height = pil_image.size
        
        bg_image = pil_image.resize((target_width, int(original_height * (target_width / original_width))), Image.Resampling.LANCZOS)
        bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=25))
        enhancer = ImageEnhance.Brightness(bg_image)
        bg_image = enhancer.enhance(0.7)
        
        left = (bg_image.width - target_width) / 2
        top = (bg_image.height - target_height) / 2
        bg_image = bg_image.crop((left, top, left + target_width, top + target_height))

        fg_height = int(target_height * 0.9)
        fg_width = int(original_width * (fg_height / original_height))

        if fg_width > int(target_width * 0.95):
            fg_width = int(target_width * 0.95)
            fg_height = int(original_height * (fg_width / original_width))

        foreground_image = pil_image.resize((fg_width, fg_height), Image.Resampling.LANCZOS)

        final_image = bg_image
        paste_x = (target_width - fg_width) // 2
        paste_y = (target_height - fg_height) // 2
        final_image.paste(foreground_image, (paste_x, paste_y))
        
        return final_image

    def crop_image(self, input_path, output_path, target_width=1280, target_height=800, verbose=True):
        """Crops or composites an image using dlib landmark-based detection."""
        try:
            pil_image = Image.open(input_path).convert('RGB')
            cv_image = np.array(pil_image)
            cv_image = cv_image[:, :, ::-1].copy()

            original_width, original_height = pil_image.size
            target_ratio = target_width / target_height
            original_ratio = original_width / original_height
            
            shapes = self.detect_faces(cv_image)
            if verbose: print(f"Detected {len(shapes)} face(s) in {input_path} using dlib.")

            use_composite = False
            crop_rect = self.calculate_smart_crop(original_width, original_height, shapes, target_width, target_height)

            is_portrait_on_landscape = original_ratio < 1.0 and target_ratio > 1.0
            if is_portrait_on_landscape and shapes:
                crop_area = (crop_rect[2] - crop_rect[0]) * (crop_rect[3] - crop_rect[1])
                original_area = original_width * original_height
                
                if original_area > 0 and ((original_area - crop_area) / original_area) > 0.60:
                    use_composite = True
                    if verbose: print("Extreme crop detected. Switching to blurred background composite.")

            if use_composite:
                final_image = self._create_blurred_background_composite(pil_image, target_width, target_height, verbose)
            else:
                if verbose: print("Standard crop processing.")
                cropped = pil_image.crop(crop_rect)
                final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

            final_image = self._enhance_image_quality(final_image, final_image.size, (target_width, target_height))
            
            save_kwargs = {'quality': 95}
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                save_kwargs.update({'optimize': True, 'progressive': True})
            
            final_image.save(output_path, **save_kwargs)
            if verbose: print(f"Saved final image to {output_path}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error processing {input_path}: {e}")
            return False
    
    def process_folder(self, input_folder, output_folder, target_width=1280, target_height=800, max_workers=None):
        """Process all images in a folder with multiprocessing support."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("No image files found in the input folder")
            return
        
        # Determine optimal number of processes for ProcessPoolExecutor
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # For CPU-intensive tasks like image processing, use fewer processes
            # to allow OpenCV internal threading to work efficiently
            max_workers = max(2, min(4, cpu_count // 2))
        
        print(f"Found {len(image_files)} image files to process")
        print(f"Using {max_workers} processes for parallel processing")
        print(f"Each process can use multiple cores via OpenCV's internal threading")
        
        # Prepare arguments for parallel processing
        processing_args = []
        for image_file in image_files:
            output_file = output_path / f"{image_file.stem}_cropped{image_file.suffix}"
            processing_args.append((image_file, output_file, target_width, target_height))
        
        # Progress tracking variables
        completed = 0
        success_count = 0
        failed_files = []
        
        start_time = time.time()
        
        # Process images in parallel using separate processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_filename = {
                executor.submit(process_single_image_worker, args): args[0].name 
                for args in processing_args
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    success, processed_filename = future.result()
                    completed += 1
                    
                    if success:
                        success_count += 1
                    else:
                        failed_files.append(processed_filename)
                    
                    # Print progress every 10 images or for the last image
                    if completed % max(1, min(10, len(image_files) // 10)) == 0 or completed == len(image_files):
                        percent = (completed / len(image_files)) * 100
                        print(f"\rProgress: {completed}/{len(image_files)} ({percent:.1f}%) - "
                              f"Success: {success_count}, Failed: {len(failed_files)}", end="", flush=True)
                        
                except Exception as e:
                    completed += 1
                    failed_files.append(filename)
                    print(f"\nError processing {filename}: {str(e)}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n\nProcessing complete!")
        print(f"Total time: {processing_time:.1f} seconds")
        print(f"Average time per image: {processing_time/len(image_files):.2f} seconds")
        print(f"Successfully processed: {success_count}/{len(image_files)} images")
        
        if failed_files:
            print(f"Failed files ({len(failed_files)}):")
            for filename in failed_files[:10]:  # Show first 10 failed files
                print(f"  - {filename}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
    
    def process_folder_single_threaded(self, input_folder, output_folder, target_width=1280, target_height=800):
        """Process all images in a folder (single-threaded version for comparison)."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("No image files found in the input folder")
            return
        
        print(f"Found {len(image_files)} image files to process (single-threaded)")
        
        start_time = time.time()
        success_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\r[{i}/{len(image_files)}] Processing {image_file.name}...", end="", flush=True)
            
            output_file = output_path / f"{image_file.stem}_cropped{image_file.suffix}"
            
            if self.crop_image(str(image_file), str(output_file), target_width, target_height):
                success_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n\nSingle-threaded processing complete!")
        print(f"Total time: {processing_time:.1f} seconds")
        print(f"Average time per image: {processing_time/len(image_files):.2f} seconds")
        print(f"Successfully processed: {success_count}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description='Face-aware photo cropper')
    parser.add_argument('--input', '-i', default='photos', 
                       help='Input folder containing photos (default: photos)')
    parser.add_argument('--output', '-o', default='output',
                       help='Output folder for cropped photos (default: output)')
    parser.add_argument('--width', '-w', type=int, default=1280,
                       help='Target width (default: 1280)')
    parser.add_argument('--height', '-ht', type=int, default=800,
                       help='Target height (default: 800)')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='Number of processes to use (default: auto-detect based on CPU cores)')
    parser.add_argument('--single-threaded', action='store_true',
                       help='Force single-threaded processing for comparison')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run both single-threaded and multi-process for comparison')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        sys.exit(1)
    
    cropper = FaceAwareCropper()
    
    if args.benchmark:
        print("🔥 BENCHMARK MODE: Running both single-threaded and multi-process")
        print("=" * 60)
        
        # Single-threaded benchmark
        print("\n🐌 Single-threaded processing:")
        cropper.process_folder_single_threaded(args.input, f"{args.output}_single", args.width, args.height)
        
        print("\n" + "=" * 60)
        
        # Multi-process benchmark
        print("\n🚀 Multi-process processing:")
        cropper.process_folder(args.input, f"{args.output}_multi", args.width, args.height, args.processes)
        
    elif args.single_threaded:
        print("🐌 Single-threaded processing mode")
        cropper.process_folder_single_threaded(args.input, args.output, args.width, args.height)
    else:
        print("🚀 Multi-process processing mode")
        cropper.process_folder(args.input, args.output, args.width, args.height, args.processes)


if __name__ == '__main__':
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()