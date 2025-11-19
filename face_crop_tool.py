#!/usr/bin/env python3
"""
Face-aware photo cropper that crops images to 1280x800 while ensuring faces are preserved.
Uses MediaPipe for state-of-the-art face detection and a smart "Safe Zone" cropping strategy.
"""

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
import mediapipe as mp_face

# Suppress MediaPipe logging
os.environ['GLOG_minloglevel'] = '2'

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
        Initializes the FaceAwareCropper using MediaPipe.
        """
        self.mp_face_detection = mp_face.solutions.face_detection
        self.mp_pose = mp_face.solutions.pose
        
        # Face Detector
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.75
        )
        
        # Pose Detector (for Body)
        # static_image_mode=True is important for photos
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1, # 0=Lite, 1=Full, 2=Heavy
            enable_segmentation=False,
            min_detection_confidence=0.75
        )

    def detect_faces(self, image):
        """
        Detects faces using MediaPipe.
        Returns a list of bounding boxes (relative 0-1 coordinates).
        """
        results = self.face_detector.process(image)
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                faces.append(bbox)
        return faces

    def detect_people(self, image):
        """
        Detects people (bodies) using MediaPipe Pose.
        Returns a list of bounding boxes (relative 0-1 coordinates) for detected bodies.
        Note: Pose only detects the most prominent person.
        """
        results = self.pose_detector.process(image)
        people = []
        
        if results.pose_landmarks:
            # Calculate bounding box from landmarks
            landmarks = results.pose_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Create a pseudo-bbox object with xmin, ymin, width, height
            class BBox:
                def __init__(self, x, y, w, h):
                    self.xmin = x
                    self.ymin = y
                    self.width = w
                    self.height = h
            
            people.append(BBox(min_x, min_y, max_x - min_x, max_y - min_y))
            
        return people

    def _get_safe_zones(self, faces, people, image_width, image_height):
        """
        Calculates two zones:
        1. Critical Zone: Faces + Padding (MUST be included)
        2. Preferred Zone: Bodies + Faces (Should be included if possible)
        """
        if not faces and not people:
            return None, None

        # --- 1. Calculate Critical Zone (Faces) ---
        c_min_x, c_min_y = 1.0, 1.0
        c_max_x, c_max_y = 0.0, 0.0
        
        has_faces = False
        if faces:
            has_faces = True
            for face in faces:
                c_min_x = min(c_min_x, face.xmin)
                c_min_y = min(c_min_y, face.ymin)
                c_max_x = max(c_max_x, face.xmin + face.width)
                c_max_y = max(c_max_y, face.ymin + face.height)
        
        # --- 2. Calculate Preferred Zone (Bodies + Faces) ---
        p_min_x, p_min_y = c_min_x, c_min_y
        p_max_x, p_max_y = c_max_x, c_max_y
        
        # If we have people, expand Preferred Zone
        if people:
            for person in people:
                is_relevant = False
                if has_faces:
                    # Check overlap with any face
                    p_x1, p_y1 = person.xmin, person.ymin
                    p_x2, p_y2 = person.xmin + person.width, person.ymin + person.height
                    
                    for face in faces:
                        fcx = face.xmin + face.width/2
                        fcy = face.ymin + face.height/2
                        if p_x1 <= fcx <= p_x2 and p_y1 <= fcy <= p_y2:
                            is_relevant = True
                            break
                else:
                    is_relevant = True
                
                if is_relevant:
                    p_min_x = min(p_min_x, person.xmin)
                    p_min_y = min(p_min_y, person.ymin)
                    p_max_x = max(p_max_x, person.xmin + person.width)
                    p_max_y = max(p_max_y, person.ymin + person.height)
                    
                    # If no faces, Critical Zone defaults to the "Head" area of the person
                    if not has_faces:
                        # Estimate head as top 15% of body?
                        # Or just use the top part of the body as critical
                        c_min_x = min(c_min_x, person.xmin)
                        c_min_y = min(c_min_y, person.ymin)
                        c_max_x = max(c_max_x, person.xmin + person.width)
                        c_max_y = max(c_max_y, person.ymin + person.height * 0.2) # Top 20% is critical

        # Convert to pixels
        def to_rect(min_x, min_y, max_x, max_y, pad_x_pct, pad_y_top_pct, pad_y_bot_pct):
            x1 = int(min_x * image_width)
            y1 = int(min_y * image_height)
            x2 = int(max_x * image_width)
            y2 = int(max_y * image_height)
            w = x2 - x1
            h = y2 - y1
            
            px = int(w * pad_x_pct)
            py_top = int(h * pad_y_top_pct)
            py_bot = int(h * pad_y_bot_pct)
            
            return (
                max(0, x1 - px),
                max(0, y1 - py_top),
                min(image_width, x2 + px),
                min(image_height, y2 + py_bot)
            )

        # Critical Zone Padding (Tight)
        critical_rect = to_rect(c_min_x, c_min_y, c_max_x, c_max_y, 0.2, 0.5, 0.1)
        
        # Preferred Zone Padding (Loose)
        preferred_rect = to_rect(p_min_x, p_min_y, p_max_x, p_max_y, 0.1, 0.2, 0.1)
        
        return critical_rect, preferred_rect

    def _create_blurred_background_composite(self, pil_image, target_width, target_height, safe_zone=None):
        """
        Creates a composite with a blurred background.
        If safe_zone is provided, it ensures the safe zone is fully visible in the foreground.
        """
        original_width, original_height = pil_image.size
        
        # 1. Create Background
        # Resize original to fill the target frame (cropping edges)
        bg_scale = max(target_width / original_width, target_height / original_height)
        bg_w = int(original_width * bg_scale)
        bg_h = int(original_height * bg_scale)
        
        bg_image = pil_image.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=30))
        enhancer = ImageEnhance.Brightness(bg_image)
        bg_image = enhancer.enhance(0.6) # Darken background slightly
        
        # Center crop the background to target size
        left = (bg_w - target_width) // 2
        top = (bg_h - target_height) // 2
        bg_image = bg_image.crop((left, top, left + target_width, top + target_height))
        
        # 2. Prepare Foreground
        # We want to fit the ENTIRE image (or safe zone) into the target frame
        # preserving aspect ratio, with no cropping of the important area.
        
        if safe_zone:
            # If we have a safe zone, we might want to crop the source image slightly 
            # to remove irrelevant edges, but ONLY if it helps the composition.
            # For now, let's keep it simple: Show the WHOLE original image 
            # (or a large crop of it) scaled to fit.
            pass

        # Scale original image to fit WITHIN target dimensions (Letterbox/Pillarbox style)
        scale = min(target_width / original_width, target_height / original_height)
        # Make it slightly smaller (95%) to have a nice border effect? 
        # Or 100% to touch edges? Let's do 100% to maximize size.
        fg_w = int(original_width * scale)
        fg_h = int(original_height * scale)
        
        foreground = pil_image.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
        
        # 3. Composite
        final_image = bg_image.copy()
        paste_x = (target_width - fg_w) // 2
        paste_y = (target_height - fg_h) // 2
        
        # Add a subtle drop shadow to foreground?
        # (Skipping for performance/simplicity, but would look nice)
        final_image.paste(foreground, (paste_x, paste_y))
        
        return final_image

    def _get_optimal_crop_1d(self, image_len, crop_len, face_min, face_max, pose_min, pose_max, axis='x'):
        """
        Calculates the optimal 1D crop start position.
        Constraints:
        1. Must be within [0, image_len - crop_len] (Image bounds)
        2. Must include [face_min, face_max] (Face safety)
        
        Optimization:
        1. Maximize overlap with [pose_min, pose_max]
        2. Tie-breaker:
           - 'x': Center on pose
           - 'y': Align with top of pose (Head/Torso preference)
        """
        # 1. Determine Valid Range (Constraints)
        # Crop must start at or before face_min
        # Crop must end at or after face_max (start >= face_max - crop_len)
        valid_start_max = min(image_len - crop_len, face_min)
        valid_start_min = max(0, face_max - crop_len)
        
        if valid_start_min > valid_start_max:
            # Should not happen if we checked dimensions beforehand, but as a fallback:
            # Center on face
            return max(0, min(image_len - crop_len, face_min + (face_max - face_min)//2 - crop_len//2))

        # 2. Determine Optimal Range (Maximize Overlap with Pose)
        pose_len = pose_max - pose_min
        
        if crop_len >= pose_len:
            # Crop is larger than Pose: We can fully include the pose.
            # Any start position in [pose_max - crop_len, pose_min] covers the pose.
            opt_min = pose_max - crop_len
            opt_max = pose_min
        else:
            # Crop is smaller than Pose: We can only include part of the pose.
            # Any start position in [pose_min, pose_max - crop_len] is fully inside the pose (max overlap = crop_len).
            opt_min = pose_min
            opt_max = pose_max - crop_len
            
        # 3. Tie-Breaker (Pick best spot within Optimal Range)
        if axis == 'x':
            # Center: Pick the middle of the optimal range
            preferred_start = (opt_min + opt_max) / 2
        else: # axis == 'y'
            # Top: Pick the top (smallest value) of the optimal range
            # This aligns the crop as high as possible on the body
            preferred_start = opt_min
            
        # 4. Apply Constraints
        final_start = max(valid_start_min, min(valid_start_max, preferred_start))
        
        return int(final_start)

    def calculate_smart_crop(self, image_width, image_height, critical_zone, preferred_zone, target_width=1280, target_height=800):
        """
        Determines the best crop coordinates.
        Prioritizes:
        1. Keeping Critical Zone (Faces) fully visible.
        2. Filling the frame (No Composite).
        3. Maximizing overlap with Preferred Zone (Body).
        """
        target_aspect = target_width / target_height
        image_aspect = image_width / image_height
        
        # Unpack zones
        cx1, cy1, cx2, cy2 = critical_zone
        px1, py1, px2, py2 = preferred_zone
        
        crit_w = cx2 - cx1
        crit_h = cy2 - cy1
        
        # --- Check if Composite is ABSOLUTELY necessary ---
        if image_aspect > target_aspect:
            # Image is wider than target
            max_crop_h = image_height
            max_crop_w = int(max_crop_h * target_aspect)
        else:
            # Image is taller than target
            max_crop_w = image_width
            max_crop_h = int(max_crop_w / target_aspect)
            
        # If Critical Zone doesn't fit in Max Crop, we MUST composite
        if crit_w > max_crop_w or crit_h > max_crop_h:
            return (0, 0, image_width, image_height), True

        # --- Calculate Optimal Crop ---
        # We use the largest possible crop to maximize context
        crop_w = max_crop_w
        crop_h = max_crop_h
        
        # Calculate X and Y positions independently using the 1D optimizer
        crop_x = self._get_optimal_crop_1d(image_width, crop_w, cx1, cx2, px1, px2, axis='x')
        crop_y = self._get_optimal_crop_1d(image_height, crop_h, cy1, cy2, py1, py2, axis='y')
        
        return (crop_x, crop_y, crop_w, crop_h), False

    def _calculate_content_aware_crop(self, image_width, image_height, target_width, target_height):
        """Fallback for no faces."""
        target_ratio = target_width / target_height
        image_ratio = image_width / image_height
        
        if image_ratio > target_ratio:
            # Image is wider than target
            crop_height = image_height
            crop_width = int(crop_height * target_ratio)
        else:
            # Image is taller than target
            crop_width = image_width
            crop_height = int(crop_width / target_ratio)
            
        # Center horizontally, but bias vertically (top 40% usually has interest)
        crop_x = (image_width - crop_width) // 2
        crop_y = int((image_height - crop_height) * 0.3) # Top-biased
        
        return (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)

    def _enhance_image_quality(self, image):
        """Apply subtle enhancement."""
        # Slight sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Slight contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        return image

    def crop_image(self, input_path, output_path, target_width=1280, target_height=800, verbose=True):
        """Main processing function."""
        try:
            pil_image = Image.open(input_path).convert('RGB')
            # Convert to numpy for MediaPipe
            cv_image = np.array(pil_image)
            
            original_width, original_height = pil_image.size
            
            # Detect Faces & People
            faces = self.detect_faces(cv_image)
            people = self.detect_people(cv_image)
            
            if verbose: print(f"Detected {len(faces)} face(s) and {len(people)} person(s) in {Path(input_path).name}")
            
            # Calculate Safe Zones
            critical_zone, preferred_zone = self._get_safe_zones(faces, people, original_width, original_height)
            
            # Determine Strategy
            if critical_zone is None:
                # Fallback if nothing detected
                crop_rect, use_composite = self.calculate_smart_crop(
                    original_width, original_height, (0,0,0,0), (0,0,0,0), target_width, target_height
                )
            else:
                crop_rect, use_composite = self.calculate_smart_crop(
                    original_width, original_height, critical_zone, preferred_zone, target_width, target_height
                )
            
            if use_composite:
                if verbose: print("  -> Using Blurred Background Composite (Faces don't fit)")
                final_image = self._create_blurred_background_composite(pil_image, target_width, target_height, safe_zone)
            else:
                if verbose: print("  -> Using Smart Crop")
                cropped = pil_image.crop(crop_rect)
                final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Final Polish
            final_image = self._enhance_image_quality(final_image)
            
            # Save
            save_kwargs = {'quality': 95}
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                save_kwargs.update({'optimize': True, 'progressive': True})
            
            final_image.save(output_path, **save_kwargs)
            return True
            
        except Exception as e:
            if verbose: print(f"Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_folder(self, input_folder, output_folder, target_width=1280, target_height=800, max_workers=None):
        """Process all images in a folder with multiprocessing."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in supported_extensions]
        
        if not image_files:
            print("No image files found.")
            return

        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
            
        print(f"Processing {len(image_files)} images with {max_workers} workers...")
        
        success_count = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_image_worker, (f, output_path / f"{f.stem}_cropped{f.suffix}", target_width, target_height))
                for f in image_files
            ]
            
            for i, future in enumerate(as_completed(futures)):
                success, name = future.result()
                if success: success_count += 1
                print(f"\rProgress: {i+1}/{len(image_files)}", end="", flush=True)
                
        print(f"\nDone! {success_count}/{len(image_files)} successful. Time: {time.time()-start_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description='MediaPipe Face-Aware Cropper')
    parser.add_argument('--input', '-i', default='photos', help='Input folder')
    parser.add_argument('--output', '-o', default='output', help='Output folder')
    parser.add_argument('--width', '-w', type=int, default=1280, help='Target width')
    parser.add_argument('--height', '-ht', type=int, default=800, help='Target height')
    parser.add_argument('--workers', '-p', type=int, default=None, help='Max workers')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input folder {args.input} not found.")
        return
        
    cropper = FaceAwareCropper()
    cropper.process_folder(args.input, args.output, args.width, args.height, args.workers)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()