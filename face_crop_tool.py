#!/usr/bin/env python3
"""
Face-aware photo cropper that crops images to target dimensions while ensuring faces are preserved.
Uses MediaPipe for face and pose detection with an aspect-ratio-first cropping strategy.

Algorithm:
1. Detect all faces in the image
2. Detect the most prominent person's pose/body
3. Build a crop window with the correct aspect ratio that maximizes context
4. If all faces cannot fit in the aspect ratio → use blurred background composite (no distortion)
5. If faces fit → optimize to include as much of the pose/body as possible
6. Crop window can pan around (not center-locked) to optimize framing
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
    """
    Crops images to target dimensions with face-aware and body-aware optimization.
    Guarantees aspect ratio preservation (no distortion).
    """

    # Configuration constants
    FACE_PADDING = 0.15  # 15% breathing room around face bounding box
    HEADROOM_PADDING_LEVELS = [0.50, 0.40, 0.30, 0.20, 0.15]  # Progressive headroom reduction to avoid composite mode
    COMPOSITE_PADDING = 0.10  # 10% breathing room for composite mode

    def __init__(self):
        """Initialize MediaPipe face and pose detectors."""
        self.mp_face_detection = mp_face.solutions.face_detection
        self.mp_pose = mp_face.solutions.pose

        # Dual Face Detectors (for better coverage)
        # Short-range model: Better for close-up selfies and children
        self.face_detector_short = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model
            min_detection_confidence=0.75
        )

        # Full-range model: Better for distant faces
        self.face_detector_full = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.75
        )

        # Pose Detector (for body detection)
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,  # Important for photos (not video)
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            enable_segmentation=False,
            min_detection_confidence=0.75
        )

    def detect_faces(self, image):
        """
        Detect all faces in the image using MediaPipe with dual models.
        Runs both short-range and full-range models for better coverage.

        Args:
            image: RGB numpy array

        Returns:
            List of face bounding boxes in relative coordinates (0-1 scale)
        """
        faces = []

        # Run short-range detector (better for close-ups and children)
        results_short = self.face_detector_short.process(image)
        if results_short.detections:
            for detection in results_short.detections:
                bbox = detection.location_data.relative_bounding_box
                faces.append(bbox)

        # Run full-range detector (better for distant faces)
        results_full = self.face_detector_full.process(image)
        if results_full.detections:
            for detection in results_full.detections:
                bbox = detection.location_data.relative_bounding_box
                # Check for duplicates (same face detected by both models)
                is_duplicate = False
                for existing_face in faces:
                    # Calculate IoU (Intersection over Union) to detect duplicates
                    x1 = max(bbox.xmin, existing_face.xmin)
                    y1 = max(bbox.ymin, existing_face.ymin)
                    x2 = min(bbox.xmin + bbox.width, existing_face.xmin + existing_face.width)
                    y2 = min(bbox.ymin + bbox.height, existing_face.ymin + existing_face.height)

                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = bbox.width * bbox.height
                        area2 = existing_face.width * existing_face.height
                        union = area1 + area2 - intersection
                        iou = intersection / union if union > 0 else 0

                        # If IoU > 50%, consider it a duplicate
                        if iou > 0.5:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    faces.append(bbox)

        return faces

    def detect_pose(self, image):
        """
        Detect the most prominent person's body/pose using MediaPipe.
        Note: MediaPipe Pose only detects one person (the most prominent).

        Args:
            image: RGB numpy array

        Returns:
            Bounding box object (with xmin, ymin, width, height) or None
        """
        results = self.pose_detector.process(image)

        if results.pose_landmarks:
            # Calculate bounding box from all pose landmarks
            landmarks = results.pose_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Create a bounding box object
            class BBox:
                def __init__(self, x, y, w, h):
                    self.xmin = x
                    self.ymin = y
                    self.width = w
                    self.height = h

            return BBox(min_x, min_y, max_x - min_x, max_y - min_y)

        return None

    def _get_combined_face_bbox(self, faces, image_width, image_height, headroom_padding=0.50):
        """
        Combine all face bounding boxes into a single bounding box with padding.

        Args:
            faces: List of face bounding boxes (relative 0-1 coordinates)
            image_width: Image width in pixels
            image_height: Image height in pixels
            headroom_padding: Extra padding above faces (0.0-1.0 as fraction of face height)

        Returns:
            Tuple (x1, y1, x2, y2) in pixels, or None if no faces
        """
        if not faces:
            return None

        # Find the bounding box that encompasses all faces
        min_x, min_y = 1.0, 1.0
        max_x, max_y = 0.0, 0.0

        for face in faces:
            min_x = min(min_x, face.xmin)
            min_y = min(min_y, face.ymin)
            max_x = max(max_x, face.xmin + face.width)
            max_y = max(max_y, face.ymin + face.height)

        # Convert to pixels
        x1 = int(min_x * image_width)
        y1 = int(min_y * image_height)
        x2 = int(max_x * image_width)
        y2 = int(max_y * image_height)

        # Add padding
        w = x2 - x1
        h = y2 - y1

        pad_x = int(w * self.FACE_PADDING)
        pad_y_bottom = int(h * self.FACE_PADDING)
        pad_y_top = int(h * (self.FACE_PADDING + headroom_padding))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y_top)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y_bottom)

        return (x1, y1, x2, y2)

    def _get_pose_bbox(self, pose, image_width, image_height):
        """
        Convert pose to pixel bounding box.

        Args:
            pose: Pose bounding box object (relative 0-1 coordinates)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple (x1, y1, x2, y2) in pixels, or None if no pose
        """
        if not pose:
            return None

        x1 = int(pose.xmin * image_width)
        y1 = int(pose.ymin * image_height)
        x2 = int((pose.xmin + pose.width) * image_width)
        y2 = int((pose.ymin + pose.height) * image_height)

        return (x1, y1, x2, y2)

    def _calculate_crop_window(self, image_width, image_height, target_width, target_height,
                                faces, pose_bbox):
        """
        Calculate the optimal crop window with exact target aspect ratio.
        Uses progressive padding reduction to avoid composite mode when possible.

        Algorithm:
        1. Determine the maximum crop window size at target aspect ratio
        2. Try progressively smaller headroom padding (50%, 40%, 30%, 20%, 15%)
        3. Return first padding level where faces fit
        4. If NO padding level works → return None (triggers composite mode)
        5. If faces fit → optimize crop position to maximize pose inclusion

        Args:
            image_width, image_height: Original image dimensions
            target_width, target_height: Desired output dimensions
            faces: List of face bounding boxes (relative 0-1 coordinates)
            pose_bbox: Pose bounding box (x1, y1, x2, y2) in pixels, or None

        Returns:
            Tuple (crop_x, crop_y, crop_width, crop_height, headroom_used) or None if faces don't fit
        """
        target_aspect = target_width / target_height
        image_aspect = image_width / image_height

        # Step 1: Calculate maximum crop window size at target aspect ratio
        if image_aspect > target_aspect:
            # Image is wider than target → height-constrained
            crop_height = image_height
            crop_width = int(crop_height * target_aspect)
        else:
            # Image is taller than target → width-constrained
            crop_width = image_width
            crop_height = int(crop_width / target_aspect)

        # Step 2: Try progressively smaller headroom levels
        for headroom in self.HEADROOM_PADDING_LEVELS:
            face_bbox = self._get_combined_face_bbox(faces, image_width, image_height, headroom)

            if not face_bbox:
                # No faces, return None to handle elsewhere
                return None

            fx1, fy1, fx2, fy2 = face_bbox
            face_width = fx2 - fx1
            face_height = fy2 - fy1

            # Check if faces fit with this headroom level
            if face_width <= crop_width and face_height <= crop_height:
                # Success! Faces fit with this headroom level
                # Step 3: Optimize crop position
                optimize_bbox = pose_bbox if pose_bbox else face_bbox

                crop_x = self._optimize_position_1d(
                    image_width, crop_width, face_bbox[0], face_bbox[2],
                    optimize_bbox[0], optimize_bbox[2], axis='x'
                )
                crop_y = self._optimize_position_1d(
                    image_height, crop_height, face_bbox[1], face_bbox[3],
                    optimize_bbox[1], optimize_bbox[3], axis='y'
                )

                return (crop_x, crop_y, crop_width, crop_height, headroom)

        # Step 4: No headroom level worked → composite mode
        return None

    def _optimize_position_1d(self, image_len, crop_len, face_min, face_max,
                               optimize_min, optimize_max, axis='x'):
        """
        Optimize crop position in one dimension (X or Y).

        Goal: Maximize overlap with optimization target (pose/body) while ensuring
              faces are fully included.

        Args:
            image_len: Image dimension (width or height)
            crop_len: Crop window dimension (width or height)
            face_min, face_max: Face bounding box extent in this dimension
            optimize_min, optimize_max: Optimization target extent (pose or face)
            axis: 'x' or 'y' for tie-breaking

        Returns:
            Crop start position (int)
        """
        # Hard constraint: Crop must fully include faces
        # crop_start <= face_min (crop starts before or at face start)
        # crop_start + crop_len >= face_max (crop ends after or at face end)

        min_start = max(0, face_max - crop_len)  # Latest possible start
        max_start = min(image_len - crop_len, face_min)  # Earliest possible start

        if min_start > max_start:
            # This should never happen if we checked face fit earlier
            # But if it does, center on face as fallback
            return max(0, min(image_len - crop_len, (face_min + face_max - crop_len) // 2))

        # Soft goal: Maximize overlap with optimization target
        optimize_len = optimize_max - optimize_min

        if crop_len >= optimize_len:
            # Crop can fully contain the optimization target
            # Position crop to fully include the target (centered on it)
            ideal_start = (optimize_min + optimize_max - crop_len) // 2
        else:
            # Crop is smaller than optimization target
            # Prioritize different parts based on axis
            if axis == 'y':
                # Y-axis: Top-bias (prioritize head/torso over legs)
                ideal_start = optimize_min
            else:
                # X-axis: Center on target (capture arms/width evenly)
                ideal_start = (optimize_min + optimize_max - crop_len) // 2

        # Apply constraints
        final_start = max(min_start, min(max_start, ideal_start))

        return int(final_start)

    def _create_composite(self, pil_image, target_width, target_height, face_bbox):
        """
        Create a composite image with blurred background (letterbox/pillarbox style).
        Used when faces are too large to fit in the target aspect ratio.

        CRITICAL: This preserves aspect ratio - no distortion occurs.

        Process:
        1. Crop to face bounding box (with padding)
        2. Scale foreground to fit target dimensions (maintains aspect ratio)
        3. Create blurred, darkened background
        4. Composite foreground onto background

        Args:
            pil_image: PIL Image object
            target_width, target_height: Target dimensions
            face_bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            PIL Image at exact target dimensions
        """
        original_width, original_height = pil_image.size

        # Step 1: Crop to face area with breathing room
        if face_bbox:
            fx1, fy1, fx2, fy2 = face_bbox
            w = fx2 - fx1
            h = fy2 - fy1

            # Add extra padding for composite look
            pad_x = int(w * self.COMPOSITE_PADDING)
            pad_y = int(h * self.COMPOSITE_PADDING)

            crop_x1 = max(0, fx1 - pad_x)
            crop_y1 = max(0, fy1 - pad_y)
            crop_x2 = min(original_width, fx2 + pad_x)
            crop_y2 = min(original_height, fy2 + pad_y)

            foreground = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            # Shouldn't happen, but use whole image as fallback
            foreground = pil_image

        # Step 2: Scale foreground to fit target (maintain aspect ratio)
        fg_width, fg_height = foreground.size
        scale = min(target_width / fg_width, target_height / fg_height)

        scaled_width = int(fg_width * scale)
        scaled_height = int(fg_height * scale)

        foreground = foreground.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        # Step 3: Create blurred background
        # Scale original image to fill target frame
        bg_scale = max(target_width / original_width, target_height / original_height)
        bg_width = int(original_width * bg_scale)
        bg_height = int(original_height * bg_scale)

        background = pil_image.resize((bg_width, bg_height), Image.Resampling.LANCZOS)
        background = background.filter(ImageFilter.GaussianBlur(radius=30))

        # Darken background
        enhancer = ImageEnhance.Brightness(background)
        background = enhancer.enhance(0.6)

        # Center crop background to target dimensions
        left = (bg_width - target_width) // 2
        top = (bg_height - target_height) // 2
        background = background.crop((left, top, left + target_width, top + target_height))

        # Step 4: Composite foreground onto background
        paste_x = (target_width - scaled_width) // 2
        paste_y = (target_height - scaled_height) // 2

        background.paste(foreground, (paste_x, paste_y))

        return background

    def _enhance_image(self, image):
        """
        Apply subtle enhancements to the final image.

        Args:
            image: PIL Image

        Returns:
            Enhanced PIL Image
        """
        # Subtle sharpness boost
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        # Subtle contrast boost
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)

        return image

    def crop_image(self, input_path, output_path, target_width=1280, target_height=800, verbose=True):
        """
        Crop and resize an image to target dimensions with face-aware optimization.

        Process:
        1. Load image and detect faces and pose
        2. Calculate face and pose bounding boxes
        3. Determine optimal crop window (or use composite if faces don't fit)
        4. Apply crop/composite
        5. Resize to exact target dimensions
        6. Enhance and save

        Args:
            input_path: Path to input image
            output_path: Path to save output image
            target_width: Target width in pixels
            target_height: Target height in pixels
            verbose: Print progress messages

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            pil_image = Image.open(input_path).convert('RGB')
            image_width, image_height = pil_image.size

            # Convert to numpy array for MediaPipe
            cv_image = np.array(pil_image)

            # Step 1: Detect faces and pose
            faces = self.detect_faces(cv_image)
            pose = self.detect_pose(cv_image)

            if verbose:
                print(f"Detected {len(faces)} face(s) and {'1 pose' if pose else '0 poses'} in {Path(input_path).name}")

            # Step 2: Calculate pose bounding box
            pose_bbox = self._get_pose_bbox(pose, image_width, image_height)

            # Step 3: Determine strategy
            if not faces:
                # No faces detected → crop based on image orientation
                target_aspect = target_width / target_height
                image_aspect = image_width / image_height
                is_portrait = image_height > image_width

                if image_aspect > target_aspect:
                    crop_height = image_height
                    crop_width = int(crop_height * target_aspect)
                else:
                    crop_width = image_width
                    crop_height = int(crop_width / target_aspect)

                # Horizontal centering (always center horizontally)
                crop_x = (image_width - crop_width) // 2

                # Vertical positioning (top-align for portraits, center for landscapes)
                if is_portrait:
                    crop_y = 0  # Top-align for portrait images (faces/heads usually at top)
                    if verbose:
                        print("  → No faces detected, using top-aligned crop (portrait)")
                else:
                    crop_y = (image_height - crop_height) // 2  # Center for landscape images
                    if verbose:
                        print("  → No faces detected, using center crop (landscape)")

                final_image = pil_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
                final_image = final_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            else:
                # Calculate optimal crop window with progressive padding reduction
                crop_window = self._calculate_crop_window(
                    image_width, image_height, target_width, target_height,
                    faces, pose_bbox
                )

                if crop_window is None:
                    # Faces don't fit even with minimum padding → composite mode
                    if verbose:
                        print("  → Faces too large, using blurred background composite")

                    # Calculate face bbox with maximum headroom for composite
                    face_bbox = self._get_combined_face_bbox(faces, image_width, image_height,
                                                              self.HEADROOM_PADDING_LEVELS[0])
                    final_image = self._create_composite(pil_image, target_width, target_height, face_bbox)

                else:
                    # Smart crop with pose optimization
                    crop_x, crop_y, crop_width, crop_height, headroom_used = crop_window

                    if verbose:
                        print(f"  → Smart crop at ({crop_x}, {crop_y}, {crop_width}x{crop_height}), headroom: {int(headroom_used*100)}%")

                    # Crop to the window (which has the correct aspect ratio)
                    final_image = pil_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

                    # Resize to target (no distortion since aspect ratios match)
                    final_image = final_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Step 4: Enhance and save
            final_image = self._enhance_image(final_image)

            # Save with high quality
            save_kwargs = {'quality': 95}
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                save_kwargs.update({'optimize': True, 'progressive': True})

            final_image.save(output_path, **save_kwargs)

            return True

        except Exception as e:
            if verbose:
                print(f"Error processing {input_path}: {e}")
                import traceback
                traceback.print_exc()
            return False

    def process_folder(self, input_folder, output_folder, target_width=1280, target_height=800, max_workers=None):
        """
        Process all images in a folder with multiprocessing.

        Args:
            input_folder: Path to input folder
            output_folder: Path to output folder
            target_width: Target width in pixels
            target_height: Target height in pixels
            max_workers: Number of parallel workers (default: CPU count - 1)
        """
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
                executor.submit(
                    process_single_image_worker,
                    (f, output_path / f"{f.stem}_cropped{f.suffix}", target_width, target_height)
                )
                for f in image_files
            ]

            for i, future in enumerate(as_completed(futures)):
                success, name = future.result()
                if success:
                    success_count += 1
                print(f"\rProgress: {i+1}/{len(image_files)}", end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\nDone! {success_count}/{len(image_files)} successful. Time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Face-Aware Photo Cropper with MediaPipe')
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
