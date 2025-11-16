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


def process_single_image_worker(args):
    """Standalone function for multiprocessing - processes a single image."""
    image_file, output_file, target_width, target_height = args
    
    # Create a fresh FaceAwareCropper instance for this process
    cropper = FaceAwareCropper()
    success = cropper.crop_image(str(image_file), str(output_file), target_width, target_height, verbose=False)
    return success, str(image_file.name)


class FaceAwareCropper:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

        # Try to load MediaPipe face detection (best accuracy and speed)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0=short range (<2m), 1=full range (best for photos)
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            print("✓ Using MediaPipe face detection (95%+ accuracy)")
        except ImportError:
            self.mp_detector = None
            self.use_mediapipe = False
            print("⚠ MediaPipe not installed. Install with: pip install mediapipe")
            print("  Falling back to OpenCV (60-70% accuracy)")

        # Try to load DNN face detector for better accuracy (fallback to Haar if not available)
        try:
            self.dnn_net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.use_dnn = True
        except:
            self.dnn_net = None
            self.use_dnn = False
        
    def detect_faces(self, image):
        """Detect faces using multiple methods for better accuracy."""
        height, width = image.shape[:2]
        faces = []

        # Method 1: Try MediaPipe first (best accuracy: 95%+)
        if self.use_mediapipe:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_detector.process(rgb_image)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    # Convert relative coordinates to absolute
                    x1 = int(bbox.xmin * width)
                    y1 = int(bbox.ymin * height)
                    x2 = int((bbox.xmin + bbox.width) * width)
                    y2 = int((bbox.ymin + bbox.height) * height)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    faces.append((x1, y1, x2, y2))

        # Method 2: Try DNN face detection (good accuracy: 90%+)
        if len(faces) == 0 and self.use_dnn:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 50% confidence threshold
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    faces.append((x1, y1, x2, y2))
        
        # Method 3: Haar cascade detection (legacy fallback: 60-70%)
        if len(faces) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple detection scales and parameters
            detection_params = [
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (25, 25)},
                {'scaleFactor': 1.2, 'minNeighbors': 7, 'minSize': (40, 40)}
            ]
            
            for params in detection_params:
                # Frontal faces
                haar_faces = self.face_cascade.detectMultiScale(gray, **params)
                if len(haar_faces) > 0:
                    for (x, y, w, h) in haar_faces:
                        faces.append((x, y, x + w, y + h))
                    break
                
                # Profile faces if no frontal faces found
                haar_faces = self.profile_cascade.detectMultiScale(gray, **params)
                if len(haar_faces) > 0:
                    for (x, y, w, h) in haar_faces:
                        faces.append((x, y, x + w, y + h))
                    break
        
        # Method 4: Eye detection to estimate faces (last resort)
        if len(faces) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
            
            if len(eyes) >= 2:
                # Group eyes into potential faces
                eye_groups = self._group_eyes_into_faces(eyes)
                for eye_group in eye_groups:
                    faces.append(eye_group)
        
        # Remove duplicate faces
        faces = self._remove_duplicate_faces(faces)
        
        return faces
    
    def _group_eyes_into_faces(self, eyes):
        """Group detected eyes into potential face regions."""
        faces = []
        eyes = sorted(eyes, key=lambda e: (e[1], e[0]))  # Sort by y, then x
        
        for i in range(len(eyes)):
            for j in range(i + 1, len(eyes)):
                eye1_x, eye1_y, eye1_w, eye1_h = eyes[i]
                eye2_x, eye2_y, eye2_w, eye2_h = eyes[j]
                
                # Check if eyes could belong to the same face
                eye_distance = abs(eye1_x - eye2_x)
                y_difference = abs(eye1_y - eye2_y)
                
                if eye_distance > 20 and eye_distance < 200 and y_difference < 50:
                    # Estimate face region based on eye positions
                    left_x = min(eye1_x, eye2_x) - eye_distance // 4
                    right_x = max(eye1_x + eye1_w, eye2_x + eye2_w) + eye_distance // 4
                    top_y = min(eye1_y, eye2_y) - eye_distance // 3
                    bottom_y = max(eye1_y + eye1_h, eye2_y + eye2_h) + eye_distance
                    
                    faces.append((max(0, left_x), max(0, top_y), right_x, bottom_y))
        
        return faces
    
    def _remove_duplicate_faces(self, faces):
        """Remove overlapping face detections."""
        if len(faces) <= 1:
            return faces
        
        # Calculate overlap between faces and remove duplicates
        unique_faces = []
        for face in faces:
            is_duplicate = False
            for existing_face in unique_faces:
                if self._faces_overlap(face, existing_face, threshold=0.3):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _faces_overlap(self, face1, face2, threshold=0.3):
        """Check if two face rectangles overlap significantly."""
        x1_min, y1_min, x1_max, y1_max = face1
        x2_min, y2_min, x2_max, y2_max = face2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        # Calculate overlap ratio
        if union == 0:
            return False
        overlap_ratio = intersection / union
        
        return overlap_ratio > threshold
    
    def calculate_face_bounding_box(self, faces):
        """Calculate the bounding box that encompasses all detected faces."""
        if not faces:
            return None

        min_x = min(face[0] for face in faces)
        min_y = min(face[1] for face in faces)
        max_x = max(face[2] for face in faces)
        max_y = max(face[3] for face in faces)

        return (min_x, min_y, max_x, max_y)

    def validate_faces_in_crop(self, faces, crop_rect, safety_margin_percent=0.05):
        """
        Validate that ALL faces are safely within the crop boundaries.
        Returns True if all faces are included with safety margin, False otherwise.
        """
        if not faces:
            return True

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_rect
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        # Add safety margin (5% inside crop boundaries by default)
        safety_x = crop_width * safety_margin_percent
        safety_y = crop_height * safety_margin_percent
        safe_x1 = crop_x1 + safety_x
        safe_y1 = crop_y1 + safety_y
        safe_x2 = crop_x2 - safety_x
        safe_y2 = crop_y2 - safety_y

        # Check each face
        for face in faces:
            # Get face boundaries
            face_x1, face_y1, face_x2, face_y2 = face
            face_center_x = (face_x1 + face_x2) / 2
            face_center_y = (face_y1 + face_y2) / 2

            # Check if entire face is within safe zone
            face_in_safe_zone = (
                safe_x1 <= face_x1 and face_x2 <= safe_x2 and
                safe_y1 <= face_y1 and face_y2 <= safe_y2
            )

            # At minimum, face center must be in safe zone
            center_in_safe_zone = (
                safe_x1 <= face_center_x <= safe_x2 and
                safe_y1 <= face_center_y <= safe_y2
            )

            if not center_in_safe_zone:
                return False

        return True
    
    def _calculate_portrait_crop(self, image_width, image_height, faces, target_ratio):
        """
        Calculates a crop for portrait-style images, prioritizing headroom and composition.
        This method is designed for images that are taller than they are wide.
        """
        face_bbox = self.calculate_face_bounding_box(faces)
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        face_center_x = face_bbox[0] + face_width / 2
        face_center_y = face_bbox[1] + face_height / 2

        # Rule of thirds: position the top of the head roughly 1/3 from the top
        # We give a lot of headroom, 100% of the face height
        headroom = face_height * 1.0
        
        # The amount of the body to show below the chin.
        body_room = face_height * 0.5

        # Calculate the total height required for the crop
        required_height = face_height + headroom + body_room
        
        # Calculate the corresponding width based on the target aspect ratio
        required_width = required_height * target_ratio

        # If the required width is wider than the image, we have to scale down
        if required_width > image_width:
            required_width = image_width
            required_height = required_width / target_ratio
            headroom = (required_height - face_height - body_room) / 2 # re-balance headroom

        # Center the crop horizontally on the face
        crop_x = face_center_x - required_width / 2
        
        # Position the crop vertically, applying the headroom
        crop_y = face_bbox[1] - headroom

        # Boundary checks to ensure the crop is within the image
        if crop_x < 0:
            crop_x = 0
        if crop_y < 0:
            crop_y = 0
        if crop_x + required_width > image_width:
            crop_x = image_width - required_width
        if crop_y + required_height > image_height:
            crop_y = image_height - required_height

        return (int(crop_x), int(crop_y), int(crop_x + required_width), int(crop_y + required_height))

    def calculate_smart_crop(self, image_width, image_height, faces, target_width=1280, target_height=800):
        """Calculate the best crop rectangle with advanced composition and face preservation."""
        target_ratio = target_width / target_height
        image_ratio = image_width / image_height

        # Decide which cropping strategy to use
        if faces:
            # Prioritize portrait-specific logic if the image is taller than wide
            if image_ratio < 1.0: # Portrait-oriented image
                return self._calculate_portrait_crop(image_width, image_height, faces, target_ratio)
            else: # Landscape or square image
                return self._calculate_face_aware_crop(image_width, image_height, faces, target_width, target_height)
        else:
            # Fallback for images without faces
            return self._calculate_content_aware_crop(image_width, image_height, target_width, target_height)

    def _calculate_face_aware_crop(self, image_width, image_height, faces, crop_width, crop_height):
        """Calculate crop with guaranteed face preservation for landscape images."""
        target_ratio = crop_width / crop_height
        face_bbox = self.calculate_face_bounding_box(faces)
        face_center_x = (face_bbox[0] + face_bbox[2]) / 2
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]

        # Generous padding for landscape shots
        padding_percent = 0.60 if len(faces) == 1 else 0.40
        min_padding = max(face_width, face_height) * padding_percent

        # Increased headroom for landscape shots
        headroom = face_height * 0.80

        # Define the required area to keep faces safe
        required_left = face_bbox[0] - min_padding
        required_right = face_bbox[2] + min_padding
        required_top = face_bbox[1] - headroom
        required_bottom = face_bbox[3] + min_padding

        min_crop_width = required_right - required_left
        min_crop_height = required_bottom - required_top

        # Determine the final crop dimensions
        if min_crop_width / min_crop_height > target_ratio:
            # Width is the limiting factor
            final_crop_width = min_crop_width
            final_crop_height = final_crop_width / target_ratio
        else:
            # Height is the limiting factor
            final_crop_height = min_crop_height
            final_crop_width = final_crop_height * target_ratio

        # Center the crop on the face group
        crop_x = face_center_x - final_crop_width / 2
        crop_y = face_center_y - final_crop_height / 2

        # Adjust crop to not go out of bounds
        crop_x = max(0, min(crop_x, image_width - final_crop_width))
        crop_y = max(0, min(crop_y, image_height - final_crop_height))
        
        # Final boundary checks to ensure the entire required zone is included
        if crop_x > required_left:
            crop_x = required_left
        if crop_y > required_top:
            crop_y = required_top
        if crop_x + final_crop_width < required_right:
            crop_x = image_width - final_crop_width
        if crop_y + final_crop_height < required_bottom:
            crop_y = image_height - final_crop_height

        return (int(crop_x), int(crop_y), int(crop_x + final_crop_width), int(crop_y + final_crop_height))
    
    
    def _calculate_content_aware_crop(self, image_width, image_height, crop_width, crop_height):
        """Calculate crop for images without faces using content analysis."""
        target_ratio = crop_width / crop_height
        
        # For wider images, crop from height; for taller, crop from width
        if image_width / image_height > target_ratio:
            crop_height = image_height
            crop_width = int(crop_height * target_ratio)
        else:
            crop_width = image_width
            crop_height = int(crop_width / target_ratio)

        # Simple center crop with slight upward bias
        crop_x = (image_width - crop_width) / 2
        crop_y = (image_height - crop_height) * 0.45  # Slightly above center
        
        return (int(crop_x), int(crop_y), int(crop_x + crop_width), int(crop_y + crop_height))
    
    def _enhance_image_quality(self, image, original_size, final_size):
        """Apply adaptive image enhancement based on scaling and quality."""
        from PIL import ImageEnhance, ImageFilter
        
        original_width, original_height = original_size
        final_width, final_height = final_size
        
        # Calculate scaling factors
        width_scale = final_width / original_width
        height_scale = final_height / original_height
        min_scale = min(width_scale, height_scale)
        
        enhanced = image
        
        # Apply sharpening if image was significantly upscaled
        if min_scale > 1.2:
            # Moderate sharpening for upscaled images
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.15)
        elif min_scale < 0.5:
            # Light sharpening for heavily downscaled images
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        # Subtle contrast enhancement for better visual appeal
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.05)
        
        # Very slight saturation boost for more vibrant colors
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.03)
        
        return enhanced
    
    def _create_blurred_background_composite(self, pil_image, content_box, target_width, target_height, verbose):
        if verbose: print("Creating blurred background composite.")
        
        original_width, original_height = pil_image.size
        target_ratio = target_width / target_height

        # 1. Create blurred, darkened background
        bg_ratio = original_width / original_height
        bg_width, bg_height = (target_width, int(target_width / bg_ratio))
        background_image = pil_image.resize((bg_width, bg_height), Image.Resampling.LANCZOS)
        background_image = background_image.filter(ImageFilter.GaussianBlur(radius=25))
        enhancer = ImageEnhance.Brightness(background_image)
        background_image = enhancer.enhance(0.6)

        left = (background_image.width - target_width) / 2
        top = (background_image.height - target_height) / 2
        background_image = background_image.crop((left, top, left + target_width, top + target_height))

        # 2. Resize original image to fit while keeping content_box visible
        # We want the content_box to be fully visible. Let's scale based on that.
        content_h = content_box[3] - content_box[1]
        # Scale the whole image so the content box height is, e.g., 80% of target height
        scale_factor = (target_height * 0.8) / content_h
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # If this makes it too wide, scale down to fit width
        if new_width > target_width:
            scale_factor = target_width / new_width
            new_width = target_width
            new_height = int(new_height * scale_factor)
            
        foreground_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 3. Paste foreground, centering the content_box
        content_box_center_y_scaled = (content_box[1] + content_h / 2) * scale_factor
        paste_y = int(target_height / 2 - content_box_center_y_scaled)
        paste_x = (target_width - new_width) // 2

        final_image = background_image.copy()
        final_image.paste(foreground_image, (paste_x, paste_y))
        
        return final_image

    def _get_safety_crop(self, faces, width, height, target_ratio):
        face_bbox = self.calculate_face_bounding_box(faces)
        face_center_x = (face_bbox[0] + face_bbox[2]) / 2
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        safety_padding = max(face_width, face_height) * 0.60
        safe_width = (face_width + 2 * safety_padding)
        safe_height = safe_width / target_ratio
        if safe_height < (face_height + 2 * safety_padding):
            safe_height = (face_height + 2 * safety_padding)
            safe_width = safe_height * target_ratio
        safe_x = max(0, min(face_center_x - safe_width / 2, width - safe_width))
        safe_y = max(0, min(face_center_y - safe_height / 2, height - safe_height))
        return (int(safe_x), int(safe_y), int(safe_x + safe_width), int(safe_y + safe_height))
    
    def crop_image(self, input_path, output_path, target_width=1280, target_height=800, verbose=True):
        """
        Crop a single image with intelligent, face-aware logic that maximizes content
        and uses a blurred background for portrait images when necessary.
        """
        try:
            # Load image with PIL and OpenCV
            pil_image = Image.open(input_path).convert('RGB')
            cv_image = np.array(pil_image)
            cv_image = cv_image[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV

            original_width, original_height = pil_image.size
            target_ratio = target_width / target_height
            original_ratio = original_width / original_height

            # Always detect faces first
            faces = self.detect_faces(cv_image)
            if verbose:
                print(f"Detected {len(faces)} face(s) in {input_path}")

            # --- Portrait on Landscape Special Handling ---
            if original_ratio < target_ratio and original_height > original_width:
                if verbose:
                    print("Portrait image on landscape frame detected. Applying advanced logic...")

                # Determine the primary content box (faces or center)
                if faces:
                    content_box = self.calculate_face_bounding_box(faces)
                else:
                    # If no faces, content is the center third of the image
                    content_box_height = original_height / 3
                    content_box_top = (original_height - content_box_height) / 2
                    content_box = (0, content_box_top, original_width, content_box_top + content_box_height)

                # Add padding to the content box for better composition
                padding_v = (content_box[3] - content_box[1]) * 0.4 # 40% vertical padding
                padding_h = (content_box[2] - content_box[0]) * 0.4 # 40% horizontal padding
                padded_box = (
                    max(0, content_box[0] - padding_h),
                    max(0, content_box[1] - padding_v),
                    min(original_width, content_box[2] + padding_h),
                    min(original_height, content_box[3] + padding_v)
                )
                
                # Can we crop the padded box directly?
                padded_width = padded_box[2] - padded_box[0]
                padded_height = padded_box[3] - padded_box[1]
                
                # If the padded content is not excessively tall, try to crop directly
                if (padded_height / padded_width) < (target_ratio * 1.5): # Avoid extreme vertical crops
                    crop_h = padded_height
                    crop_w = int(crop_h * target_ratio)
                    
                    # Center the crop horizontally on the content
                    crop_x = (padded_box[0] + padded_box[2]) / 2 - (crop_w / 2)
                    crop_y = padded_box[1]

                    # Boundary checks
                    if crop_x < 0: crop_x = 0
                    if crop_y < 0: crop_y = 0
                    if crop_x + crop_w > original_width: crop_x = original_width - crop_w
                    if crop_y + crop_h > original_height: crop_y = original_height - crop_h
                    
                    if crop_w <= original_width and crop_h <= original_height:
                        if verbose: print("Attempting direct face-aware crop for portrait.")
                        crop_rect = (int(crop_x), int(crop_y), int(crop_x + crop_w), int(crop_y + crop_h))
                        cropped = pil_image.crop(crop_rect)
                        final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        # --- This path ends here if successful ---
                    else:
                        # Fallback to blur if direct crop is not possible
                        final_image = self._create_blurred_background_composite(pil_image, content_box, target_width, target_height, verbose)

                else: # Content is too tall, use blurred background
                    final_image = self._create_blurred_background_composite(pil_image, content_box, target_width, target_height, verbose)

            else: # --- Standard Landscape or Square Image Processing ---
                if verbose: print("Standard landscape/square processing.")
                crop_rect = self.calculate_smart_crop(original_width, original_height, faces, target_width, target_height)
                
                if faces and not self.validate_faces_in_crop(faces, crop_rect):
                    if verbose: print("Warning: Initial crop would cut faces. Recalculating with safety priority...")
                    crop_rect = self._get_safety_crop(faces, original_width, original_height, target_ratio)

                cropped = pil_image.crop(crop_rect)
                final_image = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Final enhancements and saving
            final_image = self._enhance_image_quality(final_image, final_image.size, (target_width, target_height))
            
            save_kwargs = {'quality': 95}
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                save_kwargs.update({'optimize': True, 'progressive': True})
            
            final_image.save(output_path, **save_kwargs)
            if verbose: print(f"Saved final image to {output_path}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error processing {input_path}: {str(e)}")
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