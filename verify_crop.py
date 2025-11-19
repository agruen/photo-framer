import cv2
import numpy as np
from face_crop_tool import FaceAwareCropper
import argparse
import os
from pathlib import Path

def verify_crop(image_path, output_path):
    cropper = FaceAwareCropper()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Detect faces and people
    faces = cropper.detect_faces(rgb_image)
    people = cropper.detect_people(rgb_image)
    print(f"Detected {len(faces)} faces and {len(people)} people")
    
    # Draw faces
    for face in faces:
        x1 = int(face.xmin * w)
        y1 = int(face.ymin * h)
        x2 = int((face.xmin + face.width) * w)
        y2 = int((face.ymin + face.height) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Draw people
    for person in people:
        x1 = int(person.xmin * w)
        y1 = int(person.ymin * h)
        x2 = int((person.xmin + person.width) * w)
        y2 = int((person.ymin + person.height) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta for people
        
    # 2. Get Safe Zones
    critical_zone, preferred_zone = cropper._get_safe_zones(faces, people, w, h)
    
    if preferred_zone:
        px1, py1, px2, py2 = preferred_zone
        cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 2) # Blue for Preferred (Body)
        
    if critical_zone:
        cx1, cy1, cx2, cy2 = critical_zone
        cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2) # Red for Critical (Face)
        
        # 3. Calculate Crop
        crop_rect, use_composite = cropper.calculate_smart_crop(w, h, critical_zone, preferred_zone, 1280, 800)
        
        if use_composite:
            print("Decision: COMPOSITE (Blurred Background)")
            cv2.putText(image, "COMPOSITE NEEDED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print(f"Decision: CROP {crop_rect}")
            cx1, cy1, cx2, cy2 = crop_rect
            cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 255, 255), 4)
            
    cv2.imwrite(output_path, image)
    print(f"Saved debug image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output debug image path")
    args = parser.parse_args()
    
    verify_crop(args.input, args.output)
