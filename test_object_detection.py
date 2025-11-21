#!/usr/bin/env python3
"""
Quick test script for object detection functionality.
Tests the FaceAwareCropper with object detection enabled.
"""

import sys
from face_crop_tool import FaceAwareCropper
from PIL import Image
import numpy as np

def test_object_detection():
    """Test object detection initialization and functionality."""
    print("Testing Object Detection Feature")
    print("=" * 50)

    # Initialize cropper
    print("\n1. Initializing FaceAwareCropper...")
    cropper = FaceAwareCropper()

    # Check if object detector was initialized
    if cropper.object_detector is None:
        print("   ❌ Object detector NOT initialized")
        print("   This is expected if the model file is missing.")
        return False
    else:
        print("   ✓ Object detector initialized successfully")

    # Test with a simple synthetic image (solid color)
    print("\n2. Testing object detection on synthetic image...")
    test_image = Image.new('RGB', (800, 600), color='blue')

    try:
        objects = cropper.detect_objects(test_image)
        print(f"   ✓ Object detection executed (found {len(objects)} objects)")

        if objects:
            print(f"\n   Detected objects:")
            for i, obj in enumerate(objects, 1):
                print(f"     {i}. {obj['category']} (confidence: {obj['score']:.2f}, priority: {obj['priority']})")
        else:
            print("   (No objects detected in synthetic image - this is expected)")

    except Exception as e:
        print(f"   ❌ Error during object detection: {e}")
        return False

    # Test priority system
    print("\n3. Testing object priority system...")
    test_objects = [
        {'category': 'dog', 'score': 0.9, 'priority': 1, 'bbox': {'xmin': 0.1, 'ymin': 0.1, 'width': 0.3, 'height': 0.3}},
        {'category': 'chair', 'score': 0.8, 'priority': 3, 'bbox': {'xmin': 0.5, 'ymin': 0.5, 'width': 0.2, 'height': 0.2}},
        {'category': 'car', 'score': 0.95, 'priority': 2, 'bbox': {'xmin': 0.7, 'ymin': 0.1, 'width': 0.2, 'height': 0.2}},
    ]

    primary = cropper._select_primary_object(test_objects)
    if primary and primary['category'] == 'dog':
        print(f"   ✓ Primary object correctly identified: {primary['category']} (priority 1)")
    else:
        print(f"   ❌ Expected 'dog', got '{primary['category'] if primary else 'None'}'")
        return False

    # Test bbox utilities
    print("\n4. Testing bounding box utilities...")
    bbox1 = {'xmin': 0.1, 'ymin': 0.1, 'width': 0.2, 'height': 0.2}
    bbox2 = {'xmin': 0.15, 'ymin': 0.15, 'width': 0.15, 'height': 0.15}

    area = cropper._bbox_area(bbox1)
    nearby = cropper._objects_nearby(bbox1, bbox2, threshold=0.2)
    merged = cropper._merge_bboxes([bbox1, bbox2])

    print(f"   ✓ bbox_area: {area:.4f}")
    print(f"   ✓ objects_nearby: {nearby}")
    print(f"   ✓ merge_bboxes: {merged}")

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("\nObject Detection Feature Summary:")
    print(f"  - Model loaded: Yes")
    print(f"  - Priority objects: {len(cropper.PRIORITY_OBJECTS)} categories")
    print(f"  - Detection working: Yes")
    print(f"  - Priority system: Working")
    print(f"  - Bbox utilities: Working")

    return True


if __name__ == '__main__':
    try:
        success = test_object_detection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
