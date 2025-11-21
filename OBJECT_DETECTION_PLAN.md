# Intelligent Object Detection & Cropping Plan
## Adding Smart Cropping for Non-People Photos

---

## 🎯 Executive Summary

**Goal:** Extend Photo-Framer's intelligent cropping to non-people photos by detecting and centering around meaningful objects (pets, vehicles, food, landmarks) when no faces are present.

**Current State:**
- Face detection via dual MediaPipe models (75% confidence)
- Pose detection for body context
- Aspect-ratio-first cropping with progressive padding
- Fallback: Top-aligned (portrait) or center-aligned (landscape) crops when no faces

**Proposed Enhancement:**
- Add object detection layer between face detection and basic fallback
- Detect salient objects (80 COCO classes: pets, vehicles, furniture, food, etc.)
- Apply same aspect-ratio-first philosophy with object-centered crops
- Maintain performance on CPU-only Docker deployment

---

## 📊 Research Findings

### 1. Object Detection Options Analysis

| Solution | Pros | Cons | Fit Score |
|----------|------|------|-----------|
| **MediaPipe Object Detection** | • Already using MediaPipe<br>• Same dependency<br>• CPU-optimized<br>• EfficientDet-Lite models<br>• Consistent API | • Objectron deprecated<br>• Limited model selection<br>• Less active development | ⭐⭐⭐⭐ (Best) |
| **YOLOv10/YOLO26** | • State-of-art accuracy<br>• Active development (2024)<br>• 80 COCO classes<br>• Fast inference | • New dependency (Ultralytics)<br>• Heavier models<br>• GPU-optimized primarily | ⭐⭐⭐ (Good) |
| **Saliency Detection** | • Very lightweight<br>• No object classes needed<br>• Fast on CPU<br>• Works on any image | • Less precise<br>• Can focus on backgrounds<br>• No semantic understanding | ⭐⭐ (Backup) |

### 2. Recommended Approach: **Hybrid Multi-Layer Strategy**

```
Image Input
    ↓
[Layer 1: Face Detection] ← Already implemented
    ↓ (if no faces)
[Layer 2: Object Detection] ← NEW
    ↓ (if no objects)
[Layer 3: Saliency Detection] ← NEW (lightweight fallback)
    ↓ (if all fail)
[Layer 4: Basic Crop] ← Already implemented
```

---

## 🏗️ Technical Architecture

### Primary Solution: MediaPipe Object Detection

**Model Choice:** EfficientDet-Lite0
- **Classes:** 80 COCO categories (person, dog, cat, car, bicycle, chair, etc.)
- **Confidence threshold:** 0.5 (lower than faces since objects vary more)
- **Performance:** ~30-50ms per image on modern CPU
- **Model size:** ~4.5MB (lightweight)
- **Architecture:** Designed for mobile/edge devices

**COCO Categories Most Relevant for Photo Cropping:**
```python
PRIORITY_OBJECTS = {
    # Pets & Animals
    'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',

    # Vehicles
    'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',

    # Food (common in photos)
    'pizza', 'cake', 'sandwich', 'hot dog', 'donut',

    # Sports
    'sports ball', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket',

    # Large objects worth centering
    'bicycle', 'bench', 'potted plant', 'dining table', 'couch', 'bed'
}
```

**Implementation in MediaPipe:**
```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize detector
base_options = python.BaseOptions(
    model_asset_path='efficientdet_lite0.tflite'
)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=10,
    category_allowlist=PRIORITY_OBJECTS  # Filter to relevant objects
)
detector = vision.ObjectDetector.create_from_options(options)

# Detect objects
image = mp.Image.create_from_file(image_path)
detection_result = detector.detect(image)

# Results format:
# detection_result.detections[i].bounding_box  # xmin, ymin, width, height
# detection_result.detections[i].categories[0].category_name  # 'dog', 'car', etc.
# detection_result.detections[i].categories[0].score  # confidence
```

---

## 🔧 Implementation Plan

### Phase 1: Core Object Detection Integration (v7.0)

**File:** `face_crop_tool.py`

**New Methods to Add:**
```python
class FaceAwareCropper:
    def __init__(self):
        # Existing MediaPipe face/pose detectors
        # ...

        # NEW: Add object detector
        self.object_detector = self._init_object_detector()

    def _init_object_detector(self):
        """Initialize MediaPipe object detector"""
        base_options = python.BaseOptions(
            model_asset_path='models/efficientdet_lite0.tflite'
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            max_results=10,
            category_allowlist=self.PRIORITY_OBJECTS
        )
        return vision.ObjectDetector.create_from_options(options)

    def detect_objects(self, image):
        """
        Detect objects in image using MediaPipe

        Returns:
            List of dicts with:
            - bbox: (xmin, ymin, width, height) normalized [0-1]
            - category: str (e.g., 'dog', 'car')
            - score: float confidence
            - priority: int (1=highest, 3=lowest)
        """
        # Convert PIL to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(image)
        )

        # Detect
        results = self.object_detector.detect(mp_image)

        # Convert to normalized bboxes with priority
        objects = []
        for detection in results.detections:
            bbox = detection.bounding_box
            category = detection.categories[0].category_name
            score = detection.categories[0].score

            objects.append({
                'bbox': self._normalize_bbox(bbox, image.size),
                'category': category,
                'score': score,
                'priority': self._get_object_priority(category)
            })

        return objects

    def _get_object_priority(self, category):
        """
        Assign priority to object categories
        1 = Highest (pets, main subjects)
        2 = Medium (vehicles, large objects)
        3 = Low (furniture, background objects)
        """
        HIGH_PRIORITY = {'dog', 'cat', 'bird', 'horse', 'cake', 'pizza'}
        MEDIUM_PRIORITY = {'car', 'motorcycle', 'bicycle', 'airplane', 'boat', 'surfboard'}

        if category in HIGH_PRIORITY:
            return 1
        elif category in MEDIUM_PRIORITY:
            return 2
        else:
            return 3

    def _select_primary_object(self, objects):
        """
        Select the most important object(s) to center crop around

        Strategy:
        1. Sort by priority (1 > 2 > 3)
        2. Within same priority, sort by size (area)
        3. Group nearby objects of same priority
        4. Return combined bounding box
        """
        if not objects:
            return None

        # Sort by priority, then by area
        sorted_objects = sorted(
            objects,
            key=lambda x: (x['priority'], -self._bbox_area(x['bbox']))
        )

        # Take highest priority object(s)
        primary = sorted_objects[0]
        primary_priority = primary['priority']

        # Group all objects with same priority that are nearby
        grouped = [primary]
        for obj in sorted_objects[1:]:
            if obj['priority'] == primary_priority:
                # Check if nearby (within 20% of image)
                if self._objects_nearby(primary['bbox'], obj['bbox'], threshold=0.2):
                    grouped.append(obj)

        # Combine bounding boxes
        combined_bbox = self._merge_bboxes([obj['bbox'] for obj in grouped])

        return {
            'bbox': combined_bbox,
            'objects': grouped,
            'category': primary['category']
        }

    def _calculate_object_crop(self, image_size, object_group, target_aspect):
        """
        Calculate optimal crop around detected object(s)

        Similar to face crop logic but adapted for objects:
        - Less headroom needed (objects don't need top bias)
        - More breathing room (20-30% padding vs 15% for faces)
        - Center-biased rather than top-biased
        """
        width, height = image_size
        target_width, target_height = self._get_crop_dimensions(
            width, height, target_aspect
        )

        obj_bbox = object_group['bbox']
        obj_xmin, obj_ymin, obj_w, obj_h = obj_bbox

        # Convert to pixels
        obj_xmin_px = int(obj_xmin * width)
        obj_ymin_px = int(obj_ymin * height)
        obj_w_px = int(obj_w * width)
        obj_h_px = int(obj_h * height)

        # Add padding (30% breathing room for objects)
        OBJECT_PADDING = 0.30
        padding_x = int(obj_w_px * OBJECT_PADDING)
        padding_y = int(obj_h_px * OBJECT_PADDING)

        obj_xmin_px = max(0, obj_xmin_px - padding_x)
        obj_ymin_px = max(0, obj_ymin_px - padding_y)
        obj_xmax_px = min(width, obj_xmin_px + obj_w_px + 2*padding_x)
        obj_ymax_px = min(height, obj_ymin_px + obj_h_px + 2*padding_y)

        # Calculate crop window centered on object
        # (Similar to _optimize_position_1d but center-biased)

        # X-axis: center the object
        obj_center_x = (obj_xmin_px + obj_xmax_px) / 2
        crop_x = int(obj_center_x - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))

        # Y-axis: center the object (not top-biased like faces)
        obj_center_y = (obj_ymin_px + obj_ymax_px) / 2
        crop_y = int(obj_center_y - target_height / 2)
        crop_y = max(0, min(crop_y, height - target_height))

        # Verify object fits in crop
        object_fits = (
            crop_x <= obj_xmin_px and
            crop_x + target_width >= obj_xmax_px and
            crop_y <= obj_ymin_px and
            crop_y + target_height >= obj_ymax_px
        )

        if object_fits:
            return (crop_x, crop_y, target_width, target_height), 'smart_crop'
        else:
            # Fallback to composite mode (like faces)
            return self._create_composite(
                image, obj_bbox, target_width, target_height
            ), 'composite'
```

**Modified Main Pipeline:**
```python
def crop_image(self, image_path, target_width, target_height):
    """Main cropping pipeline with object detection fallback"""

    # ... (existing code: load image, get target aspect) ...

    # Layer 1: Face detection (EXISTING)
    faces = self.detect_faces(pil_image)
    pose_bbox = self.detect_pose(pil_image)

    if faces:
        # Use existing face crop logic
        crop_info, mode = self._calculate_crop_window(...)
        cropped = pil_image.crop(crop_info)
        detection_type = 'faces'

    else:
        # Layer 2: Object detection (NEW)
        objects = self.detect_objects(pil_image)

        if objects:
            primary_object = self._select_primary_object(objects)
            crop_info, mode = self._calculate_object_crop(
                pil_image.size, primary_object, target_aspect
            )
            cropped = pil_image.crop(crop_info)
            detection_type = f'object_{primary_object["category"]}'

        else:
            # Layer 3: Saliency detection (future enhancement)
            # For now, fall through to basic crop

            # Layer 4: Basic crop (EXISTING)
            crop_info = self._basic_crop(...)
            cropped = pil_image.crop(crop_info)
            detection_type = 'basic'
            mode = 'basic'

    # ... (existing code: enhance, save) ...

    return {
        'output_path': output_path,
        'detection_type': detection_type,
        'mode': mode,
        'crop_region': crop_info
    }
```

---

### Phase 2: Saliency Detection Fallback (v7.1)

**Add lightweight saliency detection for artistic/landscape photos without clear objects**

**Library:** OpenCV's Saliency API (already have OpenCV)

```python
def detect_saliency_region(self, image):
    """
    Detect salient region using OpenCV
    Fast and lightweight - no ML model needed
    """
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Use spectral residual saliency (fast)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(cv_image)

    # Threshold to get salient region
    _, thresh = cv2.threshold(
        (saliency_map * 255).astype(np.uint8),
        127, 255, cv2.THRESH_BINARY
    )

    # Find largest contour
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Normalize to [0-1]
        width, height = image.size
        return {
            'bbox': (x/width, y/height, w/width, h/height),
            'confidence': cv2.contourArea(largest) / (width * height)
        }

    return None
```

**Integration Point:**
```python
# In crop_image() between object detection and basic crop:

if not objects:
    # Layer 3: Saliency detection
    salient_region = self.detect_saliency_region(pil_image)

    if salient_region and salient_region['confidence'] > 0.15:
        # Use saliency region like an object
        crop_info, mode = self._calculate_object_crop(
            pil_image.size,
            {'bbox': salient_region['bbox']},
            target_aspect
        )
        detection_type = 'saliency'
    else:
        # Layer 4: Basic crop (fallback)
        crop_info = self._basic_crop(...)
        detection_type = 'basic'
```

---

## 📦 Dependencies & Model Files

### New Python Dependencies
Add to `requirements-server.txt`:
```txt
# Object detection (MediaPipe extension)
# (MediaPipe 0.10.14 already installed - supports object detection)
```

**No new dependencies needed!** MediaPipe already supports object detection.

### Model Files
Download and add to project:
```bash
mkdir -p models
cd models

# EfficientDet-Lite0 model (4.5MB)
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite
```

**Docker Integration:**
```dockerfile
# Add to Dockerfile
COPY models/ /app/models/
```

---

## 🎨 UI Enhancements

### Dashboard Updates

**Display object detection results in admin dashboard:**

```html
<!-- In slideshow detail page -->
<div class="photo-metadata">
    <span class="detection-badge">
        {% if photo.detection_type == 'faces' %}
            <i class="icon-face"></i> Faces ({{ photo.face_count }})
        {% elif photo.detection_type.startswith('object_') %}
            <i class="icon-object"></i> {{ photo.detection_type.split('_')[1]|title }}
        {% elif photo.detection_type == 'saliency' %}
            <i class="icon-sparkles"></i> Saliency
        {% else %}
            <i class="icon-crop"></i> Basic Crop
        {% endif %}
    </span>

    <span class="crop-mode">
        {{ photo.crop_mode|title }} Mode
    </span>
</div>
```

### Database Schema Extension

**Add to Photo model (app/models.py):**
```python
class Photo(db.Model):
    # ... existing fields ...

    # NEW FIELDS
    detection_type = db.Column(db.String(50), default='basic')
    # Values: 'faces', 'object_dog', 'object_car', 'saliency', 'basic'

    detection_confidence = db.Column(db.Float, nullable=True)
    # Average confidence score

    crop_mode = db.Column(db.String(20), default='basic')
    # Values: 'smart_crop', 'composite', 'basic'
```

**Migration:**
```bash
# Add these columns with defaults for existing photos
ALTER TABLE photo ADD COLUMN detection_type VARCHAR(50) DEFAULT 'faces';
ALTER TABLE photo ADD COLUMN detection_confidence FLOAT;
ALTER TABLE photo ADD COLUMN crop_mode VARCHAR(20) DEFAULT 'basic';
```

---

## 🚀 Performance Impact Analysis

### Estimated Processing Time Increases

**Current Pipeline (per image):**
- Face detection (dual models): ~80-120ms
- Pose detection: ~60-80ms
- Crop calculation: ~5-10ms
- Image processing: ~20-30ms
- **Total: ~165-240ms per image**

**With Object Detection:**
```
If faces found:
    → No change: ~165-240ms (skip object detection)

If no faces:
    → Add object detection: +30-50ms
    → Add saliency (if needed): +10-20ms
    → New total: ~205-310ms per image
```

**Impact:**
- **Photos with faces:** 0% slowdown (no change)
- **Photos without faces:** ~20-30% slowdown
- **Average across mixed album:** ~5-10% slowdown
- **Still within acceptable range for background processing**

### Optimization Strategies

1. **Smart Caching:**
   ```python
   # Cache detector instances (already doing this)
   # Share MediaPipe runtime across detections
   ```

2. **Model Selection:**
   ```python
   # Use EfficientDet-Lite0 (lightest)
   # Skip Lite1/Lite2 (heavier models)
   ```

3. **Conditional Detection:**
   ```python
   # Only run object detection when needed
   if not faces and not pose:
       objects = self.detect_objects(image)
   ```

4. **Batch Processing:**
   ```python
   # Already using multiprocessing in process_folder()
   # No changes needed - automatic parallelization
   ```

---

## 🧪 Testing Strategy

### Test Image Categories

Create comprehensive test suite:

```
tests/images/
├── faces/
│   ├── single_face.jpg         ← Should use face detection (no change)
│   ├── group_photo.jpg         ← Should use face detection (no change)
│   └── family_portrait.jpg     ← Should use face detection (no change)
│
├── pets/
│   ├── dog_solo.jpg            ← Should detect 'dog' object
│   ├── cat_portrait.jpg        ← Should detect 'cat' object
│   ├── multiple_dogs.jpg       ← Should group nearby dogs
│   └── bird_close.jpg          ← Should detect 'bird' object
│
├── vehicles/
│   ├── car_showcase.jpg        ← Should detect 'car' object
│   ├── motorcycle.jpg          ← Should detect 'motorcycle'
│   └── airplane.jpg            ← Should detect 'airplane'
│
├── food/
│   ├── cake_closeup.jpg        ← Should detect 'cake' (high priority)
│   ├── pizza.jpg               ← Should detect 'pizza'
│   └── dessert_table.jpg       ← Should group multiple food items
│
├── landscapes/
│   ├── mountain.jpg            ← Should use saliency (mountain peak)
│   ├── sunset.jpg              ← Should use saliency (sun/horizon)
│   └── architecture.jpg        ← May detect 'building' or saliency
│
└── edge_cases/
    ├── person_with_dog.jpg     ← Faces take priority over objects
    ├── crowd_background.jpg    ← Many objects - test grouping
    ├── abstract_art.jpg        ← Saliency fallback
    └── blank_wall.jpg          ← Basic crop fallback
```

### Automated Test Suite

```python
# tests/test_object_detection.py

def test_dog_detection():
    """Test that dog photos center on the dog"""
    cropper = FaceAwareCropper()
    result = cropper.crop_image('tests/images/pets/dog_solo.jpg', 1920, 1080)

    assert result['detection_type'] == 'object_dog'
    assert result['mode'] in ['smart_crop', 'composite']

def test_face_priority():
    """Test that faces take priority over objects"""
    result = cropper.crop_image('tests/images/edge_cases/person_with_dog.jpg', 1920, 1080)

    assert result['detection_type'] == 'faces'
    # Dog should be ignored when face is present

def test_performance_benchmark():
    """Ensure processing time is acceptable"""
    import time

    start = time.time()
    cropper.crop_image('tests/images/pets/dog_solo.jpg', 1920, 1080)
    duration = time.time() - start

    assert duration < 0.5  # Must complete in under 500ms
```

---

## 📋 Implementation Checklist

### Phase 1: Core Object Detection (Target: v7.0)

- [ ] **Setup**
  - [ ] Download EfficientDet-Lite0 model to `models/` directory
  - [ ] Update Dockerfile to copy model files
  - [ ] Verify MediaPipe version supports object detection API

- [ ] **Code Implementation**
  - [ ] Add `_init_object_detector()` method to FaceAwareCropper
  - [ ] Implement `detect_objects()` with COCO filtering
  - [ ] Implement `_get_object_priority()` for smart prioritization
  - [ ] Implement `_select_primary_object()` for grouping logic
  - [ ] Implement `_calculate_object_crop()` for object-centered crops
  - [ ] Add helper methods: `_bbox_area()`, `_objects_nearby()`, `_merge_bboxes()`
  - [ ] Integrate into `crop_image()` main pipeline (Layer 2)

- [ ] **Database**
  - [ ] Add migration for new Photo model fields
  - [ ] Update Photo model with `detection_type`, `detection_confidence`, `crop_mode`
  - [ ] Update task processing to save new metadata

- [ ] **Testing**
  - [ ] Create test image library (50+ images across categories)
  - [ ] Write unit tests for object detection
  - [ ] Write integration tests for full pipeline
  - [ ] Performance benchmarking
  - [ ] Manual QA on diverse photo sets

- [ ] **UI Updates**
  - [ ] Update admin dashboard to show detection type badges
  - [ ] Add object category icons/labels
  - [ ] Display crop mode in photo metadata

- [ ] **Documentation**
  - [ ] Update README with object detection feature
  - [ ] Add examples showing pet/vehicle detection
  - [ ] Document new Photo model fields

### Phase 2: Saliency Detection (Target: v7.1)

- [ ] **Code Implementation**
  - [ ] Implement `detect_saliency_region()` using OpenCV
  - [ ] Integrate into `crop_image()` as Layer 3
  - [ ] Add confidence threshold tuning (0.15-0.25)

- [ ] **Testing**
  - [ ] Test on landscape photos
  - [ ] Test on abstract/artistic images
  - [ ] Verify fallback behavior

### Phase 3: Future Enhancements (v8.0+)

- [ ] **Advanced Features**
  - [ ] Add user preference for object priority (e.g., "always prioritize pets")
  - [ ] Multi-object composition for group scenes
  - [ ] Rule-based logic (e.g., "center cake on birthday photos")
  - [ ] A/B testing different crop strategies per photo

- [ ] **Performance Optimization**
  - [ ] Investigate GPU acceleration for object detection (if deployed on GPU hardware)
  - [ ] Model quantization for faster CPU inference
  - [ ] Async detection pipeline

---

## 🔄 Rollback Strategy

**If object detection causes issues:**

1. **Feature Flag:**
   ```python
   # In config.py
   ENABLE_OBJECT_DETECTION = os.getenv('ENABLE_OBJECT_DETECTION', 'true').lower() == 'true'

   # In face_crop_tool.py
   if config.ENABLE_OBJECT_DETECTION and not faces:
       objects = self.detect_objects(image)
   ```

2. **Quick Disable:**
   ```bash
   # In docker-compose.yml
   environment:
     - ENABLE_OBJECT_DETECTION=false

   # Or via env var
   export ENABLE_OBJECT_DETECTION=false
   ```

3. **Database Rollback:**
   ```sql
   -- If needed, revert schema changes
   ALTER TABLE photo DROP COLUMN detection_type;
   ALTER TABLE photo DROP COLUMN detection_confidence;
   ALTER TABLE photo DROP COLUMN crop_mode;
   ```

---

## 💡 Expected Outcomes

### Success Metrics

1. **Accuracy Improvements:**
   - **Pet photos:** 80%+ properly centered on pet (vs current center crop)
   - **Vehicle photos:** 70%+ properly framed around vehicle
   - **Food photos:** 85%+ centered on main dish
   - **Overall non-face photos:** 60-70% improvement in crop quality

2. **Performance:**
   - **Processing time increase:** < 10% on mixed photo sets
   - **No impact on face photos:** 0% slowdown
   - **Acceptable latency:** < 500ms per image worst-case

3. **User Experience:**
   - **Fewer manual re-crops:** Reduce user adjustments by 40-50%
   - **Better auto-slideshows:** More professional-looking results
   - **Broader use cases:** Support pet albums, car shows, food blogs

### Example Improvements

**Before (Current System):**
```
Dog portrait photo → Center crop → Dog might be partially cut off
Car showcase → Center crop → Car might be off-center
Cake photo → Center crop → Cake might be at edge
```

**After (With Object Detection):**
```
Dog portrait photo → Object detection → Dog centered with 30% breathing room
Car showcase → Object detection → Car properly framed
Cake photo → Object detection → Cake centered (high priority)
```

---

## 🎓 Learning & Iteration

### Monitoring & Analytics

**Track detection performance in production:**

```python
# Log to database for analysis
detection_stats = {
    'total_photos': count,
    'detection_breakdown': {
        'faces': face_count,
        'object_dog': dog_count,
        'object_car': car_count,
        'saliency': saliency_count,
        'basic': basic_count
    },
    'avg_processing_time': avg_time,
    'avg_confidence': avg_confidence
}
```

**Use this data to:**
- Identify which object categories are most common
- Tune confidence thresholds
- Optimize priority rankings
- Find edge cases needing improvement

### Future ML Opportunities

Once object detection is stable, consider:

1. **Custom Model Training:**
   - Fine-tune on user's actual photo collection
   - Learn user-specific preferences (e.g., their specific pet)

2. **Aesthetic Scoring:**
   - Add aesthetic quality prediction
   - Auto-select best photos for slideshow

3. **Scene Classification:**
   - Detect "party", "wedding", "vacation" scenes
   - Apply scene-specific crop strategies

---

## 📚 References & Resources

### MediaPipe Documentation
- Object Detection Guide: https://developers.google.com/mediapipe/solutions/vision/object_detector
- Python API Reference: https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/python/vision
- Model Cards: https://developers.google.com/mediapipe/solutions/vision/object_detector#models

### COCO Dataset
- 80 Object Categories: https://cocodataset.org/#explore
- Category Definitions: https://cocodataset.org/#download

### Research Papers
- EfficientDet: Scalable and Efficient Object Detection (2020)
- Spectral Residual Saliency Detection (2007)
- MediaPipe: Framework for Building Perception Pipelines (2019)

---

## ✅ Conclusion

This plan provides a **comprehensive, well-researched approach** to adding intelligent object detection to Photo-Framer:

✅ **Minimal new dependencies** - Uses existing MediaPipe library
✅ **Performance-conscious** - < 10% overall slowdown, CPU-optimized
✅ **Incremental rollout** - Phased implementation with feature flags
✅ **Maintains existing quality** - No impact on face detection pipeline
✅ **Handles edge cases** - Multi-layer fallback (objects → saliency → basic)
✅ **Production-ready** - Docker integration, testing strategy, rollback plan

**Recommended Next Steps:**
1. Review this plan and approve approach
2. Set up test image library
3. Implement Phase 1 (core object detection)
4. Test extensively on diverse photo sets
5. Deploy behind feature flag
6. Monitor performance and iterate

**Estimated Development Time:**
- Phase 1: 2-3 days (core implementation + testing)
- Phase 2: 1 day (saliency detection)
- Testing & refinement: 2-3 days
- **Total: ~1 week for production-ready v7.0**
