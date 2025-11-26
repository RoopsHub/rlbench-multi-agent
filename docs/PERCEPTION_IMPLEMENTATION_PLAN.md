# Proper Perception Implementation Plan

## Current Problems

1. **No actual object detection** - Uses image center (assumes object is always centered)
2. **Hardcoded camera intrinsics** - fx=300 is approximate, not actual
3. **Missing coordinate transform** - Camera coords ≠ Robot base coords
4. **Single camera only** - Only using front_camera, not utilizing other viewpoints

## Proper Implementation

### Step 1: Get Camera Intrinsics from RLBench

```python
from rlbench.environment import Environment

env = Environment(...)
obs = task.reset()

# Get camera intrinsics
camera_intrinsics = obs.front_camera_intrinsics
# Returns: [fx, fy, cx, cy] where:
#   fx, fy = focal lengths (pixels)
#   cx, cy = principal point (pixels)

# Get camera pose (position + orientation in robot base frame)
camera_pose = obs.front_camera_pose
# Returns: 4x4 transformation matrix from base to camera
```

### Step 2: Object Detection with GroundingDINO

```python
import groundingdino
from groundingdino.util.inference import load_model, predict

# Load model
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "weights/groundingdino_swint_ogc.pth")

# Detect object
rgb_image = Image.open(rgb_path)
text_prompt = "red ball"

boxes, logits, phrases = predict(
    model=model,
    image=rgb_image,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25
)

# boxes: normalized [x1, y1, x2, y2] in range [0, 1]
# Convert to pixel coordinates
height, width = rgb_image.size
x1, y1, x2, y2 = boxes[0] * [width, height, width, height]

# Calculate center
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

### Step 3: 3D Localization (Pixel + Depth → 3D Camera Coords)

```python
import numpy as np

# Get depth at object center
depth_map = np.load(depth_path)
depth_value = depth_map[int(cy), int(cx)]  # meters

# Unproject to 3D camera coordinates
fx, fy, cx_offset, cy_offset = camera_intrinsics

X_cam = (cx - cx_offset) * depth_value / fx
Y_cam = (cy - cy_offset) * depth_value / fy
Z_cam = depth_value

point_camera = np.array([X_cam, Y_cam, Z_cam, 1.0])  # Homogeneous coords
```

### Step 4: Transform to Robot Base Frame

```python
# Camera pose is 4x4 transformation matrix from base to camera
# We need inverse: camera to base
T_base_to_camera = camera_pose  # From RLBench observation
T_camera_to_base = np.linalg.inv(T_base_to_camera)

# Transform point from camera frame to base frame
point_base = T_camera_to_base @ point_camera

X_base, Y_base, Z_base = point_base[:3]
```

### Step 5: Handle Multiple Cameras (Optional)

```python
# For difficult tasks, use multiple cameras and fuse detections

cameras = ['front_camera', 'wrist_camera', 'overhead_camera']
detections_3d = []

for camera_name in cameras:
    rgb = getattr(obs, f'{camera_name}_rgb')
    depth = getattr(obs, f'{camera_name}_depth')
    intrinsics = getattr(obs, f'{camera_name}_intrinsics')
    pose = getattr(obs, f'{camera_name}_pose')

    # Detect in 2D
    boxes = detect_with_groundingdino(rgb, text_prompt)

    if boxes:
        # Convert to 3D
        point_3d = unproject_and_transform(boxes[0], depth, intrinsics, pose)
        detections_3d.append(point_3d)

# Fuse detections (average or select most confident)
final_position = np.mean(detections_3d, axis=0)
```

## Implementation Priority

### Phase 1: Single Camera + Proper Intrinsics ✅ DO THIS FIRST
- Get camera intrinsics from RLBench
- Get camera pose from RLBench
- Implement proper coordinate transformation
- **Still use image center for now** (validate transform is correct)

### Phase 2: Add GroundingDINO
- Install GroundingDINO
- Integrate 2D detection
- Replace image center with actual bounding box center

### Phase 3: Multi-Camera Fusion (Optional)
- Use multiple cameras for robustness
- Fuse detections from different viewpoints

## Code Changes Needed

### 1. Update `rlbench_orchestration_server.py`

```python
@mcp.tool()
def get_camera_observation() -> dict:
    """Returns RGB, depth, intrinsics, and pose"""

    # Get camera data
    rgb = obs.front_camera_rgb
    depth = obs.front_camera_depth
    intrinsics = obs.front_camera_intrinsics  # [fx, fy, cx, cy]
    pose = obs.front_camera_pose  # 4x4 matrix

    # Save all data
    np.save(intrinsics_path, intrinsics)
    np.save(pose_path, pose)

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "intrinsics_path": str(intrinsics_path),  # NEW
        "pose_path": str(pose_path),  # NEW
        "camera_name": "front_camera"
    }
```

### 2. Update `perception_orchestration_server.py`

```python
@mcp.tool()
def detect_object_3d(text_prompt, rgb_path, depth_path,
                     intrinsics_path, pose_path):
    """Proper 3D detection with coordinate transform"""

    # Load camera calibration
    intrinsics = np.load(intrinsics_path)  # [fx, fy, cx, cy]
    pose = np.load(pose_path)  # 4x4 matrix

    # 2D Detection (Phase 1: use center, Phase 2: GroundingDINO)
    if USE_GROUNDING_DINO:
        boxes = detect_with_groundingdino(rgb_path, text_prompt)
        cx, cy = get_box_center(boxes[0])
    else:
        # Simplified: use image center
        height, width = rgb.shape[:2]
        cx, cy = width/2, height/2

    # 3D Localization
    depth = np.load(depth_path)
    depth_value = depth[int(cy), int(cx)]

    fx, fy, cx_offset, cy_offset = intrinsics
    X_cam = (cx - cx_offset) * depth_value / fx
    Y_cam = (cy - cy_offset) * depth_value / fy
    Z_cam = depth_value

    # Transform to base frame
    T_camera_to_base = np.linalg.inv(pose)
    point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
    point_base = T_camera_to_base @ point_cam

    return {
        "position_3d": point_base[:3].tolist(),  # In robot base frame!
        "camera_frame_pos": [X_cam, Y_cam, Z_cam],
        "detection_2d": {"x": cx, "y": cy}
    }
```

## Testing Strategy

1. **Validate Transform First:**
   - Use image center detection
   - Check if gripper moves to reasonable position
   - If gripper goes to wrong place → transform is broken

2. **Add GroundingDINO:**
   - Test 2D detection separately
   - Visualize bounding boxes on RGB image
   - Then integrate with 3D pipeline

3. **Multi-Camera (Later):**
   - Only if single camera isn't accurate enough
   - Compare detections from different viewpoints

## Current Status

- ❌ Using hardcoded intrinsics (fx=300)
- ❌ No coordinate transformation
- ❌ No actual object detection
- ✅ Basic pipeline structure in place

## Next Steps

1. Update `get_camera_observation()` to return intrinsics + pose
2. Update `detect_object_3d()` to use proper transformation
3. Test with image center (Phase 1)
4. Add GroundingDINO (Phase 2)
