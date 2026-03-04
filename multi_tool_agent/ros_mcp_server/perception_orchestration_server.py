#!/usr/bin/env python3
"""
Perception MCP Server for Object Detection

Using GroundingDINO for open-vocabulary detection (Phase 2)
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from PIL import Image
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict

mcp = FastMCP("perception-server")

# ==============================================================================
# Initialize GroundingDINO Model
# ==============================================================================

MODEL_DIR = Path(__file__).parent / "model"
CONFIG_PATH = MODEL_DIR / "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = MODEL_DIR / "groundingdino_swint_ogc.pth"

print(f"[Perception] Loading GroundingDINO model...", file=sys.stderr)
print(f"[Perception] Config: {CONFIG_PATH}", file=sys.stderr)
print(f"[Perception] Weights: {WEIGHTS_PATH}", file=sys.stderr)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Perception] Using device: {DEVICE}", file=sys.stderr)

try:
    GROUNDING_DINO_MODEL = load_model(
        model_config_path=str(CONFIG_PATH),
        model_checkpoint_path=str(WEIGHTS_PATH),
        device=DEVICE
    )
    print(f"[Perception] ✓ GroundingDINO model loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"[Perception] ✗ Failed to load GroundingDINO: {e}", file=sys.stderr)
    GROUNDING_DINO_MODEL = None


# ==============================================================================
# Helper: Color Verification
# ==============================================================================

def verify_color_from_bbox(image: np.ndarray, bbox: np.ndarray, label: str) -> str:
    """
    Verify actual color in bounding box using LAB color space analysis.
    LAB is more perceptually uniform than HSV and better for color discrimination.

    Corrects color labels when GroundingDINO mislabels objects.

    Args:
        image: BGR image from cv2.imread
        bbox: [x1, y1, x2, y2] bounding box coordinates
        label: Original label from GroundingDINO

    Returns:
        Corrected label with verified color
    """
    try:
        x1, y1, x2, y2 = bbox.astype(int)

        # Clamp to image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return label

        # Convert BGR to LAB color space
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        # Sample center 50% of bbox (avoid edges/shadows)
        roi_h, roi_w = roi_lab.shape[:2]
        if roi_h < 4 or roi_w < 4:
            return label  # Too small to sample

        center_roi = roi_lab[roi_h//4:3*roi_h//4, roi_w//4:3*roi_w//4]
        if center_roi.size == 0:
            return label

        # Extract LAB channels
        # L: Lightness (0-255, 0=black, 255=white)
        # a: Green-Red axis (-128 to 127, negative=green, positive=red)
        # b: Blue-Yellow axis (-128 to 127, negative=blue, positive=yellow)
        L_vals = center_roi[:,:,0].flatten()
        a_vals = center_roi[:,:,1].flatten()
        b_vals = center_roi[:,:,2].flatten()

        # Filter out very dark or very bright pixels (likely shadows or highlights)
        # Keep pixels with L in range [30, 225]
        valid_mask = (L_vals > 30) & (L_vals < 225)

        if valid_mask.sum() < 10:
            print(f"[Color Verify LAB] {label}: Not enough valid pixels, keeping original", file=sys.stderr)
            return label

        # Get median values (more robust than mean)
        median_L = np.median(L_vals[valid_mask])
        median_a = np.median(a_vals[valid_mask])
        median_b = np.median(b_vals[valid_mask])

        # Convert to float for calculations (-128 to 127 range)
        a_centered = float(median_a) - 128  # Center around 0
        b_centered = float(median_b) - 128

        print(f"[Color Verify LAB] {label}: L={median_L:.1f}, a={a_centered:.1f}, b={b_centered:.1f}", file=sys.stderr)

        # Calculate chroma (color intensity)
        chroma = np.sqrt(a_centered**2 + b_centered**2)

        # If chroma is too low, it's gray/achromatic
        if chroma < 15:
            print(f"[Color Verify LAB] {label}: Low chroma ({chroma:.1f}), achromatic object", file=sys.stderr)
            return label

        # Simple LAB color classification using a* and b* thresholds
        # Adapted for RLBench where red objects have blue tint (b < 0)
        detected_color = None

        # Red group — RLBench red always has a > 30
        if a_centered > 30:
            # RLBench bias: red objects often have b around -60 to -80 (blueish)
            if b_centered > 40:
                detected_color = "orange"
            else:
                detected_color = "red"  # Includes red with blue tint

        # Green group
        elif a_centered < -30:
            detected_color = "green"

        # Blue group (b strongly negative, a near zero)
        elif b_centered < -30:
            detected_color = "blue"

        # Yellow group (b strongly positive, a near zero)
        elif b_centered > 30:
            detected_color = "yellow"

        else:
            # Unclear color - keep original
            print(f"[Color Verify LAB] {label}: Unclear color (a={a_centered:.1f}, b={b_centered:.1f}), keeping original", file=sys.stderr)
            return label

        print(f"[Color Verify LAB] {label}: Detected='{detected_color}' (chroma={chroma:.1f})", file=sys.stderr)

        # Extract object type from label
        label_lower = label.lower()
        if "cube" in label_lower or "block" in label_lower:
            obj_type = "cube" if "cube" in label_lower else "block"
            corrected_label = f"{detected_color} {obj_type}"
        elif "sphere" in label_lower or "ball" in label_lower:
            obj_type = "sphere" if "sphere" in label_lower else "ball"
            corrected_label = f"{detected_color} {obj_type}"
        elif "tray" in label_lower or "bin" in label_lower:
            obj_type = "tray" if "tray" in label_lower else "bin"
            corrected_label = f"{detected_color} {obj_type}"
        elif "cup" in label_lower:
            corrected_label = f"{detected_color} cup"
        elif "trash" in label_lower or "rubbish" in label_lower:
            obj_type = "trash" if "trash" in label_lower else "rubbish"
            corrected_label = f"{detected_color} {obj_type}"
        elif "button" in label_lower:
            corrected_label = f"{detected_color} button"
        else:
            # Unknown object type - append detected color
            corrected_label = f"{detected_color} {label}"

        if corrected_label.lower() != label.lower():
            print(f"[Color Verify LAB] ✓ Corrected: '{label}' → '{corrected_label}'", file=sys.stderr)
        else:
            print(f"[Color Verify LAB] ✓ Verified: '{label}' (correct)", file=sys.stderr)

        return corrected_label

    except Exception as e:
        print(f"[Color Verify LAB] Error verifying {label}: {e}", file=sys.stderr)
        return label
    
# ==============================================================================
# Helper: Extract 3D position
# ==============================================================================

def _extract_3d_position(bbox, depth, point_cloud, fx, fy, cx, cy):
    """
    Helper function to extract 3D position and bounding box from depth-filtered foreground.

    Uses depth-based foreground segmentation to compute:
    - Centroid position (mean of foreground points)
    - 3D bounding box (min/max extents)
    - Dimensions (width, height, depth)

    Args:
        bbox: (x1, y1, x2, y2) 2D bounding box coordinates
        depth: Depth map array
        point_cloud: Point cloud array (H, W, 3) in base frame
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        tuple: (X_base, Y_base, Z_base, X_cam, Y_cam, Z_cam, bbox_3d_info)
        bbox_3d_info: dict with min/max/dims/volume or None
    """
    x1, y1, x2, y2 = bbox
    cx_pixel = (x1 + x2) // 2
    cy_pixel = (y1 + y2) // 2

    # Extract bounding box region
    bbox_depths = depth[y1:y2+1, x1:x2+1]
    bbox_pc = point_cloud[y1:y2+1, x1:x2+1]

    # Find minimum depth (closest point - likely the object)
    min_depth = bbox_depths.min()

    # Filter: keep only pixels within 5cm of nearest point (removes background)
    depth_threshold = 0.05  # 5cm tolerance
    foreground_mask = bbox_depths <= (min_depth + depth_threshold)

    # Get foreground positions
    foreground_positions = bbox_pc[foreground_mask]

    bbox_3d_info = None

    if len(foreground_positions) > 10:  # Need enough points for reliable bbox
        # Centroid: mean of foreground pixels
        X_base, Y_base, Z_base = foreground_positions.mean(axis=0)

        # Compute 3D bounding box extents
        min_bounds = foreground_positions.min(axis=0)
        max_bounds = foreground_positions.max(axis=0)

        # Dimensions
        width = float(max_bounds[0] - min_bounds[0])
        height = float(max_bounds[1] - min_bounds[1])
        depth_dim = float(max_bounds[2] - min_bounds[2])

        # Volume
        volume = width * height * depth_dim

        bbox_3d_info = {
            'min': min_bounds.tolist(),
            'max': max_bounds.tolist(),
            'center': [float(X_base), float(Y_base), float(Z_base)],
            'dimensions': [width, height, depth_dim],
            'volume': volume,
            'num_points': len(foreground_positions)
        }

    elif len(foreground_positions) > 0:
        # Few points - use mean but no reliable bbox
        X_base, Y_base, Z_base = foreground_positions.mean(axis=0)
    else:
        # Fallback to center pixel if filtering fails
        X_base, Y_base, Z_base = point_cloud[cy_pixel, cx_pixel]

    # Compute camera frame position (for backward compatibility)
    depth_value = depth[cy_pixel, cx_pixel]
    X_cam = (cx_pixel - cx) * depth_value / fx
    Y_cam = (cy_pixel - cy) * depth_value / fy
    Z_cam = depth_value

    return X_base, Y_base, Z_base, X_cam, Y_cam, Z_cam, bbox_3d_info


# ==============================================================================
# Main Task Object Detection
# ==============================================================================

@mcp.tool()
def detect_object_3d(text_prompt: str, rgb_path: str, depth_path: str,
                     intrinsics_path: str, pose_path: str, pointcloud_path: str = None) -> dict:
    """
    Detect object(s) in 3D using text prompt(s) with proper coordinate transformation

    PHASE 2: Supports multi-object detection with period-separated prompts
    Example: "red cube . red sphere" detects both objects and returns their 3D positions

    Args:
        text_prompt: Text description(s). Single: "red ball" or Multiple: "red cube . blue sphere"
        rgb_path: Path to RGB image
        depth_path: Path to depth map (.npy file)
        intrinsics_path: Path to camera intrinsics [fx, fy, cx, cy]
        pose_path: Path to camera pose (4x4 transformation matrix)
        pointcloud_path: Path to precomputed point cloud (optional but recommended)

    Returns:
        dict with 3D positions for all detected objects
    """
    print(f"[Tool: detect_object_3d] Detecting: {text_prompt}", file=sys.stderr)
    print(f"[Tool: detect_object_3d] RGB: {rgb_path}", file=sys.stderr)
    print(f"[Tool: detect_object_3d] Depth: {depth_path}", file=sys.stderr)

    try:
        # Load images and calibration data
        # Image is saved as BGR by RLBench server (cv2.imwrite format)
        depth = np.load(depth_path)
        intrinsics = np.load(intrinsics_path)  # [fx, fy, cx, cy]
        camera_pose = np.load(pose_path)  # 4x4 matrix: base to camera

        # Load image using GroundingDINO's load_image (returns BGR from cv2.imread)
        image_source, image_transformed = load_image(rgb_path)

        height, width = image_source.shape[:2]
        fx, fy, cx, cy = intrinsics

        # Handle negative focal lengths (CoppeliaSim convention)
        fx = abs(fx)
        fy = abs(fy)

        print(f"[Tool: detect_object_3d] Image size: {width}x{height}", file=sys.stderr)
        print(f"[Tool: detect_object_3d] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}", file=sys.stderr)

        # PHASE 2: GroundingDINO for text-based object detection
        if GROUNDING_DINO_MODEL is not None:
            print(f"[Tool: detect_object_3d] Using GroundingDINO with RGB color space", file=sys.stderr)

            # Run detection
            # Lowered thresholds to detect small/low-contrast objects (e.g., crumpled paper)
            BOX_THRESHOLD = 0.20  # Was 0.35, reduced for small objects
            TEXT_THRESHOLD = 0.15  # Was 0.25, more lenient text matching

            boxes, logits, phrases = predict(
                model=GROUNDING_DINO_MODEL,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

            if len(boxes) > 0:
                # Boxes are in normalized [cx, cy, w, h] format
                h, w = image_source.shape[:2]

                # PHASE 1: Return ALL detections instead of just best one
                print(f"[Tool: detect_object_3d] Found {len(boxes)} detection(s)", file=sys.stderr)

                all_detections = []
                for idx in range(len(boxes)):
                    box = boxes[idx].numpy()
                    confidence = logits[idx].item()
                    phrase = phrases[idx]

                    # Convert normalized [cx, cy, w, h] to pixel coordinates
                    cx_norm, cy_norm, w_norm, h_norm = box
                    cx_pixel = int(cx_norm * w)
                    cy_pixel = int(cy_norm * h)
                    w_pixel = int(w_norm * w)
                    h_pixel = int(h_norm * h)

                    # Calculate corners
                    x1 = max(0, cx_pixel - w_pixel // 2)
                    y1 = max(0, cy_pixel - h_pixel // 2)
                    x2 = min(w - 1, cx_pixel + w_pixel // 2)
                    y2 = min(h - 1, cy_pixel + h_pixel // 2)

                    # Note: Color filtering disabled - colors are random in RLBench
                    # TODO: Use RLBench ground truth for object validation instead

                    all_detections.append({
                        'phrase': phrase,
                        'confidence': confidence,
                        'center': (cx_pixel, cy_pixel),
                        'bbox': (x1, y1, x2, y2)
                    })

                    print(f"[Tool: detect_object_3d]   [{idx}] '{phrase}' at ({cx_pixel}, {cy_pixel}), conf={confidence:.3f}", file=sys.stderr)

                # Use highest confidence detection from filtered results (backward compatible)
                if len(all_detections) > 0:
                    # Find detection with highest confidence
                    best_detection = max(all_detections, key=lambda d: d['confidence'])
                    best_idx = all_detections.index(best_detection)

                    pixel_x, pixel_y = best_detection['center']
                    x1, y1, x2, y2 = best_detection['bbox']
                    confidence = best_detection['confidence']

                    # Verify and correct color for best detection
                    original_phrase = best_detection['phrase']
                    phrase = verify_color_from_bbox(image_source, np.array([x1, y1, x2, y2]), original_phrase)

                    print(f"[Tool: detect_object_3d] ✓ Using detection [{best_idx}]: '{phrase}'", file=sys.stderr)
                    print(f"[Tool: detect_object_3d]   Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]", file=sys.stderr)
                else:
                    # All detections filtered out - fallback
                    pixel_x = width // 2
                    pixel_y = height // 2
                    confidence = 0.0
                    phrase = "none"
                    print(f"[Tool: detect_object_3d] ⚠ All detections filtered out, using center: ({pixel_x}, {pixel_y})", file=sys.stderr)
            else:
                # No detection - fallback to image center
                pixel_x = width // 2
                pixel_y = height // 2
                confidence = 0.0
                phrase = "none"
                print(f"[Tool: detect_object_3d] ⚠ No objects detected, using center: ({pixel_x}, {pixel_y})", file=sys.stderr)
        else:
            # GroundingDINO not available - fallback to image center
            pixel_x = width // 2
            pixel_y = height // 2
            confidence = 0.0
            phrase = "fallback"
            print(f"[Tool: detect_object_3d] ⚠ GroundingDINO not loaded, using center: ({pixel_x}, {pixel_y})", file=sys.stderr)

        # PHASE 2: Calculate 3D positions for ALL detections
        objects_with_3d = []

        if pointcloud_path and os.path.exists(pointcloud_path):
            point_cloud = np.load(pointcloud_path)
            print(f"[Tool: detect_object_3d] Using RLBench point cloud", file=sys.stderr)

            # If we have detections, compute 3D position for each
            if 'all_detections' in locals() and len(all_detections) > 0:
                for idx, detection in enumerate(all_detections):
                    bbox = detection['bbox']

                    # Verify and correct color using LAB analysis
                    original_phrase = detection['phrase']
                    verified_phrase = verify_color_from_bbox(image_source, np.array(bbox), original_phrase)

                    X_base, Y_base, Z_base, X_cam, Y_cam, Z_cam, bbox_3d_info = _extract_3d_position(
                        bbox, depth, point_cloud, fx, fy, cx, cy
                    )

                    obj_dict = {
                        'phrase': verified_phrase,  # Use color-verified phrase
                        'confidence': detection['confidence'],
                        'position_3d': [float(X_base), float(Y_base), float(Z_base)],
                        'position_camera_frame': [float(X_cam), float(Y_cam), float(Z_cam)],
                        'detection_2d': {'x': int(detection['center'][0]), 'y': int(detection['center'][1])},
                        'bbox_2d': detection['bbox']
                    }

                    # Add 3D bounding box info if available
                    if bbox_3d_info is not None:
                        obj_dict['bbox_3d'] = bbox_3d_info

                    objects_with_3d.append(obj_dict)

                    # Enhanced logging with 3D bbox info
                    if bbox_3d_info:
                        dims = bbox_3d_info['dimensions']
                        print(f"[Tool: detect_object_3d]   [{idx}] '{verified_phrase}' pos=[{X_base:.3f}, {Y_base:.3f}, {Z_base:.3f}], dims=[{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}]m", file=sys.stderr)
                    else:
                        print(f"[Tool: detect_object_3d]   [{idx}] '{verified_phrase}' 3D pos: [{X_base:.3f}, {Y_base:.3f}, {Z_base:.3f}]", file=sys.stderr)

                # Use highest confidence detection for primary return values (backward compatible)
                best_object = objects_with_3d[best_idx]
                X_base, Y_base, Z_base = best_object['position_3d']
                X_cam, Y_cam, Z_cam = best_object['position_camera_frame']

            elif 'x1' in locals() and 'y1' in locals():  # Single detection fallback
                X_base, Y_base, Z_base, X_cam, Y_cam, Z_cam, bbox_3d_info = _extract_3d_position(
                    (x1, y1, x2, y2), depth, point_cloud, fx, fy, cx, cy
                )
            else:
                # No bounding box available (fallback mode)
                X_base, Y_base, Z_base = point_cloud[pixel_y, pixel_x]
                depth_value = depth[pixel_y, pixel_x]
                X_cam = (pixel_x - cx) * depth_value / fx
                Y_cam = (pixel_y - cy) * depth_value / fy
                Z_cam = depth_value
                print(f"[Tool: detect_object_3d] Using center pixel (no bbox)", file=sys.stderr)
        else:
            # Fallback: manual unprojection (but this has been giving wrong results)
            depth_value = depth[pixel_y, pixel_x]
            X_cam = (pixel_x - cx) * depth_value / fx
            Y_cam = (pixel_y - cy) * depth_value / fy
            Z_cam = depth_value

            # Simple transformation (this is still wrong but keeping as fallback)
            T_camera_to_base = np.linalg.inv(camera_pose)
            point_camera = np.array([X_cam, Y_cam, Z_cam, 1.0])
            point_base = T_camera_to_base @ point_camera
            X_base, Y_base, Z_base = point_base[:3]
            print(f"[Tool: detect_object_3d] WARNING: Using fallback transformation (may be incorrect)", file=sys.stderr)

        print(f"[Tool: detect_object_3d] ✓ Primary object at: [{X_base:.3f}, {Y_base:.3f}, {Z_base:.3f}]", file=sys.stderr)

        # Build return value with backward compatibility
        result = {
            "success": True,
            "object_name": text_prompt,
            "detected_phrase": phrase if 'phrase' in locals() else text_prompt,
            "position_3d": [float(X_base), float(Y_base), float(Z_base)],
            "position_camera_frame": [float(X_cam), float(Y_cam), float(Z_cam)],
            "detection_2d": {"x": int(pixel_x), "y": int(pixel_y)},
            "confidence": confidence if 'confidence' in locals() else 0.5,
            "method": "groundingdino" if GROUNDING_DINO_MODEL and 'phrase' in locals() and phrase != "none" else "fallback",
            "note": "Phase 2: Multi-object detection with 3D positions for all objects"
        }

        # PHASE 2: Add all objects with 3D positions
        if len(objects_with_3d) > 0:
            result["objects"] = objects_with_3d
            result["num_objects"] = len(objects_with_3d)
            print(f"[Tool: detect_object_3d] ✓ Returning {len(objects_with_3d)} object(s) with 3D positions", file=sys.stderr)

        # PHASE 1: Keep all_detections for backward compatibility
        if 'all_detections' in locals():
            result["all_detections"] = all_detections
            result["num_detections"] = len(all_detections)

        return result

    except Exception as e:
        print(f"[Tool: detect_object_3d] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Server Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70, file=sys.stderr)
    print("Perception MCP Server - Object Detection", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("Tools available:", file=sys.stderr)
    print("  1. detect_object_3d - Detect object and return 3D position", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("NOTE: Current implementation uses simplified detection", file=sys.stderr)
    print("TODO: Integrate GroundingDINO for open-vocabulary detection", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)

    mcp.run()
