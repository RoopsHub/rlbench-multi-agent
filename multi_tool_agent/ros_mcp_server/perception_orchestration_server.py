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
# Tool: Detect Object 3D
# ==============================================================================

@mcp.tool()
def detect_object_3d(text_prompt: str, rgb_path: str, depth_path: str,
                     intrinsics_path: str, pose_path: str, pointcloud_path: str = None) -> dict:
    """
    Detect object in 3D using text prompt with proper coordinate transformation

    CURRENT: Uses image center for detection (Phase 1 - validates transforms)
    TODO: Integrate GroundingDINO for actual object detection (Phase 2)

    Args:
        text_prompt: Text description of object (e.g., "red ball", "blue cube")
        rgb_path: Path to RGB image
        depth_path: Path to depth map (.npy file)
        intrinsics_path: Path to camera intrinsics [fx, fy, cx, cy]
        pose_path: Path to camera pose (4x4 transformation matrix)

    Returns:
        dict with 3D position [x, y, z] in robot base frame
    """
    print(f"[Tool: detect_object_3d] Detecting: {text_prompt}", file=sys.stderr)
    print(f"[Tool: detect_object_3d] RGB: {rgb_path}", file=sys.stderr)
    print(f"[Tool: detect_object_3d] Depth: {depth_path}", file=sys.stderr)

    try:
        # Load images and calibration data
        # Image is saved as BGR by RLBench server - keep as BGR throughout
        bgr = cv2.imread(str(rgb_path))
        depth = np.load(depth_path)
        intrinsics = np.load(intrinsics_path)  # [fx, fy, cx, cy]
        camera_pose = np.load(pose_path)  # 4x4 matrix: base to camera

        height, width = bgr.shape[:2]
        fx, fy, cx, cy = intrinsics

        # Handle negative focal lengths (CoppeliaSim convention)
        fx = abs(fx)
        fy = abs(fy)

        print(f"[Tool: detect_object_3d] Image size: {width}x{height}", file=sys.stderr)
        print(f"[Tool: detect_object_3d] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}", file=sys.stderr)

        # PHASE 2: GroundingDINO for text-based object detection
        if GROUNDING_DINO_MODEL is not None:
            print(f"[Tool: detect_object_3d] Using GroundingDINO with BGR color space", file=sys.stderr)

            # Use BGR directly for detection (better color distinction)
            image_source = bgr

            # Get image transform from load_image but use our BGR data
            _, image_transformed = load_image(rgb_path)

            # Run detection
            BOX_THRESHOLD = 0.35
            TEXT_THRESHOLD = 0.25

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

                # Get box with highest confidence
                # logits is 1D tensor with one score per detection
                best_idx = logits.argmax()
                box = boxes[best_idx].numpy()  # Keep normalized for now
                confidence = logits[best_idx].item()
                phrase = phrases[best_idx]

                # Convert normalized [cx, cy, w, h] to pixel coordinates
                cx_norm, cy_norm, w_norm, h_norm = box
                cx_pixel = int(cx_norm * w)
                cy_pixel = int(cy_norm * h)
                w_pixel = int(w_norm * w)
                h_pixel = int(h_norm * h)

                # Use center directly
                pixel_x = cx_pixel
                pixel_y = cy_pixel

                # Calculate corners for display
                x1 = cx_pixel - w_pixel // 2
                y1 = cy_pixel - h_pixel // 2
                x2 = cx_pixel + w_pixel // 2
                y2 = cy_pixel + h_pixel // 2

                print(f"[Tool: detect_object_3d] ✓ Detected '{phrase}' at ({pixel_x}, {pixel_y})", file=sys.stderr)
                print(f"[Tool: detect_object_3d]   Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]", file=sys.stderr)
                print(f"[Tool: detect_object_3d]   Confidence: {confidence:.3f}", file=sys.stderr)
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

        # Use RLBench's precomputed point cloud (already in world/base frame!)
        if pointcloud_path and os.path.exists(pointcloud_path):
            point_cloud = np.load(pointcloud_path)

            # IMPROVED: Use depth-filtered average instead of single center pixel
            # This handles bounding boxes that include background/foreground pixels
            if 'x1' in locals() and 'y1' in locals():  # Check if we have a valid bounding box
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

                if len(foreground_positions) > 0:
                    # Use mean of foreground pixels (more robust than single pixel)
                    X_base, Y_base, Z_base = foreground_positions.mean(axis=0)
                    print(f"[Tool: detect_object_3d] Using depth-filtered average ({len(foreground_positions)}/{foreground_mask.size} pixels)", file=sys.stderr)
                else:
                    # Fallback to center pixel if filtering fails
                    X_base, Y_base, Z_base = point_cloud[pixel_y, pixel_x]
                    print(f"[Tool: detect_object_3d] Using center pixel (filtering failed)", file=sys.stderr)
            else:
                # No bounding box available (fallback mode)
                X_base, Y_base, Z_base = point_cloud[pixel_y, pixel_x]
                print(f"[Tool: detect_object_3d] Using center pixel (no bbox)", file=sys.stderr)

            # Also compute camera frame for return value
            depth_value = depth[pixel_y, pixel_x]
            X_cam = (pixel_x - cx) * depth_value / fx
            Y_cam = (pixel_y - cy) * depth_value / fy
            Z_cam = depth_value

            print(f"[Tool: detect_object_3d] Using RLBench point cloud", file=sys.stderr)
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

        print(f"[Tool: detect_object_3d] ✓ Base frame: [{X_base:.3f}, {Y_base:.3f}, {Z_base:.3f}]", file=sys.stderr)

        return {
            "success": True,
            "object_name": text_prompt,
            "detected_phrase": phrase if 'phrase' in locals() else text_prompt,
            "position_3d": [float(X_base), float(Y_base), float(Z_base)],
            "position_camera_frame": [float(X_cam), float(Y_cam), float(Z_cam)],
            "detection_2d": {"x": int(pixel_x), "y": int(pixel_y)},
            "confidence": confidence if 'confidence' in locals() else 0.5,
            "method": "groundingdino" if GROUNDING_DINO_MODEL and 'phrase' in locals() and phrase != "none" else "fallback",
            "note": "Phase 2: Using GroundingDINO for text-based object detection with bounding boxes."
        }

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
