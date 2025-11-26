#!/usr/bin/env python3
"""
Analyze detection error by examining point cloud data at detected locations.
This script helps identify the source of the ~2cm positioning error.
"""

import numpy as np
import sys

def analyze_detection(rgb_path, depth_path, pc_path, detection_pixel, bbox, ground_truth, detected_pos):
    """Analyze a single detection case."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {rgb_path.split('/')[-1]}")
    print(f"{'='*70}")

    # Load data
    depth = np.load(depth_path)
    pc = np.load(pc_path)

    pixel_x, pixel_y = detection_pixel
    x1, y1, x2, y2 = bbox

    print(f"\nGround Truth:    [{ground_truth[0]:.3f}, {ground_truth[1]:.3f}, {ground_truth[2]:.3f}]")
    print(f"Detected (used): [{detected_pos[0]:.3f}, {detected_pos[1]:.3f}, {detected_pos[2]:.3f}]")
    print(f"Error:           [{detected_pos[0]-ground_truth[0]:.3f}, {detected_pos[1]-ground_truth[1]:.3f}, {detected_pos[2]-ground_truth[2]:.3f}]")
    print(f"Error magnitude: {np.linalg.norm(np.array(detected_pos) - np.array(ground_truth)):.3f} meters")

    print(f"\nDetection Info:")
    print(f"  Detection pixel: ({pixel_x}, {pixel_y})")
    print(f"  Bounding box: [{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1} pixels)")
    print(f"  Depth at pixel: {depth[pixel_y, pixel_x]:.4f}")

    # Sample bounding box corners and center
    print(f"\nBounding box sampling (3D positions):")
    sample_points = [
        ("Top-left", x1, y1),
        ("Top-center", (x1+x2)//2, y1),
        ("Top-right", x2, y1),
        ("Mid-left", x1, (y1+y2)//2),
        ("Center", pixel_x, pixel_y),
        ("Mid-right", x2, (y1+y2)//2),
        ("Bot-left", x1, y2),
        ("Bot-center", (x1+x2)//2, y2),
        ("Bot-right", x2, y2),
    ]

    for label, px, py in sample_points:
        if 0 <= px < pc.shape[1] and 0 <= py < pc.shape[0]:
            pos = pc[py, px]
            err = np.linalg.norm(pos - np.array(ground_truth))
            marker = " ✓" if err < 0.01 else ""
            print(f"  {label:12s} ({px:3d}, {py:3d}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] err={err:.3f}{marker}")

    # Average position within bounding box
    print(f"\nBounding box statistics:")
    bbox_positions = []
    for y in range(max(0, y1), min(pc.shape[0], y2+1)):
        for x in range(max(0, x1), min(pc.shape[1], x2+1)):
            bbox_positions.append(pc[y, x])

    if bbox_positions:
        bbox_positions = np.array(bbox_positions)
        avg_pos = bbox_positions.mean(axis=0)
        std_pos = bbox_positions.std(axis=0)
        min_pos = bbox_positions.min(axis=0)
        max_pos = bbox_positions.max(axis=0)

        print(f"  Mean:   [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}]")
        print(f"  Std:    [{std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f}]")
        print(f"  Min:    [{min_pos[0]:.3f}, {min_pos[1]:.3f}, {min_pos[2]:.3f}]")
        print(f"  Max:    [{max_pos[0]:.3f}, {max_pos[1]:.3f}, {max_pos[2]:.3f}]")
        print(f"  Range:  [{max_pos[0]-min_pos[0]:.3f}, {max_pos[1]-min_pos[1]:.3f}, {max_pos[2]-min_pos[2]:.3f}]")

        avg_err = np.linalg.norm(avg_pos - np.array(ground_truth))
        print(f"\n  Error from mean to ground truth: {avg_err:.3f} meters")
        print(f"  Mean position: [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}]")

        if avg_err < np.linalg.norm(np.array(detected_pos) - np.array(ground_truth)):
            print(f"  💡 INSIGHT: Using mean position would reduce error by {(np.linalg.norm(np.array(detected_pos) - np.array(ground_truth)) - avg_err)*1000:.1f}mm")

        # Find best pixel in bounding box
        best_err = float('inf')
        best_pixel = None
        best_pos = None
        for y in range(max(0, y1), min(pc.shape[0], y2+1)):
            for x in range(max(0, x1), min(pc.shape[1], x2+1)):
                pos = pc[y, x]
                err = np.linalg.norm(pos - np.array(ground_truth))
                if err < best_err:
                    best_err = err
                    best_pixel = (x, y)
                    best_pos = pos

        print(f"\n  Best pixel in bbox: ({best_pixel[0]}, {best_pixel[1]}) with error {best_err:.3f} meters")
        print(f"  Best position: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")
        if best_err < np.linalg.norm(np.array(detected_pos) - np.array(ground_truth)):
            improvement = (np.linalg.norm(np.array(detected_pos) - np.array(ground_truth)) - best_err) * 1000
            print(f"  💡 INSIGHT: Best pixel would reduce error by {improvement:.1f}mm")
            print(f"  Pixel offset from center: ({best_pixel[0]-pixel_x}, {best_pixel[1]-pixel_y})")


if __name__ == "__main__":
    base_path = "multi_tool_agent/ros_mcp_server/sensor_data"

    print("\n" + "="*70)
    print("DETECTION ERROR ANALYSIS")
    print("="*70)

    # Test Case 1: Pink ball detected instead of red (WRONG OBJECT)
    print("\n\n### TEST CASE 1: Pink vs Red Ball Confusion ###")
    analyze_detection(
        rgb_path=f"{base_path}/rgb_1764060533.6425023.png",
        depth_path=f"{base_path}/depth_1764060533.7157047.npy",
        pc_path=f"{base_path}/pointcloud_1764060533.7167156.npy",
        detection_pixel=(345, 318),
        bbox=(333, 306, 357, 330),
        ground_truth=[0.195, -0.090, 1.148],
        detected_pos=[0.097, 0.182, 0.856]
    )
    print("\n⚠️  NOTE: This case detected the wrong object (pink instead of red)")

    # Test Case 2: Correct red ball detection (2cm error)
    print("\n\n### TEST CASE 2: Correct Detection with 2cm Error ###")
    analyze_detection(
        rgb_path=f"{base_path}/rgb_1764060789.7572854.png",
        depth_path=f"{base_path}/depth_1764060789.8332436.npy",
        pc_path=f"{base_path}/pointcloud_1764060789.8339305.npy",
        detection_pixel=(357, 368),
        bbox=(342, 353, 372, 383),
        ground_truth=[0.355, 0.172, 0.907],
        detected_pos=[0.375, 0.167, 0.921]
    )

    print("\n\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print("""
1. **Pink vs Red Confusion**: GroundingDINO with prompt "red target"
   sometimes detects pink balls. Consider more specific prompts.

2. **Systematic Error**: Even with correct detection, ~2cm error persists.
   This analysis will show if:
   - Using bbox mean instead of center improves accuracy
   - Best pixel in bbox is significantly better
   - Object is off-center in the bounding box

3. **Next Steps**: Based on results, we can:
   - Use weighted average of bbox positions
   - Apply offset correction based on bbox position
   - Use median instead of center pixel
   - Refine bbox to object mask using color/segmentation
""")
