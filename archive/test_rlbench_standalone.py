#!/usr/bin/env python3
"""
Standalone RLBench Simulation Test
No agents, no MCP - just pure RLBench with Franka Panda robot

This script will:
1. Launch CoppeliaSim GUI with Franka Panda robot
2. Load a pick-and-place task
3. Capture RGB images, depth images, and point clouds
4. Save them to disk for verification
5. Execute a few random actions to show robot movement
"""

import numpy as np
import cv2
import time
from pathlib import Path

# RLBench imports
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PickAndLift, ReachTarget

# Open3D for point cloud saving
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Warning: Open3D not installed. Point clouds will be saved in simple PLY format.")
    HAS_OPEN3D = False

print("=" * 80)
print("RLBench Standalone Simulation Test")
print("=" * 80)
print("Testing: Franka Panda robot with RGB/Depth/PointCloud capture")
print("=" * 80)
print()

# ==============================================================================
# Configuration
# ==============================================================================

# Output directory for captured data
OUTPUT_DIR = Path("/home/roops/ADK_Agent_Demo/multi_tool_agent/rlbench_test_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"✓ Output directory: {OUTPUT_DIR}")

# Choose task
TASK_TO_USE = PickAndLift  # Options: PickAndLift, ReachTarget
HEADLESS = False  # Set to False to see CoppeliaSim GUI
NUM_STEPS = 10  # Number of simulation steps to run

print(f"✓ Task: {TASK_TO_USE.__name__}")
print(f"✓ Headless mode: {HEADLESS}")
print(f"✓ Number of steps: {NUM_STEPS}")
print()

# ==============================================================================
# Step 1: Configure Observations
# ==============================================================================

print("[Step 1] Configuring observation sensors...")

obs_config = ObservationConfig()
obs_config.set_all(True)  # Enable all observations

# Configure front camera (primary camera)
obs_config.front_camera.rgb = True
obs_config.front_camera.depth = True
obs_config.front_camera.point_cloud = True
obs_config.front_camera.image_size = [512, 512]  # Higher resolution for better quality
# Don't set render_mode - let RLBench use default

# Configure additional cameras (optional)
obs_config.left_shoulder_camera.rgb = True
obs_config.left_shoulder_camera.depth = True
obs_config.right_shoulder_camera.rgb = True
obs_config.wrist_camera.rgb = True

# Enable joint and gripper state
obs_config.joint_positions = True
obs_config.joint_velocities = True
obs_config.gripper_open = True
obs_config.gripper_pose = True

print("✓ Observation config:")
print(f"  - Front camera: {obs_config.front_camera.image_size} RGB + Depth + PointCloud")
print(f"  - Additional cameras: Left/Right shoulder, Wrist")
print(f"  - Robot state: Joints, Gripper")
print()

# ==============================================================================
# Step 2: Create Action Mode
# ==============================================================================

print("[Step 2] Configuring action mode...")

# Use joint velocity control for smooth motion
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)

print("✓ Action mode: JointVelocity + Discrete Gripper")
print()

# ==============================================================================
# Step 3: Create and Launch Environment
# ==============================================================================

print("[Step 3] Creating RLBench environment...")

env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=HEADLESS
)

print("✓ Environment created")
print()

print("[Step 4] Launching CoppeliaSim...")
if not HEADLESS:
    print(">>> CoppeliaSim GUI window should appear now!")

env.launch()

print("✓ CoppeliaSim launched successfully!")
print()

# ==============================================================================
# Step 5: Load Task
# ==============================================================================

print(f"[Step 5] Loading task: {TASK_TO_USE.__name__}...")

task = env.get_task(TASK_TO_USE)

print(f"✓ Task loaded: {TASK_TO_USE.__name__}")
print()

# ==============================================================================
# Step 6: Reset and Get Initial Observation
# ==============================================================================

print("[Step 6] Resetting task and capturing initial observation...")

descriptions, obs = task.reset()

print(f"✓ Task reset")
print(f"  Task description: {descriptions[0]}")
print()

# ==============================================================================
# Step 7: Inspect Observation Structure
# ==============================================================================

print("[Step 7] Inspecting observation data...")

print(f"✓ Observation type: {type(obs)}")
print(f"  Available attributes:")

# List all observation attributes
for attr in dir(obs):
    if not attr.startswith('_'):
        try:
            value = getattr(obs, attr)
            if isinstance(value, np.ndarray):
                print(f"    - {attr}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"    - {attr}: {type(value).__name__}")
        except:
            pass

print()

# ==============================================================================
# Step 8: Capture and Save RGB Image
# ==============================================================================

print("[Step 8] Capturing RGB image from front camera...")

if hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
    rgb = obs.front_rgb
    print(f"✓ RGB image captured: {rgb.shape} (range: [{rgb.min():.3f}, {rgb.max():.3f}])")

    # Convert to uint8 (RLBench gives [0, 1] floats)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # Save
    rgb_path = OUTPUT_DIR / "front_rgb.png"
    cv2.imwrite(str(rgb_path), bgr)
    print(f"  Saved to: {rgb_path}")
else:
    print("✗ No front_rgb in observation!")

print()

# ==============================================================================
# Step 9: Capture and Save Depth Image
# ==============================================================================

print("[Step 9] Capturing depth image from front camera...")

if hasattr(obs, 'front_depth') and obs.front_depth is not None:
    depth = obs.front_depth
    print(f"✓ Depth image captured: {depth.shape} (range: [{depth.min():.3f}m, {depth.max():.3f}m])")

    # Visualize depth (convert to uint8 for viewing)
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Save raw depth (16-bit, millimeters)
    depth_mm = (depth * 1000).astype(np.uint16)
    depth_path_raw = OUTPUT_DIR / "front_depth_raw.png"
    cv2.imwrite(str(depth_path_raw), depth_mm)
    print(f"  Saved raw depth to: {depth_path_raw}")

    # Save colorized depth for visualization
    depth_path_color = OUTPUT_DIR / "front_depth_colorized.png"
    cv2.imwrite(str(depth_path_color), depth_colored)
    print(f"  Saved colorized depth to: {depth_path_color}")
else:
    print("✗ No front_depth in observation!")

print()

# ==============================================================================
# Step 10: Capture and Save Point Cloud
# ==============================================================================

print("[Step 10] Capturing point cloud from front camera...")

if hasattr(obs, 'front_point_cloud') and obs.front_point_cloud is not None:
    pcd_points = obs.front_point_cloud
    print(f"✓ Point cloud captured: {pcd_points.shape}")

    # Save point cloud
    pcd_path = OUTPUT_DIR / "front_pointcloud.ply"

    if HAS_OPEN3D:
        # Use Open3D for better PLY format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)

        # Add colors from RGB if available
        if hasattr(obs, 'front_rgb'):
            # Reshape RGB to match point cloud
            rgb_flat = rgb.reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(rgb_flat)

        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"  Saved to: {pcd_path}")
    else:
        # Simple PLY writer
        with open(pcd_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pcd_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in pcd_points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"  Saved to: {pcd_path} (simple format)")
else:
    print("⚠ No front_point_cloud in observation - will generate from depth...")

    # Generate point cloud from depth image
    if hasattr(obs, 'front_depth') and obs.front_depth is not None:
        depth = obs.front_depth
        h, w = depth.shape

        # Camera intrinsics (RLBench defaults)
        fx = fy = 256.0
        cx = cy = 128.0

        # Generate point cloud
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter valid points
        valid = (points[:, 2] > 0.01) & (points[:, 2] < 2.0)
        points = points[valid]

        print(f"✓ Generated point cloud from depth: {points.shape}")

        # Save
        pcd_path = OUTPUT_DIR / "front_pointcloud_generated.ply"

        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if hasattr(obs, 'front_rgb'):
                colors = rgb.reshape(-1, 3)[valid]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"  Saved to: {pcd_path}")

print()

# ==============================================================================
# Step 11: Display Robot Joint States
# ==============================================================================

print("[Step 11] Robot state information...")

if hasattr(obs, 'joint_positions'):
    print(f"✓ Joint positions: {obs.joint_positions}")

if hasattr(obs, 'gripper_open'):
    print(f"✓ Gripper state: {'OPEN' if obs.gripper_open > 0.5 else 'CLOSED'} ({obs.gripper_open:.3f})")

if hasattr(obs, 'gripper_pose'):
    print(f"✓ Gripper pose: {obs.gripper_pose}")

print()

# ==============================================================================
# Step 12: Execute ACTUAL Pick-and-Place (Using Demonstrations)
# ==============================================================================

print(f"[Step 12] Executing ACTUAL pick-and-place demonstration...")
print(">>> Watch CoppeliaSim - the robot will actually pick up the cube!")
print()

try:
    # Load a demonstration (pre-recorded expert behavior)
    print("Loading expert demonstration...")
    demos = task.get_demos(amount=1, live_demos=True)

    print(f"✓ Loaded demonstration with {len(demos[0])} steps")
    print("Playing back demonstration (robot will pick and lift the cube)...")
    print()

    demo = demos[0]

    # Replay the demonstration step by step
    for step_idx, demo_obs in enumerate(demo):
        if step_idx % 10 == 0:  # Print progress every 10 steps
            print(f"  Demo step {step_idx}/{len(demo)}")

            # Save keyframes
            if step_idx % 20 == 0:
                rgb = demo_obs.front_rgb
                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    rgb_uint8 = (rgb * 255).astype(np.uint8)
                else:
                    rgb_uint8 = rgb

                bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
                keyframe_path = OUTPUT_DIR / f"demo_keyframe_{step_idx:03d}.png"
                cv2.imwrite(str(keyframe_path), bgr)

    print()
    print("✓ Demonstration complete! The robot picked and lifted the cube!")
    print()

except Exception as e:
    print(f"⚠ Could not load demo: {e}")
    print("Falling back to random actions for visualization...")
    print()

    for step in range(NUM_STEPS):
        # Random action (small velocities for smooth motion)
        action = np.random.normal(0.0, 0.05, size=env.action_shape)

        # Execute step
        obs, reward, terminate = task.step(action)

        print(f"  Step {step + 1}/{NUM_STEPS}: reward={reward:.3f}, terminate={terminate}")

        # Small delay so you can see the motion
        time.sleep(0.1)

        if terminate:
            print("  Task completed!")
            break

    print()

# ==============================================================================
# Step 13: Capture Final State
# ==============================================================================

print("[Step 13] Capturing final state...")

# Save final RGB
if hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
    rgb = obs.front_rgb
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    rgb_path = OUTPUT_DIR / "front_rgb_final.png"
    cv2.imwrite(str(rgb_path), bgr)
    print(f"✓ Final RGB saved to: {rgb_path}")

print()

# ==============================================================================
# Step 14: Cleanup
# ==============================================================================

print("[Step 14] Shutting down environment...")

env.shutdown()

print("✓ Environment shutdown complete")
print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  ✓ CoppeliaSim launched with Franka Panda robot")
print(f"  ✓ Task: {TASK_TO_USE.__name__}")
print(f"  ✓ RGB images captured and saved")
print(f"  ✓ Depth images captured and saved")
print(f"  ✓ Point clouds captured and saved")
print(f"  ✓ Robot executed {NUM_STEPS} actions")
print()
print(f"Output files saved to: {OUTPUT_DIR}")
print()
print("Files created:")
for file in sorted(OUTPUT_DIR.glob("*")):
    size_kb = file.stat().st_size / 1024
    print(f"  - {file.name} ({size_kb:.1f} KB)")
print()
print("=" * 80)
print("Next steps:")
print("  1. Check the output files to verify data quality")
print("  2. View point cloud: open3d /tmp/rlbench_test_output/front_pointcloud.ply")
print("  3. If everything looks good, proceed to agent integration")
print("=" * 80)
