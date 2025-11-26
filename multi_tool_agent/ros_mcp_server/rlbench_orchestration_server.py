#!/usr/bin/env python3
"""
RLBench MCP Server for Multi-Agent Orchestration
Task-agnostic tools for perception-based manipulation (NO DEMOS)

Tools:
- get_camera_observation: Capture RGB + depth images
- move_to_position: Move gripper to Cartesian position
- control_gripper: Open/close gripper
- lift_gripper: Lift gripper by height
- get_current_state: Get robot state
- reset_task: Reset environment
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from PIL import Image

# RLBench imports
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PickAndLift
from rlbench.backend.exceptions import InvalidActionError

# ==============================================================================
# Global State
# ==============================================================================

mcp = FastMCP("rlbench-orchestration-server")

_ENV = None
_TASK = None
_CURRENT_OBS = None
_TASK_CLASS = ReachTarget  # Default task

# Output directory for sensor data
OUTPUT_DIR = Path(__file__).parent / "sensor_data"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_environment():
    """Get or create RLBench environment"""
    global _ENV, _TASK, _CURRENT_OBS

    if _ENV is None:
        print("[RLBench] Initializing environment...", file=sys.stderr)

        # Configure observations - enable cameras
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.front_camera.image_size = [512, 512]
        obs_config.front_camera.rgb = True
        obs_config.front_camera.depth = True
        obs_config.front_camera.point_cloud = True  # Enable point cloud computation by RLBench

        # Use EndEffectorPoseViaIK for Cartesian control
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(),
            gripper_action_mode=Discrete()
        )

        _ENV = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=False  # Show CoppeliaSim
        )

        _ENV.launch()
        print("[RLBench] ✓ CoppeliaSim launched", file=sys.stderr)

        # Load task
        _TASK = _ENV.get_task(_TASK_CLASS)
        print(f"[RLBench] ✓ Loaded task: {_TASK_CLASS.__name__}", file=sys.stderr)

        # Reset to get initial observation
        descriptions, _CURRENT_OBS = _TASK.reset()
        print(f"[RLBench] ✓ Task: {descriptions[0]}", file=sys.stderr)

    return _ENV, _TASK, _CURRENT_OBS


# ==============================================================================
# Tool 1: Get Camera Observation
# ==============================================================================

@mcp.tool()
def get_camera_observation() -> dict:
    """
    Capture RGB and depth images from robot camera with calibration data

    Returns:
        dict with rgb_path, depth_path, intrinsics, and camera pose
    """
    global _CURRENT_OBS

    print("[Tool: get_camera_observation] Capturing images...", file=sys.stderr)

    env, task, obs = get_environment()

    if obs is None:
        return {"success": False, "error": "No observation available"}

    try:
        # Generate timestamp in YYYYMMDD_HHMMSS format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get RGB image and convert to BGR for entire perception pipeline
        rgb = obs.front_rgb
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Convert RGB to BGR - we'll use BGR throughout perception
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

        # Save directly as BGR (cv2.imwrite will save the BGR data as-is)
        # The perception pipeline will load and use BGR directly
        bgr_path = OUTPUT_DIR / f"image_{timestamp}.png"
        cv2.imwrite(str(bgr_path), bgr)
        print(f"[Tool: get_camera_observation] ✓ Image saved as BGR: {bgr_path}", file=sys.stderr)

        # Get depth image
        depth_image = obs.front_depth
        depth_path = OUTPUT_DIR / f"depth_{timestamp}.npy"
        np.save(depth_path, depth_image)
        print(f"[Tool: get_camera_observation] ✓ Depth saved: {depth_path}", file=sys.stderr)

        # Get camera intrinsics from misc dict (3x3 matrix)
        intrinsics_matrix = obs.misc['front_camera_intrinsics']
        # Extract [fx, fy, cx, cy] from 3x3 matrix
        intrinsics = np.array([
            intrinsics_matrix[0, 0],  # fx
            intrinsics_matrix[1, 1],  # fy
            intrinsics_matrix[0, 2],  # cx
            intrinsics_matrix[1, 2]   # cy
        ])
        intrinsics_path = OUTPUT_DIR / f"intrinsics_{timestamp}.npy"
        np.save(intrinsics_path, intrinsics)
        print(f"[Tool: get_camera_observation] ✓ Intrinsics: fx={intrinsics[0]:.1f}, fy={intrinsics[1]:.1f}", file=sys.stderr)

        # Get camera extrinsics (4x4 transformation matrix from base to camera)
        camera_pose = obs.misc['front_camera_extrinsics']
        pose_path = OUTPUT_DIR / f"camera_pose_{timestamp}.npy"
        np.save(pose_path, camera_pose)
        print(f"[Tool: get_camera_observation] ✓ Camera pose saved", file=sys.stderr)

        # Get point cloud (already transformed to world frame by RLBench!)
        point_cloud = obs.front_point_cloud
        pc_path = OUTPUT_DIR / f"pointcloud_{timestamp}.npy"
        np.save(pc_path, point_cloud)
        print(f"[Tool: get_camera_observation] ✓ Point cloud saved: shape {point_cloud.shape}", file=sys.stderr)

        return {
            "success": True,
            "rgb_path": str(bgr_path),  # Key name kept as rgb_path for compatibility
            "depth_path": str(depth_path),
            "intrinsics_path": str(intrinsics_path),
            "pose_path": str(pose_path),
            "pointcloud_path": str(pc_path),
            "image_size": list(bgr.shape[:2]),
            "camera_name": "front_camera"
        }

    except Exception as e:
        print(f"[Tool: get_camera_observation] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 2: Move to Position
# ==============================================================================

@mcp.tool()
def move_to_position(x: float, y: float, z: float, use_planning: bool = True) -> dict:
    """
    Move gripper to Cartesian position using path planning for robustness

    Args:
        x, y, z: Target position in meters
        use_planning: Use path planning (slower but robust) vs IK (fast but can fail)

    Returns:
        dict with success status
    """
    global _CURRENT_OBS, _TASK, _ENV

    print(f"[Tool: move_to_position] Moving to [{x:.3f}, {y:.3f}, {z:.3f}]", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        # Get current pose
        current_pose = obs.gripper_pose
        target_pos = np.array([x, y, z])

        # Calculate distance
        distance = np.linalg.norm(target_pos - current_pose[:3])
        print(f"[Tool: move_to_position] Distance: {distance:.3f}m", file=sys.stderr)

        if use_planning:
            # Use path planning for robust movement
            print(f"[Tool: move_to_position] Using path planning (robust for large movements)", file=sys.stderr)

            # Create planning action mode temporarily
            # Note: collision_checking=True may help reach difficult positions
            planning_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=True),
                gripper_action_mode=Discrete()
            )

            # Save current action mode
            original_action_mode = _ENV._action_mode

            # Switch to planning mode
            _ENV._action_mode = planning_mode
            task._action_mode = planning_mode

            # Preserve current gripper state (CRITICAL for maintaining grasp!)
            current_gripper_state = 1.0 if obs.gripper_open > 0.5 else 0.0
            print(f"[Tool: move_to_position] Preserving gripper state: {'open' if current_gripper_state > 0.5 else 'closed'}", file=sys.stderr)

            # Create action: target position + current orientation + current gripper state
            action = np.concatenate([target_pos, current_pose[3:7], [current_gripper_state]])

            # Execute with planning
            _CURRENT_OBS, reward, terminate = task.step(action)

            # Restore original action mode
            _ENV._action_mode = original_action_mode
            task._action_mode = original_action_mode

            final_pos = _CURRENT_OBS.gripper_pose[:3]
            final_distance = np.linalg.norm(final_pos - target_pos)
            print(f"[Tool: move_to_position] ✓ Reached via planning (error: {final_distance:.4f}m)", file=sys.stderr)

            return {
                "success": True,
                "final_position": final_pos.tolist(),
                "target_position": [x, y, z],
                "error_distance": float(final_distance),
                "task_completed": bool(terminate),
                "method": "planning"
            }

        else:
            # Use incremental IK for fine movements
            print(f"[Tool: move_to_position] Using incremental IK", file=sys.stderr)

            # Preserve current gripper state (CRITICAL for maintaining grasp!)
            current_gripper_state = 1.0 if obs.gripper_open > 0.5 else 0.0
            print(f"[Tool: move_to_position] Preserving gripper state: {'open' if current_gripper_state > 0.5 else 'closed'}", file=sys.stderr)

            max_step_size = 0.001  # 1mm max per step
            max_iterations = int(distance / max_step_size) + 100

            step_count = 0
            for i in range(max_iterations):
                current_pos = current_pose[:3]
                remaining = target_pos - current_pos
                distance_remaining = np.linalg.norm(remaining)

                if distance_remaining < 0.001:  # 1mm threshold
                    print(f"[Tool: move_to_position] ✓ Reached target (within 1mm)", file=sys.stderr)
                    break

                step_size = min(max_step_size, distance_remaining)
                direction = remaining / distance_remaining
                next_pos = current_pos + direction * step_size

                action = np.concatenate([next_pos, current_pose[3:7], [current_gripper_state]])

                _CURRENT_OBS, reward, terminate = task.step(action)

                current_pose = _CURRENT_OBS.gripper_pose
                step_count = i + 1

                if terminate:
                    print(f"[Tool: move_to_position] ✓ Task completed! (step {step_count})", file=sys.stderr)
                    break

                if (step_count % 50 == 0):
                    print(f"[Tool: move_to_position] Progress: {step_count} steps, {distance_remaining:.3f}m remaining", file=sys.stderr)

                time.sleep(0.005)

            final_pos = _CURRENT_OBS.gripper_pose[:3]
            final_distance = np.linalg.norm(final_pos - target_pos)
            print(f"[Tool: move_to_position] ✓ Reached via IK (error: {final_distance:.4f}m, steps: {step_count})", file=sys.stderr)

            return {
                "success": True,
                "final_position": final_pos.tolist(),
                "target_position": [x, y, z],
                "error_distance": float(final_distance),
                "task_completed": bool(terminate),
                "steps_executed": step_count,
                "method": "ik"
            }

    except InvalidActionError as e:
        print(f"[Tool: move_to_position] ✗ IK/Planning failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e), "suggestion": "Target may be unreachable or outside workspace"}

    except Exception as e:
        print(f"[Tool: move_to_position] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 3: Control Gripper
# ==============================================================================

@mcp.tool()
def control_gripper(action: str) -> dict:
    """
    Open or close the gripper

    Args:
        action: "open" or "close"

    Returns:
        dict with success status
    """
    global _CURRENT_OBS, _TASK

    print(f"[Tool: control_gripper] Action: {action}", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        # Get current pose
        current_pose = obs.gripper_pose

        # Set gripper value
        gripper_value = 1.0 if action == "open" else 0.0

        # Create action: keep current position, change gripper
        action_array = np.concatenate([current_pose[:7], [gripper_value]])

        _CURRENT_OBS, reward, terminate = task.step(action_array)

        print(f"[Tool: control_gripper] ✓ Gripper {action}", file=sys.stderr)

        return {
            "success": True,
            "action": action,
            "gripper_state": "open" if gripper_value > 0.5 else "closed"
        }

    except Exception as e:
        print(f"[Tool: control_gripper] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 4: Lift Gripper
# ==============================================================================

@mcp.tool()
def lift_gripper(height: float, num_steps: int = 30) -> dict:
    """
    Lift gripper up by specified height

    Args:
        height: Height to lift in meters (e.g., 0.1 for 10cm)
        num_steps: Number of interpolation steps

    Returns:
        dict with success status
    """
    global _CURRENT_OBS, _TASK

    print(f"[Tool: lift_gripper] Lifting by {height}m", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        # Get current pose
        current_pose = obs.gripper_pose
        start_z = current_pose[2]
        target_z = start_z + height

        # Lift gradually
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            z = start_z + alpha * height

            action = np.concatenate([[current_pose[0], current_pose[1], z],
                                   current_pose[3:7], [0.0]])  # Gripper closed

            _CURRENT_OBS, reward, terminate = task.step(action)

            if terminate:
                print(f"[Tool: lift_gripper] ✓ Task completed during lift!", file=sys.stderr)
                break

            time.sleep(0.01)

        final_z = _CURRENT_OBS.gripper_pose[2]
        print(f"[Tool: lift_gripper] ✓ Lifted from {start_z:.3f}m to {final_z:.3f}m", file=sys.stderr)

        return {
            "success": True,
            "start_height": float(start_z),
            "final_height": float(final_z),
            "lift_distance": float(final_z - start_z),
            "task_completed": bool(terminate)
        }

    except Exception as e:
        print(f"[Tool: lift_gripper] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 5: Get Current State
# ==============================================================================

@mcp.tool()
def get_current_state() -> dict:
    """
    Get current robot state

    Returns:
        dict with gripper position, joint positions, etc.
    """
    global _CURRENT_OBS

    env, task, obs = get_environment()

    if obs is None:
        return {"success": False, "error": "No observation available"}

    return {
        "success": True,
        "gripper_position": obs.gripper_pose[:3].tolist(),
        "gripper_orientation": obs.gripper_pose[3:7].tolist(),
        "gripper_open": bool(obs.gripper_open > 0.5),
        "joint_positions": obs.joint_positions.tolist() if hasattr(obs, 'joint_positions') else None
    }


# ==============================================================================
# Tool 6: Get Ground Truth Target Position (DEBUG)
# ==============================================================================

@mcp.tool()
def get_target_position() -> dict:
    """
    Get ground truth position of the target object (for debugging perception)

    For ReachTarget: returns target sphere position
    For PickAndLift: returns target block position AND lift target position

    Returns:
        dict with target position(s) in base frame
    """
    global _TASK

    env, task, obs = get_environment()

    try:
        # Get underlying task object from TaskEnvironment wrapper
        actual_task = task._task if hasattr(task, '_task') else task

        # Get target object from task
        if hasattr(actual_task, 'target'):
            # ReachTarget task
            target_pos = actual_task.target.get_position()
            target_name = "target sphere"
            print(f"[Tool: get_target_position] Ground truth: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]", file=sys.stderr)

            return {
                "success": True,
                "task": "ReachTarget",
                "target_name": target_name,
                "position": target_pos.tolist(),
                "note": "This is ground truth from simulation - use for debugging perception"
            }

        elif hasattr(actual_task, 'target_block'):
            # PickAndLift task
            block_pos = actual_task.target_block.get_position()
            target_name = "target block"

            # Also get lift target (success detector) position
            lift_target_pos = None
            if hasattr(actual_task, 'success_detector'):
                lift_target_pos = actual_task.success_detector.get_position()
                print(f"[Tool: get_target_position] Block: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]", file=sys.stderr)
                print(f"[Tool: get_target_position] Lift target: [{lift_target_pos[0]:.3f}, {lift_target_pos[1]:.3f}, {lift_target_pos[2]:.3f}]", file=sys.stderr)
            else:
                print(f"[Tool: get_target_position] Block: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]", file=sys.stderr)
                print(f"[Tool: get_target_position] ⚠ No success_detector found", file=sys.stderr)

            return {
                "success": True,
                "task": "PickAndLift",
                "target_name": target_name,
                "block_position": block_pos.tolist(),
                "lift_target_position": lift_target_pos.tolist() if lift_target_pos is not None else None,
                "note": "Block position is for grasping. Lift target is where to move after grasping."
            }
        else:
            return {"success": False, "error": f"Task has no target object. Available attributes: {dir(actual_task)}"}

    except Exception as e:
        print(f"[Tool: get_target_position] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 7: Reset Task
# ==============================================================================

@mcp.tool()
def reset_task(task_name: str = "ReachTarget") -> dict:
    """
    Reset environment to start new task

    Args:
        task_name: "ReachTarget" or "PickAndLift"

    Returns:
        dict with task description
    """
    global _TASK, _CURRENT_OBS, _TASK_CLASS

    print(f"[Tool: reset_task] Resetting to {task_name}...", file=sys.stderr)

    # Update task class
    if task_name == "PickAndLift":
        _TASK_CLASS = PickAndLift
    else:
        _TASK_CLASS = ReachTarget

    # Force re-initialization
    global _ENV
    if _ENV is not None:
        _TASK = _ENV.get_task(_TASK_CLASS)

    env, task, obs = get_environment()

    try:
        descriptions, _CURRENT_OBS = task.reset()
        print(f"[Tool: reset_task] ✓ Reset: {descriptions[0]}", file=sys.stderr)

        return {
            "success": True,
            "task_name": task_name,
            "description": descriptions[0]
        }

    except Exception as e:
        print(f"[Tool: reset_task] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Server Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70, file=sys.stderr)
    print("RLBench MCP Server - Multi-Agent Orchestration", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("Tools available:", file=sys.stderr)
    print("  1. get_camera_observation - Capture RGB + depth", file=sys.stderr)
    print("  2. move_to_position - Move gripper to XYZ", file=sys.stderr)
    print("  3. control_gripper - Open/close gripper", file=sys.stderr)
    print("  4. lift_gripper - Lift by height", file=sys.stderr)
    print("  5. get_current_state - Get robot state", file=sys.stderr)
    print("  6. reset_task - Reset environment", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)

    mcp.run()
