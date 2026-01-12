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
from rlbench.tasks import ReachTarget, PickAndLift, PushButton, PutRubbishInBin, StackBlocks
from rlbench.backend.exceptions import InvalidActionError

# ==============================================================================
# Global State
# ==============================================================================

mcp = FastMCP("rlbench-orchestration-server")

_ENV = None
_TASK = None
_CURRENT_OBS = None
_TASK_CLASS = None  # No default - wait for reset_task() call
_TASK_DESCRIPTION = ""  # Store task description from reset

# Output directory for sensor data
OUTPUT_DIR = Path(__file__).parent / "sensor_data"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_environment():
    """Get or create RLBench environment"""
    global _ENV, _TASK, _CURRENT_OBS

    if _ENV is None:
        print("[RLBench] Initializing environment...", file=sys.stderr)

        obs_config = ObservationConfig()

        # REQUIRED for control
        obs_config.set_all_low_dim(True)

        # Disable vision by default
        obs_config.set_all_high_dim(False)

        # Enable only needed cameras
        obs_config.front_camera.rgb = True
        obs_config.front_camera.depth = True
        obs_config.front_camera.point_cloud = True 
        obs_config.front_camera.image_size = [512, 512]

        obs_config.left_shoulder_camera.rgb = True
        obs_config.right_shoulder_camera.rgb = True
        obs_config.wrist_camera.rgb = True
        obs_config.overhead_camera.rgb = True


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
        print("[RLBench] Waiting for load_task() call to load a task...", file=sys.stderr)

    return _ENV, _TASK, _CURRENT_OBS


def parse_task_objects(task_description: str) -> list:
    """
    Parse task description to extract object names WITH COLORS for detection

    Hybrid approach: Extract both color AND shape to handle multiple objects
    Example: "pick up the red block" → ["red cube"]
    Example: "reach the red target" → ["red sphere"]

    Args:
        task_description: Natural language task description from RLBench

    Returns:
        list of object names with colors suitable for GroundingDINO prompts
    """
    desc_lower = task_description.lower()
    objects = []

    # Extract color adjectives
    colors = []
    color_keywords = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
    for color in color_keywords:
        if color in desc_lower:
            colors.append(color)

    # Object mapping: task keywords → GroundingDINO prompts
    object_keywords = [
        ('block', 'cube'),
        ('cube', 'cube'),
        ('sphere', 'sphere'),
        ('ball', 'sphere'),
        ('cup', 'cup'),
        ('button', 'button'),
        ('bin', 'bin'),
        ('trash', 'trash'),
        ('rubbish', 'trash'),
        ('target', 'sphere'),  # "target" usually refers to sphere in reach tasks
    ]

    # Synonym expansion for better GroundingDINO detection
    # Maps object names to comma-separated synonyms (visual + semantic terms)

    # Task-specific optimization: For PutRubbishInBin, use precise descriptors
    is_trash_task = 'rubbish' in desc_lower or ('bin' in desc_lower and 'trash' in desc_lower)

    object_synonyms = {
        'bin': 'basket, bin, container',  # Visual (basket) helps detect wire mesh
        'trash': 'crumpled silver paper' if is_trash_task else 'paper, trash, crumpled paper, rubbish, garbage',
        'button': 'button, switch',
        'sphere': 'sphere, ball',
        'cube': 'cube, block',
    }

    # Extract objects with their colors
    for keyword, obj_name in object_keywords:
        if keyword in desc_lower:
            # Build object description with color
            # Find which color appears before this object keyword
            obj_index = desc_lower.find(keyword)
            relevant_color = None

            # Look for color words before the object keyword
            for color in colors:
                color_index = desc_lower.find(color)
                # If color appears before object and within reasonable distance
                if color_index >= 0 and color_index < obj_index and (obj_index - color_index) < 30:
                    relevant_color = color
                    break

            if keyword == 'target' and not relevant_color and colors:
                relevant_color = colors[0]  # Use first color found in description

            # Expand with synonyms for better detection
            obj_expanded = object_synonyms.get(obj_name, obj_name)

            # Build the detection phrase with color
            if relevant_color:
                # Apply color to all synonyms
                # Example: "red" + "basket, bin" → "red basket, red bin"
                synonyms = obj_expanded.split(', ')
                colored_synonyms = [f"{relevant_color} {syn}" for syn in synonyms]
                obj_with_color = ', '.join(colored_synonyms)
            else:
                obj_with_color = obj_expanded

            if obj_with_color not in objects:  # Avoid duplicates
                objects.append(obj_with_color)

    return objects


# ==============================================================================
# Tool 1: Get Camera Observation
# ==============================================================================

@mcp.tool()
def get_camera_observation() -> dict:
    """
    Capture RGB and depth images from robot camera with calibration data
    Also returns task objects parsed from description (VoxPoser-style)

    Returns:
        dict with rgb_path, depth_path, intrinsics, camera pose, and task_objects list
    """
    global _CURRENT_OBS

    print("[Tool: get_camera_observation] Capturing images...", file=sys.stderr)

    env, task, obs = get_environment()

    if obs is None:
        return {"success": False, "error": "No observation available"}

    try:
        # Generate timestamp in YYYYMMDD_HHMMSS format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get RGB image and convert to BGR for OpenCV
        rgb = obs.front_rgb
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb

        # Convert RGB to BGR for OpenCV (matching demo code approach)
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

        # Save with OpenCV
        rgb_path = OUTPUT_DIR / f"image_{timestamp}.png"
        cv2.imwrite(str(rgb_path), bgr)
        print(f"[Tool: get_camera_observation] ✓ BGR image saved: {rgb_path}", file=sys.stderr)

        # # OPTIONAL: PIL enhancement (commented out for now)
        # from PIL import Image as PILImage, ImageEnhance
        # pil_image = PILImage.fromarray(rgb_uint8)
        # enhancer = ImageEnhance.Color(pil_image)
        # pil_image = enhancer.enhance(1.3)  # Boost color saturation by 30%
        # enhancer = ImageEnhance.Contrast(pil_image)
        # pil_image = enhancer.enhance(1.2)  # Boost contrast by 20%
        # enhancer = ImageEnhance.Brightness(pil_image)
        # pil_image = enhancer.enhance(1.1)  # Slight brightness increase
        # rgb_enhanced = np.array(pil_image)
        # bgr = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)

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

        # Parse task description to get object names WITH COLORS
        # Use the stored description from reset_task
        global _TASK_DESCRIPTION
        task_desc = _TASK_DESCRIPTION if _TASK_DESCRIPTION else "unknown task"

        task_objects = parse_task_objects(task_desc)
        detection_prompt = " . ".join(task_objects) if task_objects else "objects"

        print(f"[Tool: get_camera_observation] Task: {task_desc}", file=sys.stderr)
        print(f"[Tool: get_camera_observation] Detected objects: {task_objects}", file=sys.stderr)
        print(f"[Tool: get_camera_observation] Suggested prompt: \"{detection_prompt}\"", file=sys.stderr)

        return {
            "success": True,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "intrinsics_path": str(intrinsics_path),
            "pose_path": str(pose_path),
            "pointcloud_path": str(pc_path),
            "image_size": list(rgb_uint8.shape[:2]),
            "camera_name": "front_camera",
            "task_objects": task_objects,
            "detection_prompt": detection_prompt
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
            # Note: collision_checking=True helps reach difficult positions
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
    For PickAndLift: returns target block position AND lift target position (sphere)
    For PutRubbishInBin: returns rubbish position AND lift target position (bin)
    For SlideBlockToTarget: returns block position AND target position

    Returns:
        dict with target position(s) in base frame
    """
    global _TASK

    env, task, obs = get_environment()

    try:
        # Get underlying task object from TaskEnvironment wrapper
        actual_task = task._task if hasattr(task, '_task') else task
        task_name = actual_task.get_name()

        # Get target object from task - check by name first for ambiguous cases
        if 'slide' in task_name.lower():
            # SlideBlockToTarget task
            from pyrep.objects.shape import Shape
            from pyrep.objects.proximity_sensor import ProximitySensor

            block = Shape('block')
            target = ProximitySensor('success')
            block_pos = block.get_position()
            target_pos = target.get_position()

            print(f"[Tool: get_target_position] Block: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]", file=sys.stderr)
            print(f"[Tool: get_target_position] Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]", file=sys.stderr)

            return {
                "success": True,
                "task": "SlideBlockToTarget",
                "block_position": block_pos.tolist(),
                "position": target_pos.tolist(),
                "note": "Block position is starting position. Target position is where to slide the block."
            }

        elif hasattr(actual_task, 'target'):
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

        elif hasattr(actual_task, 'rubbish'):
            # PutRubbishInBin task
            rubbish_pos = actual_task.rubbish.get_position()

            # Get bin position - try multiple approaches
            bin_pos = None

            # Try 1: success_detector attribute
            if hasattr(actual_task, 'success_detector'):
                bin_pos = actual_task.success_detector.get_position()

            # Try 2: Look for success sensor in PyRep (used by PutRubbishInBin)
            if bin_pos is None:
                try:
                    from pyrep.objects.proximity_sensor import ProximitySensor
                    success_sensor = ProximitySensor('success')
                    bin_pos = success_sensor.get_position()
                    print(f"[Tool: get_target_position] Found bin via ProximitySensor('success')", file=sys.stderr)
                except:
                    pass

            # Try 3: Look for 'bin' shape object
            if bin_pos is None:
                try:
                    from pyrep.objects.shape import Shape
                    bin_shape = Shape('bin')
                    bin_pos = bin_shape.get_position()
                    print(f"[Tool: get_target_position] Found bin via Shape('bin')", file=sys.stderr)
                except:
                    pass

            if bin_pos is not None:
                print(f"[Tool: get_target_position] Rubbish: [{rubbish_pos[0]:.3f}, {rubbish_pos[1]:.3f}, {rubbish_pos[2]:.3f}]", file=sys.stderr)
                print(f"[Tool: get_target_position] Bin: [{bin_pos[0]:.3f}, {bin_pos[1]:.3f}, {bin_pos[2]:.3f}]", file=sys.stderr)
            else:
                print(f"[Tool: get_target_position] Rubbish: [{rubbish_pos[0]:.3f}, {rubbish_pos[1]:.3f}, {rubbish_pos[2]:.3f}]", file=sys.stderr)
                print(f"[Tool: get_target_position] ⚠ No bin position found - tried success_detector, ProximitySensor('success'), Shape('bin')", file=sys.stderr)

            return {
                "success": True,
                "task": "PutRubbishInBin",
                "rubbish_position": rubbish_pos.tolist(),
                "lift_target_position": bin_pos.tolist() if bin_pos is not None else None,
                "note": "Rubbish position is for grasping. Lift target (bin) is where to place the rubbish."
            }

        elif hasattr(actual_task, 'target_blocks'):
            # StackBlocks task
            target_blocks_positions = []
            for i, block in enumerate(actual_task.target_blocks):
                pos = block.get_position()
                target_blocks_positions.append(pos.tolist())
                print(f"[Tool: get_target_position] Target block {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", file=sys.stderr)

            # Get stacking zone position (stack_blocks_target_plane)
            stack_zone_pos = None
            try:
                from pyrep.objects.shape import Shape
                stack_plane = Shape('stack_blocks_target_plane')
                stack_zone_pos = stack_plane.get_position()
                print(f"[Tool: get_target_position] Stacking zone: [{stack_zone_pos[0]:.3f}, {stack_zone_pos[1]:.3f}, {stack_zone_pos[2]:.3f}]", file=sys.stderr)
            except:
                print(f"[Tool: get_target_position] ⚠ Could not find stack_blocks_target_plane", file=sys.stderr)

            blocks_to_stack = actual_task.blocks_to_stack if hasattr(actual_task, 'blocks_to_stack') else 2
            print(f"[Tool: get_target_position] Blocks to stack: {blocks_to_stack}", file=sys.stderr)

            return {
                "success": True,
                "task": "StackBlocks",
                "target_blocks_positions": target_blocks_positions,
                "stacking_zone_position": stack_zone_pos.tolist() if stack_zone_pos is not None else None,
                "blocks_to_stack": blocks_to_stack,
                "note": f"4 target blocks detected. Stack {blocks_to_stack} of them at stacking zone."
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
        task_name: Task to load. Options:
            - "ReachTarget": Move gripper to touch target
            - "PickAndLift": Pick up object and lift it
            - "PushButton": Push a button
            - "PutRubbishInBin": Put rubbish/trash in bin
            - "StackBlocks": Stack blocks of same color

    Returns:
        dict with task description
    """
    global _TASK, _CURRENT_OBS, _TASK_CLASS

    print(f"[Tool: reset_task] Resetting to {task_name}...", file=sys.stderr)

    # Update task class based on task name
    task_map = {
        "ReachTarget": ReachTarget,
        "PickAndLift": PickAndLift,
        "PushButton": PushButton,
        "PutRubbishInBin": PutRubbishInBin,
        "StackBlocks": StackBlocks
    }

    _TASK_CLASS = task_map.get(task_name, ReachTarget)  # Default to ReachTarget if unknown

    # Get or create environment (doesn't load task yet)
    env, task, obs = get_environment()

    # Now load the requested task
    _TASK = env.get_task(_TASK_CLASS)
    print(f"[Tool: reset_task] Loaded task: {_TASK_CLASS.__name__}", file=sys.stderr)

    try:
        descriptions, _CURRENT_OBS = _TASK.reset()

        # Store task description globally for get_camera_observation to use
        global _TASK_DESCRIPTION
        _TASK_DESCRIPTION = descriptions[0]

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
# Tool 8: Load Task 
# ==============================================================================

@mcp.tool()
def load_task(task_name: str) -> dict:
    """
    Load a specific task and get initial observation

    Use this at the start to load the task based on user's request.

    Args:
        task_name: Task to load. Options:
            - "ReachTarget": Move gripper to touch target
            - "PickAndLift": Pick up object and lift it
            - "PushButton": Push a button
            - "PutRubbishInBin": Put rubbish/trash in bin
            - "StackBlocks": Stack blocks of same color

    Returns:
        dict with task description and success status
    """
    global _TASK, _CURRENT_OBS, _TASK_CLASS

    print(f"[Tool: load_task] Loading {task_name}...", file=sys.stderr)

    # Task name mapping
    task_map = {
        "ReachTarget": ReachTarget,
        "PickAndLift": PickAndLift,
        "PushButton": PushButton,
        "PutRubbishInBin": PutRubbishInBin,
        "StackBlocks": StackBlocks
    }

    if task_name not in task_map:
        return {
            "success": False,
            "error": f"Unknown task '{task_name}'. Available: {list(task_map.keys())}"
        }

    _TASK_CLASS = task_map[task_name]

    # Get or create environment
    env, _, _ = get_environment()

    # Load the requested task
    _TASK = env.get_task(_TASK_CLASS)
    print(f"[Tool: load_task] ✓ Loaded: {_TASK_CLASS.__name__}", file=sys.stderr)

    try:
        descriptions, _CURRENT_OBS = _TASK.reset()

        # Store task description globally
        global _TASK_DESCRIPTION
        _TASK_DESCRIPTION = descriptions[0]

        print(f"[Tool: load_task] ✓ Ready: {descriptions[0]}", file=sys.stderr)

        return {
            "success": True,
            "task_name": task_name,
            "description": descriptions[0]
        }

    except Exception as e:
        print(f"[Tool: load_task] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ==============================================================================
# Tool 9: Reset Current Task
# ==============================================================================

@mcp.tool()
def reset_current_task() -> dict:
    """
    Reset the currently loaded task to retry execution

    Use this when execution fails and you want to start over with the same task.
    The task must already be loaded via load_task() first.

    Returns:
        dict with task description and success status
    """
    global _TASK, _CURRENT_OBS, _TASK_DESCRIPTION

    if _TASK is None:
        return {
            "success": False,
            "error": "No task loaded. Call load_task(task_name) first."
        }

    task_name = _TASK.get_name()
    print(f"[Tool: reset_current_task] Resetting {task_name}...", file=sys.stderr)

    try:
        descriptions, _CURRENT_OBS = _TASK.reset()
        _TASK_DESCRIPTION = descriptions[0]

        print(f"[Tool: reset_current_task] ✓ Reset: {descriptions[0]}", file=sys.stderr)

        return {
            "success": True,
            "task_name": task_name,
            "description": descriptions[0]
        }

    except Exception as e:
        print(f"[Tool: reset_current_task] ✗ Error: {e}", file=sys.stderr)
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
    print("  6. load_task - Load specific task", file=sys.stderr)
    print("  7. reset_current_task - Reset for retry", file=sys.stderr)
    print("  8. reset_task - (deprecated, use load_task)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)

    mcp.run()
