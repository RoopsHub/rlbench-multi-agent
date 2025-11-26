#!/usr/bin/env python3
"""
RLBench MCP Server - Clean Implementation for Demo-Based Execution

This server provides RLBench-specific tools for robot manipulation.
It does NOT include ROS-specific tools - those are in server.py

Tools provided:
- load_task_demo: Load expert demonstration waypoints (WITHOUT executing)
- execute_joint_position: Execute joint positions directly
- get_task_info: Get current task information
- reset_task: Reset task to initial state
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# RLBench imports
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PickAndLift

# ==============================================================================
# Global State
# ==============================================================================

mcp = FastMCP("rlbench-mcp-server")

# Global environment (initialized lazily)
_ENV = None
_TASK = None
_CURRENT_OBS = None

def get_environment():
    """Get or create the RLBench environment (singleton pattern)"""
    global _ENV, _TASK, _CURRENT_OBS

    if _ENV is None:
        print("[RLBench Server] Initializing RLBench environment...", file=sys.stderr)

        # Configure observations
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.front_camera.image_size = [256, 256]
        obs_config.front_camera.rgb = True
        obs_config.front_camera.depth = True
        obs_config.front_camera.point_cloud = True

        # Create environment with direct joint position control
        # This is more reliable than IK-based control
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(),
            gripper_action_mode=Discrete()
        )

        _ENV = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=False  # Show CoppeliaSim GUI
        )

        _ENV.launch()
        print("[RLBench Server] ✓ CoppeliaSim launched", file=sys.stderr)

        # Load default task (ReachTarget)
        _TASK = _ENV.get_task(ReachTarget)
        print("[RLBench Server] ✓ Loaded ReachTarget task", file=sys.stderr)

        # Get initial observation
        descriptions, _CURRENT_OBS = _TASK.reset()
        print(f"[RLBench Server] ✓ Task description: {descriptions[0]}", file=sys.stderr)
        print("[RLBench Server] ✓ Environment ready!", file=sys.stderr)

    return _ENV, _TASK, _CURRENT_OBS


# ==============================================================================
# Tool 1: Load Demo (WITHOUT execution)
# ==============================================================================

@mcp.tool()
def load_task_demo(task_name: str, demo_index: int = 0) -> dict:
    """
    Load expert demonstration waypoints WITHOUT executing them

    Args:
        task_name: Name of task (e.g., "ReachTarget", "PickAndLift")
        demo_index: Which demo to load (default: 0)

    Returns:
        dict with waypoints list
    """
    global _TASK, _ENV, _CURRENT_OBS

    print(f"[Tool: load_task_demo] Loading demo for task: {task_name}", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        # CRITICAL FIX: Reset FIRST, then load demo on the SAME target configuration
        print(f"[Tool: load_task_demo] Resetting task to get fresh target position...", file=sys.stderr)
        descriptions, _CURRENT_OBS = task.reset()

        if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'joint_positions'):
            start_joints = _CURRENT_OBS.joint_positions
            print(f"[Tool: load_task_demo] ✓ Task reset - robot at starting position", file=sys.stderr)
            print(f"  Starting joints: {start_joints.tolist()[:4]}... (showing first 4)", file=sys.stderr)

        # Give simulation time to settle
        time.sleep(0.5)

        # Load demo with live_demos=True to get waypoints
        # Note: This WILL execute the demo, but target position stays the same since we just reset
        print(f"[Tool: load_task_demo] Loading demo (will execute on current target)...", file=sys.stderr)
        demos = task.get_demos(amount=1, live_demos=True)

        if not demos or len(demos) == 0:
            return {
                "success": False,
                "error": "No demonstrations available"
            }

        demo = demos[0]

        # Extract waypoints from demonstration
        waypoints = []
        for demo_obs in demo:
            if hasattr(demo_obs, 'joint_positions'):
                # Extract joint positions (7 joints for Franka Panda)
                joint_positions = demo_obs.joint_positions
                gripper = 1.0 if demo_obs.gripper_open > 0.5 else 0.0

                waypoints.append({
                    "joint_positions": joint_positions.tolist(),
                    "gripper": gripper
                })

        print(f"[Tool: load_task_demo] ✓ Extracted {len(waypoints)} waypoints", file=sys.stderr)
        print(f"[Tool: load_task_demo] ✓ Demo completed - task may be finished", file=sys.stderr)
        print(f"[Tool: load_task_demo] NOTE: Waypoints are for the CURRENT target position", file=sys.stderr)

        # DO NOT reset again - target would randomize!
        print(f"[Tool: load_task_demo] ✓ Ready for manual waypoint replay (if needed)", file=sys.stderr)

        return {
            "success": True,
            "waypoints": waypoints,
            "num_waypoints": len(waypoints),
            "note": "Demo executed successfully. Task may already be complete. Waypoints are for CURRENT target position - do NOT reset before executing them!",
            "starting_joints": start_joints.tolist() if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'joint_positions') else None,
            "warning": "Manual waypoint replay might not be needed - demo already completed the task"
        }

    except Exception as e:
        print(f"[Tool: load_task_demo] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


# ==============================================================================
# Tool 2: Execute Joint Position
# ==============================================================================

@mcp.tool()
def execute_joint_position(joint_positions: list, gripper: float) -> dict:
    """
    Execute joint position directly (no IK required)

    Args:
        joint_positions: List of 7 joint angles in radians [j1, j2, j3, j4, j5, j6, j7]
        gripper: 0.0-1.0 (0=closed, 1=open)

    Returns:
        dict with success, reward, terminated
    """
    global _CURRENT_OBS, _TASK

    print(f"[Tool: execute_joint_position] Executing joint positions:", file=sys.stderr)
    print(f"  Joints: {[f'{j:.3f}' for j in joint_positions[:4]]}... (showing first 4)", file=sys.stderr)
    print(f"  Gripper: {gripper:.1f}", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        # Log current joint positions BEFORE execution
        if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'joint_positions'):
            current_joints = _CURRENT_OBS.joint_positions
            print(f"  Current joints: {[f'{j:.3f}' for j in current_joints[:4]]}... (showing first 4)", file=sys.stderr)

        # Create action: [j1, j2, j3, j4, j5, j6, j7, gripper]
        action = np.array(joint_positions + [gripper])
        print(f"  Action array shape: {action.shape}", file=sys.stderr)

        # Execute action
        _CURRENT_OBS, reward, terminate = task.step(action)

        # Log joint positions AFTER execution
        if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'joint_positions'):
            new_joints = _CURRENT_OBS.joint_positions
            print(f"  New joints: {[f'{j:.3f}' for j in new_joints[:4]]}... (showing first 4)", file=sys.stderr)

            # Calculate joint difference
            joint_diff = np.abs(new_joints - current_joints)
            print(f"  Max joint change: {np.max(joint_diff):.4f} radians", file=sys.stderr)

        # Give simulation time to render (important for visualization!)
        time.sleep(0.1)

        print(f"[Tool: execute_joint_position] ✓ Success (reward={reward:.3f}, terminate={terminate})", file=sys.stderr)

        return {
            "success": True,
            "reward": float(reward),
            "terminated": bool(terminate),
            "joints_reached": new_joints.tolist() if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'joint_positions') else None
        }

    except Exception as e:
        print(f"[Tool: execute_joint_position] ✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


# ==============================================================================
# Tool 3: Get Task Info
# ==============================================================================

@mcp.tool()
def get_task_info() -> dict:
    """
    Get current task information

    Returns:
        dict with task name, description, current state
    """
    global _TASK, _CURRENT_OBS

    if _TASK is None:
        get_environment()

    task_name = _TASK.__class__.__name__ if _TASK else "Unknown"

    # Get current gripper pose
    gripper_pose = None
    if _CURRENT_OBS and hasattr(_CURRENT_OBS, 'gripper_pose'):
        pose = _CURRENT_OBS.gripper_pose
        gripper_pose = {
            "position": pose[:3].tolist(),
            "orientation": pose[3:].tolist()
        }

    return {
        "task_name": task_name,
        "description": f"Complete the {task_name} task",
        "current_gripper_pose": gripper_pose
    }


# ==============================================================================
# Tool 4: Reset Task
# ==============================================================================

@mcp.tool()
def reset_task() -> dict:
    """
    Reset task to initial state

    Returns:
        dict with success status
    """
    global _TASK, _CURRENT_OBS

    print(f"[Tool: reset_task] Resetting task...", file=sys.stderr)

    env, task, obs = get_environment()

    try:
        descriptions, _CURRENT_OBS = task.reset()
        print(f"[Tool: reset_task] ✓ Task reset", file=sys.stderr)
        print(f"  Description: {descriptions[0]}", file=sys.stderr)

        return {
            "success": True,
            "description": descriptions[0]
        }

    except Exception as e:
        print(f"[Tool: reset_task] ✗ Error: {e}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e)
        }


# ==============================================================================
# Server Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70, file=sys.stderr)
    print("RLBench MCP Server - Demo-Based Execution", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("Tools available:", file=sys.stderr)
    print("  1. load_task_demo - Load waypoints (with auto-reset)", file=sys.stderr)
    print("  2. execute_ee_pose - Execute end-effector pose", file=sys.stderr)
    print("  3. get_task_info - Get current task info", file=sys.stderr)
    print("  4. reset_task - Reset to initial state", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)

    # Run the MCP server
    mcp.run()
