"""
Simplified Multi-Agent Orchestrator for Testing
Task: ReachTarget (no demos, perception-based)

Pipeline: Sensing → Perception → Motion
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# CoppeliaSim environment
COPPELIASIM_ROOT = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')
RLBENCH_ENV = dict(os.environ)
RLBENCH_ENV.update({
    'COPPELIASIM_ROOT': COPPELIASIM_ROOT,
    'LD_LIBRARY_PATH': COPPELIASIM_ROOT,
    'DISPLAY': os.environ.get('DISPLAY', ':0'),
    'QT_QPA_PLATFORM': 'xcb',
    'QT_QPA_PLATFORM_PLUGIN_PATH': f'{COPPELIASIM_ROOT}',
})

# ==============================================================================
# MCP Toolsets
# ==============================================================================

# RLBench tools for sensing and motion
rlbench_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(current_file_dir / "ros_mcp_server" / "rlbench_orchestration_server.py")],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
)

# Perception tools
perception_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(current_file_dir / "ros_mcp_server" / "perception_orchestration_server.py")],
        ),
        timeout=120,
    ),
)

# ==============================================================================
# Orchestrator Agent
# ==============================================================================

root_agent = Agent(
    name="OrchestratorAgent",
    model=LiteLlm(model="deepseek/deepseek-reasoner"),
    description="Orchestrates robot manipulation via Sensing → Perception → Motion",
    instruction=(
        "You are a robot orchestrator for RLBench manipulation tasks (ReachTarget and PickAndLift).\n\n"

        "**AVAILABLE TOOLS:**\n\n"

        "Sensing Tools:\n"
        "- get_camera_observation() → Returns rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path\n"
        "- get_current_state() → Returns gripper position and state\n"
        "- get_target_position() → Returns ground truth target/block position (DEBUG only)\n\n"

        "Perception Tools:\n"
        "- detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path) → Returns 3D position [x, y, z] in robot base frame\n\n"

        "Motion Tools:\n"
        "- move_to_position(x, y, z, use_planning=True) → Move gripper to target\n"
        "  - use_planning=True (default): Uses path planning (robust for all movements)\n"
        "  - use_planning=False: Uses IK only (faster but may fail for large distances)\n"
        "- control_gripper(action) → 'open' or 'close'\n"
        "- lift_gripper(height) → Lift by specified height\n"
        "- reset_task(task_name) → Reset environment ('ReachTarget' or 'PickAndLift')\n\n"

        "**TASK 1: ReachTarget**\n"
        "When the user asks to reach a target (e.g., 'reach the red ball' or 'reach the red target'):\n\n"

        "1. **Reset:** Call reset_task('ReachTarget')\n"
        "2. **Get Ground Truth (DEBUG):** Call get_target_position() to see actual target location\n"
        "3. **Sense:** Call get_camera_observation()\n"
        "4. **Perceive:** Call detect_object_3d(text_prompt='red target', ...)\n"
        "5. **Validate:** Compare detected vs ground truth\n"
        "6. **Move:** Call move_to_position(x, y, z) with detected coordinates\n"
        "7. **Report:** Task success status\n\n"

        "**TASK 2: PickAndLift**\n"
        "When the user asks to pick and lift an object (e.g., 'pick up the red block'):\n\n"

        "1. **Reset:** Call reset_task('PickAndLift')\n"
        "2. **Get Ground Truth:** Call get_target_position() to get block_position and lift_target_position\n"
        "   - SAVE lift_target_position for step 8!\n"
        "3. **Sense:** Call get_camera_observation()\n"
        "4. **Perceive Block:** Call detect_object_3d(text_prompt='red block', ...)\n"
        "5. **Approach:** Move ABOVE block: move_to_position(x, y, z+0.15) with gripper OPEN\n"
        "   - Add 15cm to z-coordinate to approach from above\n"
        "   - Ensure gripper is open: control_gripper('open') if needed\n"
        "6. **Descend:** Move DOWN to block: move_to_position(x, y, z+0.02)\n"
        "   - Stop 2cm above detected position (account for gripper height)\n"
        "7. **Grasp:** Close gripper: control_gripper('close')\n"
        "   - Wait for gripper to close fully\n"
        "8. **Lift to Target:** Move to EXACT lift target position from step 2\n"
        "   - Use lift_target_position from get_target_position()\n"
        "   - Call move_to_position(lift_x, lift_y, lift_z) with the exact coordinates\n"
        "   - Keep gripper CLOSED (don't open!)\n"
        "9. **Report:** Check if task_completed flag is true\n\n"

        "**CRITICAL NOTES FOR PickAndLift:**\n"
        "- The detected position is the SURFACE of the block\n"
        "- Account for 1.7cm perception error when approaching\n"
        "- Move to z+0.02 (2cm above surface) to account for gripper finger thickness\n"
        "- Close gripper firmly before lifting\n"
        "- MUST move to EXACT lift_target_position (not fixed height!)\n"
        "- The success condition requires: (1) block grasped AND (2) block at lift target\n"
        "- Keep gripper CLOSED during lift - opening drops the block!\n"
        "- If grasping fails, try adjusting approach height or position slightly\n\n"

        "**IMPORTANT:**\n"
        "- You MUST pass pointcloud_path from sensing to perception\n"
        "- Position from perception is already in robot base frame - use directly\n"
        "- Do NOT use get_target_position() for execution (debugging only!)\n"
        "- Execute steps sequentially and report progress\n"
        "- For PickAndLift, gripper state management is critical"
    ),
    tools=[rlbench_toolset, perception_toolset]
)

print("[Orchestrator] Ready to execute perception-based manipulation!")
print("[Orchestrator] Available tasks:")
print("  - ReachTarget: 'Reach the red target'")
print("  - PickAndLift: 'Pick up the red block'")
