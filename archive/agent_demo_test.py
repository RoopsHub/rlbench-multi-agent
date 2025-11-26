"""
Simple Demo Test Agent - For Testing RLBench Demo Execution

This is a simplified agent just for testing the demo-based approach.
The full multi-agent orchestrator is in agent_full_orchestrator.py
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

# Load environment variables
load_dotenv()

# ==============================================================================
# === CONFIGURATION ===
# ==============================================================================

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# CoppeliaSim environment variables for RLBench
COPPELIASIM_ROOT = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')

# Get current environment and add CoppeliaSim paths
RLBENCH_ENV = dict(os.environ)  # Copy current environment
RLBENCH_ENV.update({
    'COPPELIASIM_ROOT': COPPELIASIM_ROOT,
    'LD_LIBRARY_PATH': COPPELIASIM_ROOT,
    'DISPLAY': os.environ.get('DISPLAY', ':0'),  # Preserve DISPLAY
    'QT_QPA_PLATFORM': 'xcb',
    'QT_QPA_PLATFORM_PLUGIN_PATH': f'{COPPELIASIM_ROOT}',
})

# Path to RLBench MCP server
PATH_TO_RLBENCH_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_server.py"

print(f"[Config] RLBench server: {PATH_TO_RLBENCH_SERVER}")
print(f"[Config] CoppeliaSim: {COPPELIASIM_ROOT}")

# ==============================================================================
# === MCP TOOLSET ===
# ==============================================================================

demo_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_RLBENCH_SERVER)],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
    tool_filter=["load_task_demo", "execute_joint_position", "get_task_info", "reset_task"]
)

# ==============================================================================
# === AGENT ===
# ==============================================================================

# Create a simple demo execution agent
root_agent = Agent(
    name="DemoTestAgent",
    model=LiteLlm(model="deepseek/deepseek-reasoner"),
    description="Simple agent for testing RLBench demo execution",
    instruction=(
        "You are a robot control agent testing demo-based execution with JOINT POSITION control.\n\n"
        "AVAILABLE TOOLS:\n"
        "1. get_task_info() - Get current task name and joint positions\n"
        "2. load_task_demo(task_name) - Load demo waypoints with joint positions (auto-resets task after)\n"
        "3. execute_joint_position(joint_positions, gripper) - Execute joint positions directly\n"
        "4. reset_task() - Manually reset task to initial state\n\n"
        "When the user asks you to test the demo:\n"
        "1. Optionally call get_task_info to see current task state\n"
        "2. Call load_task_demo with task_name='ReachTarget'\n"
        "3. The result will contain waypoints (each with joint_positions and gripper)\n"
        "4. Execute ALL waypoints in sequence by calling execute_joint_position for each one\n"
        "5. Report progress every 10 waypoints (e.g., 'Executed waypoint 10/57, reward=X')\n"
        "6. Stop immediately if task terminates (success or failure)\n"
        "7. Summarize the final results\n\n"
        "IMPORTANT:\n"
        "- Execute ALL waypoints in order - don't skip any\n"
        "- Call execute_joint_position(joint_positions, gripper) for each\n"
        "- joint_positions is a list of 7 floats (7 joint angles in radians)\n"
        "- gripper is a float (0.0-1.0, where 0=closed, 1=open)\n"
        "- Report the reward after each execution\n"
        "- If something goes wrong, you can call reset_task() to start over\n"
    ),
    tools=[demo_toolset]
)

print(f"[Agent] Created: {root_agent.name}")
print(f"[Agent] Model: deepseek/deepseek-reasoner")
print(f"[Agent] Tools: {len(demo_toolset.tool_filter)} demo tools")
print()
print("=" * 80)
print("Ready to test! Run with: adk web multi_tool_agent/agent.py")
print("=" * 80)
