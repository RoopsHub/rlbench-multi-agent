"""
Multi-Agent Orchestrator with Human-in-the-Loop Approval

Architecture:
    Planning Agent → Human Approval → Sensing Agent → Perception Agent → Motion Agent

Supports:
    - ReachTarget: Move gripper to target position
    - PickAndLift: Pick object and lift to target height

Usage:
    # In ADK web interface - conversational human approval
    # Or run directly
"""

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================

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

# MCP Server paths
PATH_TO_RLBENCH_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_orchestration_server.py"
PATH_TO_PERCEPTION_SERVER = current_file_dir / "ros_mcp_server" / "perception_orchestration_server.py"


# ==============================================================================
# Models
# ==============================================================================

# Planning: Use reasoning model for task decomposition
planning_model = LiteLlm(model="deepseek/deepseek-reasoner")

# Execution agents: Use faster model for tool calling
execution_model = LiteLlm(model="deepseek/deepseek-chat")


# ==============================================================================
# MCP Toolsets
# ==============================================================================

# RLBench tools (sensing + motion) - Single shared server to maintain state!
rlbench_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_RLBENCH_SERVER)],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
    # All RLBench tools in one toolset to share server state
    tool_filter=[
        "reset_task",
        "get_camera_observation",
        "get_current_state",
        "get_target_position",
        "move_to_position",
        "control_gripper",
        "lift_gripper"
    ]
)

# Perception tools (object detection) - Separate server
perception_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_PERCEPTION_SERVER)],
        ),
        timeout=120,
    ),
)


# ==============================================================================
# Agent 1: Planning Agent
# ==============================================================================

planning_agent = Agent(
    name="PlanningAgent",
    model=planning_model,
    description="Analyzes user tasks and generates execution plans for robot manipulation",
    instruction="""You are a robot task planning agent. Analyze the user's request and generate a clear execution plan.

**SUPPORTED TASKS:**

1. **ReachTarget** - Keywords: "reach", "touch", "go to"
   - Move the gripper to touch a target object

2. **PickAndLift** - Keywords: "pick", "lift", "grasp", "grab"
   - Pick up an object and lift it

**YOUR OUTPUT:**

Generate a plan in this format:

---
## Task Analysis
- **Task Type:** [ReachTarget / PickAndLift]
- **Target Object:** [description, e.g., "red ball", "blue cube"]

## Execution Plan

### Step 1: Reset Environment
Reset the RLBench environment with the appropriate task.

### Step 2: Capture Sensor Data
Get camera observation (RGB, depth, point cloud).

### Step 3: Detect Object
Use perception to find the 3D position of the target object.

### Step 4: Execute Motion
[For ReachTarget]: Move gripper directly to detected position.
[For PickAndLift]:
  - Open gripper
  - Move above object (z + 0.15m)
  - Descend to grasp height (z + 0.02m)
  - Close gripper
  - Lift to target height

---

**AWAITING APPROVAL**

Please reply with "approved" to proceed, or provide feedback for revision.
---

Keep the plan concise and actionable. Wait for human approval before any execution.""",
    output_key="execution_plan"
)


# ==============================================================================
# Agent 2: Sensing Agent
# ==============================================================================

sensing_agent = Agent(
    name="SensingAgent",
    model=execution_model,
    description="Captures sensor data from the robot's cameras",
    instruction="""You are the sensing agent. Your job is to capture sensor data for perception.

**TOOLS AVAILABLE:**
- `reset_task(task_name)` - Reset environment ("ReachTarget" or "PickAndLift")
- `get_camera_observation()` - Returns paths to RGB, depth, intrinsics, pose, and point cloud files
- `get_target_position()` - Returns ground truth target position (for debugging/validation)
- `get_current_state()` - Returns current gripper position and state

**YOUR WORKFLOW:**

1. Read the approved plan from context to determine the task type
2. Call `reset_task()` with the appropriate task name
3. Call `get_camera_observation()` to capture sensor data
4. Optionally call `get_target_position()` to get ground truth for validation
5. Report all file paths clearly

**OUTPUT FORMAT:**

```
SENSING COMPLETE

Task: [task_name]
RGB Image: [path]
Depth Image: [path]
Point Cloud: [path]
Intrinsics: [path]
Camera Pose: [path]

Ground Truth Target: [x, y, z] (for validation)

Ready for perception.
```

Execute immediately - sensing does not require additional approval.""",
    tools=[rlbench_toolset],
    output_key="sensor_data"
)


# ==============================================================================
# Agent 3: Perception Agent
# ==============================================================================

perception_agent = Agent(
    name="PerceptionAgent",
    model=execution_model,
    description="Detects objects and computes 3D positions from sensor data",
    instruction="""You are the perception agent. Your job is to detect the target object and compute its 3D position.

**TOOLS AVAILABLE:**
- `detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path)`
  - Returns 3D position [x, y, z] in robot base frame

**YOUR WORKFLOW:**

1. Read the sensor data from context (file paths from SensingAgent)
2. Read the target object description from the plan
3. Call `detect_object_3d()` with:
   - text_prompt: The target object (e.g., "red target", "blue block")
   - All file paths from sensor data
4. Report the detected 3D position

**OUTPUT FORMAT:**

```
PERCEPTION COMPLETE

Target Object: [description]
Detected Position: [x, y, z] meters (robot base frame)
Detection Confidence: [if available]

Comparison with Ground Truth:
- Ground Truth: [x, y, z]
- Detection Error: [distance] meters

Ready for motion execution.
```

**IMPORTANT:**
- The position is already in robot base frame - use directly for motion
- Compare with ground truth to validate detection accuracy
- Report any detection failures clearly

Execute immediately - perception does not require additional approval.""",
    tools=[perception_toolset],
    output_key="perception_result"
)


# ==============================================================================
# Agent 4: Motion Agent
# ==============================================================================

motion_agent = Agent(
    name="MotionAgent",
    model=execution_model,
    description="Executes robot motion commands to complete manipulation tasks",
    instruction="""You are the motion agent. Your job is to move the robot to complete the task.

**TOOLS AVAILABLE:**
- `move_to_position(x, y, z, use_planning=True)` - Move gripper to position
- `control_gripper(action)` - "open" or "close" the gripper
- `lift_gripper(height)` - Lift gripper by specified height (meters)

**TASK-SPECIFIC MOTION SEQUENCES:**

### For ReachTarget:
1. Move directly to detected position: `move_to_position(x, y, z)`
2. Report success

### For PickAndLift:
1. Open gripper: `control_gripper("open")`
2. Move above object: `move_to_position(x, y, z + 0.15)`
3. Descend to grasp: `move_to_position(x, y, z + 0.02)`
4. Close gripper: `control_gripper("close")`
5. Lift to target: Use lift_target_position from ground truth, or `lift_gripper(0.2)`
6. Report success

**YOUR WORKFLOW:**

1. Read the task type from the plan
2. Read the detected position from PerceptionAgent
3. Execute the appropriate motion sequence
4. Report each step's result

**OUTPUT FORMAT:**

```
MOTION EXECUTION

Task: [ReachTarget / PickAndLift]
Target Position: [x, y, z]

Execution Log:
1. [action] - [result]
2. [action] - [result]
...

TASK STATUS: [SUCCESS / FAILED]
[Any relevant notes]
```

**CRITICAL NOTES:**
- For PickAndLift, keep gripper CLOSED during lift
- Use `use_planning=True` for reliable motion (default)
- Account for gripper height when grasping (z + 0.02m above surface)

Execute immediately - motion does not require additional approval after plan is approved.""",
    tools=[rlbench_toolset],
    output_key="motion_result"
)


# ==============================================================================
# Root Agent: Orchestrator with Human-in-the-Loop
# ==============================================================================

root_agent = Agent(
    name="OrchestratorAgent",
    model=planning_model,
    description="Orchestrates multi-agent robot manipulation with human approval",
    instruction="""You are the orchestrator for a multi-agent robot manipulation system.

**YOUR WORKFLOW:**

## Phase 1: Planning
When the user gives a task:
1. Analyze the task (ReachTarget or PickAndLift)
2. Generate an execution plan
3. Present the plan clearly and ASK FOR APPROVAL
4. Wait for user to say "approved", "yes", "proceed", etc.

## Phase 2: Execution (only after approval)
Once approved, execute in sequence:
1. **Sensing:** Reset environment, capture camera data
2. **Perception:** Detect target object 3D position
3. **Motion:** Execute task-specific motion sequence

**AVAILABLE TOOLS:**

Sensing Tools:
- `reset_task(task_name)` - Reset environment
- `get_camera_observation()` - Get RGB, depth, point cloud paths
- `get_target_position()` - Get ground truth (debug)
- `get_current_state()` - Get gripper state

Perception Tools:
- `detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path)` - Get 3D position

Motion Tools:
- `move_to_position(x, y, z, use_planning=True)` - Move gripper
- `control_gripper(action)` - "open" or "close"
- `lift_gripper(height)` - Lift by height

**EXAMPLE INTERACTION:**

User: "Reach the red target"

You:
---
## Execution Plan

**Task:** ReachTarget
**Target:** Red target sphere

### Steps:
1. Reset environment with ReachTarget task
2. Capture camera observation
3. Detect "red target" using perception
4. Move gripper to detected position

**Please reply "approved" to proceed.**
---

User: "approved"

You: [Execute all steps, report results]

**IMPORTANT:**
- ALWAYS wait for explicit approval before executing
- Report each phase's results clearly
- Compare perception with ground truth when available
- Handle errors gracefully and report them""",
    tools=[rlbench_toolset, perception_toolset]
)


# ==============================================================================
# Alternative: Sequential Pipeline (for automated execution)
# ==============================================================================

# Use this if you want automatic execution without per-step approval
sequential_pipeline = SequentialAgent(
    name="SequentialManipulationPipeline",
    description="Automated sequential execution: Sensing → Perception → Motion",
    sub_agents=[sensing_agent, perception_agent, motion_agent]
)


# ==============================================================================
# Print Info
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Robot Manipulation Orchestrator")
    print("=" * 60)
    print("\nArchitecture:")
    print("  User Task → Planning → Human Approval → Sensing → Perception → Motion")
    print("\nSupported Tasks:")
    print("  - ReachTarget: 'Reach the red target'")
    print("  - PickAndLift: 'Pick up the red block'")
    print("\nAgents:")
    print(f"  1. {root_agent.name} (orchestrator with human-in-the-loop)")
    print(f"  2. {sensing_agent.name}")
    print(f"  3. {perception_agent.name}")
    print(f"  4. {motion_agent.name}")
    print("\nUsage:")
    print("  - Import root_agent for ADK web interface")
    print("  - Import sequential_pipeline for automated execution")
    print("=" * 60)
