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

# NOTE: Switched from deepseek-reasoner to deepseek-chat due to DeepSeek API
# validation changes (Dec 2025) that break reasoning_content handling in
# Google ADK when using tools with multi-turn conversations.
# See: https://api-docs.deepseek.com/guides/thinking_with_tools

# Planning: Use chat model (supports tool calling reliably)
planning_model = LiteLlm(model="openai/gpt-5-mini")

# Execution agents: Use chat model for tool calling
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

1. **ReachTarget** - Keywords: "reach", "touch", "go to", "target"
   - Move the gripper to touch a target sphere/object
   - Scene: Contains a colored target sphere to reach

2. **PickAndLift** - Keywords: "pick and lift", "lift block", "lift cube", "block", "cube"
   - Pick up a colored block and lift it to a target height
   - Scene: Contains colored blocks and a target sphere indicating lift height

3. **PickUpCup** - Keywords: "pick up cup", "grab cup", "cup"
   - Pick up a cup from the table
   - Scene: Contains a cup on a table

4. **PushButton** - Keywords: "push button", "press button", "button"
   - Push or press a button
   - Scene: Contains a button panel with buttons

**IMPORTANT:** The task name determines which RLBench scene is loaded. Each scene has different objects:
- Use **PickAndLift** for colored blocks/cubes with specific lift targets
- Use **PickUpCup** specifically for cup manipulation
- Use **PushButton** for button pressing tasks
- Match the object mentioned by user to the correct task scene

**YOUR OUTPUT:**

Generate a plan in this format:

---
## Task Analysis
- **Task Type:** [ReachTarget / PickAndLift / PickUpCup / PushButton]
- **Target Object:** [description, e.g., "red ball", "blue cube", "cup", "button"]

## Execution Plan

### Step 1: Reset Environment
Reset the RLBench environment with the appropriate task.

### Step 2: Capture Sensor Data
Get camera observation (RGB, depth, point cloud).

### Step 3: Detect Object(s)
Use perception to find the 3D position of the target object(s).
- **ReachTarget/PickUpCup/PushButton:** Detect single object
- **PickAndLift:** Detect BOTH cube and target sphere (multi-object detection)

### Step 4: Execute Motion
Task-specific motion sequence:

[ReachTarget]:
  **Simple reach task - gripper stays OPEN**

  **Motion Sequence:**
  1. Keep gripper OPEN (do not close)
  2. Move above target (target_x, target_y, target_z + 0.1m)
  3. Descend to target position (target_x, target_y, target_z)
  4. **TASK COMPLETE** - gripper touching target

  **CRITICAL:**
  ✅ Keep gripper OPEN throughout
  ✅ Just move to target position
  ❌ DO NOT close gripper
  ❌ DO NOT grasp anything

[PickAndLift]:
  **CORRECT UNDERSTANDING: Pick cube and MOVE IT TO the sphere position (all XYZ)!**

  The sphere shows the TARGET DESTINATION where the cube should be lifted to.
  You must move the cube TO the sphere's full 3D position (X, Y, Z).

  **Detection:**
  - Sphere is translucent and hard to detect visually
  - Detect cube with camera: "red cube"
  - Get sphere position from ground truth: `get_target_position()['lift_target_position']`

  **CORRECT Motion Sequence:**
  1. Open gripper
  2. Move above cube (cube_x, cube_y, cube_z + 0.15m)
  3. Descend to grasp height (cube_x, cube_y, cube_z + 0.02m)
  4. Close gripper - grab cube
  5. Lift up (cube_x, cube_y, cube_z + 0.15m) - clear table
  6. **Move to sphere position (sphere_x, sphere_y, sphere_z)** ← Use ALL coordinates!
  7. **TASK COMPLETE** - cube is at sphere location

  **CRITICAL: Use ALL 3 coordinates of sphere position!**
  ✅ DO move to (sphere_x, sphere_y, sphere_z)
  ✅ DO use move_to_position(sphere_x, sphere_y, sphere_z)
  ❌ DO NOT open gripper (keep cube held)
  ❌ DO NOT only use sphere_z (need full XYZ!)

[PickUpCup]:
  - Open gripper
  - Move above cup (z + 0.15m)
  - Descend to cup rim (z + 0.05m)
  - Close gripper around cup
  - Lift cup upward

[PushButton]:
  **Push with CLOSED gripper (make a fist)**

  **Motion Sequence:**
  1. **Close gripper FIRST** (make fist to push with)
  2. Move above button (button_x, button_y, button_z + 0.1m)
  3. Descend to button surface (button_x, button_y, button_z)
  4. Push down slightly (button_x, button_y, button_z - 0.03m)
  5. Retract upward
  6. **TASK COMPLETE**

  **CRITICAL:**
  ✅ Close gripper BEFORE moving (step 1!)
  ✅ Push with closed fist
  ❌ DO NOT push with open gripper (will collide with mount)

[PutRubbishInBin]:
  **Multi-object detection required: trash object + bin location**

  **Detection:** Use automatic prompt (e.g., "trash . bin" or just "trash")
  - objects[0] = trash position → Pick up this object
  - objects[1] = bin position (if detected) → Place target
  - Get bin position from ground truth: `get_target_position()['lift_target_position']`

  **Motion Sequence:**
  1. Open gripper
  2. Move above trash (trash_x, trash_y, trash_z + 0.15m)
  3. Descend to grasp height (trash_x, trash_y, trash_z + 0.02m)
  4. Close gripper - grab trash
  5. Lift up (trash_z + 0.15m)
  6. Move to bin center (bin_x, bin_y, bin_z + 0.15m)
  7. Lower into bin (bin_x, bin_y, bin_z)
  8. Open gripper - release trash
  9. Retract upward

  **Key differences from PickAndLift:**
  - This DOES require moving to target XY position (the bin)
  - This DOES require opening gripper at the end (to release)
  - Target is an actual container, not just a height indicator

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
- `reset_task(task_name)` - Reset environment
  Options: "ReachTarget", "PickAndLift", "PickUpCup", "PushButton", "PutRubbishInBin"
- `get_camera_observation()` - Returns paths to RGB, depth, intrinsics, pose, and point cloud files
  Also returns `task_objects` list and `detection_prompt` string
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
**CRITICAL: Keep gripper OPEN - this is just a reach task!**
1. Ensure gripper is OPEN (do NOT close it)
2. Move above target: `move_to_position(x, y, z + 0.1)`
3. Descend to target: `move_to_position(x, y, z)`
4. Report success

### For PickAndLift:
**CRITICAL: Move cube TO sphere XYZ (use ALL coordinates)!**
1. Open gripper: `control_gripper("open")`
2. Move above cube: `move_to_position(cube_x, cube_y, cube_z + 0.15)`
3. Descend to grasp: `move_to_position(cube_x, cube_y, cube_z + 0.02)`
4. Close gripper: `control_gripper("close")`
5. Lift clear of table: `move_to_position(cube_x, cube_y, cube_z + 0.15)`
6. **Move TO sphere position**: `move_to_position(sphere_x, sphere_y, sphere_z)` - Use ALL XYZ!
7. Keep gripper CLOSED, report success
**Note**: Get sphere position from `get_target_position()['lift_target_position']`

### For PickUpCup:
1. Open gripper: `control_gripper("open")`
2. Move above cup: `move_to_position(x, y, z + 0.15)`
3. Descend to cup rim level: `move_to_position(x, y, z + 0.05)`
4. Close gripper around cup: `control_gripper("close")`
5. Lift cup: `lift_gripper(0.2)`
6. Report success

### For PushButton:
**IMPORTANT**: Gripper must be CLOSED before approaching button
1. Close gripper: `control_gripper("close")`
2. Move above button: `move_to_position(x, y, z + 0.1)`
3. Descend to button surface: `move_to_position(x, y, z)`
4. Push button (move down): `move_to_position(x, y, z - 0.03)`
5. Retract: `move_to_position(x, y, z + 0.1)`
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

**🚨 CRITICAL: PickAndLift Task Understanding 🚨**

PickAndLift: Pick cube and MOVE IT TO the sphere's position (full XYZ)!
- Sphere = TARGET DESTINATION (use ALL 3 coordinates: X, Y, Z)
- Motion: Pick cube → Lift up → **MOVE TO sphere XYZ** → DONE
- ✅ DO move to sphere's full 3D position
- ❌ DO NOT open gripper (keep holding cube)
- Sphere is translucent - use ground truth `get_target_position()` for sphere location

Example coordinates:
- Cube at [0.027, -0.360, 0.775]
- Sphere at [0.365, -0.103, 1.000]
- ✅ Move cube TO [0.365, -0.103, 1.000] ← Use ALL coordinates!
- ❌ DO NOT only use Z: [0.027, -0.360, 1.000] ← Wrong!

**YOUR WORKFLOW:**

## Phase 1: Planning
When the user gives a task:
1. Analyze the task (ReachTarget, PickAndLift, PickUpCup, PushButton, or PutRubbishInBin)
2. Generate an execution plan
3. Present the plan clearly and ASK FOR APPROVAL
4. Wait for user to say "approved", "yes", "proceed", etc.

## Phase 2: Execution (only after approval)
Once approved, execute in sequence:
1. **Sensing:** Reset environment, capture camera data
2. **Perception:** Detect target object(s) 3D position(s) - SUPPORTS MULTIPLE OBJECTS!
3. **Motion:** Execute task-specific motion sequence

**AVAILABLE TOOLS:**

Sensing Tools:
- `reset_task(task_name)` - Reset environment
- `get_camera_observation()` - Get RGB, depth, point cloud paths
  **NEW: Returns `task_objects` list and `detection_prompt` string!**
  - Automatically parses task description to extract object names WITH COLORS
  - Example: "pick up the red block and lift it up to the target" → task_objects=["red cube", "sphere"], detection_prompt="red cube . sphere"
  - Example: "reach the red target" → task_objects=["red sphere"], detection_prompt="red sphere"
  - USE the returned `detection_prompt` for detection - DON'T hardcode prompts!
- `get_target_position()` - Get ground truth (debug)
- `get_current_state()` - Get gripper state

Perception Tools:
- `detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path)` - Get 3D position(s)

  **MULTI-OBJECT DETECTION:**
  - Single object: `text_prompt="red cube"` → Returns 1 position
  - Multiple objects: `text_prompt="red cube . sphere"` → Returns multiple positions in `objects` array
  - Separate prompts with period+space: ` . `
  - **IMPORTANT: Use `detection_prompt` from `get_camera_observation()` - it's automatically generated from task description!**
  - **Color Verification**: Colors are automatically verified using HSV analysis - you'll see corrected labels in the response

Motion Tools:
- `move_to_position(x, y, z, use_planning=True)` - Move gripper
- `control_gripper(action)` - "open" or "close"
- `lift_gripper(height)` - Lift by height

**TASK SELECTION RULES:**
- Look for specific object names: "block"/"cube" → PickAndLift, "cup" → PickUpCup, "button" → PushButton, "rubbish"/"trash"+"bin" → PutRubbishInBin
- Action type: "reach"/"touch" → ReachTarget, "push"/"press" → PushButton, "put in"/"throw away" → PutRubbishInBin
- Each task loads a different scene in CoppeliaSim with appropriate objects

**GRIPPER STATE BY TASK:**

| Task | Gripper State | Why |
|------|---------------|-----|
| **ReachTarget** | Keep OPEN | Just touching target, not grasping |
| **PushButton** | CLOSE FIRST | Push with fist to avoid collision |
| **PickAndLift** | Open → Close → Keep closed | Grasp cube, keep holding |
| **PickUpCup** | Open → Close → Keep closed | Grasp cup, keep holding |
| **PutRubbishInBin** | Open → Close → Open at end | Grasp trash, release in bin |

**CRITICAL TASK UNDERSTANDING:**

**PickAndLift vs PutRubbishInBin - Know the Difference!**

| Task | Target Meaning | Motion Type | Use full XYZ? | Open Gripper? |
|------|----------------|-------------|---------------|---------------|
| **PickAndLift** | Destination position | Pick + move to target | ✅ YES - move to sphere XYZ | ❌ NO - keep closed |
| **PutRubbishInBin** | Container to place in | Pick + move + release | ✅ YES - move to bin XYZ | ✅ YES - release in bin |

PickAndLift: Pick cube → Lift up → **Move TO sphere XYZ** → DONE (keep holding)
PutRubbishInBin: Pick trash → Move to bin XYZ → Lower into bin → **Release** → DONE

**DETECTION STRATEGY - USE AUTOMATIC PROMPTS:**

**ALWAYS use `detection_prompt` from `get_camera_observation()` for detection!**

The system automatically extracts object names WITH COLORS from the task description:
- ReachTarget: "reach the red target" → "red sphere"
- PickAndLift: "pick up the red block and lift it up to the target" → "red cube"
  **SPECIAL:** Sphere is translucent! Detect cube only, use `get_target_position()['lift_target_position']` for sphere XYZ
- PickUpCup: "pick up the red cup" → "red cup"
- PushButton: "push the button" → "button"
- PutRubbishInBin: "put rubbish in bin" → "bin . trash"

**COLOR VERIFICATION**: The system automatically verifies colors using HSV analysis!
- GroundingDINO detects object shapes (cube, sphere, etc.)
- Color verification corrects mislabeled colors using actual pixel analysis
- You'll see logs like: `[Color Verify] ✓ Corrected: 'red sphere' → 'blue sphere'`
- Still recommend using `get_target_position()` to verify correct object selected

**Workflow:**
```
# Step 1: Get camera observation
camera_data = get_camera_observation()
detection_prompt = camera_data['detection_prompt']  # e.g., "cube . sphere"

# Step 2: Use automatic prompt for detection
result = detect_object_3d(detection_prompt, camera_data['rgb_path'], ...)

# Step 3: Extract positions based on task type
```

**Task-Specific Position Extraction:**

- **Single-object tasks (ReachTarget, PickUpCup, PushButton):**
  - Use `position_3d` directly: `pos = result['position_3d']`

- **PickAndLift (SPECIAL CASE - sphere is translucent!):**
  - Detect cube only: `detect_object_3d("red cube", ...)`
  - Get cube position: `cube_pos = result['position_3d']`
  - Get sphere position from ground truth: `target_data = get_target_position()`
  - Extract sphere XYZ: `sphere_pos = target_data['lift_target_position']`

  **MOTION PATTERN FOR PickAndLift:**
  ```
  # Detect cube
  cube_result = detect_object_3d("red cube", ...)
  cube_x, cube_y, cube_z = cube_result['position_3d']

  # Get sphere position from ground truth
  target_data = get_target_position()
  sphere_x, sphere_y, sphere_z = target_data['lift_target_position']

  # Pick cube
  move_to_position(cube_x, cube_y, cube_z + 0.15)  # above
  move_to_position(cube_x, cube_y, cube_z + 0.02)  # grasp
  control_gripper("close")

  # Lift and move TO sphere position (use ALL coordinates!)
  move_to_position(cube_x, cube_y, cube_z + 0.15)  # lift clear
  move_to_position(sphere_x, sphere_y, sphere_z)  # MOVE TO SPHERE!
  # DONE - keep gripper closed
  ```

  **CRITICAL:** PickAndLift = "move cube TO sphere position". Use ALL 3 coordinates!
  ✅ DO use sphere's X, Y, AND Z coordinates
  ❌ NEVER open gripper after moving to sphere
  ✅ Task completes when cube reaches sphere XYZ position

- **PutRubbishInBin:**
  - `trash_pos = objects[0]['position_3d']` → [trash_x, trash_y, trash_z]
  - `bin_pos = objects[1]['position_3d']` → [bin_x, bin_y, bin_z] (use ALL coordinates)

**EXAMPLE INTERACTIONS:**

Example 1 - ReachTarget (Keep gripper OPEN):
User: "Reach the red target"
You:
  Step 1: Get camera observation → detection_prompt = "red sphere"
  Step 2: Detect using automatic prompt: detect_object_3d("red sphere", ...)
  Step 3: Verify with ground truth: get_target_position()
  Step 4: Select detection closest to ground truth
  Step 5: Keep gripper OPEN (do not close!)
  Step 6: Move above target (target_x, target_y, target_z + 0.1)
  Step 7: Descend to target (target_x, target_y, target_z)
  Step 8: Task complete - gripper touching target

  **CRITICAL:** Gripper stays OPEN - this is just a reach task!

Example 1b - PushButton (Close gripper FIRST):
User: "Push the button"
You:
  Step 1: Get camera observation → detection_prompt = "button"
  Step 2: Detect using automatic prompt: detect_object_3d("button", ...)
  Step 3: **CLOSE GRIPPER FIRST** control_gripper("close")  ← MUST DO BEFORE MOVING!
  Step 4: Move above button (button_x, button_y, button_z + 0.1)
  Step 5: Descend to button surface (button_x, button_y, button_z)
  Step 6: Push down slightly (button_x, button_y, button_z - 0.03)
  Step 7: Retract upward
  Step 8: Task complete

  **CRITICAL:** Close gripper BEFORE moving - push with fist!

Example 2 - PickAndLift (use ground truth for sphere):
User: "Pick and lift the block"
You:
  Step 1: Get camera observation → detection_prompt = "red cube"
  Step 2: Detect cube: detect_object_3d("red cube", ...) → cube_pos = [0.027, -0.360, 0.775]
  Step 3: Get sphere from ground truth: get_target_position() → sphere_pos = [0.365, -0.103, 1.000]
  Step 4: Pick cube at (0.027, -0.360, 0.775)
  Step 5: Lift up to (0.027, -0.360, 0.925) - clear table
  Step 6: **Move TO sphere (0.365, -0.103, 1.000)** ← Use ALL coordinates!
  Step 7: Task complete - cube at sphere position (gripper stays closed)

  **CORRECT:** Move cube TO sphere's full XYZ position
  **WRONG:** Only lifting vertically without moving to sphere XY

Example 3 - Future Multi-Object:
User: "Put the trash in the bin"
You: [Select PutInBin, detect "trash . bin", pick trash from objects[0], place in objects[1]]

**IMPORTANT:**
- ALWAYS wait for explicit approval before executing
- Use multi-object detection when task requires object + target location
- Parse `objects` array to extract individual positions
- Report each phase's results clearly
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
    print("  - PickUpCup: 'Pick up the cup'")
    print("  - PushButton: 'Push the button'")
    print("  - PutRubbishInBin: 'Put rubbish in bin'")
    print("\nAgents:")
    print(f"  1. {root_agent.name} (orchestrator with human-in-the-loop)")
    print(f"  2. {sensing_agent.name}")
    print(f"  3. {perception_agent.name}")
    print(f"  4. {motion_agent.name}")
    print("\nUsage:")
    print("  - Import root_agent for ADK web interface")
    print("  - Import sequential_pipeline for automated execution")
    print("=" * 60)
