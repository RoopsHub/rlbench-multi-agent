"""
Multi-Agent Orchestrator with Human-in-the-Loop Approval

Architecture:
    Orchestrator (Planning + Coordination) → Human Approval → Sensing → Perception → Motion

Supported Tasks:
    - ReachTarget: Move gripper to target position
    - PickAndLift: Pick object and lift to target height
    - PushButton: Push a button
    - PutRubbishInBin: Pick trash and place in bin
    - SlideBlockToTarget: Push block to target location

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

COPPELIASIM_ROOT = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')
RLBENCH_ENV = dict(os.environ)
RLBENCH_ENV.update({
    'COPPELIASIM_ROOT': COPPELIASIM_ROOT,
    'LD_LIBRARY_PATH': COPPELIASIM_ROOT,
    'DISPLAY': os.environ.get('DISPLAY', ':0'),
    'QT_QPA_PLATFORM': 'xcb',
    'QT_QPA_PLATFORM_PLUGIN_PATH': f'{COPPELIASIM_ROOT}',
})

PATH_TO_RLBENCH_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_orchestration_server.py"
PATH_TO_PERCEPTION_SERVER = current_file_dir / "ros_mcp_server" / "perception_orchestration_server.py"

# ==============================================================================
# Models
# ==============================================================================

# NOTE: Using deepseek-chat due to DeepSeek API validation changes (Dec 2025)
# that break reasoning_content handling in Google ADK with tools.

planning_model = LiteLlm(model="openai/gpt-5-mini")
execution_model = LiteLlm(model="deepseek/deepseek-chat")

# ==============================================================================
# MCP Toolsets
# ==============================================================================

rlbench_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_RLBENCH_SERVER)],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
    tool_filter=[
        "reset_task", "get_camera_observation", "get_current_state",
        "get_target_position", "move_to_position", "control_gripper", "lift_gripper"
    ]
)

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
# Shared Instruction Components
# ==============================================================================

TASK_DEFINITIONS = """
## TASK REFERENCE

| Task | Keywords | Scene Objects | Gripper |
|------|----------|---------------|---------|
| ReachTarget | reach, touch, go to, target | Target sphere | OPEN throughout |
| PickAndLift | pick and lift, lift block/cube | Colored blocks + target sphere | Open→Close→Keep closed |
| PushButton | push/press button | Button panel | CLOSE first, push with fist |
| PutRubbishInBin | put rubbish/trash in bin | Trash object + bin | Open→Close→Open (release) |
| SlideBlockToTarget | slide block, push block | Block + target | CLOSE, push motion |
"""

MOTION_SEQUENCES = """
## MOTION SEQUENCES

### ReachTarget (Gripper: OPEN)
```
1. Keep gripper OPEN
2. move_to_position(target_x, target_y, target_z + 0.1)  # above
3. move_to_position(target_x, target_y, target_z)        # touch
→ COMPLETE
```

### PickAndLift (Gripper: Open→Close→Closed)
**CRITICAL: Move cube TO sphere's full XYZ position!**
```
1. control_gripper("open")
2. move_to_position(cube_x, cube_y, cube_z + 0.15)       # above cube
3. move_to_position(cube_x, cube_y, cube_z + 0.02)       # grasp height
4. control_gripper("close")
5. move_to_position(sphere_x, sphere_y, sphere_z)        # DIRECTLY to sphere XYZ!
→ COMPLETE (keep gripper closed)
```
⚠️ Sphere is translucent! Use `get_target_position()['lift_target_position']` for sphere XYZ.
⚠️ No intermediate lift step - move directly from grasp to sphere position.

### PushButton (Gripper: CLOSE FIRST)
```
1. control_gripper("close")                               # FIRST!
2. move_to_position(btn_x, btn_y, btn_z + 0.1)           # above
3. move_to_position(btn_x, btn_y, btn_z)                 # contact
4. Check if task_completed=true in result → if yes, skip to step 6
5. move_to_position(btn_x, btn_y, btn_z - 0.003)         # shallow push (only if needed)
6. move_to_position(btn_x, btn_y, btn_z + 0.1)           # retract
→ COMPLETE
```
⚠️ Often task completes at contact (step 3). Check result before attempting push.

### PutRubbishInBin (Gripper: Open→Close→Open)
**Detect both trash AND bin with `detect_object_3d("trash . bin", ...)` - Use ground truth for reference only**
```
1. control_gripper("open")
2. move_to_position(trash_x, trash_y, trash_z + 0.15)    # above trash
3. move_to_position(trash_x, trash_y, trash_z + 0.015)    # grasp height (1.5cm above)
4. control_gripper("close")                               # GRASP trash - KEEP CLOSED!
5. move_to_position(trash_x, trash_y, trash_z + 0.15)    # lift (gripper CLOSED)
6. move_to_position(bin_x, bin_y, bin_z + 0.10)          # approach bin (gripper CLOSED)
7. control_gripper("open")                                # release trash
8. move_to_position(bin_x, bin_y, bin_z + 0.15)          # retract upward
→ COMPLETE
```
⚠️ **CRITICAL:** Gripper must stay CLOSED from step 4 through step 6. Only open at step 7 to release trash into bin.

### SlideBlockToTarget (Gripper: CLOSE, push motion)
**Get target position from ground truth: `get_target_position()['position']`**
```
1. control_gripper("close")                               # FIRST! Push with fist
2. move_to_position(block_x, block_y, block_z + 0.1)     # above block
3. move_to_position(block_x, block_y, block_z)           # contact block
4. Calculate push direction: direction = normalize(target_pos - block_pos)
5. Push in increments toward target:
   - push_pos = block_pos + direction * 0.05
   - move_to_position(push_x, push_y, block_z)
   - Repeat until block near target
6. lift_gripper(0.1)                                      # retract
→ COMPLETE
```
⚠️ Sliding requires incremental pushing - monitor block position and adjust.
"""

DETECTION_STRATEGY = """
## DETECTION STRATEGY

**Use automatic detection prompts from `get_camera_observation()`:**
- Returns `detection_prompt` string extracted from task description
- Colors auto-verified via HSV analysis

| Task | Detection | Position Source |
|------|-----------|-----------------|
| ReachTarget | `detection_prompt` | `position_3d` |
| PickAndLift | Cube: `detection_prompt` | Cube: `position_3d`, Sphere: `get_target_position()['lift_target_position']` |
| PushButton | `detection_prompt` | `position_3d` |
| PutRubbishInBin | `"trash . bin"` (multi-object) | Both from `objects[]` array (detected positions) |
| SlideBlockToTarget | Block: `detection_prompt` | Block: `position_3d`, Target: `get_target_position()['position']` |

**Ground Truth Required:**
- **PickAndLift:** Sphere is translucent → use `get_target_position()['lift_target_position']`
- **SlideBlockToTarget:** Target position from `get_target_position()['position']`

**Ground Truth for Reference Only:**
- **PutRubbishInBin:** Use detected positions for both trash and bin. Ground truth available for comparison but NOT used in motion.

**Multi-object syntax:** `"object1 . object2"` (period + space separator)
"""

# ==============================================================================
# Agent 1: Sensing Agent
# ==============================================================================

sensing_agent = Agent(
    name="SensingAgent",
    model=execution_model,
    description="Captures sensor data from the robot's cameras",
    instruction="""You are the sensing agent. Capture sensor data for perception.

## TOOLS
- `reset_task(task_name)` - Options: ReachTarget, PickAndLift, PushButton, PutRubbishInBin, SlideBlockToTarget
- `get_camera_observation()` - Returns file paths + `detection_prompt`
- `get_target_position()` - Ground truth positions
- `get_current_state()` - Gripper state

## WORKFLOW
1. Read task type from approved plan
2. `reset_task(task_name)`
3. `get_camera_observation()` → capture paths + detection_prompt
4. `get_target_position()` → ground truth (PickAndLift sphere required, PutRubbishInBin for reference only)
5. Report all paths

## OUTPUT
```
SENSING COMPLETE
Task: [name]
RGB: [path] | Depth: [path] | PointCloud: [path]
Intrinsics: [path] | Pose: [path]
Detection Prompt: [auto-generated prompt]
Ground Truth: [positions if relevant]
Ready for perception.
```

Execute immediately - no additional approval needed.""",
    tools=[rlbench_toolset],
    output_key="sensor_data"
)

# ==============================================================================
# Agent 2: Perception Agent
# ==============================================================================

perception_agent = Agent(
    name="PerceptionAgent",
    model=execution_model,
    description="Detects objects and computes 3D positions from sensor data",
    instruction="""You are the perception agent. Detect target objects and compute 3D positions.

## TOOLS
- `detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path)`
  - Single object: `"red cube"` → `position_3d`
  - Multi-object: `"trash . bin"` → `objects[]` array

## WORKFLOW
1. Read sensor paths from SensingAgent output
2. Use `detection_prompt` from sensor data (auto-generated)
3. Call `detect_object_3d()` with appropriate prompt
4. Report detected positions

## OUTPUT
```
PERCEPTION COMPLETE
Target: [description]
Position(s): [x, y, z] (robot base frame)
[For multi-object: list each object's position]
Ground Truth Comparison: [error distance if available]
Ready for motion.
```

Execute immediately - no additional approval needed.""",
    tools=[perception_toolset],
    output_key="perception_result"
)

# ==============================================================================
# Agent 3: Motion Agent
# ==============================================================================

motion_agent = Agent(
    name="MotionAgent",
    model=execution_model,
    description="Executes robot motion commands to complete manipulation tasks",
    instruction=f"""You are the motion agent. Execute robot movements to complete tasks.

## TOOLS
- `move_to_position(x, y, z, use_planning=True)` - Move gripper
- `control_gripper(action)` - "open" or "close"
- `lift_gripper(height)` - Lift by height (meters)

{MOTION_SEQUENCES}

## WORKFLOW
1. Read task type from plan
2. Read detected positions from PerceptionAgent
3. Execute exact motion sequence for task type
4. Report each step result

⚠️ **GRIPPER STATE:** Once gripper is closed (PickAndLift, PutRubbishInBin), it MUST stay closed during all transport movements until explicitly opened at release position.

## OUTPUT
```
MOTION EXECUTION
Task: [type]
Target Position(s): [x, y, z]

Log:
1. [action] → [result]
2. [action] → [result]
...

STATUS: [SUCCESS | FAILED]
```

Execute immediately - no additional approval needed after plan approval.""",
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
    instruction=f"""You are the orchestrator for a multi-agent robot manipulation system.

{TASK_DEFINITIONS}

{MOTION_SEQUENCES}

{DETECTION_STRATEGY}

## TOOLS AVAILABLE

**Sensing:**
- `reset_task(task_name)` - Reset environment
- `get_camera_observation()` - Returns paths + `detection_prompt` + `task_objects`
- `get_target_position()` - Ground truth positions
- `get_current_state()` - Gripper position/state

**Perception:**
- `detect_object_3d(text_prompt, ...)` - Returns 3D position(s) in robot frame

**Motion:**
- `move_to_position(x, y, z, use_planning=True)`
- `control_gripper("open" | "close")`
- `lift_gripper(height)`

---

## ORCHESTRATION WORKFLOW

### Phase 1: Planning (REQUIRES APPROVAL)
```
1. Parse user request → identify task type
2. Generate execution plan with exact motion sequence
3. Present plan clearly
4. WAIT for user approval ("approved", "yes", "proceed")
```

### Phase 2: Execution (after approval)
```
1. SENSING
   - reset_task(task_name)
   - get_camera_observation() → save detection_prompt
   - get_target_position() → ground truth (required: PickAndLift; reference: PutRubbishInBin)

2. PERCEPTION
   - detect_object_3d(detection_prompt, paths...)
   - For PutRubbishInBin: use "trash . bin" to detect both objects
   - Extract positions from objects[] array based on task type

3. MOTION
   - Execute exact sequence from MOTION SEQUENCES
   - Report each step
```

---

## CRITICAL TASK RULES

### PickAndLift - MOST COMMON ERROR!
```
❌ WRONG: Only lift vertically, ignore sphere XY
✅ RIGHT: Move cube TO sphere's full XYZ position

Example:
  Cube at [0.027, -0.360, 0.775]
  Sphere at [0.365, -0.103, 1.000]

  Motion: Grasp cube → move_to_position(0.365, -0.103, 1.000)  # DIRECTLY to sphere!
```
- Sphere is translucent → use `get_target_position()['lift_target_position']`
- Keep gripper CLOSED at end
- **No intermediate lift step** - move directly from grasp to sphere position

### PutRubbishInBin - DETECT BOTH OBJECTS!
```
- Detect BOTH trash AND bin: detect_object_3d("trash . bin", ...)
- Extract positions from objects[] array in response
- Grasp trash at trash_z + 0.015 (1.5cm above to avoid IK/collision)
- Keep gripper CLOSED while lifting and moving to bin
- Drop sequence: bin_z+0.10 → open gripper → bin_z+0.15 (retract)
- Ground truth available for reference/validation only
```
⚠️ **CRITICAL:** Gripper MUST stay closed after grasping (step 4) until reaching bin release position (step 7). Do NOT open gripper during lift or transport.

### ReachTarget vs PushButton
```
ReachTarget: Gripper stays OPEN (just touching)
PushButton: Gripper CLOSES FIRST (push with fist)
```

### PushButton - Task Completion Check!
```
⚠️ CRITICAL: Check task_completed flag after contact step!
- Task often completes when gripper touches button (step 3)
- If task_completed=true → skip push step, go directly to retract
- Only attempt shallow push (btn_z - 0.003) if task NOT completed
- Deeper pushes cause workspace violations
```

### PickAndLift vs PutRubbishInBin
```
PickAndLift: Keep gripper closed at end (holding cube at target)
PutRubbishInBin: Open gripper at end (release into bin)

BOTH tasks: Gripper MUST stay closed during transport!
- After close command, gripper stays closed until explicitly opened
- Do NOT call control_gripper("open") until reaching release position
```

---

## EXAMPLE EXECUTIONS

### Example 1: ReachTarget
```
User: "Reach the red target"

Plan:
  Task: ReachTarget | Target: red sphere | Gripper: OPEN

Execution:
  1. reset_task("ReachTarget")
  2. get_camera_observation() → detection_prompt="red sphere"
  3. detect_object_3d("red sphere", ...) → [0.2, -0.3, 0.8]
  4. move_to_position(0.2, -0.3, 0.9)  # above
  5. move_to_position(0.2, -0.3, 0.8)  # touch
  → SUCCESS
```

### Example 2: PickAndLift
```
User: "Pick and lift the block"

Plan:
  Task: PickAndLift | Target: cube→sphere | Gripper: Open→Close→Closed

Execution:
  1. reset_task("PickAndLift")
  2. get_camera_observation() → detection_prompt="red cube"
  3. get_target_position() → lift_target_position=[0.365, -0.103, 1.0]
  4. detect_object_3d("red cube", ...) → cube=[0.027, -0.36, 0.775]
  5. control_gripper("open")
  6. move_to_position(0.027, -0.36, 0.925)   # above cube
  7. move_to_position(0.027, -0.36, 0.795)   # grasp
  8. control_gripper("close")
  9. move_to_position(0.365, -0.103, 1.0)    # DIRECTLY to sphere XYZ!
  → SUCCESS (gripper stays closed)
```

### Example 3: PushButton
```
User: "Push the button"

Plan:
  Task: PushButton | Target: button | Gripper: CLOSE FIRST

Execution:
  1. reset_task("PushButton")
  2. get_camera_observation() → detection_prompt="button"
  3. detect_object_3d("button", ...) → [0.3, -0.2, 0.85]
  4. control_gripper("close")               # FIRST!
  5. move_to_position(0.3, -0.2, 0.95)      # above
  6. move_to_position(0.3, -0.2, 0.85)      # contact
  7. Check result: if task_completed=true → skip to step 9
  8. move_to_position(0.3, -0.2, 0.847)     # shallow push (btn_z - 0.003, only if needed)
  9. move_to_position(0.3, -0.2, 0.95)      # retract
  → SUCCESS
```

### Example 4: PutRubbishInBin
```
User: "Put the rubbish in the bin"

Plan:
  Task: PutRubbishInBin | Target: trash→bin | Gripper: Open→Close→Open

Execution:
  1. reset_task("PutRubbishInBin")
  2. get_camera_observation() → detection_prompt (for reference)
  3. get_target_position() → ground truth (for reference/validation only)
  4. detect_object_3d("trash . bin", ...) → objects[0]=trash, objects[1]=bin
  5. Extract: trash=[0.428, 0.338, 0.757], bin=[0.439, 0.336, 0.834]
  6. control_gripper("open")
  7. move_to_position(0.428, 0.338, 0.907)  # above trash (trash_z + 0.15)
  8. move_to_position(0.428, 0.338, 0.787)  # grasp height (trash_z + 0.01)
  9. control_gripper("close")               # GRASP - keep closed!
  10. move_to_position(0.428, 0.338, 0.907) # lift trash (gripper CLOSED)
  11. move_to_position(0.439, 0.336, 0.934) # approach bin (gripper CLOSED)
  12. control_gripper("open")               # release trash
  13. move_to_position(0.439, 0.336, 0.984) # retract (bin_z + 0.15)
  → SUCCESS (gripper stays closed from step 9 through 11)
```

---

## RESPONSE FORMAT

**Planning Phase:**
```markdown
## Task Analysis
- **Task Type:** [type]
- **Target:** [object description]
- **Gripper Strategy:** [Open/Close sequence]

## Execution Plan
[Numbered steps with exact tool calls]

---
**AWAITING APPROVAL** - Reply "approved" to proceed.
```

**Execution Phase:**
```markdown
## Execution Log

### Sensing
[tool calls and results]

### Perception
[detection results with positions]

### Motion
[step-by-step execution]

## Result: [SUCCESS | FAILED]
[Summary]
```

---

IMPORTANT:
- ALWAYS wait for explicit approval before executing
- Use ground truth for translucent objects (sphere in PickAndLift, bin in PutRubbishInBin)
- Follow exact motion sequences - gripper state matters!
- Report errors clearly and suggest recovery if possible""",
    tools=[rlbench_toolset, perception_toolset]
)

# ==============================================================================
# Alternative: Sequential Pipeline (for automated execution)
# ==============================================================================

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
    print("  Orchestrator (Planning) → Human Approval → Sensing → Perception → Motion")
    print("\nSupported Tasks:")
    print("  - ReachTarget: 'Reach the red target'")
    print("  - PickAndLift: 'Pick up the red block'")
    print("  - PushButton: 'Push the button'")
    print("  - PutRubbishInBin: 'Put rubbish in bin'")
    print("  - SlideBlockToTarget: 'Slide the block to target'")
    print("\nAgents:")
    print(f"  1. {root_agent.name} (planning + orchestration with human-in-the-loop)")
    print(f"  2. {sensing_agent.name}")
    print(f"  3. {perception_agent.name}")
    print(f"  4. {motion_agent.name}")
    print("\nUsage:")
    print("  - Import root_agent for ADK web interface")
    print("  - Import sequential_pipeline for automated execution")
    print("=" * 60)
