"""
Multi-Agent Orchestrator with Human-in-the-Loop Approval

Architecture:
    Orchestrator (Planning + Coordination) → Human Approval → Sensing → Perception → Motion

Supported Tasks:
    - ReachTarget: Move gripper to target position
    - PickAndLift: Pick object and lift to target height
    - PushButton: Push a button
    - PutRubbishInBin: Pick trash and place in bin
    - StackBlocks: Stack 2 blocks of same color vertically

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
        "load_task", "reset_current_task", "get_camera_observation", "get_current_state",
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
| StackBlocks | stack blocks, build tower | 4 target blocks (same color) + 4 distractors | Open→Close→Open (repeated 2 times) |
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
**Detect both cube AND sphere with `detect_object_3d("red cube . sphere", ...)` - Use ground truth for reference only**
**CRITICAL: Move cube TO detected sphere's full XYZ position!**
```
1. control_gripper("open")
2. move_to_position(cube_x, cube_y, cube_z + 0.15)       # above cube
3. move_to_position(cube_x, cube_y, cube_z + 0.015)       # grasp height
4. control_gripper("close")                               # GRASP cube - KEEP CLOSED!
5. move_to_position(sphere_x, sphere_y, sphere_z)        # DIRECTLY to detected sphere XYZ!
→ COMPLETE (keep gripper closed)
```
⚠️ **Use detected sphere position from perception**, NOT ground truth! Both cube and sphere are detected via GroundingDINO.
⚠️ No intermediate lift step - move directly from grasp to sphere position.
⚠️ Gripper stays CLOSED at end (holding cube at target).

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
Often task completes at contact (step 3). Check result before attempting push.

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
**CRITICAL:** Gripper must stay CLOSED from step 4 through step 6. Only open at step 7 to release trash into bin.

### StackBlocks (Gripper: Open→Close→Open, REPEATED 2 times)
**Drop blocks from above onto stacking zone - simple and robust**
```
# STEP 0: Detect all cubes
0. detect_object_3d("red cube")
   # Green cube (LAB verified) = stacking zone [stack_x, stack_y, ignored_z]
   # Red cubes (high confidence) = target blocks
   # Select 2 red blocks with highest confidence

# CYCLE 1: Drop first block onto stacking zone
1. control_gripper("open")
2. move_to_position(block_1_x, block_1_y, block_1_z + 0.15)   # above block 1
3. move_to_position(block_1_x, block_1_y, block_1_z + 0.015)  # grasp height
4. control_gripper("close")                                    # GRASP block 1
5. move_to_position(block_1_x, block_1_y, block_1_z + 0.15)   # lift block
6. move_to_position(stack_x, stack_y, 0.80)                   # 5cm above stacking zone
7. control_gripper("open")                                     # DROP block 1 (gentle drop)
8. move_to_position(stack_x, stack_y, 0.90)                   # retract up

# CYCLE 2: Drop second block on top of first
9. move_to_position(block_2_x, block_2_y, block_2_z + 0.15)   # above block 2
10. move_to_position(block_2_x, block_2_y, block_2_z + 0.015) # grasp height
11. control_gripper("close")                                   # GRASP block 2
12. move_to_position(block_2_x, block_2_y, block_2_z + 0.15)  # lift block
13. move_to_position(stack_x, stack_y, 0.85)                  # 5cm above first block
14. control_gripper("open")                                    # DROP block 2 (gentle drop)
15. move_to_position(stack_x, stack_y, 0.90)                  # retract
→ COMPLETE (2 blocks stacked)
```
⚠️ **CRITICAL:**
- Use green cube for stack_x and stack_y ONLY (ignore green_z completely)
- Block 1 drop height: z = 0.80 (5cm above table - gentle drop, won't bounce)
- Block 2 drop height: z = 0.85 (5cm above first block)
- After dropping, retract to z = 0.90 to clear the area
- DO NOT try to place precisely - drop from low height and let gravity stack them
- Must track state: blocks_stacked = 0, 1 after each drop
- Select 2 red cubes with highest confidence
"""

DETECTION_STRATEGY = """
## DETECTION STRATEGY

**Use automatic detection prompts from `get_camera_observation()`:**
- Returns `detection_prompt` string extracted from task description
- Colors auto-verified via HSV analysis

| Task | Detection | Position Source |
|------|-----------|-----------------|
| ReachTarget | `detection_prompt` | `position_3d` |
| PickAndLift | `"red cube . sphere"` (multi-object) | Both from `objects[]` array (detected positions) |
| PushButton | `detection_prompt` | `position_3d` |
| PutRubbishInBin | `"trash . bin"` (multi-object) | Both from `objects[]` array (detected positions) |
| StackBlocks | `"red cube"` (detects all cubes including green stacking zone) | All from `objects[]` array - green cube is stacking zone, select 2 highest confidence red cubes |

**StackBlocks Special Detection:**
- Detect ALL cubes in scene: `detect_object_3d("red cube", ...)`
- LAB color verification identifies green cube as stacking zone (different color)
- Use green cube for X,Y position ONLY: [stack_x, stack_y] - ignore green_z
- Drop from low heights: Block 1 at z=0.80 (5cm above table), Block 2 at z=0.85
- Filter red cubes: select 2 blocks with highest confidence (ignore low confidence/achromatic)
- Track which blocks have been dropped (avoid re-picking)

**Ground Truth for Reference Only:**
- **All tasks:** Ground truth available for comparison/validation but NOT used in motion execution.
- System relies on perception (GroundingDINO + LAB color verification) for all object localization.

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
- `load_task(task_name)` - Load task: ReachTarget, PickAndLift, PushButton, PutRubbishInBin, StackBlocks
- `get_camera_observation()` - Returns file paths + `detection_prompt`
- `get_target_position()` - Ground truth positions
- `get_current_state()` - Gripper state

## WORKFLOW
1. Read task type from approved plan
2. `load_task(task_name)` → Loads and initializes the task
3. `get_camera_observation()` → capture paths + detection_prompt
4. `get_target_position()` → ground truth (for reference/validation only)
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

**GRIPPER STATE:** Once gripper is closed (PickAndLift, PutRubbishInBin), it MUST stay closed during all transport movements until explicitly opened at release position.

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
- `load_task(task_name)` - Load task (ReachTarget, PickAndLift, PushButton, PutRubbishInBin, StackBlocks)
- `reset_current_task()` - Reset for retry (if execution failed)
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
3. Present enhanced plan:
   - Task analysis with risk level
   - Motion sequence with WHY justifications
   - Adjustable parameters table
4. WAIT for user approval or parameter adjustments

HANDLING USER RESPONSES:
- If user approves ("approved", "yes", "proceed") → proceed to Phase 2
- If user requests parameter adjustment (e.g., "increase grasp offset to 0.02m"):
  a. Parse adjustment: extract parameter name and new value
  b. Validate: check if value is within allowed range
  c. Update plan: recalculate affected steps with new parameter
  d. Re-present updated plan
  e. WAIT for final approval (DO NOT execute without "approved")
```

PARAMETER DEFINITIONS BY TASK:
- **PutRubbishInBin**: approach_height (0.10-0.20m), grasp_offset (0.01-0.03m), bin_drop_height (0.05-0.15m)
- **PickAndLift**: approach_height (0.10-0.20m), grasp_offset (0.01-0.03m)
- **PushButton**: approach_height (0.05-0.15m), push_depth (0.001-0.005m)
- **ReachTarget**: approach_height (0.05-0.15m)
- **StackBlocks**: approach_height (0.10-0.20m), grasp_offset (0.01-0.03m), stack_offset (0.05-0.07m), stack_zone_xy ([0.0, 0.3] default)

RISK LEVEL CLASSIFICATION:
- LOW: ReachTarget, PushButton (simple, no grasping or minimal risk)
- MEDIUM: PickAndLift, PutRubbishInBin (grasping, controlled environment)
- HIGH: StackBlocks (multi-step, state tracking, precision stacking, 8 objects)
```

### Phase 2: Execution (after approval)
```
1. SENSING
   - load_task(task_name)
   - get_camera_observation() → save detection_prompt
   - get_target_position() → ground truth (for reference/validation only)

2. PERCEPTION
   - detect_object_3d(detection_prompt, paths...)
   - For PickAndLift: use "red cube . sphere" to detect both objects
   - For PutRubbishInBin: use "trash . bin" to detect both objects
   - Extract positions from objects[] array based on task type

3. MOTION
   - Execute exact sequence from MOTION SEQUENCES
   - Use detected positions (NOT ground truth)
   - Report each step
```

---

## CRITICAL TASK RULES

### PickAndLift - DETECT BOTH OBJECTS!
```
WRONG: Only lift vertically, ignore sphere XY
RIGHT: Move cube TO detected sphere's full XYZ position

Example:
  Cube detected at [0.064, 0.227, 0.773]
  Sphere detected at [0.207, 0.210, 0.996]

  Motion: Grasp cube → move_to_position(0.207, 0.210, 0.996)  # DIRECTLY to detected sphere!
```
- Detect BOTH cube AND sphere: detect_object_3d("red cube . sphere", ...)
- Extract both positions from objects[] array in response
- Move cube to detected sphere position (NOT ground truth!)
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
**CRITICAL:** Gripper MUST stay closed after grasping (step 4) until reaching bin release position (step 7). Do NOT open gripper during lift or transport.

### StackBlocks - DROP FROM ABOVE APPROACH!
```
- Detect ALL cubes: detect_object_3d("red cube", ...)
- LAB verification identifies green cube as stacking zone, red cubes as targets
- Extract X,Y from green cube ONLY: stack_x = green_x, stack_y = green_y
- IGNORE green_z completely - use fixed drop heights instead
- Filter red cubes: select 2 with highest confidence (ignore achromatic/low confidence)
- Execute 2 pick-drop cycles:
  * Cycle 1: Drop block from z=0.80 (5cm above table - gentle drop)
  * Cycle 2: Drop block from z=0.85 (5cm above first block)
- Must track blocks_stacked state (0 → 1 → 2)
- Gripper opens at drop height - gravity does the rest
- Ground truth available for reference/validation only
```
⚠️ **CRITICAL:**
- This is a LOOP task - repeat pick-drop sequence 2 times
- Use green cube for X,Y coordinates ONLY - NEVER use green_z
- Fixed drop heights: 0.80 for first block, 0.85 for second block (low and gentle)
- DO NOT try to place precisely - drop from low height and let gravity stack
- Track which blocks have been dropped (don't re-pick same block)

### ReachTarget vs PushButton
```
ReachTarget: Gripper stays OPEN (just touching)
PushButton: Gripper CLOSES FIRST (push with fist)
```

### PushButton - Task Completion Check!
```
CRITICAL: Check task_completed flag after contact step!
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
  1. load_task("ReachTarget")
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
  1. load_task("PickAndLift")
  2. get_camera_observation() → detection_prompt (for reference)
  3. get_target_position() → ground truth (for reference/validation only)
  4. detect_object_3d("red cube . sphere", ...) → objects[0]=cube, objects[1]=sphere
  5. Extract: cube=[0.064, 0.227, 0.773], sphere=[0.207, 0.210, 0.996]
  6. control_gripper("open")
  7. move_to_position(0.064, 0.227, 0.923)   # above cube (cube_z + 0.15)
  8. move_to_position(0.064, 0.227, 0.773)   # grasp (cube_z + 0.015)
  9. control_gripper("close")                # GRASP - keep closed!
  10. move_to_position(0.207, 0.210, 0.996)  # DIRECTLY to detected sphere XYZ!
  → SUCCESS (gripper stays closed, using detected positions)
```

### Example 3: PushButton
```
User: "Push the button"

Plan:
  Task: PushButton | Target: button | Gripper: CLOSE FIRST

Execution:
  1. load_task("PushButton")
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

### Example 4: PutRubbishInBin (ENHANCED HITL)
```
User: "Put the rubbish in the bin"

PHASE 1: PLANNING (Enhanced Approval)
---
## Task Analysis
- **Task Type:** PutRubbishInBin
- **Target Objects:** trash (crumpled paper), bin (basket)
- **Gripper Strategy:** Open → Close → Open (release)
- **Risk Level:** MEDIUM (grasping task in controlled environment)

## Motion Plan with Justifications

1. control_gripper("open")
   WHY: Prepare gripper for grasping, ensure fingers are clear

2. move_to_position(trash_x, trash_y, trash_z + 0.15)
   WHY: Safe approach height prevents collision with scene objects

3. move_to_position(trash_x, trash_y, trash_z + 0.015)
   WHY: Grasp height optimized for crumpled paper (1.5cm clearance avoids IK/collision)

4. control_gripper("close")
   WHY: Secure grasp on trash object
   CRITICAL: Gripper MUST remain closed until step 7

5. move_to_position(trash_x, trash_y, trash_z + 0.15)
   WHY: Lift trash clear of surrounding objects (gripper CLOSED)

6. move_to_position(bin_x, bin_y, bin_z + 0.10)
   WHY: Position 10cm above bin for safe release (gripper CLOSED)

7. control_gripper("open")
   WHY: Release trash into bin

8. move_to_position(bin_x, bin_y, bin_z + 0.15)
   WHY: Retract gripper clear of bin

## Adjustable Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Approach Height | 0.15m | [0.10-0.20]m | Height above objects for safe approach |
| Grasp Offset | 0.015m | [0.01-0.03]m | Clearance above trash for grasping |
| Bin Drop Height | 0.10m | [0.05-0.15]m | Release height above bin rim |

**To modify:** Reply with adjustment (e.g., "increase grasp offset to 0.02m")
**To approve:** Reply "approved" or "proceed"

NOTE: Object positions will be detected after approval.

**AWAITING APPROVAL**
---

User: "approved"

PHASE 2: EXECUTION
---
### Sensing
  1. load_task("PutRubbishInBin") → success
  2. get_camera_observation() → paths captured
  3. get_target_position() → ground truth for reference

### Perception
  4. detect_object_3d("trash . bin", ...) → 2 objects detected
  5. Extract positions:
     - trash: [0.428, 0.338, 0.757] (conf: 0.85)
     - bin: [0.439, 0.336, 0.834] (conf: 0.91)

### Motion (Using approved parameters: default values)
  6. control_gripper("open") → success
  7. move_to_position(0.428, 0.338, 0.907) → success (approach)
  8. move_to_position(0.428, 0.338, 0.772) → success (grasp height)
  9. control_gripper("close") → success (grasped)
  10. move_to_position(0.428, 0.338, 0.907) → success (lift)
  11. move_to_position(0.439, 0.336, 0.934) → success (to bin)
  12. control_gripper("open") → success (release)
  13. move_to_position(0.439, 0.336, 0.984) → success (retract)

## Result: SUCCESS
Task completed successfully with default parameters.
```

### Example 4b: PutRubbishInBin (With Parameter Adjustment)
```
User: "Put the rubbish in the bin"

[Agent presents enhanced plan with default parameters]

User: "increase grasp offset to 0.02m"

Agent: "Updating parameters:
- Grasp Offset: 0.015m → 0.02m ✓ (within range [0.01-0.03]m)

Updated Step 3:
3. move_to_position(trash_x, trash_y, trash_z + 0.02)
   WHY: Grasp height with increased clearance (2cm) for safer approach

All other steps remain unchanged. Please confirm: Reply 'approved' to proceed."

User: "approved"

[Agent executes with grasp_offset = 0.02m]
  → Step 8 uses: move_to_position(0.428, 0.338, 0.777)  # trash_z + 0.02
```

---

## RESPONSE FORMAT

**Planning Phase (BEFORE Perception):**
```markdown
## Task Analysis
- **Task Type:** [type]
- **Target Objects:** [what will be detected]
- **Gripper Strategy:** [Open/Close sequence]
- **Risk Level:** [LOW/MEDIUM/HIGH based on task complexity]

## Motion Plan with Justifications

Follow the exact motion sequence from MOTION SEQUENCES section, adding WHY for each step:

Example for PutRubbishInBin:
1. control_gripper("open")
   WHY: Prepare gripper for grasping, ensure fingers are clear

2. move_to_position(trash_x, trash_y, trash_z + 0.15)
   WHY: Safe approach height prevents collision with scene objects

3. move_to_position(trash_x, trash_y, trash_z + 0.015)
   WHY: Grasp height optimized (1.5cm clearance avoids IK/collision)

4. control_gripper("close")
   WHY: Secure grasp on trash object
   CRITICAL: Gripper MUST remain closed until step 7

5. move_to_position(trash_x, trash_y, trash_z + 0.15)
   WHY: Lift trash clear of surrounding objects (gripper CLOSED)

6. move_to_position(bin_x, bin_y, bin_z + 0.10)
   WHY: Position 10cm above bin for safe release (gripper CLOSED)

7. control_gripper("open")
   WHY: Release trash into bin

8. move_to_position(bin_x, bin_y, bin_z + 0.15)
   WHY: Retract gripper clear of bin

## Adjustable Parameters

Show task-specific parameters that can be modified:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Approach Height | 0.15m | [0.10-0.20]m | Height above objects for safe approach |
| Grasp Offset | 0.015m | [0.01-0.03]m | Clearance above object (for trash/paper) |
| Bin Drop Height | 0.10m | [0.05-0.15]m | Release height above bin rim |

**To modify parameters:** Reply with adjustment, e.g., "increase grasp offset to 0.02m"
**To approve as-is:** Reply "approved" or "proceed"

NOTE: Actual object positions will be determined by perception after approval.

---
**AWAITING APPROVAL** - Reply "approved" to proceed, or suggest parameter adjustments.
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
    print("  - StackBlocks: 'Stack 2 red blocks'")
    print("\nAgents:")
    print(f"  1. {root_agent.name} (planning + orchestration with human-in-the-loop)")
    print(f"  2. {sensing_agent.name}")
    print(f"  3. {perception_agent.name}")
    print(f"  4. {motion_agent.name}")
    print("\nUsage:")
    print("  - Import root_agent for ADK web interface")
    print("  - Import sequential_pipeline for automated execution")
    print("=" * 60)
