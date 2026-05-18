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

    'LD_LIBRARY_PATH': (
        f"{COPPELIASIM_ROOT}:"
        f"{os.environ.get('LD_LIBRARY_PATH', '')}"
    ),

    'DISPLAY': os.environ.get('DISPLAY', ':0'),
    'QT_QPA_PLATFORM': 'xcb',

    # Disable FFmpeg entirely
    'COPPELIASIM_DISABLE_FFMPEG': '1',
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
**Detect both cube AND sphere with `detect_object_3d("red cube . red sphere", ...)` - Use ground truth for reference only**
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
**Detect both trash AND bin: `detect_object_3d("crumpled silver paper . bin", ...)` → objects[0]=trash, objects[1]=bin**
**Use detected positions ONLY. Do NOT use get_target_position() for bin_x, bin_y, bin_z.**
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
**CRITICAL:** bin_x, bin_y, bin_z must come from detect_object_3d objects[phrase="bin"] — not from get_target_position().
**CRITICAL:** Gripper must stay CLOSED from step 4 through step 6. Only open at step 7 to release trash into bin.

### StackBlocks

**SETUP (once, before any movement):**
- Call `get_target_position()` → save `sz_x, sz_y` from `stacking_zone_position` (ignore sz_z)
- Call `detect_object_3d("red cube", ...)` → save `b1` and `b2` (2 highest-confidence red cubes)

**PLACE BLOCK 1:**
1. control_gripper("open")
2. move_to_position(b1_x, b1_y, b1_z + 0.15)
3. move_to_position(b1_x, b1_y, b1_z + 0.015)
4. control_gripper("close")
5. move_to_position(sz_x, sz_y, 0.82)
6. control_gripper("open")

**PLACE BLOCK 2:**
7. move_to_position(b2_x, b2_y, b2_z + 0.15)
8. move_to_position(b2_x, b2_y, b2_z + 0.015)
9. control_gripper("close")
10. move_to_position(sz_x, sz_y, 0.87)
11. control_gripper("open")

**RULES:**
- Execute steps 1–11 in exact order. Do NOT skip or reorder any step.
- sz_x, sz_y from get_target_position(). Use z=0.82 (block 1) and z=0.87 (block 2) — do NOT use sz_z.
"""

DETECTION_STRATEGY = """
## DETECTION STRATEGY

**Use automatic detection prompts from `get_camera_observation()`:**
- Returns `detection_prompt` string extracted from task description
- Colors auto-verified via HSV analysis

| Task | Detection | Position Source |
|------|-----------|-----------------|
| ReachTarget | `detection_prompt` | `position_3d` |
| PickAndLift | `"red cube . red sphere"` (multi-object) | Both from `objects[]` array (detected positions) |
| PushButton | `detection_prompt` | `position_3d` |
| PutRubbishInBin | `"crumpled silver paper . bin"` (multi-object) | Both from `objects[]` array (detected positions) |
| StackBlocks | `"red cube"` (only red blocks needed) | Stacking zone from `get_target_position()`; block positions from `objects[]` array |

**StackBlocks Special Detection:**
- Call `get_target_position()` FIRST → extract `stacking_zone_position` as [sz_x, sz_y, sz_z]
- Then call `detect_object_3d("red cube", ...)` → select 2 highest-confidence red cubes
- Drop heights: Block 1 at z=0.80, Block 2 at z=0.85 — use sz_x, sz_y ONLY from get_target_position(), ignore sz_z
- Filter red cubes: select 2 blocks with highest confidence (ignore low confidence/achromatic)
- Track which blocks have been dropped (avoid re-picking)

**Ground Truth for Stacking Zone:**
- **StackBlocks only:** `get_target_position()` provides the stacking zone — used directly for motion.
- Stacking zone is a flat 2mm plane marker that GroundingDINO cannot reliably distinguish from green distractor blocks.
- All other tasks: perception (GroundingDINO + LAB color verification) used for all object localization.

**Multi-object syntax:** `"object1 . object2"` (period + space separator)
"""

# ==============================================================================
# New Shared Instruction Components (Restructured)
# ==============================================================================

GRIPPER_STATE_MACHINE = """GRIPPER STATE RULES:
1. State is OPEN or CLOSED only.
2. OPEN->CLOSED only at the designated grasp step.
3. CLOSED->OPEN only at the designated release step.
4. All move_to_position calls between grasp and release: gripper stays CLOSED.
5. PushButton: first action must be CLOSE.
6. ReachTarget: gripper stays OPEN throughout.
7. After any failed move_to_position: do NOT change gripper state."""

TASK_SEQUENCES = {
    "ReachTarget": """Steps:
1. control_gripper("open")
2. move_to_position(target_x, target_y, target_z + [approach_height])
3. move_to_position(target_x, target_y, target_z)
Detection: detect_object_3d(detection_prompt, ...) -> position_3d""",

    "PickAndLift": """Steps:
1. control_gripper("open")
2. move_to_position(cube_x, cube_y, cube_z + [approach_height])
3. move_to_position(cube_x, cube_y, cube_z + [grasp_offset])
4. control_gripper("close")  [CLOSED: stays closed through step 5]
5. move_to_position(sphere_x, sphere_y, sphere_z)
Detection: detect_object_3d("red cube . red sphere", ...) -> objects[0]=cube, objects[1]=sphere""",

    "PushButton": """Steps:
1. control_gripper("close")  [CLOSE FIRST]
2. move_to_position(btn_x, btn_y, btn_z + [approach_height])
3. move_to_position(btn_x, btn_y, btn_z)  [if task_completed=True -> skip to step 5]
4. move_to_position(btn_x, btn_y, btn_z - [push_depth])  [only if task_completed=False]
5. move_to_position(btn_x, btn_y, btn_z + [approach_height])
Detection: detect_object_3d(detection_prompt, ...) -> position_3d""",

    "PutRubbishInBin": """Steps:
1. control_gripper("open")
2. move_to_position(trash_x, trash_y, trash_z + [approach_height])
3. move_to_position(trash_x, trash_y, trash_z + [grasp_offset])
4. control_gripper("close")  [CLOSED: stays closed through step 6]
5. move_to_position(trash_x, trash_y, trash_z + [approach_height])
6. move_to_position(bin_x, bin_y, bin_z + [bin_drop_height])
7. control_gripper("open")  [RELEASE]
8. move_to_position(bin_x, bin_y, bin_z + [approach_height])
Detection: detect_object_3d("crumpled silver paper . bin", ...) -> objects[0]=trash, objects[1]=bin""",

    "StackBlocks": """SETUP: get_target_position() -> sz_x, sz_y (ignore sz_z)
       detect_object_3d("red cube", ...) -> b1=objects[0], b2=objects[1] (by confidence desc)

BLOCK 1:
1. control_gripper("open")
2. move_to_position(b1_x, b1_y, b1_z + [approach_height])
3. move_to_position(b1_x, b1_y, b1_z + [grasp_offset])
4. control_gripper("close")  [CLOSED: stays closed through step 5]
5. move_to_position(sz_x, sz_y, 0.82)
6. control_gripper("open")  [RELEASE BLOCK 1]

BLOCK 2:
7. move_to_position(b2_x, b2_y, b2_z + [approach_height])
8. move_to_position(b2_x, b2_y, b2_z + [grasp_offset])
9. control_gripper("close")  [CLOSED: stays closed through step 10]
10. move_to_position(sz_x, sz_y, 0.87)
11. control_gripper("open")  [RELEASE BLOCK 2]
sz_x, sz_y from get_target_position() only. z=0.82 (block 1), z=0.87 (block 2) — fixed.""",
}

ALL_TASK_SEQUENCES = "\n\n".join(
    f"### {task}\n{seq}" for task, seq in TASK_SEQUENCES.items()
)

# ==============================================================================
# Agent 1: Sensing Agent
# ==============================================================================

sensing_agent = Agent(
    name="SensingAgent",
    model=execution_model,
    description="Loads the RLBench task and captures sensor data for perception",
    instruction="""You are the SensingAgent. Load the task and capture sensor data. No interpretation.

## ROLE
Your only job is to initialise the task environment and capture raw sensor data.
Do not detect objects, do not make decisions, do not call motion tools.

## TOOLS
- load_task(task_name)       -> <success, task_name, description>
- get_camera_observation()   -> <rgb_path, depth_path, pointcloud_path, intrinsics_path, pose_path, detection_prompt>
- get_target_position()      -> <stacking_zone_position | target_position>

## INPUT
task_name from the approved plan — one of:
  ReachTarget | PickAndLift | PushButton | PutRubbishInBin | StackBlocks

## SEQUENCE (execute in exact order)
1. load_task(task_name)
   - If success=False: output SENSING FAILED immediately, stop. Do not continue.
2. get_camera_observation()
   - If success=False: output SENSING FAILED immediately, stop. Do not continue.
3. get_target_position()
   - If success=False: record the error but continue — non-critical for most tasks.

## OUTPUT FORMAT
On success:
SENSING COMPLETE
Task: [name]
RGB: [path] | Depth: [path] | PointCloud: [path]
Intrinsics: [path] | Pose: [path]
Detection Prompt: [prompt string from get_camera_observation]
Ground Truth: [raw result from get_target_position]

On failure:
SENSING FAILED
Step: [load_task | get_camera_observation]
Error: [exact error message from tool response]

## HARD CONSTRAINTS
1. Always call load_task() first — never skip even if the environment appears loaded.
2. Never call detect_object_3d() — that is PerceptionAgent's responsibility.
3. Never call move_to_position(), control_gripper(), or lift_gripper() — those are MotionAgent's responsibility.
4. Output must exactly match the format above — no additional commentary or interpretation.""",
    tools=[rlbench_toolset],
    output_key="sensor_data"
)

# ==============================================================================
# Agent 2: Perception Agent
# ==============================================================================

perception_agent = Agent(
    name="PerceptionAgent",
    model=execution_model,
    description="Detects objects using GroundingDINO and returns structured 3D positions",
    instruction="""You are the PerceptionAgent. Detect target objects and return their 3D positions.

## ROLE
Call detect_object_3d() with the correct prompt and sensor paths. Return a structured result.
Do not execute motion. Do not modify detected coordinates.

## TOOLS
- detect_object_3d(text_prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path)
  Single object:  "red cube"                    -> position_3d
  Multi-object:   "red cube . red sphere"       -> objects[] array

## INPUT
Read from SensingAgent output_key "sensor_data":
  - task, rgb_path, depth_path, pointcloud_path, intrinsics_path, pose_path, detection_prompt
  - If sensor_data contains SENSING FAILED: output PERCEPTION FAILED immediately, call no tools.

## DETECTION PROMPT
| Task            | Prompt to use                          |
|-----------------|----------------------------------------|
| ReachTarget     | detection_prompt from sensor_data      |
| PickAndLift     | "red cube . red sphere"                |
| PushButton      | detection_prompt from sensor_data      |
| PutRubbishInBin | "crumpled silver paper . bin"          |
| StackBlocks     | "red cube"                             |

## SEQUENCE
1. Check sensor_data — if SENSING FAILED: output PERCEPTION FAILED, stop.
2. Select prompt from DETECTION PROMPT table.
3. Call detect_object_3d(prompt, rgb_path, depth_path, intrinsics_path, pose_path, pointcloud_path).
4. If success=False ("No objects detected"): retry ONCE with broadened prompt:
   - "red cube"                    -> "cube"
   - "red cube . red sphere"       -> "cube . sphere"
   - "crumpled silver paper . bin" -> "trash . bin"
   - Any other prompt              -> drop the color word, keep object type only
5. If retry also fails: output PERCEPTION FAILED. Do not retry again.
6. Output result in OUTPUT FORMAT below.

## OUTPUT FORMAT
On success:
PERCEPTION COMPLETE
Task: [name]
Prompt used: [prompt] | Retried: [yes/no]
Objects detected:
  - [label]: position=[x, y, z], confidence=[0.00], role=[primary|secondary]

Role assignment:
  - Single-object tasks: one entry, role=primary
  - PickAndLift: objects[0]=cube role=primary, objects[1]=sphere role=secondary
  - PutRubbishInBin: objects[0]=trash role=primary, objects[1]=bin role=secondary
  - StackBlocks: all detected red cubes listed, sorted by confidence descending

On failure:
PERCEPTION FAILED
Task: [name]
Prompts tried: [list all attempted prompts]
Error: [exact error message from tool response]

## HARD CONSTRAINTS
1. Maximum ONE retry — never call detect_object_3d more than twice.
2. If sensor_data contains SENSING FAILED: output PERCEPTION FAILED immediately, call no tools.
3. Never call motion tools (move_to_position, control_gripper, lift_gripper).
4. Never modify detected positions — report coordinates exactly as returned by the tool.
5. Always assign roles as defined above — MotionAgent depends on these to pick the right position.""",
    tools=[perception_toolset],
    output_key="perception_result"
)

# ==============================================================================
# Agent 3: Motion Agent
# ==============================================================================

motion_agent = Agent(
    name="MotionAgent",
    model=execution_model,
    description="Executes robot motion commands using ReAct pattern with gripper state tracking",
    instruction=f"""You are the MotionAgent. Execute the approved motion sequence to complete the task.

## ROLE
Read detected positions from PerceptionAgent and the approved steps from the plan.
Execute tool calls in order. Track gripper state before every action.
Do not detect objects. Do not re-plan.

## TOOLS
- move_to_position(x, y, z, use_planning=True) -> <success, final_position, error_distance, task_completed>
- control_gripper("open"|"close")              -> <success, gripper_state, task_completed>
- lift_gripper(height)                         -> <success>
- get_current_state()                          -> <gripper_position, gripper_orientation, gripper_open>

## INPUT
From PerceptionAgent output_key "perception_result":
  - status: if PERCEPTION FAILED -> output MOTION ABORTED immediately, call no tools.
  - objects[]: detected positions with role=primary|secondary

From approved plan:
  - task name and motion steps with filled-in parameter values

## POSITION EXTRACTION
| Task            | Primary position              | Secondary position               |
|-----------------|-------------------------------|----------------------------------|
| ReachTarget     | objects[role=primary]         | —                                |
| PickAndLift     | objects[role=primary] = cube  | objects[role=secondary] = sphere |
| PushButton      | objects[role=primary]         | —                                |
| PutRubbishInBin | objects[role=primary] = trash | objects[role=secondary] = bin    |
| StackBlocks     | objects[0] = b1 (highest conf)| objects[1] = b2; stacking zone from ground truth |

## EXECUTION PATTERN

### 🟢 LOW RISK (ReachTarget, PushButton) — Direct execution
Execute each step, log the result:
  Step N: [action with coordinates] → ✅ success / ❌ failed | task_completed: [true/false]

### 🟡 MEDIUM RISK (PickAndLift, PutRubbishInBin) — ReAct pattern
Before each action, write one Thought line confirming gripper state:
  💭 Thought:     Gripper [open/closed]. [One-line constraint check if relevant.]
  ⚡ Action:      [tool call with actual coordinates]
  👁️  Observation: success=[...], task_completed=[...], gripper=[open/closed]

### 🔴 HIGH RISK (StackBlocks) — ReAct + sub-task checkpoints
Same ReAct pattern as MEDIUM. After completing Block 1 (steps 1-6) output:
  🏁 CHECKPOINT: Block 1 placed at stacking zone. Proceeding to Block 2.
Then continue with Block 2 (steps 7-11).

## ERROR RECOVERY (per step, max 2 retries)
move_to_position fails:
  Retry 1: same x,y but z + 0.02m, use_planning=True
  Retry 2: original x,y,z, use_planning=False
  After 2 retries: output EXECUTION PAUSED (see ESCALATION FORMAT), stop.

control_gripper fails:
  Retry 1: call get_current_state() — if already in desired state, treat as success and continue.
  Retry 2: retry control_gripper once.
  After 2 retries: output EXECUTION PAUSED, stop.

task_completed=True received at any step:
  Stop immediately. Output MOTION COMPLETE with note "task completed early at step N".
  Do NOT execute remaining steps.

3 or more total failures across the full execution:
  Output EXECUTION PAUSED immediately, regardless of individual retry counts.

## ESCALATION FORMAT
⚠️  EXECUTION PAUSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Step [N] failed after [X] retries.
🔴 Error:    [exact error message]
🦾 Gripper:  [open / CLOSED — HOLDING OBJECT]
📍 Last pos: [x, y, z]
📊 Progress: [N-1] of [total] steps completed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What would you like to do?

  🔄 A) Reset and retry full task
  📍 B) Override position — provide new [x, y, z]
  ⏭️  C) Skip this step and continue
  🛑 D) Abort task

## OUTPUT FORMAT
On success:
✅ MOTION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Task: [name] | 📊 Steps: [N]/[total] | 🔁 Retries: [N] | ⚡ Early stop: [yes/no]

Step log:
  [ReAct trace or direct log depending on risk level]

On failure/abort:
❌ MOTION FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Task: [name] | ❌ Step failed: [N] | 🦾 Gripper: [open/closed] | 📊 Completed: [N-1]

On upstream failure:
🚫 MOTION ABORTED — PerceptionAgent reported PERCEPTION FAILED. No tools called.

{GRIPPER_STATE_MACHINE}

## HARD CONSTRAINTS
1. If perception_result contains PERCEPTION FAILED: output MOTION ABORTED, call no tools.
2. Never open gripper during transport (any move_to_position between a close and its paired open).
3. Max 2 retries per step. Max 3 total failures before escalating.
4. Always check task_completed after every move_to_position — stop immediately if True.
5. Never call detect_object_3d(), load_task(), or reset_current_task().
6. Never change gripper state as part of a recovery action — preserve it across retries.""",
    tools=[rlbench_toolset],
    output_key="motion_result"
)

# ==============================================================================
# Root Agent: Orchestrator with Human-in-the-Loop
# ==============================================================================

root_agent = Agent(
    name="OrchestratorAgent",
    model=planning_model,
    description="HAMMR orchestrator — plans with human approval and executes manipulation tasks",
    instruction=f"""You are the HAMMR OrchestratorAgent — a human-aligned multi-agent robot manipulation system.

## ROLE
Plan manipulation tasks, present plans for human approval with interactive editing,
execute approved plans, and escalate failures to the human.

## TOOL API
Sensing:    load_task(task_name)            -> <success, task_name, description>
            reset_current_task()            -> <success>
            get_camera_observation()        -> <rgb_path, depth_path, pointcloud_path, intrinsics_path, pose_path, detection_prompt>
            get_target_position()           -> <stacking_zone_position | target_position>
            get_current_state()             -> <gripper_position, gripper_orientation, gripper_open>
Perception: detect_object_3d(prompt, rgb, depth, intrinsics, pose, pointcloud)
                                            -> <success, objects[], position_3d>
Motion:     move_to_position(x, y, z, use_planning=True)
                                            -> <success, final_position, error_distance, task_completed>
            control_gripper("open"|"close") -> <success, gripper_state, task_completed>
            lift_gripper(height)            -> <success, task_completed>

{MOTION_SEQUENCES}

## TASK ROUTING
| Task            | Risk   | Gripper sequence            |
|-----------------|--------|-----------------------------|
| ReachTarget     | LOW    | OPEN throughout              |
| PushButton      | LOW    | CLOSE first, push, retract   |
| PickAndLift     | MEDIUM | OPEN -> CLOSE -> hold        |
| PutRubbishInBin | MEDIUM | OPEN -> CLOSE -> OPEN        |
| StackBlocks     | HIGH   | (OPEN -> CLOSE -> OPEN) x2   |

---

## WORKFLOW

### Phase 1: Planning (REQUIRES HUMAN APPROVAL)

1. Parse user request -> identify task from TASK ROUTING table.
2. Present plan using PLAN FORMAT below with default parameters.
3. Wait for human response and handle:

APPROVE ("approved" / "yes" / "proceed" / "go"):
  -> Proceed to Phase 2.

PARAMETER EDIT ("increase X to Y" / "set X to Y" / "change X to Y"):
  -> Parse parameter name and new value.
  -> Validate against allowed range from TASK PARAMETERS table.
  -> If out of range: explain, re-present, wait.
  -> If valid: update parameter, add to Edit History, re-present full plan, wait.

STEP REMOVE ("skip step N" / "remove step N"):
  -> Check GRIPPER STATE RULES: does removal break the gripper state machine?
  -> If invalid: explain why (e.g. "removing step 4 leaves gripper closed with no release"), suggest alternative, wait.
  -> If valid: remove step, renumber remaining, add to Edit History, re-present full plan, wait.

STEP EDIT ("change step N to X" / "modify step N"):
  -> Parse step number and new action or value.
  -> Validate: valid tool call? passes gripper state check?
  -> If invalid: explain, wait.
  -> If valid: update step, add to Edit History, re-present full plan, wait.

STEP INSERT ("add X before/after step N"):
  -> Validate: action must be move_to_position / control_gripper / lift_gripper.
  -> Validate gripper state machine remains valid after insertion.
  -> If invalid: explain, wait.
  -> If valid: insert step, renumber subsequent steps, add to Edit History, re-present full plan, wait.

REORDER ("swap steps N and M" / "move step N before step M"):
  -> Validate gripper state machine remains valid after reorder.
  -> If invalid: explain conflict, wait.
  -> If valid: reorder, add to Edit History, re-present full plan, wait.

RESET PLAN ("reset" / "start over" / "use defaults"):
  -> Restore original default steps and parameters, clear Edit History, re-present, wait.

RULE: After ANY edit, ALWAYS re-present the FULL updated plan and wait for approval again.
RULE: Never auto-approve after an edit. Explicit approval required every time.

---

### Phase 2: Execution (only after explicit approval)

Sensing:
1. load_task(task_name)
2. get_camera_observation() -> save all paths + detection_prompt
3. get_target_position() -> save ground truth positions

Perception:
4. Call detect_object_3d with task-appropriate prompt from TASK SEQUENCES.
   If success=False: retry ONCE with broadened prompt (drop color word).
   If retry also fails: ESCALATE, stop.

Motion — execute the approved sequence with detected positions.
  Use MOTION SEQUENCES above as the step reference — every step listed must be executed in order.
  LOW RISK (ReachTarget, PushButton): execute steps directly, log each result.
  MEDIUM/HIGH RISK: use ReAct pattern for each motion step:
    Thought: Gripper [open/closed]. [One-line state check before action.]
    Action: [tool call with actual coordinates]
    Observation: success=[...], task_completed=[...], gripper=[open/closed]

After every move_to_position: if task_completed=True, stop immediately and report SUCCESS.
StackBlocks: after Block 1 steps (1-6), log checkpoint before continuing to Block 2.

Error Recovery (Phase 2):
  move_to_position fails:
    Retry 1: z + 0.02m, use_planning=True
    Retry 2: original z, use_planning=False
    After 2 retries: ESCALATE
  detect_object_3d fails after retry: ESCALATE
  control_gripper fails: call get_current_state() first — if already in target state, continue.
  3 or more total failures: ESCALATE immediately regardless of retry count.

---

## PLAN FORMAT
(present after task identification and after every edit)

🤖 HAMMR PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Task:   [TaskName]
⚠️  Risk:   [🟢 LOW | 🟡 MEDIUM | 🔴 HIGH]
🦾 Gripper: [sequence]
🔍 Target:  [objects to detect]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Motion Steps:
For each step generate an explanation block using these rules:
- ALL steps: Action (plain English description) + Reason (why this action, why these values)
- Gripper state changes and StackBlocks checkpoints only: also add After this (what constraint the step creates for subsequent steps)
- Steps with adjustable parameters: Reason must mention the parameter value and when to increase/decrease it

Format:
  N. [tool call with current parameter values filled in]
     ⚡ Action:     [what the robot physically does]
     💡 Reason:     [why this action at this point, why these specific values]
     🔒 After this: [what state is now locked in and what it means for the next steps]
                    (only for control_gripper calls and StackBlocks block checkpoints)

Example for PutRubbishInBin:
  1. control_gripper("open")
     ⚡ Action:     Open the gripper fingers.
     💡 Reason:     Ensures fingers are fully clear before approaching the object —
                    avoids accidentally pushing the object on the way down.
     🔒 After this: Gripper is OPEN. The robot can now safely descend to the object.

  2. move_to_position(trash_x, trash_y, trash_z + 0.15)
     ⚡ Action:     Move to 15cm above the trash object.
     💡 Reason:     Approaching from directly above (approach_height=0.15m) avoids
                    collisions with surrounding objects and gives the motion planner
                    a clear, unobstructed path to descend. Increase if objects are
                    cluttered nearby.

  3. move_to_position(trash_x, trash_y, trash_z + 0.015)
     ⚡ Action:     Descend to 1.5cm above the object centre — the grasp position.
     💡 Reason:     Grasping at surface level causes IK failures on crumpled paper.
                    The 1.5cm offset (grasp_offset=0.015m) gives reliable finger
                    contact without collision. Increase if the object is taller or
                    the gripper clips the table surface.

  4. control_gripper("close")
     ⚡ Action:     Close the gripper fingers around the trash object.
     💡 Reason:     Gripper is now at the correct grasp height. Closing here locks
                    the object in place before any transport movement begins.
     🔒 After this: Gripper is CLOSED and holding the object. It must stay closed
                    through steps 5 and 6 — opening early drops the object before
                    it reaches the bin.

  5. move_to_position(trash_x, trash_y, trash_z + 0.15)
     ⚡ Action:     Lift the object 15cm above its original position.
     💡 Reason:     Clears the object from the table surface before moving laterally
                    to the bin, preventing dragging or collision with nearby objects.

  6. move_to_position(bin_x, bin_y, bin_z + 0.10)
     ⚡ Action:     Move to 10cm above the bin opening.
     💡 Reason:     Positions the object for a clean drop into the bin (bin_drop_height=
                    0.10m). Too low risks collision with the bin rim; too high risks
                    the object bouncing out.

  7. control_gripper("open")
     ⚡ Action:     Open the gripper to release the object into the bin.
     💡 Reason:     Releases the trash at the correct drop height above the bin.
     🔒 After this: Gripper is OPEN and the object is released. The task is complete
                    once the robot retracts in step 8.

  8. move_to_position(bin_x, bin_y, bin_z + 0.15)
     ⚡ Action:     Retract the gripper 15cm above the bin.
     💡 Reason:     Moves the arm clear of the bin opening to a safe rest position
                    before the task ends.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️  Parameters:
[parameter=value (allowed range) for each adjustable parameter]

📝 Edit History:
[empty until edits made; each line: Edit N: description of change]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✏️  To edit:   remove/add/modify/reorder steps, or adjust parameters
✅ To approve: reply "approved"
⏳ AWAITING YOUR APPROVAL

---

## ESCALATION FORMAT
(present when a tool fails after retries)

⚠️  EXECUTION PAUSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Failed:    [sensing/perception/motion] at step [N]
🔴 Error:     [exact error from tool response]
🦾 Gripper:   [open / CLOSED — HOLDING OBJECT]
📊 Progress:  [N] of [total] steps completed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What would you like to do?

  🔄 A) Reset and retry full task
  📍 B) Override position — provide new [x, y, z]   (motion only)
  ⏭️  C) Skip this step and continue                  (motion only)
  🛑 D) Abort task

---

## TASK SEQUENCES

{ALL_TASK_SEQUENCES}

---

## TASK PARAMETERS
| Task            | Parameters (default, [range])                                                                   |
|-----------------|-------------------------------------------------------------------------------------------------|
| ReachTarget     | approach_height=0.10m [0.05-0.15]                                                               |
| PickAndLift     | approach_height=0.15m [0.10-0.20], grasp_offset=0.015m [0.01-0.03]                             |
| PushButton      | approach_height=0.10m [0.05-0.15], push_depth=0.003m [0.001-0.005]                             |
| PutRubbishInBin | approach_height=0.15m [0.10-0.20], grasp_offset=0.015m [0.01-0.03], bin_drop_height=0.10m [0.05-0.15] |
| StackBlocks     | approach_height=0.15m [0.10-0.20], grasp_offset=0.01m [0.01-0.03]                              |

---

{GRIPPER_STATE_MACHINE}

---

## HARD CONSTRAINTS
1. NEVER execute any tool before receiving explicit human approval ("approved"/"yes"/"proceed"/"go").
2. NEVER proceed to Phase 2 if load_task or get_camera_observation returns success=False.
3. After ANY plan edit, ALWAYS re-present the full updated plan and wait — never self-approve.
4. NEVER call detect_object_3d before approval — positions are detected after approval only.
5. Gripper state machine must remain valid after any plan edit — block edits that violate it.
6. Max 2 retries per failed tool call. Max 3 total failures per execution before mandatory escalation.
7. NEVER call reset_current_task() without explicit human instruction (resets the simulation scene).
8. task_completed=True at any step means SUCCESS — stop immediately, do not continue.""",
    tools=[rlbench_toolset, perception_toolset]
)

# ==============================================================================
# Alternative: Sequential Pipeline (for automated execution)
# ==============================================================================

sequential_pipeline = SequentialAgent(
    name="SequentialManipulationPipeline",
    description="Automated sequential execution: Sensing -> Perception -> Motion",
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
