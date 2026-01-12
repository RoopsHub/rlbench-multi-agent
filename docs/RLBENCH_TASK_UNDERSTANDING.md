# RLBench Task Implementation Guide

This document provides a comprehensive guide to understanding and replicating the multi-agent framework's RLBench task implementations.

---

## Overview

The framework implements 5 RLBench tasks using a multi-agent orchestration approach with open-vocabulary object detection. All tasks follow a consistent pattern: **Planning → Approval → Sensing → Perception → Motion**.

---

## Implemented Tasks

### Task Summary

| Task | Complexity | Objects | Gripper Sequence | Success Criteria |
|------|-----------|---------|------------------|------------------|
| ReachTarget | LOW | 1 sphere | Open (constant) | Touch target sphere |
| PushButton | LOW | 1 button | Open (constant) | Press button |
| PickAndLift | MEDIUM | 2 (cube + sphere) | Open→Close→Closed | Grasp cube, lift to sphere |
| PutRubbishInBin | MEDIUM | 2 (trash + bin) | Open→Close→Open | Place trash in bin |
| StackBlocks | HIGH | 8 cubes | Multiple cycles | Stack 2 red cubes on green |

---

## Task Specifications

### 1. ReachTarget

**Objective:** Move end-effector to touch a colored target sphere

**RLBench Source:** `rlbench/tasks/reach_target.py`

**Scene Setup:**
- 1 target sphere (changes color per variation)
- 2 distractor spheres (different colors)
- Random spawn positions within boundaries
- Minimum object separation: 0.2m

**Detection Strategy:**
- Prompt: Single object from task description (e.g., "red sphere")
- Use `detection_prompt` from camera observation

**Motion Sequence:**
```python
1. control_gripper("open")  # Gripper stays open
2. move_to_position(target_x, target_y, target_z + 0.10)  # Approach
3. move_to_position(target_x, target_y, target_z)  # Touch target
```

**Success Condition:** Proximity sensor detects gripper tip touching target sphere

**Key Parameters:**
- Approach height: 0.10m [0.05-0.15m]

---

### 2. PickAndLift

**Objective:** Pick up colored cube and lift it to sphere position

**RLBench Source:** `rlbench/tasks/pick_and_lift.py`

**Scene Setup:**
- 1 target cube (red)
- 1 target sphere (red) - lift destination
- Random spawn positions

**Detection Strategy:**
- Prompt: "red cube . red sphere" (multi-object detection)
- Detects both objects in single call
- Extract positions from `objects[]` array

**Motion Sequence:**
```python
1. control_gripper("open")
2. move_to_position(cube_x, cube_y, cube_z + 0.15)  # Approach above cube
3. move_to_position(cube_x, cube_y, cube_z + 0.015)  # Grasp height
4. control_gripper("close")  # Grasp cube
5. move_to_position(sphere_x, sphere_y, sphere_z)  # Lift to sphere
```

**Success Condition:**
- Cube is grasped AND
- Cube is lifted to sphere position

**Key Parameters:**
- Approach height: 0.15m [0.10-0.20m]
- Grasp offset: 0.015m [0.01-0.03m]

**Critical Note:** Use detected sphere position, NOT ground truth

---

### 3. PushButton

**Objective:** Push button with end-effector

**RLBench Source:** `rlbench/tasks/push_button.py`

**Scene Setup:**
- 1 button to push
- Random button position and orientation

**Detection Strategy:**
- Prompt: "button" from task description
- Single object detection

**Motion Sequence:**
```python
1. control_gripper("open")  # Gripper stays open
2. move_to_position(button_x, button_y, button_z + 0.10)  # Approach
3. move_to_position(button_x, button_y, button_z - 0.002)  # Push down
```

**Success Condition:** Button is pressed (pushed down)

**Key Parameters:**
- Approach height: 0.10m [0.05-0.15m]
- Push depth: 0.002m [0.001-0.005m]

---

### 4. PutRubbishInBin

**Objective:** Pick up trash and place it in bin

**RLBench Source:** `rlbench/tasks/put_rubbish_in_bin.py`

**Scene Setup:**
- 1 crumpled paper (trash)
- 1 bin (basket)
- Random positions

**Detection Strategy:**
- Prompt: "trash . bin" (multi-object detection)
- Detects both objects simultaneously

**Motion Sequence:**
```python
1. control_gripper("open")
2. move_to_position(trash_x, trash_y, trash_z + 0.15)  # Approach trash
3. move_to_position(trash_x, trash_y, trash_z + 0.015)  # Grasp height
4. control_gripper("close")  # Grasp trash
5. move_to_position(trash_x, trash_y, trash_z + 0.15)  # Lift trash
6. move_to_position(bin_x, bin_y, bin_z + 0.15)  # Move to bin
7. move_to_position(bin_x, bin_y, bin_z + 0.10)  # Position over bin
8. control_gripper("open")  # Release trash
9. move_to_position(bin_x, bin_y, bin_z + 0.15)  # Retract
```

**Success Condition:** Trash is inside bin

**Key Parameters:**
- Approach height: 0.15m [0.10-0.20m]
- Grasp offset: 0.015m [0.01-0.03m]
- Bin drop height: 0.10m [0.05-0.15m]

---

### 5. StackBlocks

**Objective:** Stack 2 red cubes on green cube (stacking zone)

**RLBench Source:** `rlbench/tasks/stack_blocks.py`

**Scene Setup:**
- 1 green cube (stacking zone)
- 2 red cubes (to be stacked)
- 5 additional cubes (various colors)
- Total: 8 objects

**Detection Strategy:**
- Prompt: "red cube" (detects all cubes including green)
- Filter: Select 2 highest confidence red cubes
- Stacking zone: Green cube at fixed position [0.0, 0.3]

**Motion Sequence (per cube):**
```python
# For cube 1:
1. control_gripper("open")
2. move_to_position(cube1_x, cube1_y, cube1_z + 0.15)  # Approach
3. move_to_position(cube1_x, cube1_y, cube1_z + 0.015)  # Grasp
4. control_gripper("close")
5. move_to_position(cube1_x, cube1_y, cube1_z + 0.15)  # Lift
6. move_to_position(0.0, 0.3, green_z + 0.055)  # Move to stack zone
7. control_gripper("open")  # Release
8. move_to_position(0.0, 0.3, green_z + 0.15)  # Retract

# Repeat for cube 2 (stack on cube 1)
```

**Success Condition:** Both red cubes stacked on green cube

**Key Parameters:**
- Approach height: 0.15m [0.10-0.20m]
- Grasp offset: 0.015m [0.01-0.03m]
- Stack offset: 0.055m [0.05-0.07m]
- Stack zone XY: [0.0, 0.3]

**State Tracking:** Must track which cubes have been stacked to adjust stack height

---

## Control Strategy

### Action Space

All tasks use **MoveArmThenGripper** action mode:

```python
MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(),
    gripper_action_mode=Discrete()
)
```

**Action Format:**
- Position control: (x, y, z) in meters, base frame coordinates
- Gripper states: "open" (1.0) or "close" (0.0)
- Orientation: Maintained automatically by IK solver

### Motion Planning Principles

1. **Approach from above:** Always move to approach height before descending
2. **Grasp with offset:** Add small offset (0.015m) above object for safe grasping
3. **Retract before moving:** Lift object before lateral movement
4. **Sequential execution:** Complete one motion before starting next

### Ground Truth Usage

**Important:** Ground truth is available via `get_target_position()` but should ONLY be used for:
- Validation and debugging
- Comparing detected vs actual positions
- Evaluating perception accuracy

**Never use ground truth for actual task execution** - this violates the perception-based requirement.

---

## Perception System

### GroundingDINO Integration

The framework uses GroundingDINO for open-vocabulary object detection:

**Key Features:**
- Text-prompted detection (no training required)
- Multi-object detection with "." separator (e.g., "red cube . red sphere")
- Confidence scores for each detection
- 2D bounding boxes in image space

### 3D Position Extraction

**Pipeline:**
1. GroundingDINO detects 2D bounding box in RGB image
2. Extract depth values within bounding box
3. Filter outliers and compute median depth
4. Project to 3D using camera intrinsics
5. Transform from camera frame to base frame

**Camera Intrinsics (RLBench):**
- Resolution: 512x512
- Focal length: fx = fy (varies by camera)
- Principal point: cx ≈ 256, cy ≈ 256

### Color Verification (Optional)

For improved accuracy, HSV color verification can be applied:
- Convert detected region to HSV color space
- Check if dominant color matches expected color
- Reject detections with wrong colors

---

## Implementation Patterns

### Task Loading Pattern

```python
# 1. Load task
result = load_task("TaskName")

# 2. Get camera observation
obs = get_camera_observation()
# Returns: RGB paths, depth paths, detection_prompt

# 3. Get ground truth (reference only)
gt = get_target_position()
# Returns: Actual object position for validation
```

### Detection Pattern

```python
# Single object
result = detect_object_3d(
    detection_prompt="red sphere",
    rgb_paths=[...],
    depth_paths=[...],
    point_cloud_paths=[...]
)
# Returns: objects[0] with position_3d

# Multi-object
result = detect_object_3d(
    detection_prompt="red cube . red sphere",
    rgb_paths=[...],
    depth_paths=[...],
    point_cloud_paths=[...]
)
# Returns: objects[0] = cube, objects[1] = sphere
```

### Motion Pattern

```python
# Approach → Grasp → Move sequence
control_gripper("open")
move_to_position(x, y, z + approach_height)  # Safe approach
move_to_position(x, y, z + grasp_offset)     # Grasp position
control_gripper("close")                      # Grasp object
move_to_position(x, y, z + approach_height)  # Lift up
move_to_position(target_x, target_y, target_z)  # Move to destination
control_gripper("open")                      # Release (if needed)
```

---

## Adding New Tasks

To implement a new RLBench task:

### 1. Understand Task Requirements

- Read RLBench task source code (`rlbench/tasks/<task_name>.py`)
- Identify: objects, success conditions, gripper strategy
- Determine detection strategy (single vs multi-object)

### 2. Update Agent Instructions

In `agent.py`, add to `MOTION_SEQUENCES`:
```python
### TaskName (Gripper: sequence)
**Detection prompt**
```
1. control_gripper(state)
2. move_to_position(...)
...
```
```

### 3. Update Detection Strategy

In `DETECTION_STRATEGY` table, specify:
- Detection prompt format
- Position source (single object vs objects[] array)

### 4. Update Task Loading

In `rlbench_orchestration_server.py`:
- Add task name to `SUPPORTED_TASKS`
- Update task parsing in `parse_task_objects()` if needed

### 5. Define Parameters

Add adjustable parameters to planning template:
- Approach heights
- Grasp offsets
- Task-specific parameters

### 6. Test and Iterate

- Test with ground truth comparison
- Validate detection accuracy
- Adjust motion parameters as needed

---

## Key Technical Details

### Environment Configuration

```python
# Observation Config
obs_config.set_all_low_dim(True)  # Required for control
obs_config.set_all_high_dim(False)  # Disable by default
obs_config.front_camera.rgb = True  # Enable RGB
obs_config.front_camera.depth = True  # Enable depth
obs_config.front_camera.point_cloud = True  # Enable point cloud
obs_config.front_camera.image_size = [512, 512]
```

### Environment Variables

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export COPPELIASIM_DISABLE_FFMPEG=1  # Prevent FFmpeg issues
```

### MCP Tools Available

**Sensing:**
- `load_task(task_name)` - Initialize RLBench task
- `get_camera_observation()` - Capture RGB, depth, detection prompt
- `get_target_position()` - Get ground truth (validation only)

**Perception:**
- `detect_object_3d(prompt, rgb_paths, depth_paths, pcd_paths)` - 3D object detection

**Motion:**
- `move_to_position(x, y, z)` - Cartesian position control
- `control_gripper(state)` - Gripper control ("open"/"close")

---

## Common Issues and Solutions

### Issue 1: Detection Outside Workspace

**Symptom:** "Target is outside of workspace" error

**Cause:** Incorrect coordinate transformation from camera to base frame

**Solution:**
- Verify camera extrinsics are correctly applied
- Use ground truth comparison to identify transformation error
- Check coordinate frame conventions (X-forward, Y-left, Z-up)

### Issue 2: Low Detection Confidence

**Symptom:** Objects not detected or low confidence scores

**Solution:**
- Adjust GroundingDINO thresholds (box_threshold, text_threshold)
- Improve detection prompts (add color and shape)
- Enable HSV color verification

### Issue 3: Gripper Not Grasping

**Symptom:** Gripper closes but doesn't grasp object

**Solution:**
- Increase grasp_offset for better positioning
- Ensure approach height is sufficient
- Check object is within gripper reach

### Issue 4: Multi-Object Detection Order

**Symptom:** Objects detected in wrong order

**Solution:**
- Detection order follows prompt order ("A . B" → objects[0]=A, objects[1]=B)
- Verify prompt matches expected order
- Use confidence scores to validate detections

---

## Performance Benchmarks

Based on our implementation:

| Task | Success Rate | Avg Steps | Key Challenge |
|------|-------------|-----------|---------------|
| ReachTarget | High | 3 | Detection accuracy |
| PushButton | High | 3 | Button alignment |
| PickAndLift | Medium | 5 | Multi-object detection |
| PutRubbishInBin | Medium | 9 | Trash detection (deformable) |
| StackBlocks | Medium | 16 | State tracking, precision |

---

## Academic Validation

For research replication, ensure:

1. **Task-agnostic:** Framework works across multiple tasks without task-specific hardcoding
2. **Perception-based:** Uses camera observations, ground truth only for validation
3. **Multi-agent:** Clear separation between sensing, perception, and motion agents
4. **Benchmarked:** Evaluated on standard RLBench tasks for comparability
5. **Success metrics:** Report success rates over multiple trials

---

## References

**RLBench Tasks:**
- Task source code: `/RLBench/rlbench/tasks/`
- Documentation: https://github.com/stepjam/RLBench

**Detection Model:**
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Paper: arXiv:2303.05499

**Framework:**
- Google ADK: https://github.com/google/adk
- MCP Protocol: https://github.com/modelcontextprotocol

---

## Quick Start for Replication

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up CoppeliaSim:**
   ```bash
   export COPPELIASIM_ROOT=/path/to/CoppeliaSim
   ```

3. **Run a task:**
   ```bash
   python -m google.adk.ui.web --agent-module multi_tool_agent.agent --agent-name root_agent
   ```

4. **Execute task:**
   - Navigate to http://localhost:8000
   - Enter: "Pick up the red cube and lift it"
   - Review plan
   - Type "approved" to execute

5. **Monitor execution:**
   - Sensing phase: Camera data captured
   - Perception phase: Objects detected with positions
   - Motion phase: Robot executes motion sequence

---

**Last Updated:** 2026-01-12

**Status:** Production-ready - 5 tasks implemented and validated
