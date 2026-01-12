# RLBench Task Implementation Guide

A practical guide for replicating the multi-agent framework's RLBench task implementations.

---

## Overview

The framework implements 5 RLBench tasks using multi-agent orchestration with open-vocabulary object detection. Pattern: **Planning → Approval → Sensing → Perception → Motion**.

---

## Implemented Tasks

| Task | Complexity | Objects | Gripper | Detection Prompt |
|------|-----------|---------|---------|------------------|
| ReachTarget | LOW | 1 sphere | Open | "red sphere" |
| PushButton | LOW | 1 button | Open | "button" |
| PickAndLift | MEDIUM | 2 objects | Open→Close→Closed | "red cube . red sphere" |
| PutRubbishInBin | MEDIUM | 2 objects | Open→Close→Open | "trash . bin" |
| StackBlocks | HIGH | 8 cubes | Multiple cycles | "red cube" |

---

## Task Specifications

### 1. ReachTarget
**Objective:** Touch colored target sphere with end-effector

**Motion:**
```python
control_gripper("open")
move_to_position(target_x, target_y, target_z + 0.10)  # Approach
move_to_position(target_x, target_y, target_z)  # Touch
```

**Parameters:** Approach height: 0.10m [0.05-0.15m]

---

### 2. PickAndLift
**Objective:** Pick cube and lift to sphere position

**Detection:** "red cube . red sphere" → objects[0]=cube, objects[1]=sphere

**Motion:**
```python
control_gripper("open")
move_to_position(cube_x, cube_y, cube_z + 0.15)  # Approach
move_to_position(cube_x, cube_y, cube_z + 0.015)  # Grasp height
control_gripper("close")
move_to_position(sphere_x, sphere_y, sphere_z)  # Lift to sphere
```

**Parameters:** Approach: 0.15m [0.10-0.20m], Grasp offset: 0.015m [0.01-0.03m]

**Critical:** Use detected sphere position, NOT ground truth

---

### 3. PushButton
**Objective:** Push button with end-effector

**Motion:**
```python
control_gripper("open")
move_to_position(button_x, button_y, button_z + 0.10)  # Approach
move_to_position(button_x, button_y, button_z - 0.002)  # Push
```

**Parameters:** Approach: 0.10m [0.05-0.15m], Push depth: 0.002m [0.001-0.005m]

---

### 4. PutRubbishInBin
**Objective:** Pick trash and place in bin

**Detection:** "trash . bin" → objects[0]=trash, objects[1]=bin

**Motion:**
```python
control_gripper("open")
move_to_position(trash_x, trash_y, trash_z + 0.15)  # Approach
move_to_position(trash_x, trash_y, trash_z + 0.015)  # Grasp
control_gripper("close")
move_to_position(trash_x, trash_y, trash_z + 0.15)  # Lift
move_to_position(bin_x, bin_y, bin_z + 0.15)  # Move to bin
move_to_position(bin_x, bin_y, bin_z + 0.10)  # Position over bin
control_gripper("open")  # Release
move_to_position(bin_x, bin_y, bin_z + 0.15)  # Retract
```

**Parameters:** Approach: 0.15m, Grasp offset: 0.015m, Drop height: 0.10m

---

### 5. StackBlocks
**Objective:** Stack 2 red cubes on green cube

**Detection:** "red cube" → Select 2 highest confidence red cubes, green at [0.0, 0.3]

**Motion (per cube):**
```python
control_gripper("open")
move_to_position(cube_x, cube_y, cube_z + 0.15)  # Approach
move_to_position(cube_x, cube_y, cube_z + 0.015)  # Grasp
control_gripper("close")
move_to_position(cube_x, cube_y, cube_z + 0.15)  # Lift
move_to_position(0.0, 0.3, green_z + 0.055)  # Stack position
control_gripper("open")
move_to_position(0.0, 0.3, green_z + 0.15)  # Retract
```

**Parameters:** Approach: 0.15m, Grasp offset: 0.015m, Stack offset: 0.055m

**Note:** Track stacked cubes to adjust height for second cube

---

## Control Strategy

**Action Mode:** MoveArmThenGripper with EndEffectorPoseViaIK

**Action Format:**
- Position: (x, y, z) in meters, base frame
- Gripper: "open" (1.0) or "close" (0.0)
- Orientation: Maintained by IK solver

**Motion Principles:**
1. Approach from above (safe height)
2. Grasp with offset (0.015m above object)
3. Retract before moving laterally
4. Sequential execution

**Ground Truth:** Available via `get_target_position()` but use ONLY for validation, not execution

---

## Perception System

**GroundingDINO:** Text-prompted object detection
- Single object: "red sphere"
- Multi-object: "red cube . red sphere" (separator: ".")
- Returns: 2D bounding boxes + confidence scores

**3D Position Extraction:**
1. Detect 2D bounding box (GroundingDINO)
2. Extract depth within box
3. Project to 3D (camera intrinsics)
4. Transform to base frame (camera extrinsics)

**Camera Config:** 512x512 resolution, fx=fy (varies), cx≈256, cy≈256

---

## Implementation Patterns

### Task Loading
```python
load_task("TaskName")
obs = get_camera_observation()  # RGB, depth, detection_prompt
gt = get_target_position()  # Ground truth (validation only)
```

### Detection
```python
# Single object
detect_object_3d("red sphere", rgb_paths, depth_paths, pcd_paths)
# Returns: objects[0] with position_3d

# Multi-object
detect_object_3d("red cube . red sphere", rgb_paths, depth_paths, pcd_paths)
# Returns: objects[0]=cube, objects[1]=sphere
```

### Motion
```python
control_gripper("open")
move_to_position(x, y, z + 0.15)  # Approach
move_to_position(x, y, z + 0.015)  # Grasp
control_gripper("close")
move_to_position(x, y, z + 0.15)  # Lift
move_to_position(target_x, target_y, target_z)  # Move
control_gripper("open")  # Release
```

---

## Adding New Tasks

1. **Read RLBench source:** `rlbench/tasks/<task_name>.py`
2. **Update agent.py:** Add motion sequence to `MOTION_SEQUENCES`
3. **Update detection:** Add to `DETECTION_STRATEGY` table
4. **Update server:** Add to `SUPPORTED_TASKS` in `rlbench_orchestration_server.py`
5. **Define parameters:** Add adjustable parameters to planning template
6. **Test:** Validate with ground truth comparison

---

## Environment Setup

### Configuration
```python
obs_config.set_all_low_dim(True)  # Required
obs_config.front_camera.rgb = True
obs_config.front_camera.depth = True
obs_config.front_camera.point_cloud = True
obs_config.front_camera.image_size = [512, 512]
```

### Environment Variables
```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export COPPELIASIM_DISABLE_FFMPEG=1
```

### MCP Tools
- **Sensing:** `load_task()`, `get_camera_observation()`, `get_target_position()`
- **Perception:** `detect_object_3d()`
- **Motion:** `move_to_position()`, `control_gripper()`

---

## Common Issues

**Detection outside workspace:** Check coordinate transformation (camera→base frame)

**Low detection confidence:** Adjust GroundingDINO thresholds, improve prompts

**Gripper not grasping:** Increase grasp_offset, verify approach height

**Multi-object order:** Follows prompt order ("A . B" → objects[0]=A, objects[1]=B)

---

## Performance

| Task | Success Rate | Steps | Key Challenge |
|------|-------------|-------|---------------|
| ReachTarget | High | 3 | Detection accuracy |
| PushButton | High | 3 | Button alignment |
| PickAndLift | Medium | 5 | Multi-object detection |
| PutRubbishInBin | Medium | 9 | Deformable object |
| StackBlocks | Medium | 16 | Precision stacking |

---

## Quick Start

1. **Install:** `pip install -r requirements.txt`
2. **Setup:** `export COPPELIASIM_ROOT=/path/to/CoppeliaSim`
3. **Run:** `python -m google.adk.ui.web --agent-module multi_tool_agent.agent --agent-name root_agent`
4. **Execute:** Navigate to http://localhost:8000, enter task (e.g., "Pick up the red cube"), review plan, type "approved"
5. **Monitor:** Sensing → Perception → Motion phases execute sequentially

---

## References

- **RLBench:** https://github.com/stepjam/RLBench
- **GroundingDINO:** https://github.com/IDEA-Research/GroundingDINO (arXiv:2303.05499)
- **Google ADK:** https://github.com/google/adk
- **MCP:** https://github.com/modelcontextprotocol

---

**Last Updated:** 2026-01-12 | **Status:** Production - 5 tasks validated
