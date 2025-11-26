# Multi-Agent Orchestration Plan - Task Agnostic

## Goal
Build platform-independent multi-agent orchestration for robot manipulation using:
- Planning → Human Approval → Sensing → Perception → Motion
- Works for multiple tasks: ReachTarget, PickAndLift
- NO demos - uses perception-based planning

## Architecture

### 1. Planning Agent
- **Input:** User task description ("reach the red ball", "pick and lift the blue cube")
- **Output:** Execution plan specifying which agents to use
- **Model:** DeepSeek R1 Reasoner

### 2. Human Approval
- **Method:** Display plan in adk web chat
- **User Action:** Approve/reject via chat
- **Proceed only after approval**

### 3. Sensing Agent
- **Tools:**
  - `get_camera_observation()` → Returns RGB + depth images
- **Output:** File paths to sensor data

### 4. Perception Agent
- **Tools:**
  - `detect_object_3d(text_prompt, rgb_path, depth_path)` → Returns 3D position [x, y, z]
  - Uses GroundingDINO for 2D detection + depth for 3D localization
- **Output:** Target object 3D coordinates

### 5. Motion Agent
- **Tools (task-specific):**

  **For ReachTarget:**
  - `move_to_position(x, y, z)` → Move gripper to target

  **For PickAndLift:**
  - `move_to_position(x, y, z)` → Reach object
  - `control_gripper(action)` → action = "close" or "open"
  - `lift_gripper(height)` → Lift by specified height

## Implementation Steps

### Phase 1: Update RLBench MCP Server
- [x] Remove demo-based tools
- [ ] Add `get_camera_observation()`
- [ ] Add `move_to_position(x, y, z)` with automatic IK/motion planning
- [ ] Add `control_gripper(action)`
- [ ] Add `lift_gripper(height)`

### Phase 2: Create Perception MCP Server
- [ ] Implement `detect_object_3d(text_prompt, rgb_path, depth_path)`
- [ ] Integrate GroundingDINO for 2D detection
- [ ] Use depth map to get 3D coordinates
- [ ] Return bounding box + 3D centroid

### Phase 3: Update Orchestrator
- [ ] Update tool filters for new RLBench tools
- [ ] Update Sensing Agent instructions
- [ ] Update Perception Agent instructions
- [ ] Update Motion Agent instructions
- [ ] Add task classification logic

### Phase 4: Testing
- [ ] Test ReachTarget end-to-end
- [ ] Test PickAndLift end-to-end
- [ ] Measure success rates
- [ ] Compare with MALMM baseline

## Key Design Decisions

1. **No Demos:** All tasks use perception-based planning
2. **Simple Motion:** Use direct Cartesian targets, let RLBench handle IK
3. **Task Agnostic:** Same pipeline, different tool sequences
4. **Platform Independent:** MCP abstraction allows ROS ↔ RLBench ↔ Real Robot
5. **Human-in-Loop:** Planning requires approval before execution

## Expected Outcomes

- Validate multi-agent orchestration framework
- Demonstrate platform independence via MCP
- Show task generalization (ReachTarget + PickAndLift)
- Research contribution: MCP-based abstraction for robotic manipulation
